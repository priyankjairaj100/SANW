"""Text-only, pinned CLIP towers for a future last-block adaptation experiment.

This module never loads images or chooses a training or evaluation split.
Frozen prefix outputs are float32. Causal trimming only removes positions
strictly after each batch's last end-of-text token.

Architecture follows OpenCLIP 2.32.0's ResidualAttentionBlock and CLIP text
forward. The tokenizer is loaded from that installed wheel without importing
its unrelated vision dependencies. No vision tower is constructed here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


WEIGHTS = {
    "vit_b32": {
        "filename": "vit_b32_laion.safetensors",
        "sha256": "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6",
        "bytes": 605143316,
        "dimension": 512,
        "activation": "gelu",
        "source": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "revision": "1a25a446712ba5ee05982a381eed697ef9b435cf",
    },
    "rn50": {
        "filename": "RN50_openai.pt",
        "sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
        "bytes": 255827503,
        "dimension": 1024,
        "activation": "quick_gelu",
        "source": "OpenAI CLIP official RN50",
    },
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tokenizer():
    distribution = importlib.metadata.distribution("open_clip_torch")
    if distribution.version != "2.32.0":
        raise RuntimeError("Tokenizer requires OpenCLIP 2.32.0")
    path = distribution.locate_file("open_clip/tokenizer.py")
    spec = importlib.util.spec_from_file_location("sanw_open_clip_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleTokenizer()


class QuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


class TextBlock(nn.Module):
    def __init__(self, quick_gelu=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln_2 = nn.LayerNorm(512)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 2048)),
            ("gelu", QuickGELU() if quick_gelu else nn.GELU()),
            ("c_proj", nn.Linear(2048, 512)),
        ]))

    def forward(self, value, mask):
        normed = self.ln_1(value)
        value = value + self.attn(normed, normed, normed,
                                  attn_mask=mask, need_weights=False)[0]
        return value + self.mlp(self.ln_2(value))


class TextTower(nn.Module):
    def __init__(self, dimension, quick_gelu):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(77, 512))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([
            TextBlock(quick_gelu) for _ in range(12)
        ])
        self.ln_final = nn.LayerNorm(512)
        self.text_projection = nn.Parameter(torch.empty(512, dimension))
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("attention_mask", torch.full((77, 77), float("-inf")).triu_(1),
                             persistent=False)

    def prefix(self, tokens, trim=True):
        """Return frozen first-11-block activations and each EOT position."""
        ends = tokens.argmax(dim=-1)
        length = int(ends.max()) + 1 if trim else tokens.shape[1]
        if length > 77 or tokens.ndim != 2:
            raise ValueError("Expected a batch of CLIP token sequences")
        value = self.token_embedding(tokens[:, :length]) + self.positional_embedding[:length]
        mask = self.attention_mask[:length, :length]
        for block in self.transformer.resblocks[:-1]:
            value = block(value, mask)
        return value, ends

    def suffix(self, prefix, ends):
        """Last attention block, layer normalization, EOT pooling, projection."""
        length = prefix.shape[1]
        value = self.transformer.resblocks[-1](prefix, self.attention_mask[:length, :length])
        value = self.ln_final(value)
        value = value[torch.arange(len(value), device=value.device), ends]
        return F.normalize(value @ self.text_projection, dim=-1)

    def forward(self, tokens, trim=True):
        return self.suffix(*self.prefix(tokens, trim=trim))

    def enable_last_block(self, projection=False):
        """Expose a restricted future adaptation family; no optimizer is run."""
        self.requires_grad_(False)
        self.transformer.resblocks[-1].requires_grad_(True)
        self.ln_final.requires_grad_(True)
        self.text_projection.requires_grad_(projection)
        return [p for p in self.parameters() if p.requires_grad]


def load_text_tower(encoder, model_directory):
    identity = WEIGHTS[encoder]
    weight = Path(model_directory) / identity["filename"]
    if weight.stat().st_size != identity["bytes"] or digest(weight) != identity["sha256"]:
        raise RuntimeError("Encoder weight does not match the frozen scientific provenance")
    tower = TextTower(identity["dimension"], identity["activation"] == "quick_gelu")
    wanted = set(tower.state_dict())
    if encoder == "vit_b32":
        from safetensors import safe_open
        with safe_open(weight, framework="pt", device="cpu") as archive:
            state = {key: archive.get_tensor(key).float() for key in wanted}
    else:
        archive = torch.jit.load(str(weight), map_location="cpu")
        official_state = archive.state_dict()
        state = {key: official_state[key].float() for key in wanted}
        del archive, official_state
    tower.load_state_dict(state, strict=True)
    tower.eval().requires_grad_(False)
    return tower
