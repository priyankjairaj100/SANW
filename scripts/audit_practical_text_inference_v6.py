#!/usr/bin/env python3
"""Independent canonical text inference from pinned tensors.

No production scorer, encoder, training, tokenizer, or evaluator is imported.
The audit reconstructs every transformer operation using PyTorch functions.
It verifies that every suffix reference equals the original pinned text tower.
Inputs are explicit token IDs and checkpoint tensor dictionaries, not labels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


# Independent copies of the declared scientific input identities.
PINNED_WEIGHTS = {
    "vit_b32": ("vit_b32_laion.safetensors", "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6", 605143316, 512, False),
    "rn50": ("RN50_openai.pt", "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762", 255827503, 1024, True),
}

BLOCK_KEYS = (
    "ln_1.weight", "ln_1.bias", "attn.in_proj_weight", "attn.in_proj_bias",
    "attn.out_proj.weight", "attn.out_proj.bias", "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias",
)


def _read_pinned_weights(encoder, model_directory):
    if encoder not in PINNED_WEIGHTS:
        raise ValueError("Unknown encoder")
    filename, expected_hash, expected_size, dimension, quick = PINNED_WEIGHTS[encoder]
    weight_path = Path(model_directory) / filename
    with weight_path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if weight_path.stat().st_size != expected_size or actual_hash != expected_hash:
        raise ValueError("Independent audit rejected unpinned pretrained weights")
    keys = {"token_embedding.weight", "positional_embedding", "ln_final.weight", "ln_final.bias", "text_projection"}
    keys.update(f"transformer.resblocks.{layer}.{key}" for layer in range(12) for key in BLOCK_KEYS)
    if encoder == "vit_b32":
        from safetensors import safe_open
        with safe_open(weight_path, framework="pt", device="cpu") as stored:
            tensors = {name: stored.get_tensor(name).float() for name in keys}
    else:
        script_archive = torch.jit.load(str(weight_path), map_location="cpu")
        stored = script_archive.state_dict()
        tensors = {name: stored[name].detach().float().clone() for name in keys}
        del script_archive, stored
    if tensors["text_projection"].shape != (512, dimension):
        raise ValueError("Pinned projection has an unexpected dimension")
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid pretrained tensor: {name}")
    return tensors, dimension, quick


def _block(value, parameters, prefix, mask, quick):
    """One pre-normalized attention/MLP block, reconstructed functionally."""
    get = lambda name: parameters[prefix + name]
    normalized = F.layer_norm(value, (512,), get("ln_1.weight"), get("ln_1.bias"), 1e-5)
    sequence = normalized.transpose(0, 1)
    # The floating causal mask disables MultiheadAttention's native fast path.
    # This functional call reproduces that declared attention operation without
    # constructing or calling the production TextBlock or TextTower classes.
    attention, _ = F.multi_head_attention_forward(
        query=sequence, key=sequence, value=sequence,
        embed_dim_to_check=512, num_heads=8,
        in_proj_weight=get("attn.in_proj_weight"), in_proj_bias=get("attn.in_proj_bias"),
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
        out_proj_weight=get("attn.out_proj.weight"), out_proj_bias=get("attn.out_proj.bias"),
        training=False, key_padding_mask=None, need_weights=False, attn_mask=mask,
        use_separate_proj_weight=False, average_attn_weights=True, is_causal=False,
    )
    value = value + attention.transpose(0, 1)
    normalized = F.layer_norm(value, (512,), get("ln_2.weight"), get("ln_2.bias"), 1e-5)
    hidden = F.linear(normalized, get("mlp.c_fc.weight"), get("mlp.c_fc.bias"))
    activated = hidden * torch.sigmoid(1.702 * hidden) if quick else F.gelu(hidden, approximate="none")
    return value + F.linear(activated, get("mlp.c_proj.weight"), get("mlp.c_proj.bias"))


def _validated_suffixes(checkpoint_states, pretrained):
    if not checkpoint_states:
        raise ValueError("At least one checkpoint tensor dictionary is required")
    keys = {"projection", "final_norm.weight", "final_norm.bias", "reference_norm.weight", "reference_norm.bias"}
    keys.update(prefix + key for prefix in ("block.", "reference_block.") for key in BLOCK_KEYS)
    result = []
    for state in checkpoint_states:
        if set(state) != keys:
            raise ValueError("Checkpoint suffix tensor keys differ from the declared architecture")
        copied = {}
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError(f"Suffix tensor must be CPU float32: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite suffix tensor: {name}")
            copied[name] = tensor.detach()
        for key in BLOCK_KEYS:
            expected = pretrained[f"transformer.resblocks.11.{key}"]
            if not torch.equal(copied[f"reference_block.{key}"], expected):
                raise ValueError(f"Frozen reference differs from pinned encoder: {key}")
            if copied[f"block.{key}"].shape != expected.shape:
                raise ValueError(f"Learned suffix shape differs: {key}")
        for key in ("weight", "bias"):
            if not torch.equal(copied[f"reference_norm.{key}"], pretrained[f"ln_final.{key}"]):
                raise ValueError("Frozen reference normalization differs from pinned encoder")
            if copied[f"final_norm.{key}"].shape != pretrained[f"ln_final.{key}"].shape:
                raise ValueError("Learned final normalization shape differs")
        if not torch.equal(copied["projection"], pretrained["text_projection"]):
            raise ValueError("Text projection differs from pinned encoder")
        result.append(copied)
    return result


def _suffix(prefix, end, mask, parameters, quick, *, reference):
    block_name = "reference_block." if reference else "block."
    norm_name = "reference_norm." if reference else "final_norm."
    value = _block(prefix, parameters, block_name, mask, quick)
    value = F.layer_norm(value, (512,), parameters[norm_name + "weight"], parameters[norm_name + "bias"], 1e-5)
    end_state = value[torch.arange(1), torch.tensor([end], dtype=torch.long)]
    projected = end_state @ parameters["projection"]
    return F.normalize(projected, p=2.0, dim=-1, eps=1e-12)


def independent_encode(encoder, model_directory: Path, checkpoint_states: list[dict], tokens: torch.Tensor) -> dict:
    """Return independent learned/reference/delta arrays shaped [state,N,D].

    ``tokens`` must be CPU int64 [N,77], with standard CLIP start/end tokens.
    Prefix computation uses exactly one caption, trimmed through its own EOT.
    Duplicate complete token sequences reuse identical outputs.
    The prefix hash covers each first-seen full token row and its prefix bytes.
    """
    torch.set_num_threads(1)
    tokens = torch.as_tensor(tokens)
    if tokens.device.type != "cpu" or tokens.dtype != torch.int64 or tokens.ndim != 2 or tokens.shape[1] != 77 or not len(tokens):
        raise ValueError("Expected nonempty CPU int64 token IDs shaped [N,77]")
    if torch.any(tokens < 0) or torch.any(tokens >= 49408):
        raise ValueError("Token ID outside the CLIP vocabulary")
    if not torch.all(tokens[:, 0] == 49406) or not torch.all((tokens == 49407).sum(dim=1) == 1):
        raise ValueError("Expected exactly one EOT and a leading SOT per caption")
    pretrained, dimension, quick = _read_pinned_weights(encoder, model_directory)
    states = _validated_suffixes(checkpoint_states, pretrained)
    output = {name: np.empty((len(states), len(tokens), dimension), dtype=np.float32)
              for name in ("delta", "learned", "reference")}
    seen, prefix_hasher = {}, hashlib.sha256()
    with torch.inference_mode():
        for index in range(len(tokens)):
            key = tokens[index].numpy().tobytes()
            if key in seen:
                for field in output:
                    output[field][:, index] = output[field][:, seen[key]]
                continue
            end = int(torch.nonzero(tokens[index] == 49407, as_tuple=False)[0, 0])
            caption = tokens[index:index + 1, :end + 1]
            mask = torch.full((end + 1, end + 1), float("-inf"), dtype=torch.float32).triu_(1)
            prefix = F.embedding(caption, pretrained["token_embedding.weight"]) + pretrained["positional_embedding"][:end + 1]
            for layer in range(11):
                prefix = _block(prefix, pretrained, f"transformer.resblocks.{layer}.", mask, quick)
            prefix_hasher.update(key)
            prefix_hasher.update(prefix.numpy().tobytes())
            old = _suffix(prefix, end, mask, states[0], quick, reference=True)
            for model_index, state in enumerate(states):
                new = _suffix(prefix, end, mask, state, quick, reference=False)
                if new.shape != (1, dimension) or not torch.isfinite(new).all() or not torch.isfinite(old).all():
                    raise ValueError("Independent suffix output is invalid")
                output["learned"][model_index, index] = new[0].numpy()
                output["reference"][model_index, index] = old[0].numpy()
                output["delta"][model_index, index] = (new - old)[0].numpy()
            seen[key] = index
    output["unique_token_sequences"] = len(seen)
    output["canonical_prefix_sha256"] = prefix_hasher.hexdigest()
    return output
