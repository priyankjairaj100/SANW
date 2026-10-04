"""Development-only last-text-block adaptation with one bounded canonical score.

Only the trainable last block and final layer normalization are updated. Frozen
images, token embeddings, first eleven blocks, and projection retain provenance.
The base uses the existing normalized feature archive. The correction subtracts
a paired frozen re-encoding, making initialization exactly zero despite cache
roundoff. Neither scoring nor parameterization depends on the gallery or task.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .practical_text_v6 import WEIGHTS, load_text_tower, digest
from .practical_training import (PracticalTrainingConfig, TrainingExamples,
                                 evaluate_development, input_paths, select_development_epoch)
from .review_training import SourceRetrievalPool
from .training import (FeatureDataset, atomic_json, atomic_torch_save,
                       canonical_json, seed_everything, sha256_file)


@dataclasses.dataclass(frozen=True)
class TextTrainingConfig(PracticalTrainingConfig):
    """Shared scientific loss settings plus CPU-friendly text minibatches."""
    learning_rate: float = 1e-5
    batch_size: int = 32
    threads: int = 2
    text_batch_size: int = 64

    def validate(self):
        super().validate()
        if self.text_batch_size < 1:
            raise ValueError("Positive text minibatch size required")


class LastTextBlock(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.block = copy.deepcopy(tower.transformer.resblocks[-1]).requires_grad_(True)
        self.final_norm = copy.deepcopy(tower.ln_final).requires_grad_(True)
        # These reference tensors never enter the optimizer. Their autograd
        # flags match the learned copy solely to force identical CPU kernels.
        # The complete reference output is detached before forming any loss.
        self.reference_block = copy.deepcopy(tower.transformer.resblocks[-1]).requires_grad_(True)
        self.reference_norm = copy.deepcopy(tower.ln_final).requires_grad_(True)
        self.register_buffer("projection", tower.text_projection.detach().clone())
        self.register_buffer("mask", tower.attention_mask.detach().clone(), persistent=False)

    def _embed(self, prefix, ends, block, norm):
        value = block(prefix, self.mask[:prefix.shape[1], :prefix.shape[1]])
        value = norm(value)[torch.arange(len(value)), ends]
        return F.normalize(value @ self.projection, dim=-1)

    def forward(self, prefix, ends):
        # Identical batching and arithmetic guarantee an exact zero at start.
        learned = self._embed(prefix, ends, self.block, self.final_norm)
        # PyTorch can select different kernels from parameter autograd flags.
        # Both copies use matching flags; the detached reference has no path
        # in the eventual loss graph and is excluded from the optimizer.
        reference_prefix = prefix.detach()
        reference = self._embed(reference_prefix, ends, self.reference_block, self.reference_norm).detach()
        return learned - reference

    def learned_named_parameters(self):
        return [(name, parameter) for name, parameter in self.named_parameters()
                if not name.startswith("reference_")]


class PrefixCache:
    def __init__(self, directory, encoder, verify=True):
        self.directory = Path(directory)
        self.receipt = json.loads((self.directory / "receipt.json").read_text())
        if self.receipt["encoder"] != encoder or not self.receipt["test_and_calibration_excluded"]:
            raise ValueError("Wrong or impermissible prefix cache")
        if self.receipt["weight_identity"] != WEIGHTS[encoder]:
            raise ValueError("Prefix cache does not match the pinned encoder weights")
        if verify:
            for name, record in self.receipt["files"].items():
                path = self.directory / name
                if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"Prefix cache changed: {name}")
        self.values = np.load(self.directory / "prefix_values.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.directory / "offsets.npy", allow_pickle=False)
        self.tokens = np.load(self.directory / "tokens.npy", mmap_mode="r", allow_pickle=False)
        self.keys = np.load(self.directory / "keys.npy", allow_pickle=False).tolist()
        self.teacher = np.load(self.directory / "teacher_features.npy", mmap_mode="r", allow_pickle=False)
        self.lengths = np.diff(self.offsets)
        self.key_to_index = {key: index for index, key in enumerate(self.keys)}
        self.feature_to_index = {}
        for index, value in enumerate(self.teacher):
            key = value.tobytes()
            if key in self.feature_to_index:
                previous = self.feature_to_index[key]
                if not np.array_equal(self.tokens[previous], self.tokens[index]):
                    raise ValueError("Distinct tokenizations collided in a frozen feature vector")
            else:
                self.feature_to_index[key] = index

    def indices_for_vectors(self, values):
        if values.dtype != torch.float32 or values.device.type != "cpu":
            raise ValueError("Canonical frozen text features must be float32 CPU")
        return [self.feature_to_index[value.tobytes()] for value in values.detach().numpy()]

    def encode(self, model, indices, batch_size):
        """Unique length-sorted sequences; autograd tracks only the suffix."""
        indices = sorted(set(indices), key=lambda i: (int(self.lengths[i]), i))
        parts = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            length = max(int(self.lengths[index]) for index in batch)
            values = torch.zeros(len(batch), length, 512)
            for row, index in enumerate(batch):
                values[row, :self.lengths[index]] = torch.from_numpy(
                    np.array(self.values[self.offsets[index]:self.offsets[index + 1]], copy=True))
            ends = torch.tensor([int(self.lengths[index]) - 1 for index in batch])
            parts.append(model(values, ends))
        if not parts:
            raise ValueError("Cannot encode an empty caption set")
        return PreparedTextScores(self, indices, torch.cat(parts), epsilon=None)


class PreparedTextScores(nn.Module):
    """Scorer adapter for the shared loss/evaluator after one text encoding."""
    def __init__(self, cache, indices, delta, epsilon):
        super().__init__()
        self.cache, self.delta, self.epsilon = cache, delta, epsilon
        self.positions = {index: offset for offset, index in enumerate(indices)}
        self._retrieval_matrix = None
        self._retrieval_key = None

    def differences(self, texts):
        indices = self.cache.indices_for_vectors(texts)
        return self.delta[[self.positions[index] for index in indices]]

    def score_pairs(self, images, texts):
        if self.epsilon is None or self.epsilon <= 0:
            raise ValueError("Set the strictly positive score budget first")
        if len(images) != len(texts):
            raise ValueError("Paired scoring requires equal batch lengths")
        delta = self.differences(texts).double()
        baseline = (images.double() * texts.double()).sum(dim=-1)
        change = (images.double() * delta).sum(dim=-1)
        return baseline + self.epsilon * torch.tanh(change / self.epsilon)

    def score_matrix(self, images, texts, pair_chunk=None):
        if self.epsilon is None or self.epsilon <= 0:
            raise ValueError("Set the strictly positive score budget first")
        delta = self.differences(texts).double()
        baseline = images.double() @ texts.double().T
        change = images.double() @ delta.T
        return baseline + self.epsilon * torch.tanh(change / self.epsilon)

    @torch.no_grad()
    def exact_topk(self, images, texts, k=1, direction="i2t", query_chunk=None, pair_chunk=None):
        # Both directions use this identical dense matrix. No learned shortlist.
        if direction not in ("i2t", "t2i"):
            raise ValueError("Unknown retrieval direction")
        key = (images.data_ptr(), texts.data_ptr(), tuple(images.shape), tuple(texts.shape))
        if self._retrieval_key != key:
            self._retrieval_matrix = self.score_matrix(images, texts)
            self._retrieval_key = key
        scores = self._retrieval_matrix if direction == "i2t" else self._retrieval_matrix.T
        if not 1 <= k <= scores.shape[1]:
            raise ValueError("Invalid top-k")
        if k == 1:
            indices = scores.argmax(dim=1, keepdim=True)
        else:
            indices = torch.argsort(scores, descending=True, stable=True, dim=1)[:, :k]
        return scores.gather(1, indices), indices


def training_caption_indices(train, local_indices):
    """All captions used by shared loss, encoded once per image minibatch."""
    source = {j for i in local_indices for j in train.source_by_image[i]}
    source.update(train.best_source[local_indices].tolist())
    source.update(train.i2t_negative[local_indices].reshape(-1).tolist())
    result = {train.source_global[j] for j in source}
    for i in local_indices:
        for group in train.composition[i]:
            result.update(group)
    return sorted(result)


def load_context(repository, cache_directory, config):
    paths = input_paths(repository, config.encoder)
    dimension = WEIGHTS[config.encoder]["dimension"]
    data = FeatureDataset(repository, paths["manifest"], paths["features"], paths["metadata"], dimension)
    pool = SourceRetrievalPool(repository, paths["development_manifest"], paths["development_features"],
                               paths["development_metadata"], data, dimension)
    data.images = F.normalize(data.images, dim=-1)
    data.texts = F.normalize(data.texts, dim=-1)
    pool.images = F.normalize(pool.images, dim=-1)
    pool.texts = F.normalize(pool.texts, dim=-1)
    cache = PrefixCache(cache_directory, config.encoder)
    for name, record in cache.receipt["inputs"].items():
        if digest(paths[name]) != record["sha256"]:
            raise ValueError(f"Prefix cache no longer matches run inputs: {name}")
    train = TrainingExamples(data, config.hard_negative_count)
    tower = load_text_tower(config.encoder, repository / "data/practical_v6_models")
    model = LastTextBlock(tower)
    del tower
    return paths, data, pool, cache, train, model


def run_text_training(repository, output, cache_directory, config, token_protocol=None):
    """Run only when a separate root decision authorizes this new experiment."""
    config.validate()
    repository, output = Path(repository).resolve(), Path(output).resolve()
    parent_protocol = repository / "results/practical_v6/protocol_v1.json"
    protocol_path = Path(token_protocol) if token_protocol is not None else repository / "results/practical_v6/token_protocol_v1.json"
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text())
    parent = json.loads(parent_protocol.read_text())
    if protocol["parent_protocol"]["sha256"] != digest(parent_protocol):
        raise ValueError("Token protocol parent identity changed")
    for name in ("practical_gate", "test_lock"):
        if protocol[name] != parent[name]:
            raise ValueError(f"Token protocol weakened inherited {name}")
    for name in ("composition_objective", "retrieval_objective", "selection", "neutral_policy"):
        if protocol["development"][name] != parent["development"][name]:
            raise ValueError(f"Token protocol changed shared development rule: {name}")
    for name, value in protocol["pilot_config"].items():
        if getattr(config, name) != value:
            raise ValueError(f"Configuration differs from token protocol: {name}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output already contains a run")
    seed_everything(config.seed, config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    paths, data, pool, cache, train, model = load_context(repository, cache_directory, config)
    parameters = [parameter for _, parameter in model.learned_named_parameters()]
    initial = {name: parameter.detach().clone() for name, parameter in model.learned_named_parameters()}
    source_names = ("src/gcr/practical_text_v6.py", "src/gcr/practical_text_training_v6.py",
                    "scripts/run_practical_text_training_v6.py", "scripts/cache_practical_text_prefix_v6.py",
                    "src/gcr/practical_training.py", "src/gcr/training.py", "src/gcr/review_training.py")
    tokenizer_distribution = importlib.metadata.distribution("open_clip_torch")
    tokenizer_files = {name: digest(tokenizer_distribution.locate_file("open_clip/" + name))
                       for name in ("tokenizer.py", "bpe_simple_vocab_16e6.txt.gz")}
    identity = {"schema": "sanw_practical_text_last_block_v1", "config": dataclasses.asdict(config),
                "weight_identity": WEIGHTS[config.encoder],
                "inputs": {name: {"path": str(path.relative_to(repository)), "sha256": digest(path)} for name, path in paths.items()},
                "prefix_cache_receipt": {"path": str(Path(cache_directory) / "receipt.json"), "sha256": digest(Path(cache_directory) / "receipt.json")},
                "source_sha256": {name: digest(repository / name) for name in source_names},
                "tokenizer": {"package": "open_clip_torch", "version": tokenizer_distribution.version, "files": tokenizer_files},
                "score": "s0 + epsilon*tanh(dot(frozen_image, normalized_new_text-normalized_reference_text)/epsilon)",
                "baseline": "float64 dot product of original float32-normalized cached features",
                "reference_policy": "frozen and learned suffix use identical first11block prefixes, length-sorted batching and float32 kernels",
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "optimizer_updates_before_epoch_zero": 0, "held_out_evaluation": "prohibited_by_runner",
                "loss_and_selection": "shared_practical_training_implementation_bound_by_source_hash",
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    identity["parent_development_protocol"] = {"path": str(parent_protocol.relative_to(repository)), "sha256": digest(parent_protocol)}
    identity["token_development_protocol"] = {"path": str(protocol_path.relative_to(repository)), "sha256": digest(protocol_path)}
    ledger_hash = hashlib.sha256(canonical_json(identity)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_hash})
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                  foreach=False, fused=False)
    dev_texts = sorted({j for i in data.split_indices["validation"] for j in data.pairs[i]})
    dev_indices = cache.indices_for_vectors(data.texts[dev_texts]) + cache.indices_for_vectors(pool.texts)
    permutation_generator = torch.Generator().manual_seed(config.seed)
    history, steps = [], 0
    started = time.monotonic()
    for epoch in range(config.epochs + 1):
        epoch_started = time.monotonic()
        train_summary = {}
        if epoch:
            model.train()
            permutation = torch.randperm(len(train.images), generator=permutation_generator).tolist()
            totals, total_images = {}, 0
            for offset in range(0, len(permutation), config.batch_size):
                batch = permutation[offset:offset + config.batch_size]
                text_indices = training_caption_indices(train, batch)
                cache_indices = cache.indices_for_vectors(data.texts[text_indices])
                optimizer.zero_grad(set_to_none=True)
                scorer = cache.encode(model, cache_indices, config.text_batch_size)
                scorer.epsilon = config.epsilon
                loss, values = train.batch_loss(scorer, batch, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite text adaptation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                total_images += len(batch)
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value * len(batch)
            train_summary = {name: value / total_images for name, value in totals.items()}
        checkpoint = output / "checkpoints" / f"epoch_{epoch:02d}.pt"
        update_norm = float(torch.sqrt(sum((parameter.detach() - initial[name]).double().square().sum()
                                          for name, parameter in model.learned_named_parameters())))
        atomic_torch_save(checkpoint, {"schema": "sanw_practical_text_last_block_v1", "state_dict": model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict(), "config": dataclasses.asdict(config),
                                      "weight_identity": WEIGHTS[config.encoder], "epoch": epoch, "optimizer_steps": steps,
                                      "update_norm": update_norm, "ledger_sha256": ledger_hash,
                                      "torch_rng_state": torch.get_rng_state(), "permutation_rng_state": permutation_generator.get_state()})
        model.eval()
        with torch.inference_mode():
            scorer = cache.encode(model, dev_indices, config.text_batch_size)
            scorer.epsilon = config.epsilon
            if epoch == 0 and bool(scorer.delta.ne(0).any()):
                raise RuntimeError("Initial learned and reference encodings are not exactly identical")
            composition, retrieval = evaluate_development(scorer, data, pool, output, epoch, config)
            maximum_delta = float(scorer.delta.abs().max())
        row = {"epoch": epoch, "optimizer_steps": steps, "update_norm": update_norm,
               "training": train_summary, "composition": composition, "retrieval": retrieval,
               "maximum_development_embedding_component_change": maximum_delta,
               "checkpoint": {"path": str(checkpoint.relative_to(output)), "sha256": digest(checkpoint)},
               "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        atomic_json(output / "history.json", {"ledger_sha256": ledger_hash, "epochs": history})
        atomic_json(output / "selection.json", select_development_epoch(history))
        print(json.dumps({"event": "text_epoch_complete", "encoder": config.encoder, "epoch": epoch,
                          "optimizer_steps": steps, "paired_joint_accuracy": composition["paired_joint_accuracy"],
                          "i2t_r1": retrieval["i2t_r1"], "t2i_r1": retrieval["t2i_r1"],
                          "epoch_seconds": row["epoch_seconds"]}), flush=True)
    completion = {"status": "completed_development_only", "ledger_sha256": ledger_hash,
                  "epochs": config.epochs, "optimizer_steps": steps, "selection": select_development_epoch(history),
                  "history_sha256": digest(output / "history.json"), "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "completion.json", completion)
    return completion
