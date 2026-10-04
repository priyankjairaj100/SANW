"""Canonical single-caption text inference for the practical-v6 token family.

Every prefix and suffix call has batch dimension one and length equal to that
caption's end-of-text position plus one. No neighboring caption, dataset role,
gallery size, or image enters text encoding. Repeated token sequences reuse the
same returned float32 learned-minus-reference vector.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def normalized_cache(values):
    """Exactly one final float32 normalization, matching the original evaluator."""
    return F.normalize(torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)), dim=-1).numpy()


def encode_token_rows(tower, suffix_models, tokens, progress=None):
    """Encode tokens once through the frozen prefix and all selected suffixes.

    Returns state-major delta/reference/learned arrays in original text order.
    Models and tower must be CPU float32, eval mode. No gradients are retained.
    The optional progress callback receives counts only, never test outcomes.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    if tokens.ndim != 2 or tokens.shape[1] != 77 or not len(tokens):
        raise ValueError("Expected nonempty [caption,77] CLIP token IDs")
    tower.eval()
    for model in suffix_models:
        model.eval()
    if not suffix_models:
        raise ValueError("At least one selected suffix is required")
    reference_state = suffix_models[0].state_dict()
    for model in suffix_models[1:]:
        state = model.state_dict()
        for name, value in reference_state.items():
            if name.startswith("reference_") or name == "projection":
                if name not in state or not torch.equal(value, state[name]):
                    raise ValueError("Suffixes do not share the same frozen reference")
    dimension = int(tower.text_projection.shape[1])
    delta = np.empty((len(suffix_models), len(tokens), dimension), dtype=np.float32)
    learned, reference = np.empty_like(delta), np.empty_like(delta)
    seen = {}
    prefix_hasher = hashlib.sha256()
    with torch.inference_mode():
        for index in range(len(tokens)):
            key = tokens[index].numpy().tobytes()
            if key in seen:
                original = seen[key]
                delta[:, index], learned[:, index], reference[:, index] = delta[:, original], learned[:, original], reference[:, original]
                continue
            prefix, ends = tower.prefix(tokens[index:index + 1], trim=True)
            if prefix.shape[0] != 1 or prefix.shape[1] != int(ends[0]) + 1:
                raise ValueError("Noncanonical prefix shape")
            prefix_hasher.update(key)
            prefix_hasher.update(prefix.numpy().tobytes())
            anchor = suffix_models[0]
            old = anchor._embed(prefix, ends, anchor.reference_block, anchor.reference_norm)
            for model_index, model in enumerate(suffix_models):
                new = model._embed(prefix, ends, model.block, model.final_norm)
                if new.shape != (1, dimension) or not torch.isfinite(new).all() or not torch.isfinite(old).all():
                    raise ValueError("Invalid suffix output")
                learned[model_index, index] = new[0].numpy()
                reference[model_index, index] = old[0].numpy()
                delta[model_index, index] = (new - old)[0].numpy()
            seen[key] = index
            if progress is not None and (len(seen) % 100 == 0 or index + 1 == len(tokens)):
                progress({"caption_rows_seen": index + 1, "unique_token_sequences": len(seen), "caption_rows": len(tokens)})
    return {"delta": delta, "learned": learned, "reference": reference,
            "unique_token_sequences": len(seen), "canonical_prefix_sha256": prefix_hasher.hexdigest()}


def paired_scores(images, baseline_texts, delta, epsilon):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite residual budget required")
    image = np.asarray(images, dtype=np.float64)
    baseline = np.asarray(baseline_texts, dtype=np.float64)
    difference = np.asarray(delta, dtype=np.float64)
    if image.shape != baseline.shape or image.shape != difference.shape:
        raise ValueError("Paired scoring shapes differ")
    return np.sum(image * baseline, axis=1) + epsilon * np.tanh(np.sum(image * difference, axis=1) / epsilon)


def score_matrix(images, baseline_texts, delta, epsilon):
    image = np.asarray(images, dtype=np.float64)
    baseline = np.asarray(baseline_texts, dtype=np.float64)
    difference = np.asarray(delta, dtype=np.float64)
    if baseline.shape != difference.shape or image.shape[1] != baseline.shape[1]:
        raise ValueError("Matrix scoring shapes differ")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite residual budget required")
    return image @ baseline.T + epsilon * np.tanh((image @ difference.T) / epsilon)
