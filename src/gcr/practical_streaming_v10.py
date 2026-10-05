"""Exact full-gallery v9 mathematics with disk-backed frozen scores.

This module changes storage and reduction order, not the learning objective.
Every query retains every training-gallery candidate. Float64 final rankings use
the existing canonical pair scorer after conservative numerical screening.
The fitted scorer, joint relation loss, and stochastic fitting loop stay in the
unchanged v8/v9 modules. No development or test data are loaded here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .practical_constrained_v8 import Edge, FullGalleryConstraints
from .practical_constrained_evaluation_v8 import canonical_pair_scores, canonical_project


def _array(value, name):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _hash_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    raw = memoryview(value).cast("B")
    for offset in range(0, len(raw), 16 << 20):
        digest.update(raw[offset:offset + (16 << 20)])
    return digest.hexdigest()


def _hash_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            result.update(block)
    return result.hexdigest()


class FrozenScoreCache:
    """Two immutable float64 NPY orientations; no resident NxM copies required."""

    @classmethod
    def create(cls, directory, images, texts, owner, block_size=128):
        directory = Path(directory)
        images, texts = _array(images, "images"), _array(texts, "texts")
        owner = np.ascontiguousarray(owner, dtype=np.int64)
        if not len(images) or not len(texts) or images.shape[1] != texts.shape[1] or owner.shape != (len(texts),):
            raise ValueError("Frozen-cache dimensions disagree")
        if np.any(owner < 0) or np.any(owner >= len(images)) or not np.all(np.bincount(owner, minlength=len(images))):
            raise ValueError("Every gallery caption needs one valid owner and every image needs a caption")
        if block_size < 1 or int(block_size) != block_size:
            raise ValueError("Positive integer cache block size required")
        if directory.exists():
            if not (directory / "metadata.json").exists():
                raise FileExistsError("Refusing to reuse an incomplete frozen-score cache")
            return cls.open(directory, images, texts, owner)
        directory.mkdir(parents=True)
        build_started = time.monotonic()
        i2t_path, t2i_path = directory / "image_to_text.npy", directory / "text_to_image.npy"
        i2t = np.lib.format.open_memmap(i2t_path, mode="w+", dtype=np.float64, shape=(len(images), len(texts)))
        t2i = np.lib.format.open_memmap(t2i_path, mode="w+", dtype=np.float64, shape=(len(texts), len(images)))
        positive = np.sum(images[owner] * texts, axis=1, dtype=np.float64)
        best = np.empty(len(images), dtype=np.int64)
        for i in range(len(images)):
            columns = np.flatnonzero(owner == i)
            best[i] = columns[int(np.argmax(positive[columns]))]
        best_score = positive[best]
        # Match the v9 correction of near-zero ownership margins, preserving
        # exact tied frozen-correct queries under the canonical reduction order.
        corrected_pairs = 0
        for lo in range(0, len(images), block_size):
            hi = min(lo + block_size, len(images))
            block = images[lo:hi] @ texts.T
            owned_columns = np.flatnonzero((owner >= lo) & (owner < hi))
            block[owner[owned_columns] - lo, owned_columns] = positive[owned_columns]
            near = ((np.abs(block - best_score[lo:hi, None]) < 1e-10)
                    | (np.abs(block - positive[None, :]) < 1e-10))
            rows, columns = np.nonzero(near)
            for start in range(0, len(rows), 4096):
                rr, cc = rows[start:start + 4096], columns[start:start + 4096]
                block[rr, cc] = np.sum(images[lo + rr] * texts[cc], axis=1, dtype=np.float64)
            corrected_pairs += len(rows)
            i2t[lo:hi] = block
            t2i[:, lo:hi] = block.T
        i2t.flush(); t2i.flush()
        del i2t, t2i
        np.savez_compressed(directory / "ownership.npz", owner=owner, best_positive=best, positive_score=positive)
        names = ("image_to_text.npy", "text_to_image.npy", "ownership.npz")
        metadata = {"schema": "sanw_frozen_score_cache_v10", "images": len(images), "texts": len(texts),
                    "dimension": images.shape[1], "dtype": "float64", "block_size": block_size,
                    "image_features_sha256": _hash_array(images), "text_features_sha256": _hash_array(texts),
                    "owner_sha256": _hash_array(owner), "near_zero_pairs_canonically_corrected": corrected_pairs,
                    "files": {name: {"sha256": _hash_file(directory / name), "bytes": (directory / name).stat().st_size}
                              for name in names}, "build_seconds": time.monotonic() - build_started,
                    "scope": "specified_training_gallery_only", "gallery_order_preserved": True}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return cls.open(directory, images, texts, owner)

    @classmethod
    def open(cls, directory, images, texts, owner):
        obj = cls()
        obj.directory = Path(directory)
        obj.metadata = json.loads((obj.directory / "metadata.json").read_text())
        metadata = obj.metadata
        if metadata.get("schema") != "sanw_frozen_score_cache_v10":
            raise ValueError("Wrong frozen-cache schema")
        for key, value in (("image_features_sha256", np.ascontiguousarray(images, dtype=np.float64)),
                           ("text_features_sha256", np.ascontiguousarray(texts, dtype=np.float64)),
                           ("owner_sha256", np.ascontiguousarray(owner, dtype=np.int64))):
            if metadata.get(key) != _hash_array(value):
                raise ValueError("Frozen cache input identity differs")
        for name, record in metadata["files"].items():
            path = obj.directory / name
            if path.stat().st_size != record["bytes"] or _hash_file(path) != record["sha256"]:
                raise ValueError("Frozen cache content hash differs")
        obj.i2t = np.load(obj.directory / "image_to_text.npy", mmap_mode="r", allow_pickle=False)
        obj.t2i = np.load(obj.directory / "text_to_image.npy", mmap_mode="r", allow_pickle=False)
        if (obj.i2t.shape != (len(images), len(texts)) or obj.t2i.shape != (len(texts), len(images))
                or obj.i2t.dtype != np.float64 or obj.t2i.dtype != np.float64):
            raise ValueError("Frozen-cache orientation shape differs")
        with np.load(obj.directory / "ownership.npz", allow_pickle=False) as z:
            obj.owner, obj.best_positive, obj.positive_score = z["owner"], z["best_positive"], z["positive_score"]
        if (not np.array_equal(obj.owner, owner) or obj.best_positive.shape != (len(images),)
                or obj.positive_score.shape != (len(texts),) or not np.isfinite(obj.positive_score).all()):
            raise ValueError("Frozen-cache ownership differs")
        return obj

    def image_rows(self, indices):
        return np.asarray(self.i2t[indices])

    def validate_inputs(self, images, texts, owner):
        for key, value in (("image_features_sha256", images), ("text_features_sha256", texts),
                           ("owner_sha256", np.ascontiguousarray(owner, dtype=np.int64))):
            if self.metadata.get(key) != _hash_array(value):
                raise ValueError("Frozen cache input identity differs")

    def text_columns(self, indices):
        # Transposed disk orientation makes each text query contiguous on disk.
        return np.asarray(self.t2i[indices]).T


def cached_exact_retrieval(images, texts, owner, cache, scorer=None, query_block=64):
    """All-gallery retrieval with cached screening and canonical near-winners."""
    images, texts = _array(images, "images"), _array(texts, "texts")
    owner = np.asarray(owner, dtype=np.int64)
    cache.validate_inputs(images, texts, owner)
    if scorer is None:
        left, right = np.zeros((len(images), 1)), np.zeros((len(texts), 1))
    else:
        left = canonical_project(scorer.image_coordinates(images), scorer.coefficient)
        right = scorer.text_coordinates(texts)
    dimension, rank = images.shape[1], left.shape[1]
    if (not isinstance(query_block, int) or query_block < 1
            or (dimension + rank + 4) * np.finfo(np.float64).eps >= .001):
        raise ValueError("Invalid numerical-screen dimensions or block size")
    raw = {}
    for direction, queries, gallery, qlow, glow, base in (
        ("i2t", images, texts, left, right, cache.i2t),
        ("t2i", texts, images, right, left, cache.t2i),
    ):
        winners = np.empty(len(queries), dtype=np.int64)
        scores = np.empty(len(queries), dtype=np.float64)
        counts = np.empty(len(queries), dtype=np.int64)
        gallery_norm = float(np.linalg.norm(gallery, axis=1).max())
        gallery_low_norm = float(np.linalg.norm(glow, axis=1).max())
        for lo in range(0, len(queries), query_block):
            hi = min(lo + query_block, len(queries))
            q, ql = queries[lo:hi], qlow[lo:hi]
            approximate = np.asarray(base[lo:hi]) + ql @ glow.T
            guard = (32 * (dimension + rank + 4) * np.finfo(np.float64).eps
                     * (1 + np.linalg.norm(q, axis=1) * gallery_norm
                        + np.linalg.norm(ql, axis=1) * gallery_low_norm))
            if not np.isfinite(approximate).all() or not np.isfinite(guard).all():
                raise ValueError("Nonfinite full-gallery screening score or bound")
            for row in range(hi - lo):
                possible = np.flatnonzero(approximate[row] >= approximate[row].max() - 2 * guard[row])
                query = np.full(len(possible), lo + row, dtype=np.int64)
                ii, tt = (query, possible) if direction == "i2t" else (possible, query)
                canonical = canonical_pair_scores(images[ii], texts[tt], left[ii], right[tt])
                best = int(np.argmax(canonical))
                winners[lo + row], scores[lo + row], counts[lo + row] = possible[best], canonical[best], len(possible)
        raw[direction + "_top_indices"] = winners
        raw[direction + "_top_scores"] = scores
        raw[direction + "_candidate_counts"] = counts
        raw[direction + "_correct"] = (owner[winners] == np.arange(len(images)) if direction == "i2t" else winners == owner)
    return raw


class StreamingFullGalleryConstraints:
    """The v8 finite-training feasible set, scanned in bounded-memory blocks."""
    def __init__(self, images, source_texts, owner, scorer, cache, retention_fraction=.5, block_size=128):
        self.images, self.texts = _array(images, "images"), _array(source_texts, "source texts")
        self.owner = np.ascontiguousarray(owner, dtype=np.int64)
        if not 0 < retention_fraction <= 1 or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("Invalid retention fraction or block size")
        if not np.array_equal(cache.owner, self.owner) or cache.i2t.shape != (len(images), len(source_texts)):
            raise ValueError("Cache gallery does not match constraint gallery")
        cache.validate_inputs(self.images, self.texts, self.owner)
        self.cache, self.scorer, self.gamma, self.block_size = cache, scorer, retention_fraction, block_size
        self.x, self.y = scorer.image_coordinates(images), scorer.text_coordinates(source_texts)
        self.best_positive = cache.best_positive
        baseline = cached_exact_retrieval(self.images, self.texts, self.owner, cache)
        self.frozen_i2t, self.frozen_t2i = baseline["i2t_top_indices"], baseline["t2i_top_indices"]
        self.protect_i2t, self.protect_t2i = baseline["i2t_correct"], baseline["t2i_correct"]
        self.active = set()

    def _blocks(self, coefficient):
        left = canonical_project(self.x, coefficient)
        i_positive = np.sum(left * self.y[self.best_positive], axis=1)
        t_positive = np.sum(left[self.owner] * self.y, axis=1)
        best_base = self.cache.positive_score[self.best_positive]
        for lo in range(0, len(self.images), self.block_size):
            hi = min(lo + self.block_size, len(self.images))
            rows = np.arange(lo, hi)
            base = self.cache.image_rows(slice(lo, hi))
            residual = left[lo:hi] @ self.y.T
            i_margin = best_base[lo:hi, None] - base
            t_margin = self.cache.positive_score[None, :] - base
            near = (np.abs(i_margin) < 1e-10) | (np.abs(t_margin) < 1e-10)
            ii, tt = np.nonzero(near)
            for start in range(0, len(ii), 4096):
                rr, cc = ii[start:start + 4096], tt[start:start + 4096]
                residual[rr, cc] = np.sum(left[lo + rr] * self.y[cc], axis=1)
            i_delta = i_positive[lo:hi, None] - residual
            t_delta = t_positive[None, :] - residual
            owned = self.owner[None, :] == rows[:, None]
            yield lo, hi, base, residual, i_margin, t_margin, i_delta, t_delta, owned

    def scan(self, coefficient, tolerance=1e-12, add=True, canonical=False):
        t_worst = np.zeros(len(self.texts), dtype=np.int64)
        t_min = np.full(len(self.texts), np.inf)
        t_max = np.full(len(self.texts), -np.inf)
        t_top = np.zeros(len(self.texts), dtype=np.int64)
        i_top = np.empty(len(self.images), dtype=np.int64)
        checked, violated, minimum, new_edges = 0, 0, np.inf, []
        for lo, hi, base, residual, mi, mt, di, dt, owned in self._blocks(coefficient):
            si, st = (1 - self.gamma) * mi + di, (1 - self.gamma) * mt + dt
            si[owned | ~self.protect_i2t[lo:hi, None]] = np.inf
            st[owned | ~self.protect_t2i[None, :]] = np.inf
            checked += int(np.isfinite(si).sum() + np.isfinite(st).sum())
            violated += int((si < -tolerance).sum() + (st < -tolerance).sum())
            minimum = min(minimum, float(si.min()), float(st.min()))
            iw = si.argmin(axis=1)
            new_edges.extend(Edge(0, lo + row, int(iw[row])) for row in range(hi - lo) if si[row, iw[row]] < -tolerance)
            tw = st.argmin(axis=0)
            tv = st[tw, np.arange(len(self.texts))]
            improve = tv < t_min
            t_min[improve], t_worst[improve] = tv[improve], lo + tw[improve]
            scores = base + residual
            i_top[lo:hi] = scores.argmax(axis=1)
            ti = scores.argmax(axis=0)
            ts = scores[ti, np.arange(len(self.texts))]
            improve = ts > t_max
            t_max[improve], t_top[improve] = ts[improve], lo + ti[improve]
        new_edges.extend(Edge(1, int(q), int(t_worst[q])) for q in np.flatnonzero(t_min < -tolerance))
        if add:
            self.active.update(new_edges)
        if canonical:
            old = self.scorer.coefficient
            try:
                self.scorer.coefficient = np.asarray(coefficient)
                raw = cached_exact_retrieval(self.images, self.texts, self.owner, self.cache, self.scorer)
            finally:
                self.scorer.coefficient = old
            ic, tc = raw["i2t_correct"], raw["t2i_correct"]
        else:
            ic, tc = self.owner[i_top] == np.arange(len(self.images)), t_top == self.owner
        lost_i, lost_t = int(np.sum(self.protect_i2t & ~ic)), int(np.sum(self.protect_t2i & ~tc))
        return {"protected_i2t_queries": int(self.protect_i2t.sum()), "protected_t2i_queries": int(self.protect_t2i.sum()),
                "source_gallery_images": len(self.images), "source_gallery_texts": len(self.texts),
                "checked_constraints": checked, "violated_constraints": violated,
                "min_constraint_slack": None if not np.isfinite(minimum) else float(minimum),
                "active_constraints": len(self.active), "lost_frozen_correct_i2t": lost_i,
                "lost_frozen_correct_t2i": lost_t, "current_i2t_correct": int(ic.sum()), "current_t2i_correct": int(tc.sum()),
                "ranking_checked_canonically": bool(canonical), "ranking_preserved": lost_i == 0 and lost_t == 0,
                "feasible_with_tolerance": minimum >= -tolerance, "storage_backend": "streamed_float64_full_gallery_v10"}

    def edge_factors(self, edges=None):
        edges = sorted(self.active) if edges is None else list(edges)
        left, right, rhs = [], [], []
        for edge in edges:
            q, n = edge.query, edge.negative
            if edge.direction == 0:
                if not self.protect_i2t[q] or self.owner[n] == q:
                    raise ValueError("Invalid protected image edge")
                left.append(self.x[q]); right.append(self.y[self.best_positive[q]] - self.y[n])
                margin = self.cache.positive_score[self.best_positive[q]] - self.cache.i2t[q, n]
            elif edge.direction == 1:
                if not self.protect_t2i[q] or self.owner[q] == n:
                    raise ValueError("Invalid protected text edge")
                left.append(self.x[self.owner[q]] - self.x[n]); right.append(self.y[q])
                margin = self.cache.positive_score[q] - self.cache.t2i[q, n]
            else:
                raise ValueError("Unknown edge direction")
            rhs.append(-(1 - self.gamma) * margin)
        return (np.asarray(left, dtype=np.float64).reshape(-1, self.x.shape[1]),
                np.asarray(right, dtype=np.float64).reshape(-1, self.y.shape[1]), np.asarray(rhs))

    penalty = FullGalleryConstraints.penalty

    def radial_feasibility_factor(self, coefficient):
        factor = 1.0
        for lo, hi, _, _, mi, mt, di, dt, owned in self._blocks(coefficient):
            for margin, delta, valid in ((mi, di, ~owned & self.protect_i2t[lo:hi, None]),
                                         (mt, dt, ~owned & self.protect_t2i[None, :])):
                required = valid & (delta < 0)
                if np.any(required):
                    factor = min(factor, float(np.min((1 - self.gamma) * margin[required] / -delta[required])))
        return max(0.0, min(1.0, factor))


class StreamingFullGallerySourceLoss:
    """Exactly the v9 source objective; stream queries, never sample negatives."""
    def __init__(self, constraints, logit_scale, query_block=64):
        if not np.isfinite(logit_scale) or logit_scale <= 0 or not isinstance(query_block, int) or query_block < 1:
            raise ValueError("Positive fixed logit scale and query block required")
        self.c, self.scale, self.query_block = constraints, float(logit_scale), int(query_block)

    def _loss_gradient_block(self, coefficient, images):
        from .practical_joint_v9 import _softmax
        c = self.c
        text_queries = np.flatnonzero(np.isin(c.owner, images))
        image_logits = self.scale * (c.cache.image_rows(images) + (c.x[images] @ coefficient) @ c.y.T)
        anchors = c.best_positive[images]
        other_owned = c.owner[None, :] == images[:, None]
        other_owned[np.arange(len(images)), anchors] = False
        image_logits[other_owned] = -np.inf
        image_prob, image_lse = _softmax(image_logits, 1)
        i_loss = np.mean(image_lse - image_logits[np.arange(len(images)), anchors])
        image_prob[np.arange(len(images)), anchors] -= 1
        image_derivative = self.scale * image_prob / len(images)
        gradient_i = c.x[images].T @ (image_derivative @ c.y)
        text_logits = self.scale * (c.cache.text_columns(text_queries) + (c.x @ coefficient) @ c.y[text_queries].T)
        text_prob, text_lse = _softmax(text_logits, 0)
        owners = c.owner[text_queries]
        denominator = len(images) * len(c.texts) / len(c.images)
        t_loss = np.sum(text_lse - text_logits[owners, np.arange(len(text_queries))]) / denominator
        text_prob[owners, np.arange(len(text_queries))] -= 1
        text_derivative = self.scale * text_prob / denominator
        gradient_t = c.x.T @ (text_derivative @ c.y[text_queries])
        return float((i_loss + t_loss) / 2), (gradient_i + gradient_t) / 2

    def loss_gradient(self, coefficient, image_indices=None):
        c = self.c
        images = np.arange(len(c.images)) if image_indices is None else np.asarray(image_indices, dtype=np.int64)
        if not len(images) or len(np.unique(images)) != len(images) or np.any(images < 0) or np.any(images >= len(c.images)):
            raise ValueError("Require distinct valid owner queries")
        value, gradient = 0.0, np.zeros_like(coefficient)
        for lo in range(0, len(images), self.query_block):
            batch = images[lo:lo + self.query_block]
            block_value, block_gradient = self._loss_gradient_block(coefficient, batch)
            weight = len(batch) / len(images)
            value += weight * block_value
            gradient += weight * block_gradient
        return float(value), gradient
