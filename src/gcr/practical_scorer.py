"""Bounded, pair-conditioned correction of a frozen image--text score.

Input embeddings must receive the experiment's frozen preprocessing before this
module is called.  The module never silently normalizes embeddings or estimates
means.  Supply image/text means estimated from training examples only.

The residual network normally uses float32.  Its tanh output is converted to
float64 before multiplication by epsilon.  Frozen dot products also use float64,
matching the original evaluator and giving one canonical score in train/dev/test.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor, nn


class BoundedPairScorer(nn.Module):
    """A shared scorer for every gallery size and both retrieval directions.

    Features comprise the centered Hadamard product, a bias-free projected
    interaction, and the frozen cosine.  Only the final output is bounded.
    ``score_pairs`` is paired scoring; ``score_matrix`` scores a Cartesian product.
    ``config()`` plus ``state_dict()`` suffices to reconstruct the scorer.
    """

    def __init__(
        self,
        dimension: int,
        epsilon: float = 0.01,
        hidden: int = 128,
        rank: int = 64,
        image_mean=None,
        text_mean=None,
    ) -> None:
        super().__init__()
        for name, value in (("dimension", dimension), ("hidden", hidden), ("rank", rank)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(epsilon)) or float(epsilon) < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        self.dimension = int(dimension)
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.epsilon = float(epsilon)
        self.register_buffer("image_mean", self._make_mean(image_mean, "image_mean"))
        self.register_buffer("text_mean", self._make_mean(text_mean, "text_mean"))
        self.image_projection = nn.Linear(self.dimension, self.rank, bias=False)
        self.text_projection = nn.Linear(self.dimension, self.rank, bias=False)
        self.pair_hidden = nn.Linear(self.dimension + self.rank + 1, self.hidden)
        self.activation = nn.GELU()
        self.output = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _make_mean(self, mean, name: str) -> Tensor:
        if mean is None:
            return torch.zeros(self.dimension, dtype=torch.float32)
        result = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        if result.numel() != self.dimension:
            raise ValueError(f"{name} must contain dimension entries")
        result = result.reshape(self.dimension)
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def config(self) -> dict:
        """Constructor arguments; training-only means are stored in state_dict."""
        return {
            "dimension": self.dimension,
            "epsilon": self.epsilon,
            "hidden": self.hidden,
            "rank": self.rank,
        }

    def _validate_inputs(self, images: Tensor, texts: Tensor, *, paired: bool) -> None:
        for name, value in (("images", images), ("texts", texts)):
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor")
            if value.shape[1] != self.dimension:
                raise ValueError(f"{name} has the wrong embedding dimension")
            if value.dtype != self.image_mean.dtype or value.device != self.image_mean.device:
                raise ValueError(f"{name} must match the scorer's dtype and device")
        if paired and images.shape[0] != texts.shape[0]:
            raise ValueError("paired images and texts must have the same row count")

    def residual_pairs(self, images: Tensor, texts: Tensor) -> Tensor:
        """Return bounded paired corrections [N], using no gallery context.

        The network uses the module dtype.  The returned float64 correction has
        magnitude at most ``epsilon``, including when epsilon is not exactly
        representable in float32.  This operation remains differentiable.
        """
        self._validate_inputs(images, texts, paired=True)
        centered_images = images - self.image_mean
        centered_texts = texts - self.text_mean
        hadamard = math.sqrt(self.dimension) * centered_images * centered_texts
        # A shared deterministic scale gives projected products the same order
        # as the unprojected interaction; the projections contain no biases.
        projection_scale = math.sqrt(self.dimension / self.rank)
        projected_images = self.image_projection(centered_images) * projection_scale
        projected_texts = self.text_projection(centered_texts) * projection_scale
        interaction = math.sqrt(self.rank) * projected_images * projected_texts
        frozen_cosine = (images * texts).sum(dim=-1, keepdim=True)
        features = torch.cat((hadamard, interaction, frozen_cosine), dim=-1)
        raw = self.output(self.activation(self.pair_hidden(features))).squeeze(-1)
        return torch.tanh(raw).to(torch.float64) * self.epsilon

    def score_pairs(self, images: Tensor, texts: Tensor) -> Tensor:
        """Return canonical float64 paired scores [N]."""
        residual = self.residual_pairs(images, texts)
        frozen = (images.to(torch.float64) * texts.to(torch.float64)).sum(dim=-1)
        return frozen + residual

    def forward(self, images: Tensor, texts: Tensor) -> Tensor:
        return self.score_pairs(images, texts)

    def score_matrix(self, images: Tensor, texts: Tensor, pair_chunk: int = 8192) -> Tensor:
        """Return all canonical scores [image_count, text_count].

        Pair chunks bound temporary feature memory and preserve gradients.
        Float32 linear kernels can differ by ordinary roundoff across chunk sizes.
        No approximate nearest-neighbor search or gallery-dependent score is used.
        """
        self._validate_inputs(images, texts, paired=False)
        _positive_integer(pair_chunk, "pair_chunk")
        image_count, text_count = images.shape[0], texts.shape[0]
        total = image_count * text_count
        if total == 0:
            return torch.empty((image_count, text_count), dtype=torch.float64, device=images.device)
        chunks = []
        for start in range(0, total, pair_chunk):
            flat = torch.arange(start, min(start + pair_chunk, total), device=images.device)
            chunks.append(self.score_pairs(images[flat // text_count], texts[flat % text_count]))
        return torch.cat(chunks).reshape(image_count, text_count)

    def score(self, images: Tensor, texts: Tensor, pair_chunk: int = 8192) -> Tensor:
        """Alias for Cartesian-product scoring, not paired scoring."""
        return self.score_matrix(images, texts, pair_chunk=pair_chunk)

    @torch.no_grad()
    def exact_topk(
        self,
        images: Tensor,
        texts: Tensor,
        k: int = 1,
        direction: Literal["i2t", "t2i"] = "i2t",
        query_chunk: int = 64,
        pair_chunk: int = 8192,
    ) -> tuple[Tensor, Tensor]:
        """Return exact top-k (scores, gallery indices), each shaped [Q,k].

        ``i2t`` queries images and indexes texts.  ``t2i`` queries texts and
        indexes images, while retaining the scorer's image/text argument roles.
        Scores descend; exact score ties use ascending gallery index.

        If b_k is the kth largest frozen score, every possible new top-k entry
        has frozen score >= b_k - 2*epsilon.  We rescore this inclusive shortlist.
        A conservative float64 dot-product roundoff guard enlarges the shortlist.
        This is exact bounded retrieval, not a fixed-size approximate shortlist.
        As in dense scoring, network evaluations retain ordinary float32 roundoff.
        """
        self._validate_inputs(images, texts, paired=False)
        for name, value in (("k", k), ("query_chunk", query_chunk), ("pair_chunk", pair_chunk)):
            _positive_integer(value, name)
        if direction not in ("i2t", "t2i"):
            raise ValueError("direction must be 'i2t' or 't2i'")
        queries, gallery = (images, texts) if direction == "i2t" else (texts, images)
        if k > gallery.shape[0]:
            raise ValueError("k must not exceed the gallery size")
        if not torch.isfinite(images).all() or not torch.isfinite(texts).all():
            raise ValueError("retrieval embeddings must be finite")
        result_scores = torch.empty((queries.shape[0], k), dtype=torch.float64, device=images.device)
        result_indices = torch.empty((queries.shape[0], k), dtype=torch.long, device=images.device)
        gallery64 = gallery.to(torch.float64)
        gallery_norm = torch.linalg.vector_norm(gallery64, dim=1).max()
        unit_roundoff = torch.finfo(torch.float64).eps / 2
        dimension_error = self.dimension * unit_roundoff
        gamma = dimension_error / (1 - dimension_error) if dimension_error < 1 else math.inf
        for start in range(0, queries.shape[0], query_chunk):
            query = queries[start:start + query_chunk]
            query64 = query.to(torch.float64)
            frozen = query64 @ gallery64.T
            kth = torch.topk(frozen, k, dim=1, sorted=True).values[:, -1]
            # The two baseline implementations can accumulate dot products in a
            # different order.  Cauchy--Schwarz bounds their absolute roundoff.
            scale = torch.linalg.vector_norm(query64, dim=1) * gallery_norm
            guard = 8 * gamma * scale + 16 * unit_roundoff * (1 + kth.abs() + self.epsilon)
            threshold = kth - 2 * self.epsilon - guard
            candidate_rows, candidate_columns = torch.nonzero(
                frozen >= threshold[:, None], as_tuple=True
            )
            scores = []
            for offset in range(0, candidate_rows.numel(), pair_chunk):
                rows = candidate_rows[offset:offset + pair_chunk]
                columns = candidate_columns[offset:offset + pair_chunk]
                if direction == "i2t":
                    scores.append(self.score_pairs(query[rows], gallery[columns]))
                else:
                    scores.append(self.score_pairs(gallery[columns], query[rows]))
            candidate_scores = torch.cat(scores)
            if not torch.isfinite(candidate_scores).all():
                raise FloatingPointError("nonfinite score in exact retrieval")
            counts = torch.bincount(candidate_rows, minlength=query.shape[0]).tolist()
            offset = 0
            for row, count in enumerate(counts):
                # nonzero returns row-major, ascending gallery indices. Stable
                # score sorting therefore gives the documented tie order.
                values = candidate_scores[offset:offset + count]
                indices = candidate_columns[offset:offset + count]
                selected = torch.argsort(values, descending=True, stable=True)[:k]
                result_scores[start + row] = values[selected]
                result_indices[start + row] = indices[selected]
                offset += count
        return result_scores, result_indices


def _positive_integer(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def exact_topk(
    scorer: BoundedPairScorer,
    images: Tensor,
    texts: Tensor,
    k: int = 1,
    direction: Literal["i2t", "t2i"] = "i2t",
    query_chunk: int = 64,
    pair_chunk: int = 8192,
) -> tuple[Tensor, Tensor]:
    """Functional wrapper around :meth:`BoundedPairScorer.exact_topk`."""
    return scorer.exact_topk(images, texts, k, direction, query_chunk, pair_chunk)
