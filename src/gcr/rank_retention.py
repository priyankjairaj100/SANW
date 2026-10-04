"""Exact forward-KL distance to losing every relevant item from the top k.

These are post-fit diagnostics on the *same complete gallery*. A KL penalty
measured on a training minibatch does not certify a held-out gallery. Ties
between relevant and irrelevant items are treated as failures. Mathematical
thresholds are exact; NumPy floating-point evaluations are not interval proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RankProjection:
    threshold: float
    projection: np.ndarray
    teacher_correct: bool
    cutoff: float
    active_relevant: np.ndarray
    selected_irrelevant: np.ndarray
    active_irrelevant: np.ndarray
    k: int


@dataclass(frozen=True)
class RankCertificate:
    threshold: float
    divergence: float
    certified: bool
    teacher_correct: bool
    student_correct: bool
    k: int


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional distribution")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    total = q.sum()
    if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-6):
        raise ValueError(f"{name} probabilities must sum to one")
    # Account for ordinary float32/64 summation error, never clip probabilities.
    return q / total


def _mask(relevant: np.ndarray | Sequence[int], size: int) -> np.ndarray:
    values = np.asarray(relevant)
    if values.dtype == np.bool_:
        if values.shape != (size,):
            raise ValueError("boolean relevance mask must match the gallery")
        mask = values.copy()
    else:
        if values.ndim != 1 or values.size == 0:
            raise ValueError("relevance indices must be a nonempty vector")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("relevance indices must be integers")
        if np.any(values < 0) or np.any(values >= size):
            raise ValueError("relevance index outside the gallery")
        mask = np.zeros(size, dtype=bool)
        mask[values] = True
    if not np.any(mask):
        raise ValueError("at least one relevant item is required")
    return mask


def _validate_k(k: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _top_wrong(q: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    wrong = np.flatnonzero(~mask)
    if wrong.size < k:
        return wrong
    if k == 1:
        return wrong[np.array([np.argmax(q[wrong])])]
    # Stable tie identities are immaterial to the cost, but useful in receipts.
    cutoff = np.partition(q[wrong], wrong.size - k)[wrong.size - k]
    above = wrong[q[wrong] > cutoff]
    equal = wrong[q[wrong] == cutoff]
    selected = np.concatenate([above, equal[: k - above.size]])
    return selected[np.lexsort((selected, -q[selected]))]


def _correct(q: np.ndarray, mask: np.ndarray, k: int) -> bool:
    # A relevant item must strictly outrank all but at most k-1 irrelevant items.
    return bool(np.count_nonzero(q[~mask] >= np.max(q[mask])) < k)


def forward_kl(teacher_probs: np.ndarray, student_probs: np.ndarray) -> float:
    """KL(teacher || student), with 0 log(0/p)=0 and q>0,p=0 => infinity."""
    q = _probabilities(teacher_probs, "teacher")
    p = _probabilities(student_probs, "student")
    if q.shape != p.shape:
        raise ValueError("teacher and student must use the identical gallery")
    positive = q > 0
    if np.any(p[positive] == 0):
        return float("inf")
    value = np.sum(q[positive] * (np.log(q[positive]) - np.log(p[positive])))
    if not np.isfinite(value):
        raise ValueError("nonfinite KL arithmetic; no certificate can be evaluated")
    return float(max(0.0, value))


def rank_failure_projection(
    teacher_probs: np.ndarray,
    relevant: np.ndarray | Sequence[int],
    k: int = 1,
) -> RankProjection:
    """Project onto the closed event of k wrong items tying/beating every right item.

    The closest failure uses the k teacher-largest irrelevant items. Pool the
    too-large relevant and too-small selected irrelevant probabilities at a
    shared arithmetic mean; leave every other probability unchanged. For k=1,
    this is a linear gallery scan and a sort of the relevant probabilities.
    The optional general-k path scans at most r+k breakpoints, where r is the
    number of relevant items. If fewer than k irrelevant items exist, failure
    is impossible and the distance is infinity.
    """
    q = _probabilities(teacher_probs, "teacher")
    mask = _mask(relevant, q.size)
    k = _validate_k(k)
    selected = _top_wrong(q, mask, k)
    empty = np.empty(0, dtype=np.int64)
    if selected.size < k:
        return RankProjection(float("inf"), q.copy(), True, float("nan"),
                              empty, selected, empty, k)
    correct = _correct(q, mask, k)
    if not correct:
        return RankProjection(0.0, q.copy(), False, float("nan"),
                              empty, selected, empty, k)
    rel = np.flatnonzero(mask)
    a = q[rel]
    b = q[selected]
    if k == 1:
        descending = rel[np.argsort(-a, kind="stable")]
        pooled_sum = float(b[0])
        pooled_count = 1
        cutoff = pooled_sum
        for index in descending:
            if q[index] <= cutoff:
                break
            pooled_sum += float(q[index])
            pooled_count += 1
            cutoff = pooled_sum / pooled_count
    else:
        # F(t)=sum(a-t)_+ - sum(t-b)_+ is monotone. Between consecutive
        # breakpoints its zero is the arithmetic mean of active coordinates.
        lower = float(np.min(b))
        upper = float(np.max(a))
        knots = np.unique(np.concatenate(([lower, upper], a, b)))
        knots = knots[(knots >= lower) & (knots <= upper)]
        cutoff = None
        for left, right in zip(knots[:-1], knots[1:]):
            midpoint = left + (right - left) / 2
            pooled = np.concatenate([a[a > midpoint], b[b < midpoint]])
            candidate = float(np.mean(pooled))
            if left <= candidate <= right:
                cutoff = candidate
                break
        if cutoff is None:
            # Roundoff at a breakpoint: bracket the root, then use its active
            # arithmetic mean. This branch is not needed in exact arithmetic.
            left, right = lower, upper
            for _ in range(64):
                midpoint = left + (right - left) / 2
                value = np.maximum(a - midpoint, 0).sum() - np.maximum(midpoint - b, 0).sum()
                if value > 0:
                    left = midpoint
                else:
                    right = midpoint
            midpoint = left + (right - left) / 2
            pooled = np.concatenate([a[a >= midpoint], b[b <= midpoint]])
            cutoff = float(np.mean(pooled))
    active_rel = rel[a > cutoff]
    active_wrong = selected[b < cutoff]
    pooled_indices = np.concatenate([active_rel, active_wrong])
    projection = q.copy()
    projection[pooled_indices] = cutoff
    positive_pool = pooled_indices[q[pooled_indices] > 0]
    threshold = np.sum(q[positive_pool] * np.log(q[positive_pool] / cutoff))
    return RankProjection(float(max(0.0, threshold)), projection, True, cutoff,
                          active_rel, selected, active_wrong, k)


def _passes(divergence: float, threshold: float, atol: float, rtol: float) -> bool:
    if atol < 0 or rtol < 0 or not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("certificate tolerances must be finite and nonnegative")
    if np.isnan(divergence) or np.isnan(threshold) or divergence < 0:
        raise ValueError("certificate arithmetic must not contain NaN or negative divergence")
    if threshold == float("inf"):
        return True  # Reserved for galleries with impossible failure.
    return bool(divergence + atol + rtol * abs(threshold) < threshold)


def rank_retention_certificate(
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    relevant: np.ndarray | Sequence[int],
    k: int = 1,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> RankCertificate:
    """Certify retention only under the strict full-gallery KL threshold."""
    q = _probabilities(teacher_probs, "teacher")
    p = _probabilities(student_probs, "student")
    if q.shape != p.shape:
        raise ValueError("teacher and student must use the identical gallery")
    mask = _mask(relevant, q.size)
    projection = rank_failure_projection(q, mask, k)
    divergence = forward_kl(q, p)
    return RankCertificate(projection.threshold, divergence,
                           _passes(divergence, projection.threshold, atol, rtol),
                           projection.teacher_correct, _correct(p, mask, k), k)


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    # Subtract before dividing to avoid exponentiating large absolute logits.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            shifted = (logits - np.max(logits)) / temperature
            log_probs = shifted - np.log(np.sum(np.exp(shifted)))
    except FloatingPointError as exc:
        raise ValueError("unsafe logit range or temperature; no certificate can be evaluated") from exc
    if not np.all(np.isfinite(log_probs)):
        raise ValueError("nonfinite log probabilities; no certificate can be evaluated")
    return log_probs


def rank_retention_diagnostics(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    relevance: np.ndarray | Sequence[Sequence[int]],
    *,
    temperature: float = 2.0,
    k: int = 1,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Per-query full-gallery certificates, with O(gallery_size) extra memory.

    Inputs have shape (queries, candidates). Relevance is either a boolean
    array of that shape or one integer-index sequence per query. KL uses
    teacher-to-student probabilities at `temperature`, without a T**2 factor.
    Native logits are used for the separate shift-invariant drift certificate.
    Both certificates use pessimistic treatment of cross-relevance ties.
    """
    teacher = np.asarray(teacher_logits)
    student = np.asarray(student_logits)
    if teacher.ndim != 2 or teacher.shape != student.shape or teacher.shape[1] == 0:
        raise ValueError("teacher/student logits must have the same (queries, gallery) shape")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    k = _validate_k(k)
    if len(relevance) != teacher.shape[0]:
        raise ValueError("one relevance mask or index vector is required per query")
    bool_keys = ("certified", "teacher_correct", "student_correct", "logit_drift_certified",
                 "pairwise_certified", "combined_certified")
    float_keys = ("threshold", "divergence", "probability_margin", "logit_margin",
                  "logit_drift_oscillation", "logit_drift_linf_centered", "pairwise_threshold")
    output = {key: np.zeros(teacher.shape[0], dtype=bool) for key in bool_keys}
    output.update({key: np.zeros(teacher.shape[0], dtype=np.float64) for key in float_keys})
    _passes(0, 0, atol, rtol)  # Validate even on empty inputs.
    for row in range(teacher.shape[0]):
        z = np.asarray(teacher[row], dtype=np.float64)
        v = np.asarray(student[row], dtype=np.float64)
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(v)):
            raise ValueError("diagnostic logits must be finite")
        mask = _mask(relevance[row], z.size)
        logq = _log_softmax(z, temperature)
        logp = _log_softmax(v, temperature)
        q = np.exp(logq)
        projection = rank_failure_projection(q, mask, k)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                raw_divergence = float(np.sum(q * (logq - logp)))
        except FloatingPointError as exc:
            raise ValueError("unsafe KL arithmetic; no certificate can be evaluated") from exc
        if not np.isfinite(raw_divergence):
            raise ValueError("nonfinite KL arithmetic; no certificate can be evaluated")
        divergence = max(0.0, raw_divergence)
        selected = projection.selected_irrelevant
        if selected.size < k:
            probability_margin = logit_margin = pairwise_threshold = float("inf")
        else:
            a, b = float(np.max(q[mask])), float(np.min(q[selected]))
            probability_margin = a - b
            pooled_pair = (a + b) / 2
            pairwise_threshold = (a * np.log(a / pooled_pair)
                                  + (b * np.log(b / pooled_pair) if b > 0 else 0.0)
                                  if a > b else 0.0)
            # Use native-score order for ties/underflow-safe rank decisions.
            wrong = z[~mask]
            kth = np.partition(wrong, wrong.size - k)[wrong.size - k]
            try:
                with np.errstate(over="raise", invalid="raise"):
                    logit_margin = float(np.max(z[mask]) - kth)
            except FloatingPointError as exc:
                raise ValueError("unsafe native logit margin; no certificate can be evaluated") from exc
        try:
            with np.errstate(over="raise", invalid="raise"):
                drift = v - z
                oscillation = float(np.max(drift) - np.min(drift))
        except FloatingPointError as exc:
            raise ValueError("unsafe native logit drift; no certificate can be evaluated") from exc
        if not np.isfinite(oscillation):
            raise ValueError("nonfinite native logit drift; no certificate can be evaluated")
        output["threshold"][row] = projection.threshold
        output["divergence"][row] = divergence
        output["certified"][row] = _passes(divergence, projection.threshold, atol, rtol)
        output["teacher_correct"][row] = _correct(z, mask, k)
        output["student_correct"][row] = _correct(v, mask, k)
        output["probability_margin"][row] = probability_margin
        output["logit_margin"][row] = logit_margin
        output["logit_drift_oscillation"][row] = oscillation
        output["logit_drift_linf_centered"][row] = oscillation / 2
        output["logit_drift_certified"][row] = _passes(oscillation, logit_margin, atol, rtol)
        output["pairwise_threshold"][row] = pairwise_threshold
        output["pairwise_certified"][row] = _passes(divergence, pairwise_threshold, atol, rtol)
        output["combined_certified"][row] = (output["certified"][row]
                                                or output["logit_drift_certified"][row])
    return output


def failures_from_total_kl(
    thresholds: np.ndarray, total_kl: float, *, atol: float = 1e-12, rtol: float = 1e-10,
) -> int:
    """Sharp worst-case count from a full-gallery total KL budget.

    Only supply positive thresholds for teacher-correct queries. Every lost
    query costs at least its own threshold, so an adversary can lose at most
    the largest m for which the m smallest thresholds sum to <= total_kl.
    This is a mathematical budget bound, not a minibatch extrapolation.
    """
    costs = np.asarray(thresholds, dtype=np.float64)
    if costs.ndim != 1 or np.any(np.isnan(costs)) or np.any(costs <= 0):
        raise ValueError("thresholds must be a vector of strictly positive costs")
    if np.isnan(total_kl) or total_kl < 0:
        raise ValueError("total_kl must be nonnegative")
    # Infinite costs correspond to galleries with impossible failure.
    finite = np.sort(costs[np.isfinite(costs)])
    _passes(0, 0, atol, rtol)
    if np.isinf(total_kl):
        return int(finite.size)
    budget = total_kl + atol + rtol * total_kl
    return int(np.searchsorted(np.cumsum(finite), budget, side="right"))
