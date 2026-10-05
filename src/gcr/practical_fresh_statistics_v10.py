"""Exact paired-image bootstrap endpoints for variable-denominator composition.

Each image contributes one rational mean change across the three fixed states.
Images, not triples or training seeds, are sampled uniformly with replacement.
The integer path computes every bootstrap numerator exactly. The fallback uses
certified floating intervals ONLY to screen order-statistic candidates, then
regenerates and evaluates every candidate exactly before percentile calculation.
No numerical epsilon decides a scientific inequality.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np


REPLICATES = 100000
BOOTSTRAP_SEED = 20261007
FAMILY_SIZE = 80
TAIL = Fraction(1, 3200)
DRAW_BLOCK = 250
MAX_IMAGES = 1 << 20


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _validated_counts(numerators, denominators):
    n, d = np.asarray(numerators), np.asarray(denominators)
    if (n.ndim != 1 or d.shape != n.shape or not 0 < len(n) <= MAX_IMAGES
            or n.dtype.kind not in "iu" or d.dtype.kind not in "iu"):
        raise ValueError("Require nonempty aligned native integer vectors, at most 2^20 images")
    limit = np.iinfo(np.int64).max
    # Bounds are checked in Python integers before any narrowing conversion.
    if (int(n.min()) < -limit or int(n.max()) > limit or int(d.min()) <= 0 or int(d.max()) > limit):
        raise ValueError("Counts must fit signed int64, with positive denominators")
    n, d = n.astype(np.int64), d.astype(np.int64)
    if np.any(d % 3) or np.any(n < -d) or np.any(n > d):
        raise ValueError("Require |numerator| <= denominator = 3 * positive triplet count")
    return n, d


def _percentile_plan(size, probability):
    position = probability * (size - 1)
    lower = position.numerator // position.denominator
    return lower, min(lower + 1, size - 1), position - lower


def _interpolate(order_values, plan, denominator):
    lower, upper, weight = plan
    return ((1 - weight) * Fraction(int(order_values[lower]), denominator)
            + weight * Fraction(int(order_values[upper]), denominator))


def _roundoff_envelope(ratios):
    """Absolute mean-error bound for float(Fraction), sum(float64), then /N.

Let M=max|rounded input|, e0=max exact input rounding error, u=2^-53,
gamma=(N-1)u/(1-(N-1)u). Then the mean error is bounded by
e0 + gamma*M + u*M*(1+gamma). A sum with any ordering needs no more than
N-1 nontrivial additions, so the bound covers NumPy's pairwise reduction too.
Inputs originate from int64 ratios, hence nonzero rounded inputs have magnitude
>=2^-63 and summation lattice spacing >=2^-115. N<=2^20 ensures cancellation
and final division remain normal (or exact zero); no underflow term is needed.
Every bound construction here uses Fraction until the final outward rounding.
"""
    values = np.array([float(value) for value in ratios], dtype=np.float64)
    rounded = [Fraction.from_float(float(value)) for value in values]
    magnitude = max(map(abs, rounded))
    input_error = max(abs(a - b) for a, b in zip(ratios, rounded))
    u = Fraction(1, 1 << 53)
    steps = len(ratios) - 1
    gamma = steps * u / (1 - steps * u)
    exact = input_error + gamma * magnitude + u * magnitude * (1 + gamma)
    upper_float = float(exact)
    if Fraction.from_float(upper_float) < exact:
        upper_float = float(np.nextafter(upper_float, np.inf))
    return values, exact, upper_float


def _candidate_sets(lower, upper, ranks):
    """Certified kth-rank screening, including all possible ties.

The kth exact value lies between kth(lower) and kth(upper), denoted L,U.
Every interval wholly below L precedes it, every interval wholly above U
follows it, and all remaining intervals are retained. Sorting retained exact
values and subtracting the number wholly below L therefore recovers rank k.
Strict exclusion inequalities retain intervals touching either boundary.
"""
    result = {}
    for rank in sorted(ranks):
        low = float(np.partition(lower, rank)[rank])
        high = float(np.partition(upper, rank)[rank])
        below = upper < low
        possible = np.flatnonzero((upper >= low) & (lower <= high))
        local_rank = rank - int(below.sum())
        if low > high or not 0 <= local_rank < len(possible):
            raise ArithmeticError("Invalid certified percentile candidate enclosure")
        result[rank] = (possible, local_rank)
    return result


def _bootstrap_core(n, d, *, replicates=REPLICATES, seed=BOOTSTRAP_SEED, force_fallback=False):
    """Internal configurable size permits exhaustive small synthetic tests."""
    n, d = _validated_counts(n, d)
    if type(replicates) is not int or replicates < 2:
        raise ValueError("Require at least two bootstrap replicates")
    ratios = [Fraction(int(a), int(b)) for a, b in zip(n, d)]
    count = len(ratios)
    common = math.lcm(*(value.denominator for value in ratios))
    scaled = [value.numerator * (common // value.denominator) for value in ratios]
    denominator = common * count
    mean = Fraction(sum(scaled), denominator)
    plans = [_percentile_plan(replicates, probability) for probability in (TAIL, 1 - TAIL)]
    ranks = {rank for low, high, _ in plans for rank in (low, high)}
    rng = np.random.default_rng(seed)
    metadata = {"common_denominator_bits": common.bit_length(), "images_per_draw": count,
                "draw_block": DRAW_BLOCK, "exact_fallback_draws": 0,
                "candidate_rank_counts": {}, "bootstrap_rng": "numpy.default_rng.PCG64.integers"}
    if not force_fallback and max(map(abs, scaled)) * count <= np.iinfo(np.int64).max:
        integer = np.asarray(scaled, dtype=np.int64)
        samples_n = np.empty(replicates, dtype=np.int64)
        for start in range(0, replicates, DRAW_BLOCK):
            draws = rng.integers(0, count, size=(min(DRAW_BLOCK, replicates - start), count))
            samples_n[start:start + len(draws)] = integer[draws].sum(axis=1, dtype=np.int64)
        ordered = np.sort(samples_n)
        order_values = {rank: int(ordered[rank]) for rank in ranks}
        # Float draws are descriptive; the endpoint numerators above stay exact.
        samples = np.array([float(Fraction(int(value), denominator)) for value in samples_n])
        metadata.update({"path": "common_denominator_int64_exact_accumulation",
                         "floating_screen_error_bound": None, "floating_screen_error_bound_exact": None})
    else:
        values, exact_error, error = _roundoff_envelope(ratios)
        samples = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, DRAW_BLOCK):
            draws = rng.integers(0, count, size=(min(DRAW_BLOCK, replicates - start), count))
            samples[start:start + len(draws)] = values[draws].sum(axis=1, dtype=np.float64) / count
        lower = np.nextafter(samples - error, -np.inf)
        upper = np.nextafter(samples + error, np.inf)
        candidates = _candidate_sets(lower, upper, ranks)
        selected = np.zeros(replicates, dtype=bool)
        for possible, _ in candidates.values():
            selected[possible] = True
        # Equal exact ratios share an integer coefficient; grouping changes only
        # the exact arithmetic cost, never the owner-index resampling stream.
        group_lookup, group_coefficients, inverse = {}, [], []
        for coefficient in scaled:
            if coefficient not in group_lookup:
                group_lookup[coefficient] = len(group_coefficients)
                group_coefficients.append(coefficient)
            inverse.append(group_lookup[coefficient])
        inverse = np.asarray(inverse, dtype=np.int64)
        rng = np.random.default_rng(seed)
        exact_draws = {}
        for start in range(0, replicates, DRAW_BLOCK):
            draws = rng.integers(0, count, size=(min(DRAW_BLOCK, replicates - start), count))
            for position in np.flatnonzero(selected[start:start + len(draws)]):
                counts = np.bincount(inverse[draws[position]], minlength=len(group_coefficients))
                value = sum(int(c) * a for c, a in zip(counts, group_coefficients) if c)
                sample_index = start + int(position)
                fraction = Fraction(value, denominator)
                if not (Fraction.from_float(float(lower[sample_index])) <= fraction
                        <= Fraction.from_float(float(upper[sample_index]))):
                    raise ArithmeticError("Exact fallback value escaped its certified interval")
                exact_draws[sample_index] = value
        order_values = {rank: sorted(exact_draws[int(i)] for i in possible)[local_rank]
                        for rank, (possible, local_rank) in candidates.items()}
        metadata.update({"path": "certified_float_screen_then_exact_integer_candidate_order_statistics",
                         "floating_screen_error_bound": error, "floating_screen_error_bound_exact": fraction_record(exact_error),
                         "exact_fallback_draws": len(exact_draws),
                         "candidate_rank_counts": {str(rank): len(possible) for rank, (possible, _) in candidates.items()},
                         "exact_value_groups": len(group_coefficients)})
    low, high = (_interpolate(order_values, plan, denominator) for plan in plans)
    return mean, low, high, samples, metadata


def image_balanced_bootstrap(numerators, denominators, image_ids):
    """Fixed100k paired-image bootstrap of three-seed per-image rational changes.

numerators: summed correct-count differences across seeds17,29,43.
denominators: three times that image's original eligible triplet count.
image_ids: distinct image IDs; there must be exactly one observation per image.
No triple-weighted pooling and no resampling of seeds occurs.
"""
    n, d = _validated_counts(numerators, denominators)
    ids = np.asarray(image_ids)
    if ids.shape != n.shape or ids.dtype.kind not in "USiu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Require one distinct native string/integer image ID per observation")
    order = np.argsort(ids, kind="stable")
    mean, low, high, samples, metadata = _bootstrap_core(n[order], d[order])
    effect = {"difference": float(mean), "exact_difference": fraction_record(mean),
              "ci_lower": float(low), "ci_upper": float(high),
              "exact_ci_lower": fraction_record(low), "exact_ci_upper": fraction_record(high),
              "replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "family_size": FAMILY_SIZE,
              "familywise_alpha": .05, "tail_probability": float(TAIL), "exact_tail_probability": fraction_record(TAIL),
              "image_clusters": len(n), "items": len(n), "fixed_training_seeds": [17, 29, 43],
              "algorithm": "direct_uniform_paired_image_cluster_mean_of_rational_fixed_three_seed_changes",
              "cluster_order": "ascending unique image ID", "arithmetic": metadata,
              "conditioning": "fixed selected states and fixed galleries; eligible images uniformly resampled; no training-seed resampling",
              "boundary_comparison": "exact_rational_linear_percentile; equality_fails_strict_inequality",
              "sample_storage": "float64 descriptive draws; exact endpoints reconstructed from the same owner-index draw stream, never gated from rounded draws"}
    return effect, samples


def _exact_entry(effect, name):
    record = effect["exact_" + name]
    if (set(record) != {"numerator", "denominator"} or type(record["numerator"]) is not int
            or type(record["denominator"]) is not int or record["denominator"] <= 0):
        raise ValueError("Invalid exact rational effect record")
    value = Fraction(record["numerator"], record["denominator"])
    if float(value) != effect[name]:
        raise ValueError("Displayed effect differs from its exact rational value")
    return value


def exact_lower_exceeds(effect, threshold):
    if type(threshold) is not int and not isinstance(threshold, Fraction):
        raise TypeError("Scientific threshold must be an exact Fraction or integer")
    return _exact_entry(effect, "ci_lower") > threshold


def exact_mean_nonnegative(effect):
    return _exact_entry(effect, "difference") >= 0
