"""Integer paired-outcome accumulation and exact linear percentile boundaries.

The estimand, random image-cluster resampling, seed, family and tail are unchanged.
Exact boundaries prevent rounded thirds from turning equality into improvement.
"""
from fractions import Fraction
import numpy as np


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def exact_percentile(numerators, denominators, probability):
    n, d = np.asarray(numerators, dtype=np.int64), np.asarray(denominators, dtype=np.int64)
    if (n.ndim != 1 or n.shape != d.shape or not len(n) or np.any(d <= 0)
            or np.any(n < -d) or np.any(n > d) or int(d.max()) > (1 << 30)):
        raise ValueError("Percentile requires bounded signed integer ratios")
    order = np.argsort(n.astype(np.float64) / d, kind="stable")
    a, b = n[order], d[order]
    # Products fit int64 by the explicit bound. Certify ordering exactly, even
    # when distinct ratios would round to the same floating-point number.
    if np.any(a[:-1] * b[1:] > a[1:] * b[:-1]):
        order = np.asarray(sorted(range(len(n)), key=lambda i: Fraction(int(n[i]), int(d[i]))))
    position = probability * (len(n) - 1)
    lower = position.numerator // position.denominator
    weight = position - lower
    lo = Fraction(int(n[order[lower]]), int(d[order[lower]]))
    hi = Fraction(int(n[order[min(lower + 1, len(n) - 1)]]), int(d[order[min(lower + 1, len(n) - 1)]]))
    return (1 - weight) * lo + weight * hi


def integer_cluster_bootstrap(numerators, item_denominator, clusters, *, grouped=False):
    numerator, clusters = np.asarray(numerators), np.asarray(clusters)
    if (numerator.ndim != 1 or numerator.dtype.kind != "i" or not len(numerator)
            or clusters.shape != numerator.shape or type(item_denominator) is not int or item_denominator <= 0
            or np.any(numerator < -item_denominator) or np.any(numerator > item_denominator)):
        raise ValueError("Require integer per-item paired numerators and a common positive denominator")
    _, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros(len(counts), dtype=np.int64)
    np.add.at(sums, inverse, numerator)
    rng = np.random.default_rng(20261007)
    n, d = np.empty(100000, dtype=np.int64), np.empty(100000, dtype=np.int64)
    if grouped:
        if not np.all(counts == counts[0]):
            raise ValueError("Grouped resampling requires equal cluster sizes")
        values, frequencies = np.unique(sums, return_counts=True)
        for start in range(0, len(n), 4096):
            draws = rng.multinomial(len(sums), frequencies / len(sums), size=min(4096, len(n) - start))
            n[start:start + len(draws)] = draws @ values
        d[:] = item_denominator * len(numerator)
        algorithm = "grouped_multinomial_exact_equal_cluster_distribution_integer_accumulation"
    else:
        for start in range(0, len(n), 250):
            draws = rng.integers(0, len(counts), size=(min(250, len(n) - start), len(counts)))
            n[start:start + len(draws)] = sums[draws].sum(axis=1, dtype=np.int64)
            d[start:start + len(draws)] = item_denominator * counts[draws].sum(axis=1, dtype=np.int64)
        algorithm = "direct_paired_image_cluster_ratio_integer_accumulation"
    tail = Fraction(1, 3200)
    low, high = exact_percentile(n, d, tail), exact_percentile(n, d, 1 - tail)
    difference = Fraction(int(numerator.sum()), item_denominator * len(numerator))
    effect = {"difference": float(difference), "exact_difference": fraction_record(difference),
              "ci_lower": float(low), "ci_upper": float(high),
              "exact_ci_lower": fraction_record(low), "exact_ci_upper": fraction_record(high),
              "replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80, "familywise_alpha": .05,
              "tail_probability": float(tail), "image_clusters": len(counts), "items": len(numerator),
              "algorithm": algorithm, "boundary_comparison": "exact_rational_linear_percentile; equality_fails_strict_inequality"}
    return effect, n.astype(np.float64) / d


def selected_development_uncertainty(paired):
    """Exact retrieval CI boundaries; unchanged descriptive composition intervals."""
    from .practical_constrained_evaluation_v8 import paired_cluster_bootstrap
    from .practical_inner_evaluation_v9 import METRICS, exact_changes, rational
    exact = exact_changes(paired)
    effects, samples = {}, {}
    for metric in METRICS:
        if metric in ("i2t", "t2i"):
            d = paired[f"{metric}_difference_denominators"]
            if not np.all(d == d[0]) or int(d[0]) not in (1, 3):
                raise ValueError("Development retrieval requires one state or the fixed three-seed mean")
            effects[metric], samples[metric] = integer_cluster_bootstrap(
                paired[f"{metric}_difference_numerators"], int(d[0]), paired[f"{metric}_cluster_ids"], grouped=True)
        else:
            effects[metric], samples[metric] = paired_cluster_bootstrap(paired[f"{metric}_difference"], paired[f"{metric}_cluster_ids"])
        effects[metric]["difference"] = float(rational(exact[metric]))
        effects[metric]["conditioning"] = "fixed selected states; paired image clusters resampled; composition intervals descriptive"
    return effects, samples


def lower_exceeds(effect, threshold):
    if not isinstance(threshold, (Fraction, int)):
        raise TypeError("Strict statistical boundary must be an exact Fraction or integer")
    entry = effect["exact_ci_lower"]
    lower = Fraction(int(entry["numerator"]), int(entry["denominator"]))
    if float(lower) != effect["ci_lower"]:
        raise ValueError("Displayed lower endpoint differs from its exact rational value")
    return lower > threshold
