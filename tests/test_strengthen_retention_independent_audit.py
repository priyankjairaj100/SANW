"""Known factorial and resampling examples for the independent retention audit."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_strengthen_retention_independent import (
    PRIMARY_SEED, FACTORIAL_SEED, RATES, SEEDS, POLICIES,
    bootstrap_matrix, factorial_deltas, factorial_plan,
)


def test_directional_factorial_has_the_declared_signs_and_interaction():
    # A pure image intervention creates only an image main effect.
    source, supported = np.array([[.8]]), np.array([[.4]])
    image, reverse = np.array([[.8]]), np.array([[.4]])
    effects = factorial_deltas(source, supported, image, reverse)
    assert effects["image_main_effect"][0, 0] == pytest.approx(.4)
    assert effects["reverse_main_effect"][0, 0] == pytest.approx(0.)
    assert effects["interaction"][0, 0] == pytest.approx(0.)
    # An isolated source/source increment creates the expected interaction.
    effects = factorial_deltas(np.array([[.9]]), supported, image, reverse)
    assert effects["interaction"][0, 0] == pytest.approx(.1)


@pytest.mark.parametrize("seed", [PRIMARY_SEED, FACTORIAL_SEED])
def test_all_100000_draws_use_the_original_family_seed(seed):
    values = np.tile(np.array([1., 0., -1.]), (3, 1))
    samples = bootstrap_matrix([values], ["a", "a", "b"], seed=seed)[:, 0]
    draws = np.random.default_rng(seed).integers(0, 2, size=(100000, 2))
    table = {(0, 0): .5, (0, 1): 0., (1, 0): 0., (1, 1): -1.}
    expected = np.asarray([table[tuple(pair)] for pair in draws])
    assert np.array_equal(samples, expected)


def test_factorial_plan_ignores_wise_variants_and_requires_complete_original_grid():
    states = [{"method": method, "learning_rate": rate, "seed": seed, "epoch": epoch,
               "state_id": f"{method}-{rate}-{seed}-{epoch}", "alpha": 1., "draw_id": None}
              for method in POLICIES for rate in RATES for seed in SEEDS for epoch in range(11)]
    states.extend({"method": "wise_ft", "learning_rate": 1e-4, "seed": 17, "epoch": 3,
                   "state_id": f"wise-{alpha}", "alpha": alpha, "draw_id": None} for alpha in (.25, .5))
    plan = factorial_plan({"states": states})
    assert len(plan) == 36
    assert all(row["epoch"] == 10 for row in plan)
    with pytest.raises(ValueError, match="grid is incomplete"):
        factorial_plan({"states": states[1:]})
