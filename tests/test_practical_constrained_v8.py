"""Small, targeted checks for the new convex score and finite-gallery certificate."""
from dataclasses import replace
import unittest

import numpy as np

from gcr.practical_constrained_v8 import (
    CompositionExamples, ConstrainedBilinearScorer, Edge, FitConfig,
    FullGalleryConstraints, fit, radius_headroom, repair_feasibility,
)


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def fixture(seed=3):
    rng = np.random.default_rng(seed)
    images = unit(rng.normal(size=(8, 10)))
    source = unit(images + 0.03 * rng.normal(size=images.shape))
    other = unit(rng.normal(size=(16, 10)))
    texts = np.concatenate((source, other))
    scorer = ConstrainedBilinearScorer.from_training(images, texts, 4)
    composition = CompositionExamples(images, texts, [[8+i] for i in range(8)], [[16+i] for i in range(8)], scorer)
    constraints = FullGalleryConstraints(images, source, np.arange(8), scorer)
    return scorer, composition, constraints, images, texts


class ConstrainedTests(unittest.TestCase):
    def test_centering_and_pair_batch_invariance(self):
        scorer, _, _, images, texts = fixture()
        scorer.coefficient[:] = np.random.default_rng(9).normal(size=(4, 4)) * .01
        residual = scorer.image_coordinates(images) @ scorer.coefficient @ scorer.text_coordinates(texts).T
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-16)
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=1e-16)
        whole = scorer.score_pairs(images, texts[:len(images)])
        separate = np.asarray([scorer.score_pairs(images[i:i+1], texts[i:i+1])[0] for i in range(len(images))])
        np.testing.assert_array_equal(whole, separate)
        np.testing.assert_allclose(whole, scorer.score_matrix(images, texts[:len(images)]).diagonal(), atol=1e-15)

    def test_composition_gradient_finite_difference(self):
        scorer, composition, _, _, _ = fixture()
        config = FitConfig(rank=4)
        coefficient = np.random.default_rng(1).normal(size=(4, 4)) * .02
        value, gradient = composition.loss_gradient(coefficient, composition.eligible, config)
        self.assertTrue(np.isfinite(value))
        for ij in [(0, 0), (1, 3), (3, 2)]:
            plus, minus = coefficient.copy(), coefficient.copy()
            plus[ij] += 1e-6; minus[ij] -= 1e-6
            numerical = (composition.loss_gradient(plus, composition.eligible, config)[0] - composition.loss_gradient(minus, composition.eligible, config)[0]) / 2e-6
            self.assertAlmostEqual(numerical, gradient[ij], places=7)

    def test_full_gallery_repair_and_penalty_gradient(self):
        scorer, _, constraints, _, _ = fixture()
        coefficient = -4 * np.eye(4)
        before = constraints.scan(coefficient)
        self.assertGreater(before["violated_constraints"], 0)
        value, gradient = constraints.penalty(coefficient)
        self.assertGreater(value, 0)
        plus, minus = coefficient.copy(), coefficient.copy()
        plus[1, 1] += 1e-6; minus[1, 1] -= 1e-6
        numerical = (constraints.penalty(plus)[0] - constraints.penalty(minus)[0]) / 2e-6
        self.assertAlmostEqual(numerical, gradient[1, 1], places=7)
        fixed, cert = repair_feasibility(coefficient, constraints, FitConfig(rank=4, radius=2, repair_rounds=2, directional_passes=1))
        self.assertTrue(cert["ranking_preserved"])
        self.assertTrue(cert["feasible_with_tolerance"])
        self.assertLessEqual(np.linalg.norm(fixed), 2 + 1e-14)

    def test_tied_correct_queries_and_wrong_predictions(self):
        # Image 0 has a tied correct top caption through the lowest-index rule.
        # Caption 1 has a wrong frozen image prediction and must not be protected.
        images = np.array([[1., 0.], [0., 1.]])
        texts = np.array([[1., 0.], [1., 0.]])
        scorer = ConstrainedBilinearScorer(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
        constraints = FullGalleryConstraints(images, texts, [0, 1], scorer)
        self.assertTrue(constraints.protect_i2t[0])
        self.assertFalse(constraints.protect_t2i[1])
        self.assertEqual(constraints.i_margin[0, 1], 0)
        zero = constraints.scan(np.zeros((2, 2)))
        self.assertTrue(zero["ranking_preserved"])
        left, right, rhs = constraints.edge_factors([Edge(0, 0, 1)])
        self.assertEqual(rhs[0], 0)
        np.testing.assert_array_equal(right, np.zeros((1, 2)))

    def test_radius_bound_and_headroom(self):
        scorer, composition, _, images, texts = fixture()
        scorer.coefficient[:] = np.random.default_rng(7).normal(size=(4, 4)) * .1
        residual = scorer.score_matrix(images, texts) - images @ texts.T
        self.assertLessEqual(np.abs(residual).max(), scorer.unit_input_residual_bound())
        report = radius_headroom(scorer, composition, [.01, .1])
        self.assertLessEqual(report["radii"][0]["optimistic_repairable_failure_mass"], report["radii"][1]["optimistic_repairable_failure_mass"])

    def test_stochastic_fit_has_nonzero_feasible_selected_state(self):
        config = FitConfig(rank=4, epochs=3, batch_size=2, radius=.1, learning_rate=.01, repair_rounds=1, directional_passes=1)
        scorer, composition, constraints, _, _ = fixture()
        report = fit(scorer, composition, constraints, config)
        self.assertGreater(np.linalg.norm(scorer.coefficient), 0)
        self.assertTrue(report["final_certificate"]["ranking_preserved"])
        self.assertFalse(report["development_or_test_used"])
        other, composition2, constraints2, _, _ = fixture()
        fit(other, composition2, constraints2, replace(config, seed=29))
        self.assertFalse(np.array_equal(scorer.coefficient, other.coefficient))


if __name__ == "__main__":
    unittest.main()
