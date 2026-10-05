"""Independent synthetic review; never accesses research data or fits a model."""
import unittest

import numpy as np

from gcr.practical_constrained_v8 import (
    CompositionExamples, ConstrainedBilinearScorer, Edge, FitConfig,
    FullGalleryConstraints,
)
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer, exact_retrieval


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def scalar_scores(images, texts, scorer):
    """Independent pair enumeration, fixed reductions, no gallery GEMM."""
    result = np.empty((len(images), len(texts)), dtype=np.float64)
    for i, image in enumerate(images):
        x = np.einsum("d,dr->r", image - scorer.image_mean, scorer.image_basis, optimize=False)
        xa = np.einsum("r,rs->s", x, scorer.coefficient, optimize=False)
        for j, text in enumerate(texts):
            y = np.einsum("d,dr->r", text - scorer.text_mean, scorer.text_basis, optimize=False)
            result[i, j] = np.sum(image * text) + np.sum(xa * y)
    return result


class IndependentConstrainedReview(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4821)
        self.images = unit(rng.normal(size=(5, 7)))
        self.source = unit(np.repeat(self.images, 2, axis=0) + .05 * rng.normal(size=(10, 7)))
        self.texts = np.concatenate((self.source, unit(rng.normal(size=(9, 7)))))
        self.owner = np.repeat(np.arange(5), 2)
        self.scorer = ConstrainedBilinearScorer.from_training(self.images, self.texts, 3)
        self.config = FitConfig(rank=3, temperature=.17)

    def test_full_gradient_and_convexity_with_unequal_image_pair_counts(self):
        positives = [[0, 1], [2], [4, 5, 10], [6], [8, 9]]
        negatives = [[11], [11, 12, 13], [14, 15], [16, 17, 18], [10, 11]]
        examples = CompositionExamples(self.images, self.texts, positives, negatives, self.scorer)
        rng = np.random.default_rng(57)
        a, b = rng.normal(size=(2, 3, 3)) * .07

        def objective(matrix):
            loss, grad = examples.loss_gradient(matrix, examples.eligible, self.config)
            return loss + self.config.ridge * np.sum(matrix * matrix) / 2, grad + self.config.ridge * matrix

        value, gradient = objective(a)
        for i, j in np.ndindex(a.shape):
            plus, minus = a.copy(), a.copy()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            numerical = (objective(plus)[0] - objective(minus)[0]) / 2e-6
            self.assertAlmostEqual(numerical, gradient[i, j], places=8)
        # Average images equally, rather than weighting larger pair sets more.
        singleton_mean = np.mean([examples.loss_gradient(a, [i], self.config)[0] for i in examples.eligible])
        self.assertAlmostEqual(value - self.config.ridge * np.sum(a * a) / 2, singleton_mean, places=14)
        fraction = .371
        self.assertLessEqual(objective(fraction * a + (1-fraction) * b)[0], fraction * value + (1-fraction) * objective(b)[0] + 1e-14)

    def test_every_protected_query_competitor_is_scanned(self):
        constraints = FullGalleryConstraints(self.images, self.source, self.owner, self.scorer, .5)
        a = -3 * np.eye(3)
        result = constraints.scan(a)
        x, y = self.scorer.image_coordinates(self.images), self.scorer.text_coordinates(self.source)
        baseline = scalar_scores(self.images, self.source, self.scorer)
        residual = np.array([[np.sum((x[i] @ a) * y[j]) for j in range(len(y))] for i in range(len(x))])
        slacks, expected_edges = [], []
        for i in range(len(x)):
            if self.owner[baseline[i].argmax()] != i:
                continue
            owned = np.flatnonzero(self.owner == i)
            positive = owned[np.argmax(baseline[i, owned])]
            for j in np.flatnonzero(self.owner != i):
                slack = .5 * (baseline[i, positive] - baseline[i, j]) + residual[i, positive] - residual[i, j]
                slacks.append(slack)
                expected_edges.append(Edge(0, i, int(j)))
        for j in range(len(y)):
            if baseline[:, j].argmax() != self.owner[j]:
                continue
            for i in np.flatnonzero(np.arange(len(x)) != self.owner[j]):
                slack = .5 * (baseline[self.owner[j], j] - baseline[i, j]) + residual[self.owner[j], j] - residual[i, j]
                slacks.append(slack)
                expected_edges.append(Edge(1, j, int(i)))
        self.assertEqual(result["checked_constraints"], len(slacks))
        self.assertEqual(result["violated_constraints"], int((np.asarray(slacks) < -1e-12).sum()))
        self.assertAlmostEqual(result["min_constraint_slack"], min(slacks), places=13)
        left, right, rhs = constraints.edge_factors(expected_edges)
        np.testing.assert_allclose(np.sum((left @ a) * right, axis=1) - rhs, slacks, atol=2e-15, rtol=1e-14)

    def test_constraint_penalty_gradient_all_coordinates(self):
        constraints = FullGalleryConstraints(self.images, self.source, self.owner, self.scorer)
        a = -3 * np.eye(3)
        constraints.scan(a)
        value, gradient = constraints.penalty(a)
        self.assertGreater(value, 0)
        for i, j in np.ndindex(a.shape):
            plus, minus = a.copy(), a.copy()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            numerical = (constraints.penalty(plus)[0] - constraints.penalty(minus)[0]) / 2e-6
            self.assertAlmostEqual(numerical, gradient[i, j], places=8)

    def test_canonical_frozen_protection_including_roundoff_sensitive_ties(self):
        rng = np.random.default_rng(187)
        # Adjacent representable vectors expose BLAS reduction-order ties.
        for _ in range(32):
            image = unit(rng.normal(size=(1, 32)))[0]
            text = unit(rng.normal(size=(1, 32)))[0]
            adjacent = text.copy()
            adjacent[0] = np.nextafter(adjacent[0], np.inf)
            images, texts = np.stack((image, -image)), np.stack((text, adjacent))
            scorer = ConstrainedBilinearScorer(np.zeros(32), np.zeros(32), np.eye(32), np.eye(32), np.zeros((32, 32)))
            baseline = scalar_scores(images, texts, scorer)
            constraints = FullGalleryConstraints(images, texts, [0, 1], scorer)
            np.testing.assert_array_equal(constraints.frozen_i2t, baseline.argmax(axis=1))
            np.testing.assert_array_equal(constraints.frozen_t2i, baseline.argmax(axis=0))
            self.assertTrue(constraints.scan(scorer.coefficient, canonical=True)["ranking_preserved"])

    def test_tolerance_is_not_a_substitute_for_direct_ranking(self):
        images = np.eye(2)
        texts = np.array([[1., 0.], [1., 1.]])
        scorer = ConstrainedBilinearScorer(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
        constraints = FullGalleryConstraints(images, texts, [0, 1], scorer)
        coefficient = np.array([[0., 1e-14], [0., 0.]])
        certificate = constraints.scan(coefficient, tolerance=1e-12, canonical=True)
        self.assertTrue(certificate["feasible_with_tolerance"])
        self.assertTrue(certificate["ranking_checked_canonically"])
        self.assertFalse(certificate["ranking_preserved"])
        self.assertEqual(certificate["lost_frozen_correct_i2t"], 1)

    def test_scoring_and_retrieval_match_independent_full_enumeration(self):
        self.scorer.coefficient = np.random.default_rng(910).normal(size=(3, 3)) * .03
        scorer = CanonicalScorer(self.scorer.image_mean, self.scorer.text_mean, self.scorer.image_basis,
                                 self.scorer.text_basis, self.scorer.coefficient)
        expected = scalar_scores(self.images, self.source, self.scorer)
        for block in (1, 2, 20):
            _, raw = exact_retrieval(self.images, self.source, self.owner, scorer, query_block=block)
            np.testing.assert_array_equal(raw["i2t_top_indices"], expected.argmax(axis=1))
            np.testing.assert_array_equal(raw["t2i_top_indices"], expected.argmax(axis=0))
            np.testing.assert_array_equal(raw["i2t_top_scores"], expected.max(axis=1))
            np.testing.assert_array_equal(raw["t2i_top_scores"], expected.max(axis=0))
        np.testing.assert_array_equal(self.scorer.score_pairs(self.images, self.source[::2]), scorer.pair_scores(self.images, self.source[::2]))

    def test_double_centering_and_operator_bound(self):
        self.scorer.coefficient = np.random.default_rng(78).normal(size=(3, 3)) * .07
        coordinates_i = self.scorer.image_coordinates(self.images)
        coordinates_t = self.scorer.text_coordinates(self.texts)
        residual = coordinates_i @ self.scorer.coefficient @ coordinates_t.T
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-17)
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=1e-17)
        self.assertLessEqual(np.abs(residual).max(), self.scorer.unit_input_residual_bound())


if __name__ == "__main__":
    unittest.main()
