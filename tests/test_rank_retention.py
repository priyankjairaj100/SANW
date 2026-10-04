"""Independent constrained-optimization and edge-case checks for rank geometry."""
import itertools
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from gcr.rank_retention import (
    failures_from_total_kl,
    forward_kl,
    rank_failure_projection,
    rank_retention_certificate,
    rank_retention_diagnostics,
)


def optimize_failure(q, relevant, k):
    """Enumerate every wrong subset; solve each primal without pooling formula."""
    q = np.asarray(q, dtype=float)
    relevant = np.asarray(relevant, dtype=int)
    wrong = np.setdiff1d(np.arange(q.size), relevant)
    best = np.inf
    for subset in itertools.combinations(wrong, k):
        # All relevant-vs-selected-wrong inequalities are explicit.
        matrix = np.zeros((len(subset) * relevant.size, q.size))
        for row, (j, i) in enumerate(itertools.product(subset, relevant)):
            matrix[row, j] = 1
            matrix[row, i] = -1
        def objective(p):
            return np.sum(xlogy(q, q)) - np.sum(xlogy(q, p))
        def gradient(p):
            return -np.divide(q, p, out=np.zeros_like(q), where=p > 0)
        results = []
        for start in [np.full(q.size, 1 / q.size), .8 * q + .2 / q.size]:
            result = minimize(
                objective, start, jac=gradient,
                method="SLSQP", bounds=[(1e-13, 1)] * q.size,
                constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1,
                              "jac": lambda p: np.ones(p.size)},
                             {"type": "ineq", "fun": lambda p, matrix=matrix: matrix @ p,
                              "jac": lambda p, matrix=matrix: matrix}],
                options={"ftol": 2e-12, "maxiter": 1000},
            )
            if result.success and np.min(matrix @ result.x) >= -1e-7:
                results.append(result)
        if not results:
            raise AssertionError("Independent optimizer failed from both starts")
        best = min(best, min(result.fun for result in results))
    return best


class RankRetentionTests(unittest.TestCase):
    def test_exact_pair_and_strict_boundary(self):
        q = np.array([.6, .3, .1])
        projection = rank_failure_projection(q, [0])
        expected = .6 * np.log(.6 / .45) + .3 * np.log(.3 / .45)
        self.assertAlmostEqual(projection.threshold, expected)
        np.testing.assert_allclose(projection.projection, [.45, .45, .1])
        boundary = rank_retention_certificate(q, projection.projection, [0])
        self.assertFalse(boundary.certified)
        self.assertFalse(boundary.student_correct)
        self.assertTrue(rank_retention_certificate(q, q, [0]).certified)

    def test_multi_positive_strictly_improves_pair_bound(self):
        q = np.array([.35, .3, .25, .1])
        exact = rank_failure_projection(q, [0, 1, 2])
        pair = .35 * np.log(.35 / .225) + .1 * np.log(.1 / .225)
        self.assertGreater(exact.threshold, pair)
        np.testing.assert_allclose(exact.projection, [.25] * 4)
        self.assertAlmostEqual(exact.threshold, forward_kl(q, exact.projection))
        self.assertFalse(rank_retention_certificate(q, exact.projection, [0, 1, 2]).certified)

    def test_teacher_incorrect_tied_and_impossible(self):
        for q in ([.3, .7], [.5, .5], [0, 1]):
            p = rank_failure_projection(np.array(q), [0])
            self.assertEqual(p.threshold, 0)
            self.assertFalse(p.teacher_correct)
        for relevant, k in [([0, 1], 1), ([0], 2)]:
            p = rank_failure_projection(np.array([.4, .6]), relevant, k)
            self.assertEqual(p.threshold, np.inf)
            self.assertTrue(p.teacher_correct)
            self.assertTrue(rank_retention_certificate(np.array([.4, .6]), np.array([0, 1]), relevant, k).certified)

    def test_zero_probabilities_and_active_pool_ties(self):
        q = np.array([.5, .5, 0, 0])
        p = rank_failure_projection(q, [0, 1])
        np.testing.assert_allclose(p.projection, [1 / 3, 1 / 3, 1 / 3, 0])
        self.assertAlmostEqual(p.threshold, np.log(1.5))
        q = np.array([.45, .25, .15, .15])
        p = rank_failure_projection(q, [0, 1])
        np.testing.assert_allclose(p.projection, [.3, .25, .3, .15])
        self.assertTrue(np.isfinite(p.threshold))
        self.assertEqual(forward_kl(np.array([1., 0]), np.array([0., 1.])), np.inf)
        self.assertEqual(forward_kl(np.array([1., 0]), np.array([1., 0])), 0)

    def test_topk_projection_and_monotone_distance(self):
        q = np.array([.30, .15, .50, .05])
        projection = rank_failure_projection(q, [0, 1], k=2)
        np.testing.assert_allclose(projection.projection, [.175, .15, .5, .175])
        self.assertFalse(rank_failure_projection(q, [0, 1], k=1).teacher_correct)
        self.assertTrue(projection.teacher_correct)
        rng = np.random.default_rng(874)
        for _ in range(50):
            q = rng.dirichlet(np.ones(9))
            values = [rank_failure_projection(q, [0, 1, 2], k).threshold for k in range(1, 8)]
            self.assertTrue(np.all(np.diff(values[:6]) >= -1e-12))
            self.assertEqual(values[-1], np.inf)

    def test_exhaustive_primal_optimizer(self):
        rng = np.random.default_rng(429)
        cases = [(np.array(q), relevant, k) for q, relevant, k in [
            ([.35, .30, .25, .10], [0, 1, 2], 1),
            ([.5, .5, 0, 0], [0, 1], 1),
            ([.30, .15, .50, .05], [0, 1], 2),
            ([.5, .25, .125, .125], [0, 1], 1),
            ([.4, .2, .2, .2], [0], 2),
        ]]
        for n in [3, 4, 5, 6]:
            for count in [1, 2]:
                for k in range(1, min(3, n - count) + 1):
                    for _ in range(2):
                        cases.append((rng.dirichlet(np.ones(n)), list(range(count)), k))
        for q, relevant, k in cases:
            with self.subTest(q=q, relevant=relevant, k=k):
                exact = rank_failure_projection(q, relevant, k)
                value = optimize_failure(q, relevant, k)
                self.assertAlmostEqual(exact.threshold, value, delta=2e-7)
                self.assertAlmostEqual(np.sum(exact.projection), 1, places=13)
                self.assertAlmostEqual(exact.threshold, forward_kl(q, exact.projection), places=12)
                self.assertFalse(rank_retention_certificate(q, exact.projection, relevant, k).certified)

    def test_random_certificates_never_assert_failure(self):
        rng = np.random.default_rng(1989)
        for _ in range(200):
            q, p = rng.dirichlet(np.ones(8), size=2)
            for k in [1, 2, 4]:
                result = rank_retention_certificate(q, p, [0, 1, 2], k)
                if result.certified:
                    self.assertTrue(result.teacher_correct)
                    self.assertTrue(result.student_correct)

    def test_logits_shift_temperature_and_extreme_scores(self):
        q = np.array([[2., 1., 0.], [1000., 0., -1000.]])
        p = q + np.array([[.01, -.01, 0], [0, .01, 0]])
        diagnostics = rank_retention_diagnostics(q, p, [[0], [0]], temperature=2)
        shifted = rank_retention_diagnostics(q + 17, p - 200, [[0], [0]], temperature=2)
        for key in diagnostics:
            np.testing.assert_allclose(diagnostics[key], shifted[key], atol=1e-12)
        self.assertTrue(np.all(diagnostics["certified"]))
        self.assertTrue(np.all(diagnostics["logit_drift_certified"]))
        self.assertTrue(np.all(np.isfinite(diagnostics["divergence"])))
        extreme_student = np.array([[-1000., 1000., 0.]])
        extreme = rank_retention_diagnostics(q[:1], extreme_student, [[0]], temperature=1)
        self.assertTrue(np.isfinite(extreme["divergence"][0]))
        self.assertFalse(extreme["certified"][0])

    def test_zero_minibatch_kl_does_not_certify_full_gallery(self):
        teacher = np.array([[2., 1., 0.]])
        student = np.array([[2., 1., 10.]])
        batch = rank_retention_diagnostics(teacher[:, :2], student[:, :2], [[0]], temperature=1)
        full = rank_retention_diagnostics(teacher, student, [[0]], temperature=1)
        self.assertEqual(batch["divergence"][0], 0)
        self.assertTrue(batch["certified"][0])
        self.assertFalse(full["student_correct"][0])
        self.assertFalse(full["certified"][0])

    def test_average_kl_does_not_certify_every_query(self):
        q = np.array([.51, .49])
        p = q[::-1]
        divergence = forward_kl(q, p)
        threshold = rank_failure_projection(q, [0]).threshold
        self.assertLess(divergence / 100, threshold)
        self.assertFalse(rank_retention_certificate(q, p, [0]).student_correct)
        self.assertEqual(failures_from_total_kl(np.array([.1, .2, .4, np.inf]), .3), 2)
        self.assertEqual(failures_from_total_kl(np.array([.125, .25, .5, np.inf]), .375), 2)
        self.assertEqual(failures_from_total_kl(np.array([.125, np.inf]), np.inf), 1)

    def test_logit_and_kl_certificates_are_complementary(self):
        q = np.array([.6, .3999, .0001])
        p = np.array([.6, .399, .001])
        first = rank_retention_diagnostics(np.log(q)[None], np.log(p)[None], [[0]], temperature=1)
        self.assertTrue(first["certified"][0])
        self.assertFalse(first["logit_drift_certified"][0])
        z = np.log(np.array([.34, .33, .33]))[None]
        second = rank_retention_diagnostics(z, z + [0, .029, .029], [[0]], temperature=1)
        self.assertFalse(second["certified"][0])
        self.assertTrue(second["logit_drift_certified"][0])
        self.assertTrue(second["combined_certified"][0])
        self.assertTrue(second["student_correct"][0])

    def test_finite_inputs_with_overflow_fail_closed(self):
        teacher = np.array([[1e308, -1e308]])
        student = -teacher
        with self.assertRaisesRegex(ValueError, "unsafe logit range"):
            rank_retention_diagnostics(teacher, student, [[0]], temperature=1)
        # Positive finite temperatures can still exceed the safe arithmetic range.
        with self.assertRaisesRegex(ValueError, "unsafe logit range"):
            rank_retention_diagnostics(np.array([[1., -1.]]), np.array([[-1., 1.]]),
                                       [[0]], temperature=np.nextafter(0., 1.))
        # A harmless within-query offset must not overflow cross-model drift.
        teacher = np.array([[1e308, 1e308]])
        student = -teacher
        with self.assertRaisesRegex(ValueError, "unsafe native logit drift"):
            rank_retention_diagnostics(teacher, student, [[0]], temperature=1)

    def test_invalid_inputs(self):
        for q in [[-.1, 1.1], [.2, .2], [np.nan, 1], []]:
            with self.assertRaises(ValueError):
                rank_failure_projection(np.array(q), [0])
        for relevant in [[], [2], [-1], [.1], [False, False]]:
            with self.assertRaises(ValueError):
                rank_failure_projection(np.array([.6, .4]), relevant)
        for k in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                rank_failure_projection(np.array([.6, .4]), [0], k)


if __name__ == "__main__":
    unittest.main()
