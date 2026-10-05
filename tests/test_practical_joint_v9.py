"""Objective gradients, unbiased full-gallery query sampling, and certificates."""
from dataclasses import replace
import unittest
import numpy as np

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import FullGallerySourceLoss, JointCompositionExamples, JointFitConfig, fit_joint


def unit(values):
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def fixture():
    rng = np.random.default_rng(44)
    images = unit(rng.normal(size=(5, 8)))
    owner = np.asarray([0, 0, 1, 2, 2, 2, 3, 4, 4])
    sources = unit(images[owner] + .1 * rng.normal(size=(len(owner), 8)))
    texts = np.concatenate((sources, unit(rng.normal(size=(10, 8)))))
    model = ConstrainedBilinearScorer.from_training(images, texts, 3)
    constraints = FullGalleryConstraints(images, sources, owner, model)
    original = [[int(j) for j in np.flatnonzero(owner == i)] for i in range(len(images))]
    supported = [[len(sources)+i] for i in range(len(images))]
    negative = [[len(sources)+len(images)+i] for i in range(len(images))]
    composition = JointCompositionExamples(images, texts, original, supported, negative, model)
    retrieval = FullGallerySourceLoss(constraints, 8)
    return model, constraints, composition, retrieval


class JointTests(unittest.TestCase):
    def assert_gradient(self, function, point):
        _, gradient = function(point)
        for index in np.ndindex(point.shape):
            plus, minus = point.copy(), point.copy()
            plus[index] += 1e-6; minus[index] -= 1e-6
            numerical = (function(plus)[0] - function(minus)[0]) / 2e-6
            self.assertAlmostEqual(numerical, gradient[index], places=6)

    def test_all_retrieval_gradient_coordinates(self):
        model, _, _, retrieval = fixture()
        point = np.random.default_rng(5).normal(size=model.coefficient.shape) * .04
        self.assert_gradient(retrieval.loss_gradient, point)
        self.assert_gradient(lambda a: retrieval.loss_gradient(a, [0, 2]), point)

    def test_query_sampling_unbiased_with_unequal_owner_counts(self):
        model, constraints, _, retrieval = fixture()
        point = np.random.default_rng(9).normal(size=model.coefficient.shape) * .03
        full_value, full_gradient = retrieval.loss_gradient(point)
        batches = [retrieval.loss_gradient(point, [i]) for i in range(len(constraints.images))]
        self.assertAlmostEqual(np.mean([x[0] for x in batches]), full_value, places=13)
        np.testing.assert_allclose(np.mean([x[1] for x in batches], axis=0), full_gradient, atol=1e-14)

    def test_full_gallery_not_in_batch_denominator(self):
        _, constraints, _, retrieval = fixture()
        value, _ = retrieval.loss_gradient(np.zeros((3, 3)), [0])
        self.assertGreater(value, 0)
        # With only one owner's queries, in-batch negatives would yield zero CE.
        self.assertEqual(len(constraints.texts), 9)

    def test_all_joint_gradient_coordinates(self):
        model, _, composition, _ = fixture()
        config = JointFitConfig(rank=3)
        point = np.random.default_rng(3).normal(size=model.coefficient.shape) * .04
        self.assert_gradient(lambda a: composition.loss_gradient(a, composition.eligible, config), point)

    def test_actual_joint_objective_not_mean_pair_loss(self):
        _, _, composition, _ = fixture()
        config = JointFitConfig(rank=3)
        zero = np.zeros((3, 3))
        value, _ = composition.loss_gradient(zero, [0], config)
        source, supported, negative = composition.frozen[0]
        manual = []
        for a in source:
            for b in supported:
                for n in negative:
                    manual.append(np.logaddexp(0, np.logaddexp((n-a+config.composition_margin)/config.temperature,
                                                             (n-b+config.composition_margin)/config.temperature)))
        self.assertAlmostEqual(value, np.mean(manual), places=13)

    def test_joint_query_sampling_with_ineligible_owner(self):
        model, _, composition, _ = fixture()
        supported = list(composition.supported)
        supported[0] = []
        missing = JointCompositionExamples(composition.images, composition.texts, composition.source,
                                          supported, composition.negative, model)
        config = JointFitConfig(rank=3)
        full_value, full_gradient = missing.loss_gradient(model.coefficient, missing.eligible, config)
        singletons = [missing.query_batch_loss_gradient(model.coefficient, [i], config) for i in range(len(missing.images))]
        self.assertEqual(singletons[0][0], 0)
        self.assertAlmostEqual(np.mean([x[0] for x in singletons]), full_value, places=13)
        np.testing.assert_allclose(np.mean([x[1] for x in singletons], axis=0), full_gradient, atol=1e-14)

    def test_real_seed_variation_fixed_multiplier_and_feasibility(self):
        config = JointFitConfig(rank=3, epochs=3, batch_size=2, radius=.1, repair_rounds=1, directional_passes=1)
        model, constraints, composition, retrieval = fixture()
        result = fit_joint(model, composition, retrieval, constraints, config)
        expected = config.composition_weight * result["initial_retrieval_gradient_norm"] / result["initial_joint_gradient_norm"]
        self.assertEqual(result["fixed_composition_multiplier"], expected)
        self.assertTrue(all(row["fixed_composition_multiplier"] == expected for row in result["history"]))
        self.assertTrue(result["final_certificate"]["ranking_checked_canonically"])
        self.assertTrue(result["final_certificate"]["ranking_preserved"])
        self.assertGreater(np.linalg.norm(model.coefficient), 0)
        second, constraints2, composition2, retrieval2 = fixture()
        fit_joint(second, composition2, retrieval2, constraints2, replace(config, seed=29))
        self.assertFalse(np.array_equal(model.coefficient, second.coefficient))


if __name__ == "__main__":
    unittest.main()
