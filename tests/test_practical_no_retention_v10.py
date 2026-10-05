"""The matched control removes retention and keeps the objective/update budget."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import JointFitConfig
from gcr.practical_no_retention_v10 import fit_joint_without_retention
from test_practical_joint_v9 import fixture


class NoRetentionTests(unittest.TestCase):
    def test_exact_manual_objective_updates_selection_and_seed(self):
        model, constraints, composition, retrieval = fixture()
        config = JointFitConfig(rank=3, radius=.3, composition_weight=.25, epochs=3, batch_size=2)
        coefficient = np.zeros((3, 3))
        _, gr = retrieval.loss_gradient(coefficient)
        _, gc = composition.loss_gradient(coefficient, composition.eligible, config)
        multiplier = config.composition_weight*np.linalg.norm(gr)/max(np.linalg.norm(gc), 1e-12)
        rng, states, objectives = np.random.default_rng(config.seed), [], []
        for epoch in range(1, config.epochs+1):
            order = rng.permutation(len(constraints.images))
            for start in range(0, len(order), config.batch_size):
                query = order[start:start+config.batch_size]
                _, gr = retrieval.loss_gradient(coefficient, query)
                _, gc = composition.query_batch_loss_gradient(coefficient, query, config)
                gradient = gr+multiplier*gc+config.ridge*coefficient
                norm = np.linalg.norm(gradient)
                if norm > config.gradient_clip:
                    gradient *= config.gradient_clip/norm
                coefficient -= config.learning_rate/math.sqrt(epoch)*gradient
                norm = np.linalg.norm(coefficient)
                if norm > config.radius:
                    coefficient *= config.radius/norm
            lr, _ = retrieval.loss_gradient(coefficient)
            lc, _ = composition.loss_gradient(coefficient, composition.eligible, config)
            objectives.append(lr+multiplier*lc+config.ridge*np.sum(coefficient**2)/2)
            states.append(coefficient.copy())
        saved = []
        with patch.object(constraints, 'penalty', side_effect=AssertionError('Retention penalty forbidden')):
            with patch.object(constraints, 'scan', wraps=constraints.scan) as scan:
                result = fit_joint_without_retention(model, composition, retrieval, constraints, config,
                                                     lambda row, point: saved.append(point))
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(scan.call_args.kwargs, {'add': False, 'canonical': True})
        for observed, expected in zip(saved, states):
            np.testing.assert_array_equal(observed, expected)
        best = int(np.argmin(objectives))
        np.testing.assert_array_equal(model.coefficient, states[best])
        self.assertEqual(result['selected_epoch'], best+1)
        self.assertEqual(result['optimizer_steps'], 9)
        self.assertEqual(result['fixed_composition_multiplier'], multiplier)
        self.assertFalse(result['retention_enforced'])
        self.assertNotIn('final_certificate', result)
        second, c2, j2, r2 = fixture()
        fit_joint_without_retention(second, j2, r2, c2, replace(config, seed=29))
        self.assertFalse(np.array_equal(second.coefficient, model.coefficient))

    def test_actual_training_losses_are_reported_and_never_repaired_or_filtered(self):
        images = np.eye(2)
        model = ConstrainedBilinearScorer.from_training(images, images, 2)
        constraints = FullGalleryConstraints(images, images, np.arange(2), model)
        # Deliberately harmful differentiable objective isolates removal of the
        # retention mechanism; this is a unit-test oracle, not a training arm.
        retrieval = SimpleNamespace(loss_gradient=lambda point, image_indices=None:
                                    (float(4*np.trace(point)), 4*np.eye(2)))
        composition = SimpleNamespace(eligible=np.arange(2), summary=lambda point: {},
                                      loss_gradient=lambda point, indices, config: (0., np.zeros_like(point)),
                                      query_batch_loss_gradient=lambda point, indices, config: (0., np.zeros_like(point)))
        config = JointFitConfig(rank=2, radius=10, epochs=1, batch_size=2, learning_rate=1)
        with patch.object(constraints, 'penalty', side_effect=AssertionError('Retention penalty forbidden')):
            result = fit_joint_without_retention(model, composition, retrieval, constraints, config)
        np.testing.assert_array_equal(model.coefficient, -4*np.eye(2))
        diagnostic = result['final_training_retention_diagnostic']
        self.assertEqual(diagnostic['lost_frozen_correct_i2t'], 2)
        self.assertEqual(diagnostic['lost_frozen_correct_t2i'], 2)
        self.assertFalse(diagnostic['ranking_preserved'])
        self.assertFalse(diagnostic['feasible_with_tolerance'])
        self.assertFalse(result['finite_training_retention_guarantee_claimed'])
        self.assertTrue(diagnostic['ranking_checked_canonically'])
        self.assertEqual(len(constraints.active), 0)
        self.assertEqual(result['selected_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
