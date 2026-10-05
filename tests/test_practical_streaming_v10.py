"""Numerical equivalence of streamed full galleries and the frozen v9 objective."""
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer, exact_retrieval
from gcr.practical_joint_v9 import FullGallerySourceLoss, JointCompositionExamples, JointFitConfig, fit_joint
from gcr.practical_streaming_v10 import FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss, cached_exact_retrieval


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(414)
        self.images = unit(rng.normal(size=(7, 9)))
        self.owner = np.asarray([3, 1, 0, 0, 4, 2, 3, 3, 2, 6, 6, 5, 6])
        self.texts = unit(self.images[self.owner] + .3 * rng.normal(size=(len(self.owner), 9)))
        self.all_texts = np.concatenate((self.texts, unit(rng.normal(size=(14, 9)))))
        self.model = ConstrainedBilinearScorer.from_training(self.images, self.all_texts, 4)
        self.point = rng.normal(size=(4, 4)) * .4
        cache_started = time.monotonic()
        self.cache = FrozenScoreCache.create(Path(self.tmp.name) / 'cache', self.images, self.texts, self.owner, block_size=3)
        self.cache_elapsed = time.monotonic() - cache_started
        self.dense = FullGalleryConstraints(self.images, self.texts, self.owner, self.model)
        self.stream = StreamingFullGalleryConstraints(self.images, self.texts, self.owner, self.model, self.cache, block_size=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_build_timer_bounded_by_outer_elapsed(self):
        self.assertGreaterEqual(self.cache.metadata['build_seconds'], 0)
        self.assertLessEqual(self.cache.metadata['build_seconds'], self.cache_elapsed)

    def test_disk_orientations_ownership_and_frozen_protection(self):
        self.assertIsInstance(self.cache.i2t, np.memmap)
        self.assertFalse(self.cache.i2t.flags.writeable)
        np.testing.assert_array_equal(self.cache.i2t, self.cache.t2i.T)
        np.testing.assert_allclose(self.cache.i2t, self.dense.frozen, atol=2e-16, rtol=0)
        for name in ('best_positive', 'protect_i2t', 'protect_t2i', 'frozen_i2t', 'frozen_t2i'):
            np.testing.assert_array_equal(getattr(self.stream, name), getattr(self.dense, name))

    def test_full_and_sampled_query_losses_all_gradients(self):
        dense = FullGallerySourceLoss(self.dense, 13)
        for block in (1, 2, 3, 64):
            streamed = StreamingFullGallerySourceLoss(self.stream, 13, query_block=block)
            for selected in (None, [5, 0, 6], [2], [2, 0, 5, 6, 1]):
                with self.subTest(block=block, selected=selected):
                    dv, dg = dense.loss_gradient(self.point, selected)
                    sv, sg = streamed.loss_gradient(self.point, selected)
                    self.assertAlmostEqual(dv, sv, places=12)
                    np.testing.assert_allclose(dg, sg, rtol=2e-13, atol=2e-14)
        streamed = StreamingFullGallerySourceLoss(self.stream, 13, query_block=2)
        _, analytic = streamed.loss_gradient(self.point)
        for index in np.ndindex(self.point.shape):
            plus, minus = self.point.copy(), self.point.copy()
            plus[index] += 1e-6; minus[index] -= 1e-6
            finite = (streamed.loss_gradient(plus)[0] - streamed.loss_gradient(minus)[0]) / 2e-6
            self.assertAlmostEqual(analytic[index], finite, places=6)

    def test_complete_constraint_scan_penalty_and_radial_factor(self):
        for coefficient in (np.zeros_like(self.point), self.point, 4*self.point):
            dense = self.dense.scan(coefficient, canonical=True)
            stream = self.stream.scan(coefficient, canonical=True)
            for key in dense:
                if key == 'min_constraint_slack':
                    self.assertAlmostEqual(dense[key], stream[key], places=13)
                else:
                    self.assertEqual(dense[key], stream[key], key)
            self.assertEqual(self.dense.active, self.stream.active)
            dv, dg = self.dense.penalty(coefficient)
            sv, sg = self.stream.penalty(coefficient)
            self.assertAlmostEqual(dv, sv, places=13)
            np.testing.assert_allclose(dg, sg, atol=2e-15)
            self.assertAlmostEqual(self.dense.radial_feasibility_factor(coefficient),
                                   self.stream.radial_feasibility_factor(coefficient), places=13)
            for df, sf in zip(self.dense.edge_factors(), self.stream.edge_factors()):
                np.testing.assert_allclose(df, sf, atol=2e-16)

    def test_canonical_retrieval_matches_exhaustive_scorer_at_every_batch_size(self):
        self.model.coefficient = self.point
        canonical = CanonicalScorer(self.model.image_mean, self.model.text_mean, self.model.image_basis,
                                    self.model.text_basis, self.point)
        for scorer, reference in ((None, None), (self.model, canonical)):
            _, dense = exact_retrieval(self.images, self.texts, self.owner, reference)
            for block in (1, 3, 64):
                stream = cached_exact_retrieval(self.images, self.texts, self.owner, self.cache, scorer, query_block=block)
                for direction in ('i2t', 't2i'):
                    for name in ('top_indices', 'top_scores', 'correct'):
                        np.testing.assert_array_equal(dense[direction+'_'+name], stream[direction+'_'+name])

    def test_identical_caption_columns_and_exact_ties_preserve_manifest_order(self):
        images = unit(np.asarray([[1., 1., 0], [1., 1., 0], [0, 0, 1]]))
        texts = images[[1, 0, 2, 0, 1]]
        owner = np.asarray([1, 0, 2, 0, 1])
        model = ConstrainedBilinearScorer.from_training(images, texts, 2)
        cache = FrozenScoreCache.create(Path(self.tmp.name)/'ties', images, texts, owner, block_size=1)
        constraints = StreamingFullGalleryConstraints(images, texts, owner, model, cache, block_size=2)
        np.testing.assert_array_equal(constraints.best_positive, [1, 0, 2])
        self.assertEqual(cache.i2t.shape[1], 5)
        _, dense = exact_retrieval(images, texts, owner)
        stream = cached_exact_retrieval(images, texts, owner, cache)
        for direction in ('i2t', 't2i'):
            np.testing.assert_array_equal(dense[direction+'_top_indices'], stream[direction+'_top_indices'])
        certificate = constraints.scan(np.zeros((2, 2)), canonical=True)
        self.assertTrue(certificate['ranking_preserved'])
        self.assertEqual(certificate['min_constraint_slack'], 0)

    def test_input_identity_and_cache_corruption_rejected(self):
        wrong = self.images.copy(); wrong[0, 0] += 1e-15
        with self.assertRaises(ValueError):
            FrozenScoreCache.open(self.cache.directory, wrong, self.texts, self.owner)
        with self.assertRaises(ValueError):
            StreamingFullGalleryConstraints(wrong, self.texts, self.owner, self.model, self.cache)
        with self.assertRaises(ValueError):
            cached_exact_retrieval(wrong, self.texts, self.owner, self.cache)
        with (self.cache.directory/'image_to_text.npy').open('r+b') as stream:
            stream.seek(-1, 2); value = stream.read(1); stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaises(ValueError):
            FrozenScoreCache.open(self.cache.directory, self.images, self.texts, self.owner)

    def test_unchanged_joint_fit_fixed_multiplier_and_numerical_state_equivalence(self):
        source = [np.flatnonzero(self.owner == i) for i in range(7)]
        supported = [[len(self.texts)+i] for i in range(7)]
        supported[1] = []
        negative = [[len(self.texts)+7+i] for i in range(7)]
        composition = JointCompositionExamples(self.images, self.all_texts, source, supported, negative, self.model)
        config = JointFitConfig(rank=4, radius=.15, composition_weight=.25, epochs=2, batch_size=3,
                                repair_rounds=2, directional_passes=1)
        streamed = StreamingFullGallerySourceLoss(self.stream, 13, query_block=2)
        result_s = fit_joint(self.model, composition, streamed, self.stream, config)
        state_s = self.model.coefficient.copy()
        self.model.coefficient[:] = 0
        result_d = fit_joint(self.model, composition, FullGallerySourceLoss(self.dense, 13), self.dense, config)
        self.assertAlmostEqual(result_s['fixed_composition_multiplier'], result_d['fixed_composition_multiplier'], places=13)
        self.assertAlmostEqual(result_s['selected_training_objective'], result_d['selected_training_objective'], places=12)
        np.testing.assert_allclose(state_s, self.model.coefficient, atol=2e-13, rtol=2e-12)
        self.assertEqual(result_s['optimizer_steps'], 6)
        self.assertTrue(result_s['final_certificate']['ranking_preserved'])


if __name__ == '__main__':
    unittest.main()
