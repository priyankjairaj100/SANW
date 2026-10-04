"""Synthetic-only independent checks, with no held-out benchmark access."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import evaluate_practical_v6 as production
import audit_practical_v6 as independent
from gcr.practical_scorer import BoundedPairScorer


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.images = torch.nn.functional.normalize(torch.randn(7, 9), dim=1).numpy()
        self.texts = torch.nn.functional.normalize(torch.randn(35, 9), dim=1).numpy()
        self.images[1] = self.images[0]
        self.texts[1] = self.texts[0]
        self.model = BoundedPairScorer(9, epsilon=.09, hidden=12, rank=5)
        with torch.no_grad():
            self.model.output.weight.normal_(0, .5)
            self.model.output.bias.fill_(.2)
        self.model.eval()
        self.arrays = {"image_features": self.images, "text_features": self.texts,
                       "image_ids": np.asarray([f"i{i}" for i in range(7)]),
                       "text_ids": np.asarray([f"t{i}" for i in range(35)])}
        self.manifest = {"pairs": [{"image_id": f"i{i // 5}", "text_id": f"t{i}", "relation": "source"} for i in range(35)]}
        self.dataset = {"arrays": self.arrays, "manifest": self.manifest}
        self.checkpoint = {"model_config": self.model.config(), "state_dict": self.model.state_dict()}

    def test_sparse_top_one_equals_all_pairs_and_independent_audit(self):
        metrics, raw = production.retrieval(self.model, self.dataset, 3)
        base = self.images.astype(np.float64) @ self.texts.astype(np.float64).T
        ii, tt = np.indices(base.shape)
        complete = base + production.residual(self.model, self.images[ii.ravel()], self.texts[tt.ravel()]).reshape(base.shape)
        np.testing.assert_array_equal(raw["i2t_top_indices"], np.argmax(complete, axis=1))
        np.testing.assert_array_equal(raw["t2i_top_indices"], np.argmax(complete, axis=0))
        independent.Audit().retrieval(raw, self.manifest, self.arrays, self.checkpoint, metrics)

    def test_frozen_ties_and_ownership_are_exact(self):
        metrics, raw = production.retrieval(None, self.dataset, 2)
        base = self.images.astype(np.float64) @ self.texts.astype(np.float64).T
        np.testing.assert_array_equal(raw["t2i_top_indices"], np.argmax(base, axis=0))
        np.testing.assert_array_equal(raw["i2t_top_indices"], np.argmax(base, axis=1))
        independent.Audit().retrieval(raw, self.manifest, self.arrays, None, metrics)

    def test_both_positive_triplets_and_equal_score_failure(self):
        self.manifest["triplets"] = [{"id": "x", "image_id": "i0", "category": "test", "positive1_id": "t0", "positive2_id": "t1", "negative_id": "t0"},
                                     {"id": "y", "image_id": "i2", "category": "test", "positive1_id": "t3", "positive2_id": "t4", "negative_id": "t5"}]
        metrics, raw = production.triplets(self.model, self.dataset)
        self.assertFalse(raw["correct"][0])
        independent.Audit().triplets(raw, self.manifest, self.arrays, self.checkpoint, metrics)

    def test_canonical_pair_outputs_ignore_chunk_and_duplicate_position(self):
        torch.manual_seed(611)
        model = BoundedPairScorer(512, epsilon=.01, hidden=128, rank=64)
        with torch.no_grad():
            model.output.weight.normal_()
        images = torch.nn.functional.normalize(torch.randn(603, 512), dim=1).numpy()
        texts = torch.nn.functional.normalize(torch.randn(603, 512), dim=1).numpy()
        for position in (37, 257, 513, 602):
            images[position], texts[position] = images[0], texts[0]
        first = production.residual(model, images, texts, chunk=256)
        second = production.residual(model, images, texts, chunk=37)
        third = independent.Audit().residual(images, texts, {"model_config": model.config(), "state_dict": model.state_dict()})
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)
        self.assertTrue(all(first[0] == first[i] for i in (37, 257, 513, 602)))

    def test_independent_bootstrap_all_values_variable_cluster_sizes(self):
        old_p, old_a = production.REPLICATES, independent.NBOOT
        production.REPLICATES = independent.NBOOT = 1003
        try:
            delta = np.asarray([[1, 0, -1, 0, 1], [0, 1, -1, 1, 0], [1, 1, 0, -1, 0]], dtype=float)
            clusters = np.asarray(["b", "a", "a", "c", "c"])
            summary, a = production.bootstrap(delta, clusters)
            b = independent.Audit().bootstrap(delta, clusters)
            np.testing.assert_allclose(a, b, atol=2e-15, rtol=0)
            self.assertEqual(summary["family_size"], 80)
        finally:
            production.REPLICATES, independent.NBOOT = old_p, old_a


if __name__ == "__main__":
    unittest.main()
