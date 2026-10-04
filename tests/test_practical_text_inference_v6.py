"""Canonical token-family invariants on synthetic token sequences only."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_text_v6 import load_text_tower
from gcr.practical_text_training_v6 import LastTextBlock
from gcr.practical_text_inference_v6 import encode_token_rows, normalized_cache, paired_scores, score_matrix


class CanonicalTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(692)
        cls.tower = load_text_tower("vit_b32", ROOT / "data/practical_v6_models")
        cls.initial = LastTextBlock(cls.tower).eval()
        cls.trained = LastTextBlock(cls.tower).eval()
        with torch.no_grad():
            cls.trained.final_norm.bias.add_(torch.linspace(-.0001, .0001, 512))
        cls.tokens = torch.zeros((5, 77), dtype=torch.long)
        for row, length in enumerate((9, 17, 23, 10, 9)):
            cls.tokens[row, 0] = 49406
            cls.tokens[row, 1:length - 1] = torch.randint(10, 49000, (length - 2,))
            cls.tokens[row, length - 1] = 49407
        cls.tokens[4] = cls.tokens[0]
        cls.result = encode_token_rows(cls.tower, [cls.initial, cls.trained], cls.tokens)

    def test_initial_delta_is_bitwise_zero(self):
        self.assertEqual(np.count_nonzero(self.result["delta"][0]), 0)
        self.assertGreater(np.max(np.abs(self.result["delta"][1])), 0)

    def test_duplicate_and_gallery_order_invariance(self):
        order = [3, 4, 1, 0, 2]
        reordered = encode_token_rows(self.tower, [self.trained], self.tokens[order])
        np.testing.assert_array_equal(reordered["delta"][0], self.result["delta"][1, order])
        np.testing.assert_array_equal(self.result["delta"][:, 0], self.result["delta"][:, 4])
        self.assertEqual(self.result["unique_token_sequences"], 4)

    def test_same_score_zero_baseline_and_bounded_change(self):
        x = normalized_cache(np.random.default_rng(733).normal(size=(5, 512)).astype(np.float32))
        t = self.result["reference"][0]
        delta = self.result["delta"][1]
        base = np.sum(x.astype(float) * t.astype(float), axis=1)
        np.testing.assert_array_equal(paired_scores(x, t, np.zeros_like(delta), .01), base)
        score = paired_scores(x, t, delta, .01)
        self.assertTrue(np.all(np.abs(score - base) <= .01))
        np.testing.assert_allclose(np.diag(score_matrix(x, t, delta, .01)), score, rtol=0, atol=2e-16)

    def test_reference_changes_are_rejected(self):
        broken = LastTextBlock(self.tower).eval()
        with torch.no_grad():
            broken.reference_norm.bias.add_(.1)
        with self.assertRaises(ValueError):
            encode_token_rows(self.tower, [self.initial, broken], self.tokens)


if __name__ == "__main__":
    unittest.main()
