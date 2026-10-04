"""Synthetic score and worker invariants; never reads benchmark outcomes."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from gcr.practical_text_v6 import load_text_tower
from gcr.practical_text_training_v6 import LastTextBlock
from gcr.practical_text_parallel_v6 import encode_parallel
from gcr.practical_text_inference_v6 import encode_token_rows
import evaluate_practical_text_v6 as production
from audit_practical_text_v6 import Audit, independently_encode


class TokenEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(839)
        self.images = rng.normal(size=(7, 9)).astype(np.float32)
        self.texts = rng.normal(size=(35, 9)).astype(np.float32)
        self.images /= np.linalg.norm(self.images, axis=1, keepdims=True)
        self.texts /= np.linalg.norm(self.texts, axis=1, keepdims=True)
        self.texts[1] = self.texts[0]
        self.images[1] = self.images[0]
        self.delta = rng.normal(size=(35, 9)).astype(np.float32) * .03
        self.delta[1] = self.delta[0]
        self.arrays = {"image_features": self.images, "text_features": self.texts,
                       "image_ids": np.asarray([f"i{i}" for i in range(7)]), "text_ids": np.asarray([f"t{i}" for i in range(35)])}
        self.manifest = {"pairs": [{"image_id": f"i{i // 5}", "text_id": f"t{i}", "relation": "source"} for i in range(35)]}
        self.dataset = {"manifest": self.manifest, "arrays": self.arrays}

    def test_full_gallery_matches_independent_and_zero_baseline(self):
        for delta in (self.delta, np.zeros_like(self.delta)):
            metrics, raw = production.retrieval(self.dataset, delta, .01, block_size=3)
            Audit().token_retrieval(raw, self.manifest, self.arrays, delta, .01, metrics)
        baseline = self.images.astype(float) @ self.texts.astype(float).T
        np.testing.assert_array_equal(raw["i2t_top_indices"], baseline.argmax(axis=1))
        np.testing.assert_array_equal(raw["t2i_top_indices"], baseline.argmax(axis=0))

    def test_both_positive_endpoint_and_exact_tie_failure(self):
        self.manifest["triplets"] = [{"id": "a", "image_id": "i0", "category": "test", "positive1_id": "t0", "positive2_id": "t1", "negative_id": "t0"},
                                     {"id": "b", "image_id": "i3", "category": "test", "positive1_id": "t2", "positive2_id": "t3", "negative_id": "t4"}]
        metrics, raw = production.triplets(self.dataset, self.delta, .01)
        self.assertFalse(raw["correct"][0])
        Audit().token_triplets(raw, self.manifest, self.arrays, self.delta, .01, metrics)

    def test_parallel_canonical_encoding_and_independent_prefixes(self):
        tower = load_text_tower("vit_b32", ROOT / "data/practical_v6_models")
        model = LastTextBlock(tower)
        with torch.no_grad():
            model.final_norm.bias.add_(torch.linspace(-.0001, .0001, 512))
        tokens = torch.zeros((5, 77), dtype=torch.long)
        for i in range(5):
            tokens[i, 0] = 49406
            tokens[i, 1:9 + i] = torch.arange(20, 28 + i)
            tokens[i, 9 + i] = 49407
        tokens[4] = tokens[0]
        direct = encode_token_rows(tower, [model], tokens)
        with tempfile.TemporaryDirectory(prefix="synthetic_token_eval_", dir=ROOT / "results/practical_v6") as temp:
            temp = Path(temp)
            checkpoint = temp / "checkpoint.pt"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            parallel = encode_parallel("vit_b32", ROOT / "data/practical_v6_models", [checkpoint], tokens.numpy(), temp / "parallel", workers=2)
            for field in ("delta", "learned", "reference"):
                np.testing.assert_array_equal(direct[field], parallel[field])
            independent = independently_encode("vit_b32", [{"path": str(checkpoint), "sha256": production.digest(checkpoint)}],
                                                tokens.numpy(), parallel["canonical_prefix_chunks"], workers=2)
            for field in ("delta", "learned", "reference"):
                np.testing.assert_array_equal(independent[field], parallel[field])


if __name__ == "__main__":
    unittest.main()
