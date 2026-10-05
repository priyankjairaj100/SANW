"""Synthetic v7 scoring, tie and study-isolation tests; no benchmark inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import evaluate_practical_text_v7 as production
import evaluate_practical_text_v6 as prior
from audit_practical_text_v7 import Audit


class V7EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(931)
        images = rng.normal(size=(7, 9)).astype(np.float32)
        texts = rng.normal(size=(35, 9)).astype(np.float32)
        images /= np.linalg.norm(images, axis=1, keepdims=True)
        texts /= np.linalg.norm(texts, axis=1, keepdims=True)
        images[1], texts[1] = images[0], texts[0]
        self.delta = rng.normal(size=texts.shape).astype(np.float32) * .03
        self.delta[1] = self.delta[0]
        self.arrays = {"image_features": images, "text_features": texts,
                       "image_ids": np.asarray([f"i{i}" for i in range(7)]),
                       "text_ids": np.asarray([f"t{i}" for i in range(35)])}
        self.manifest = {"pairs": [{"image_id": f"i{i // 5}", "text_id": f"t{i}", "relation": "source"} for i in range(35)]}
        self.dataset = {"arrays": self.arrays, "manifest": self.manifest}

    def test_full_galleries_all_alphas_and_independent_audit(self):
        for alpha in (0., .1, .2, .35, .5, .75, 1.):
            metrics, raw = production.retrieval(self.dataset, self.delta, .01, alpha, 3)
            Audit().token_retrieval(raw, self.manifest, self.arrays, self.delta, .01, alpha, metrics)
            complete = production.score_matrix(self.arrays["image_features"], self.arrays["text_features"], self.delta, .01, alpha)
            np.testing.assert_array_equal(raw["i2t_top_indices"], complete.argmax(axis=1))
            np.testing.assert_array_equal(raw["t2i_top_indices"], complete.argmax(axis=0))
            self.assertLessEqual(metrics["maximum_score_correction"], alpha * .01 + 1e-15)

    def test_zero_is_bitwise_frozen_and_one_is_v6(self):
        for alpha, delta in ((0., np.zeros_like(self.delta)), (1., self.delta)):
            _, original = prior.retrieval(self.dataset, delta, .01, 3)
            _, calibrated = production.retrieval(self.dataset, self.delta, .01, alpha, 3)
            self.assertEqual(set(original), set(calibrated))
            for field in original:
                np.testing.assert_array_equal(original[field], calibrated[field])

    def test_alpha_multiplies_after_tanh_and_duplicates_are_invariant(self):
        images = self.arrays["image_features"]
        texts = self.arrays["text_features"]
        base = images.astype(float) @ texts.astype(float).T
        correction = .01 * np.tanh((images.astype(float) @ self.delta.astype(float).T) / .01)
        actual = production.score_matrix(images, texts, self.delta, .01, .35)
        np.testing.assert_array_equal(actual, base + .35 * correction)
        wrong = base + .01 * np.tanh(.35 * (images.astype(float) @ self.delta.astype(float).T) / .01)
        self.assertGreater(float(np.max(np.abs(actual - wrong))), .001)
        np.testing.assert_array_equal(actual[0], actual[1])
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        perm = np.arange(len(texts))[::-1]
        reordered = production.score_matrix(images, texts[perm], self.delta[perm], .01, .35)
        np.testing.assert_array_equal(actual, reordered[:, perm])

    def test_strict_two_positive_caption_gate(self):
        self.manifest["triplets"] = [
            {"id": "a", "image_id": "i0", "category": "synthetic", "positive1_id": "t0", "positive2_id": "t1", "negative_id": "t0"},
            {"id": "b", "image_id": "i3", "category": "synthetic", "positive1_id": "t2", "positive2_id": "t3", "negative_id": "t4"},
        ]
        for alpha in (0., .35, 1.):
            metrics, raw = production.triplets(self.dataset, self.delta, .01, alpha)
            self.assertFalse(raw["correct"][0])
            Audit().token_triplets(raw, self.manifest, self.arrays, self.delta, .01, alpha, metrics)
            if alpha == 1:
                _, old = prior.triplets(self.dataset, self.delta, .01)
                for field in old:
                    np.testing.assert_array_equal(raw[field], old[field])

    def test_lock_rejects_manifest_from_old_study(self):
        manifest = {"encoder": "vit_b32", "protocol_sha256": "x", "family": production.FAMILY,
                    "test_outcomes_used_for_selection": False, "current_test_outcomes_used_for_selection": False,
                    "evaluation_status": production.EVALUATION_STATUS, "historical_and_v6_test_exposure": True,
                    "study_schema": "sanw_practical_text_last_block_v1"}
        with tempfile.TemporaryDirectory(dir=ROOT / "results/practical_v7") as temporary:
            filename = Path(temporary) / "manifest.json"
            filename.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Wrong v7 study schema"):
                production.verify_manifest(filename, "x")

    def test_old_checkpoint_cannot_be_relabelled(self):
        identity = {"schema": production.STUDY_SCHEMA, "source_factory": production.SOURCE_FACTORY,
                    "protocol": {"path": "protocol", "sha256": "p"}, "source_sha256": {},
                    "config": {"encoder": "vit_b32", "seed": 17}}
        ledger = {"identity": identity, "ledger_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        state = {"alpha": .35, "training_ledger": {"path": "ledger", "sha256": "l"},
                 "seed": 17, "checkpoint": "old_checkpoint"}
        with patch.object(production, "verified_record", side_effect=[ledger, {"source_hashes": {}, "fit_config": {}}]), patch.object(production.torch, "load", return_value={"schema": "sanw_practical_text_last_block_v1"}):
            with self.assertRaisesRegex(ValueError, "Checkpoint belongs to a different study"):
                production.validate_state_study(state, {"encoder": "vit_b32"}, "p")

    def test_invalid_alpha_rejected(self):
        for alpha in (-.1, 1.01, np.inf, np.nan):
            with self.assertRaises(ValueError):
                production.score_matrix(self.arrays["image_features"], self.arrays["text_features"], self.delta, .01, alpha)

    def test_capture_workers_preserve_v6_encodings_and_chunk_hashes(self):
        from gcr.practical_text_v6 import load_text_tower
        from gcr.practical_text_training_v6 import LastTextBlock
        from gcr.practical_text_parallel_v6 import encode_parallel as original
        from gcr.practical_text_parallel_v7 import encode_parallel as capturing
        tower = load_text_tower("vit_b32", ROOT / "data/practical_v6_models")
        model = LastTextBlock(tower)
        with torch.no_grad():
            model.final_norm.bias.add_(torch.linspace(-.0001, .0001, 512))
        tokens = np.zeros((4, 77), dtype=np.int64)
        for index in range(4):
            tokens[index, 0] = 49406
            tokens[index, 1:5 + index] = np.arange(20, 24 + index)
            tokens[index, 5 + index] = 49407
        tokens[3] = tokens[0]
        with tempfile.TemporaryDirectory(dir=ROOT / "results/practical_v7") as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "suffix.pt"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            expected = original("vit_b32", ROOT / "data/practical_v6_models", [checkpoint], tokens, temporary / "original", workers=2)
            actual = capturing("vit_b32", ROOT / "data/practical_v6_models", [checkpoint], tokens, temporary / "captured", workers=2)
            for field in ("delta", "learned", "reference"):
                np.testing.assert_array_equal(actual[field], expected[field])
            self.assertEqual(actual["canonical_prefix_chunks"], expected["canonical_prefix_chunks"])
            self.assertEqual(actual["canonical_prefix_sha256"], expected["canonical_prefix_sha256"])
            self.assertEqual(len(actual["prefix_cache_chunks"]), 2)
            for chunk in actual["prefix_cache_chunks"]:
                receipt = chunk["receipt"]
                self.assertEqual(production.digest(receipt["path"]), receipt["sha256"])


if __name__ == "__main__":
    unittest.main()
