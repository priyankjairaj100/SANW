"""Synthetic checks for selection integrity and clustered replication inference."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_strengthen_replication import (POLICIES, RATES, SEEDS,
    verify_selection_plan, state_id)
from analyze_strengthen_replication import primary_analysis, selected_summaries
from gcr.strengthen_replication import NonlinearResidualAdapter, load_replication_adapter


def fixture():
    states = []
    native_epochs, retrieval_epochs = (0, 5, 10), (3, 0, 7)
    for method in POLICIES:
        for rate in RATES:
            for si, seed in enumerate(SEEDS):
                for epoch in range(11):
                    sid = state_id(method, rate, seed, epoch)
                    states.append({"state_id": sid, "method": method, "learning_rate": rate,
                        "seed": seed, "epoch": epoch, "checkpoint": sid,
                        "checkpoint_retained": True,
                        "validation": {
                            "native": {"score": .9 - .001 * abs(epoch - native_epochs[si]) - (.1 if rate == .001 else 0)},
                            "source_retrieval": {"score": .7 + rate - .001 * abs(epoch - retrieval_epochs[si])}}})
    selections = []
    for selector, rate, epochs in (("native", .0001, native_epochs), ("source_retrieval", .001, retrieval_epochs)):
        for method in POLICIES:
            for seed, epoch in zip(SEEDS, epochs):
                sid = state_id(method, rate, seed, epoch)
                selections.append({"selector": selector, "method": method, "learning_rate": rate,
                    "seed": seed, "epoch": epoch, "candidate_state_id": sid,
                    "state_id": sid if epoch else "frozen"})
    wanted = {row["state_id"] for row in selections if row["epoch"]}
    evaluated = [{"state_id": "frozen", "checkpoint": None, "epoch": 0}]
    evaluated += [row for row in states if row["epoch"] == 10 or row["state_id"] in wanted]
    return {"states": evaluated, "selections": selections}, {"states": states}


class SelectionTests(unittest.TestCase):
    def test_full_grid_selection_and_frozen_alias(self):
        manifest, full = fixture()
        self.assertEqual(verify_selection_plan(manifest, full)["selected_seed_states"], 30)
        states = {row["state_id"]: row for row in manifest["states"]}
        metrics = {sid: {"example": float(i)} for i, sid in enumerate(states)}
        summaries = selected_summaries(states, manifest["selections"], metrics)
        self.assertEqual(len(summaries["strategies"]), 10)
        native = summaries["strategies"][0]
        self.assertEqual([row["seed"] for row in native["selected_states"]], list(SEEDS))
        self.assertEqual(native["metrics"]["example"]["values"][0], metrics["frozen"]["example"])

    def test_tied_rate_cannot_change(self):
        manifest, full = fixture()
        manifest["selections"][0]["learning_rate"] = .0003
        with self.assertRaises(ValueError):
            verify_selection_plan(manifest, full)

    def test_missing_or_duplicate_epoch_rejected(self):
        manifest, full = fixture()
        full["states"][-1] = copy.deepcopy(full["states"][0])
        with self.assertRaises(ValueError):
            verify_selection_plan(manifest, full)

    def test_unselected_test_state_rejected(self):
        manifest, full = fixture()
        unwanted = next(row for row in full["states"] if row["epoch"] == 1)
        manifest["states"].append(unwanted)
        with self.assertRaises(ValueError):
            verify_selection_plan(manifest, full)


class BootstrapTests(unittest.TestCase):
    def test_all_twelve_against_direct_cluster_gather(self):
        # This synthetic example has five captions per image.
        rng = np.random.default_rng(919)
        images = np.array([f"image_{i:02d}" for i in range(13)])
        texts = np.array([f"text_{i:02d}_{j}" for i in range(13) for j in range(5)])
        owners = np.repeat(images, 5)
        states, predictions, arrays = {}, {}, {}
        for method in POLICIES:
            for rate in RATES:
                for seed in SEEDS:
                    sid = state_id(method, rate, seed, 10)
                    states[sid] = {"state_id": sid, "method": method, "learning_rate": rate, "seed": seed, "epoch": 10}
                    ranks_i = rng.integers(1, 4, len(images))
                    ranks_t = rng.integers(1, 4, len(texts))
                    predictions[(sid, "e_vil_test1000")] = {"image_ids": images, "text_ids": texts,
                        "text_source_image_ids": owners, "i2t_ranks": ranks_i, "t2i_ranks": ranks_t}
                    arrays[(method, rate, seed, "i2t.r1")] = (ranks_i == 1).astype(float)
                    arrays[(method, rate, seed, "t2i.r1")] = (ranks_t == 1).astype(float)
        for protocol_name in ("STRENGTHEN_REPLICATION_PROTOCOL.json", "STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json"):
            protocol = json.loads((ROOT / "docs" / protocol_name).read_text())
            effects, samples = primary_analysis(states, predictions, protocol)
            self.assertEqual(len(effects), 12)
            seed = protocol["primary_analysis"]["bootstrap_seed"]
            draws = np.random.default_rng(seed).integers(0, 13, size=(10000, 13))
            for effect in effects:
                rate, metric = effect["learning_rate"], effect["metric"]
                left = np.array([arrays[("supported", rate, s, metric)] for s in SEEDS])
                methods = ["source"] if effect["right"] == "source" else POLICIES[2:]
                right = np.array([[arrays[(m, rate, s, metric)] for s in SEEDS] for m in methods]).mean(axis=0)
                delta = (left - right).mean(axis=0)
                by_image = delta if metric == "i2t.r1" else delta.reshape(13, 5).mean(axis=1)
                direct = by_image[draws].mean(axis=1)
                np.testing.assert_allclose(samples[effect["effect_id"]], direct, rtol=0, atol=2e-16)
                tail = .05 / 24
                interval = np.quantile(direct, [tail, 1-tail], method="linear")
                np.testing.assert_allclose([effect["ci_lower"], effect["ci_upper"]], interval, rtol=0, atol=2e-16)
                self.assertAlmostEqual(effect["difference"], float(delta.mean()), places=14)
        missing = dict(states)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            primary_analysis(missing, predictions, protocol)


class NonlinearScoringTests(unittest.TestCase):
    def test_shared_backend_matches_direct_stable_ranks(self):
        from evaluate_allocation_distillation import score_state
        rng = np.random.default_rng(37)
        images = rng.normal(size=(4, 6)).astype(np.float32)
        texts = rng.normal(size=(8, 6)).astype(np.float32)
        images /= np.linalg.norm(images, axis=1, keepdims=True)
        texts /= np.linalg.norm(texts, axis=1, keepdims=True)
        image_ids, text_ids = np.array([f"i{i}" for i in range(4)]), np.array([f"t{i}" for i in range(8)])
        pairs = [{"image_id": image_ids[j // 2], "text_id": text_ids[j], "relation": "source"} for j in range(8)]
        dataset = {"name": "e_vil_test1000", "manifest": {"pairs": pairs},
            "features": {"image_features": images, "text_features": texts,
                         "image_ids": image_ids, "text_ids": text_ids}}
        torch.manual_seed(71)
        adapter = NonlinearResidualAdapter(6, 3)
        _, frozen = score_state(None, dataset, 2)
        _, initialized = score_state(adapter, dataset, 2)
        for key in frozen:
            np.testing.assert_array_equal(frozen[key], initialized[key])
        with torch.no_grad():
            adapter.image[2].weight.copy_(torch.tensor(rng.normal(scale=.1, size=(6, 3)), dtype=torch.float32))
            adapter.text[2].weight.copy_(torch.tensor(rng.normal(scale=.1, size=(6, 3)), dtype=torch.float32))
        restored = load_replication_adapter({"adapter": {"kind": "nonlinear", "dimension": 6, "bottleneck": 3},
                                            "state_dict": adapter.state_dict()})
        _, actual = score_state(restored, dataset, 2)
        with torch.no_grad():
            ai, at = restored(torch.from_numpy(images), torch.from_numpy(texts))
            matrix = (ai.double() @ at.double().T).numpy()
        expected_i = []
        for i, row in enumerate(matrix):
            order = np.argsort(-row, kind="stable").tolist()
            expected_i.append(min(order.index(2*i), order.index(2*i+1)) + 1)
        expected_t = [np.argsort(-matrix[:, j], kind="stable").tolist().index(j // 2) + 1 for j in range(8)]
        np.testing.assert_array_equal(actual["i2t_ranks"], expected_i)
        np.testing.assert_array_equal(actual["t2i_ranks"], expected_t)


if __name__ == "__main__":
    unittest.main()
