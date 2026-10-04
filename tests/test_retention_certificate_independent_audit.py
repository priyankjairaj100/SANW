"""Independent certificate arithmetic and meaningful failure-detection checks."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from audit_retention_certificates_independent import (
    BOOLS, FAMILIES, compare_arrays, expected_states, forward_features,
    independent_rows, independent_summary, r1_threshold,
)
from diagnose_strengthen_retention import diagnose_direction, coverage_summary
from gcr.adapters import ResidualAdapter
from evaluate_study import adapted_features


def test_positive_set_projection_uses_all_active_positives():
    q = np.array([.4, .35, .25])
    exact = float(np.sum(q * np.log(q * 3)))
    pair = .4 * np.log(.4 / .325) + .25 * np.log(.25 / .325)
    assert r1_threshold(q, [0, 1]) == pytest.approx(exact, abs=1e-15)
    assert exact > pair
    assert r1_threshold(q, [2]) == 0.
    assert np.isinf(r1_threshold(q, [0, 1, 2]))


@pytest.mark.parametrize("scale", [1., 3.7, 100.])
def test_independent_recomputation_matches_production_without_shared_math(scale):
    rng = np.random.default_rng(708)
    tq, tc = rng.normal(size=(17, 7)), rng.normal(size=(23, 7))
    sq, sc = tq + .001 * rng.normal(size=tq.shape), tc + .001 * rng.normal(size=tc.shape)
    labels = [sorted(rng.choice(23, size=5, replace=False).tolist()) for _ in tq]
    actual = independent_rows(tq @ tc.T, sq @ sc.T, labels, scale)
    expected = diagnose_direction(tq, tc, sq, sc, labels, scale, block_size=5)
    compare_arrays(actual, expected, defaultdict(float))
    summary = independent_summary(actual)
    production = coverage_summary(expected)
    assert summary["counts"] == production["counts"]
    assert summary["certificate_coverage"] == production["certificate_coverage"]
    assert summary["tie_disagreements"] == production["tie_disagreements"]


def test_ties_do_not_turn_benchmark_success_into_certificate_success():
    scores = np.array([[1., 1.], [1., 0.], [0., 1.]])
    raw = independent_rows(scores, scores, [[0], [0], [0]], 100.)
    np.testing.assert_array_equal(raw["benchmark_teacher_ranks"], [1, 1, 2])
    np.testing.assert_array_equal(raw["teacher_correct"], [False, True, False])
    np.testing.assert_array_equal(raw["certified"], [False, True, False])
    summary = independent_summary(raw)
    assert summary["tie_disagreements"]["teacher_stable_vs_pessimistic"] == 1
    assert summary["certificate_coverage"]["certified"]["fraction_of_all_queries"] == pytest.approx(1 / 3)


def test_full_gallery_and_kl_direction_are_material():
    teacher, student = np.array([[.8, .0, .6]]), np.array([[.8, .0, 1.2]])
    mini = independent_rows(teacher[:, :2], student[:, :2], [[0]], 30.)
    full = independent_rows(teacher, student, [[0]], 30.)
    assert mini["certified"][0]
    assert full["teacher_correct"][0] and not full["student_correct"][0]
    assert not full["certified"][0]
    z, v = teacher[0] * 15., student[0] * 15.
    q, p = np.exp(z - z.max()), np.exp(v - v.max())
    q, p = q / q.sum(), p / p.sum()
    expected = float(np.sum(q * np.log(q / p)))
    reverse = float(np.sum(p * np.log(p / q)))
    assert full["divergence"][0] == pytest.approx(expected, abs=1e-13)
    assert abs(expected - reverse) > .1
    assert abs(full["divergence"][0] - 4 * expected) > 1.


def test_tampered_masks_ranks_and_numerics_fail():
    raw = independent_rows(np.array([[2., 0.], [2., 0.]]), np.array([[2., 0.], [0., 2.]]), [[0], [0]], 1.)
    for key, value in (("certified", True), ("benchmark_student_ranks", 1), ("threshold", 99.)):
        wrong = {key: value.copy() for key, value in raw.items()}
        wrong[key][1] = value
        with pytest.raises(ValueError, match="differs|mismatch"):
            compare_arrays(raw, wrong, defaultdict(float))
    wrong = {key: value.copy() for key, value in raw.items()}
    wrong["certified"][1] = True
    with pytest.raises(ValueError, match="contradicts"):
        independent_summary(wrong)


def test_independent_forward_matches_declared_float32_operation():
    rng = np.random.default_rng(8)
    features = {"image_features": rng.normal(size=(7, 5)).astype(np.float32),
                "text_features": rng.normal(size=(13, 5)).astype(np.float32)}
    adapter = ResidualAdapter(5)
    with torch.no_grad():
        adapter.image.weight[:] = torch.from_numpy(rng.normal(size=(5, 5)).astype(np.float32))
        adapter.text.weight[:] = torch.from_numpy(rng.normal(size=(5, 5)).astype(np.float32))
    for weights, model in ((None, None), (adapter.state_dict(), adapter)):
        expected, actual = adapted_features(features, model), forward_features(features, weights)
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(left.astype(np.float64), right)


def test_both_tolerances_require_all_seven_families_and_three_seeds():
    manifest = {"states": [{"state_id": str(seed), "seed": seed} for seed in (17, 29, 43)], "selections": [
        {"state_id": str(seed), "family": family, "seed": seed, "tolerance_pp": tolerance}
        for tolerance in (0., 1.) for family in FAMILIES for seed in (17, 29, 43)]}
    states = expected_states(manifest)
    assert len(states) == 3 and len(states[0]["selection_roles"]) == 14
    manifest["selections"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        expected_states(manifest)


def test_artifact_audit_binds_receipts_and_rejects_modified_source(tmp_path, monkeypatch):
    """Exercise the complete artifact path with a small gallery substituted only at loading."""
    import json
    import audit_retention_certificates_independent as module
    from audit_allocation_distillation_independent import Audit, canonical_digest, digest
    from diagnose_strengthen_retention import selected_diagnostic_states

    def document(path, value):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, sort_keys=True) + "\n")
        return digest(destination)

    def archive(path, values):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **values)
        return {"predictions": path, "predictions_sha256": digest(destination)}

    sources = {}
    for path in module.DIAGNOSTIC_SOURCES:
        sources[path] = document(path, {"fixture": path})
    encoder, protocol, lock_hash = "vit_b32", module.PROTOCOL_SHA256, "locked-selections"
    features = {"image_features": np.eye(512, dtype=np.float32)[:2],
                "text_features": np.repeat(np.eye(512, dtype=np.float32)[:2], 2, axis=0)}
    identity = {"image_ids": np.array(["i0", "i1"]), "text_ids": np.array(["t0", "t1", "t2", "t3"]),
                "text_source_image_ids": np.array(["i0", "i0", "i1", "i1"])}
    relevance = ([[0, 1], [2, 3]], [[0], [0], [1], [1]])
    features.update({key: value for key, value in identity.items() if key != "text_source_image_ids"})
    monkeypatch.setattr(module, "load_dataset", lambda *args: (features, relevance, identity))
    adapted = forward_features(features)
    raw, summaries, benchmark = dict(identity), {}, dict(identity)
    for direction, index in (("i2t", 0), ("t2i", 1)):
        values = diagnose_direction(adapted[index], adapted[1 - index], adapted[index], adapted[1 - index], relevance[index], 10.)
        raw.update({f"{direction}_{key}": value for key, value in values.items()})
        summaries[direction] = coverage_summary(values)
        benchmark[f"{direction}_ranks"] = values["benchmark_teacher_ranks"]
    ledger_identity = {"inputs": {"logit_scale": 10.}, "source_sha256": {}}
    ledger = {"identity": ledger_identity, "ledger_sha256": canonical_digest(ledger_identity)}
    ledger_file_hash = document("training/ledger.json", ledger)
    states = []
    for seed in (17, 29, 43):
        checkpoint = {"ledger_sha256": ledger["ledger_sha256"], "protocol_sha256": protocol,
                      "method": "source", "epoch": 0, "seed": seed, "learning_rate": .0001,
                      "state_dict": {"image.weight": torch.zeros(512, 512), "text.weight": torch.zeros(512, 512)}}
        path = f"training/seed_{seed}.pt"
        torch.save(checkpoint, tmp_path / path)
        states.append({"state_id": f"source_{seed}", "method": "source", "seed": seed, "epoch": 0,
                       "learning_rate": .0001, "update_norm": 0., "checkpoint": path,
                       "checkpoint_sha256": digest(tmp_path / path)})
    selection_hashes = {label: document(f"training/selection_{label}.json", {"label": label}) for label in ("primary", "sensitivity")}
    manifest = {"encoder": encoder, "protocol_sha256": protocol, "ledger_sha256": ledger["ledger_sha256"],
                "states": states, "matched_controls_complete": True, "test_outcomes_used_for_selection": False,
                "selection_sha256": selection_hashes, "selections": [
                    {"state_id": f"source_{seed}", "family": family, "seed": seed, "tolerance_pp": tolerance}
                    for tolerance in (0., 1.) for family in FAMILIES for seed in (17, 29, 43)]}
    manifest_hash = document("training/state_manifest.json", manifest)
    selected = selected_diagnostic_states(manifest)
    lock = {"encoders": {encoder: {"manifest": "training/state_manifest.json", "manifest_sha256": manifest_hash,
                                    "selection_sha256": selection_hashes, "ledger_sha256": ledger["ledger_sha256"]}}}
    config_hash = document("datasets.json", {"encoder": encoder})
    prescore = {"protocol_sha256": protocol, "encoder": encoder, "manifest_sha256": manifest_hash,
                "selection_lock_sha256": lock_hash, "source_hashes": sources,
                "ledger_file_sha256": ledger_file_hash, "input_hashes": {name: {} for name in module.DATASETS}}
    prescore_hash = document("evaluation/prescore_receipt.json", prescore)
    scored = []
    for state in [{"state_id": "frozen"}] + states:
        datasets = {name: archive(f"evaluation/{state['state_id']}/{name}.npz", benchmark) for name in module.DATASETS}
        scored.append({**state, "datasets": datasets})
    evaluation = {**prescore, "status": "complete", "prescore_receipt": "evaluation/prescore_receipt.json",
                  "prescore_receipt_sha256": prescore_hash, "runs": scored}
    evaluation_hash = document("evaluation/index.json", evaluation)
    receipt = {"protocol_sha256": protocol, "encoder": encoder, "manifest_sha256": manifest_hash,
               "selection_lock_sha256": lock_hash, "source_hashes": sources, "temperature": 2.,
               "native_logit_scale": 10., "k": 1, "scope": "fixture", "evidence_type": "fixture",
               "atol": module.ATOL, "rtol": module.RTOL, "selection_modified": False, "success_gates_modified": False,
               "dataset_config_sha256": config_hash, "evaluation_index": "evaluation/index.json",
               "evaluation_index_sha256": evaluation_hash, "input_hashes": prescore["input_hashes"], "states": selected}
    receipt_hash = document("diagnostics/prediagnostic_receipt.json", receipt)
    diagnostic_runs = []
    for state in selected:
        datasets = {}
        for name in module.DATASETS:
            record = archive(f"diagnostics/{state['state_id']}/{name}.npz", raw)
            metadata = {"predictions_sha256": record["predictions_sha256"], "summary": summaries,
                        "provenance": {"state": state, "dataset": name, "prediagnostic_receipt_sha256": receipt_hash}}
            path = f"diagnostics/{state['state_id']}/{name}.json"
            record.update(metadata=path, metadata_sha256=document(path, metadata), summary=summaries)
            datasets[name] = record
        diagnostic_runs.append({**state, "datasets": datasets})
    index = {key: receipt[key] for key in ("protocol_sha256", "encoder", "source_hashes", "temperature",
                                         "native_logit_scale", "k", "scope", "evidence_type")}
    index.update(status="complete", prediagnostic_receipt="diagnostics/prediagnostic_receipt.json",
                 prediagnostic_receipt_sha256=receipt_hash, state_count=3, archive_count=6,
                 runs=diagnostic_runs, false_exact_kl_certificates=0)
    document("diagnostics/index.json", index)
    encoder, result = module.audit_index(Audit(tmp_path), "diagnostics/index.json", "datasets.json", lock, lock_hash, 2, defaultdict(float))
    assert encoder == "vit_b32"
    assert result["counts"]["unchanged_states"] == 3
    assert result["counts"]["state_query_observations"] == 36
    assert "nonzero_update" not in result["coverage_counts"]
    document("src/gcr/rank_retention.py", {"modified": True})
    with pytest.raises(ValueError, match="hash mismatch"):
        module.audit_index(Audit(tmp_path), "diagnostics/index.json", "datasets.json", lock, lock_hash, 2, defaultdict(float))
