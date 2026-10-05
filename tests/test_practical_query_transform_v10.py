"""Synthetic deployment algebra, numerical ties, gallery isolation and identity guards."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_practical_query_transform_v10 as diagnostic
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer


def toy():
    rng = np.random.default_rng(1789)
    d, r = 7, 3
    u = np.linalg.qr(rng.normal(size=(d, r)))[0]
    v = np.linalg.qr(rng.normal(size=(d, r)))[0]
    # Nonzero unequal means and nonsymmetric A expose transpose/offset errors.
    model = ConstrainedBilinearScorer(rng.normal(size=d), rng.normal(size=d), u, v, rng.normal(size=(r, r)))
    return model, rng.normal(size=(5, d)), rng.normal(size=(9, d))


def test_two_directions_equal_dense_bilinear_score_and_preserve_gallery_bytes():
    model, images, texts = toy()
    before = images.tobytes(), texts.tobytes()
    images.flags.writeable = texts.flags.writeable = False
    transform = diagnostic.QueryTransform(model)
    b = model.image_basis @ model.coefficient @ model.text_basis.T
    dense = images @ texts.T + (images - model.image_mean) @ b @ (texts - model.text_mean).T
    qi, ci = transform.image_queries(images)
    qt, ct = transform.text_queries(texts)
    np.testing.assert_allclose(qi @ texts.T + ci[:, None], dense, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(images @ qt.T + ct[None, :], dense, atol=2e-13, rtol=2e-13)
    assert np.max(np.abs((qi @ texts.T) - dense)) > 1e-2  # Offsets cannot be omitted for score equality.
    assert np.max(np.abs((images @ qt.T) - dense)) > 1e-2
    np.testing.assert_array_equal((qi @ texts.T).argmax(axis=1), dense.argmax(axis=1))
    np.testing.assert_array_equal((images @ qt.T).argmax(axis=0), dense.argmax(axis=0))
    assert not np.allclose(np.linalg.norm(qi, axis=1), 1)  # No post-transform normalization.
    assert (images.tobytes(), texts.tobytes()) == before


def test_canonical_full_gallery_comparison_on_both_directions_and_blocks():
    model, images, texts = toy()
    transform = diagnostic.QueryTransform(model)
    for direction, size in (("i2t", len(images)), ("t2i", len(texts))):
        queries = diagnostic.fixed_query_indices(size, 3)
        result = diagnostic.compare_direction(direction, images, texts, model, transform, queries, gallery_block=2)
        assert result["query_count"] == 3
        assert result["pairs_compared"] == 3 * (len(texts) if direction == "i2t" else len(images))
        assert result["max_absolute_score_error_after_offset"] < 1e-12
        assert result["raw_top_disagreements"] == result["offset_restored_top_disagreements"] == 0


def test_near_tie_rank_change_is_reported_without_replacing_deployed_top():
    model = ConstrainedBilinearScorer(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
    images = np.array([[1., 0.]])
    texts = np.array([[1., 0.], [1., 1.]])
    class PerturbedTransform:
        def image_queries(self, values):
            return values + np.array([0., 1e-12]), np.zeros(len(values))
    result = diagnostic.compare_direction("i2t", images, texts, model, PerturbedTransform(), np.array([0]), gallery_block=1)
    row = result["queries"][0]
    assert row["canonical_top_index"] == 0 and row["transformed_raw_top_index"] == 1
    assert row["offset_restored_top_index"] == 1 and row["near_tie"] is True
    assert row["canonical_top_two_margin"] == row["canonical_score_regret_of_raw_top"] == 0
    assert result["raw_top_disagreements"] == result["offset_restored_top_disagreements"] == 1


def test_manifest_refuses_heldout_rows_duplicate_owners_or_caption_associations():
    manifest = {"images": [{"id": "i0", "split": "train"}, {"id": "i1", "split": "train"}],
                "texts": [{"id": f"t{i}"} for i in range(10)],
                "pairs": [{"image_id": f"i{i//5}", "text_id": f"t{i}", "relation": "source"} for i in range(10)]}
    _, _, rows = diagnostic.source_layout(manifest, expected_images=2)
    assert rows.tolist() == list(range(10))
    manifest["images"][1]["split"] = "validation"
    with pytest.raises(ValueError, match="exclusively training"):
        diagnostic.source_layout(manifest, expected_images=2)
    manifest["images"][1]["split"] = "train"
    manifest["pairs"].append(dict(manifest["pairs"][0]))
    with pytest.raises(ValueError, match="multiple associations"):
        diagnostic.source_layout(manifest, expected_images=2)


def test_hash_guard_detects_changed_source_or_state(tmp_path):
    source = tmp_path / "state.npz"
    source.write_bytes(b"original checkpoint")
    records = {str(source): diagnostic.digest(source)}
    diagnostic.verify_unchanged(records)
    source.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="hash differs"):
        diagnostic.verify_unchanged(records)


def test_deterministic_subset_has_no_duplicate_queries_or_outcome_dependency():
    np.testing.assert_array_equal(diagnostic.fixed_query_indices(10, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(diagnostic.fixed_query_indices(3, 8), [0, 1, 2])
    np.testing.assert_array_equal(diagnostic.fixed_query_indices(10, 1), [0])
    with pytest.raises(ValueError):
        diagnostic.fixed_query_indices(10, 0)


def test_selected_state_guard_rejects_incomplete_budget_and_tampered_checkpoint(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    source = tmp_path / "source.py"
    source.write_text("# synthetic protocol source\n")
    config = {"rank": 2, "epochs": 2, "batch_size": 6000, "radius": 1.0}
    protocol = {"study": "sanw_practical_v10", "encoders": ["vit_b32", "rn50"], "seeds": [17, 29, 43],
                "fit_config": config, "source_sha256": {"source.py": diagnostic.digest(source)},
                "training_inputs": {"vit_b32": {"synthetic": True}}}
    protocol_path = write("protocol.json", protocol)
    protocol_sha = diagnostic.digest(protocol_path)
    identity = {"study": "sanw_practical_v10", "mode": "full", "family": "joint", "encoder": "vit_b32",
                "config": {**config, "seed": 17}, "protocol_sha256": protocol_sha,
                "source_sha256": protocol["source_sha256"], "fit_gallery_image_count": 6000, "fit_gallery_text_count": 30000,
                "official_development_or_benchmarks_used": False,
                "training_provenance": {"inputs": protocol["training_inputs"]["vit_b32"], "heldout_used": False, "fit_split": "train"}}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    write("ledger.json", {"identity": identity, "ledger_sha256": ledger_sha})
    model = ConstrainedBilinearScorer(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), .5*np.eye(2))
    selected, epoch = tmp_path / "selected.npz", tmp_path / "epoch.npz"
    model.save(selected)
    epoch.write_bytes(selected.read_bytes())
    certificate = {"ranking_checked_canonically": True, "ranking_preserved": True, "feasible_with_tolerance": True}
    history = [{"epoch": e, "optimizer_steps": e, "training_objective": 3-e, "nonzero": True, "certificate": certificate} for e in (1, 2)]
    epoch_record = {"path": "epoch.npz", "sha256": diagnostic.digest(epoch), "ledger_sha256": ledger_sha}
    completion = {key: identity[key] for key in ("study", "mode", "family", "encoder", "config", "protocol_sha256")}
    completion.update({"ledger_sha256": ledger_sha, "optimizer_steps": 2, "history": history,
                       "checkpoint_history": [{**row, "checkpoint": epoch_record} for row in history],
                       "selected_epoch": 2, "selected_training_objective": 1,
                       "selected_checkpoint": {"path": "selected.npz", "sha256": diagnostic.digest(selected)}})
    path = write("completion.json", completion)
    diagnostic.verify_completed_state(tmp_path, protocol_path, protocol_sha, path)
    completion["optimizer_steps"] = 1
    write("completion.json", completion)
    with pytest.raises(ValueError, match="budget/history"):
        diagnostic.verify_completed_state(tmp_path, protocol_path, protocol_sha, path)
    completion["optimizer_steps"] = 2
    write("completion.json", completion)
    model.coefficient = .25*np.eye(2)
    model.save(selected)
    with pytest.raises(ValueError, match="hash differs"):
        diagnostic.verify_completed_state(tmp_path, protocol_path, protocol_sha, path)
