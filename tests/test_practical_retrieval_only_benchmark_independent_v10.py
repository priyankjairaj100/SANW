"""Independent synthetic checks for release isolation and matched fit metadata."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_practical_retrieval_only_benchmark_v10 as supplement


def rec(path):
    return {"path": str(path), "sha256": "h", "bytes": 1}


@pytest.mark.parametrize("core,supplement_name", [("same", "same"), ("a", "a/b"), ("a/b", "a")])
def test_release_rejects_core_supplement_aliases_and_nesting(tmp_path, core, supplement_name):
    args = SimpleNamespace(core_output_root=str(tmp_path/core), supplement_output_root=str(tmp_path/supplement_name),
                           lock="lock", output="release", additional_locks=[])
    with patch.object(supplement, "root_path", side_effect=lambda p: Path(p).resolve()):
        with pytest.raises(ValueError, match="nonnested"):
            supplement.release(args, {"core_lock": rec("core-lock")})


@pytest.mark.parametrize("extra", ["core", "core/child", "supplement", "supplement/child", "."])
def test_release_rejects_all_additional_root_overlap_forms(tmp_path, extra):
    args = SimpleNamespace(core_output_root=str(tmp_path/"core"), supplement_output_root=str(tmp_path/"supplement"),
                           lock="lock", output="release", additional_locks=["lab"])
    with patch.object(supplement, "root_path", side_effect=lambda p: Path(p).resolve()), \
         patch.object(supplement, "verify_additional_lock", return_value={"family": "labclip", "output_root": str(tmp_path/extra)}):
        with pytest.raises(ValueError, match="nonnested"):
            supplement.release(args, {"core_lock": rec("core-lock")})


def test_release_rejects_existing_additional_outcomes(tmp_path):
    extra = tmp_path/"lab"; extra.mkdir(); (extra/"prediction.npz").write_bytes(b"synthetic")
    args = SimpleNamespace(core_output_root=str(tmp_path/"core"), supplement_output_root=str(tmp_path/"supplement"),
                           lock="lock", output="release", additional_locks=["lab"])
    with patch.object(supplement, "root_path", side_effect=lambda p: Path(p).resolve()), \
         patch.object(supplement, "verify_additional_lock", return_value={"family": "labclip", "output_root": str(extra)}):
        with pytest.raises(ValueError, match="precede"):
            supplement.release(args, {"core_lock": rec("core-lock")})


@pytest.mark.parametrize("changed", ["retrieval_logit_scale", "streaming"])
def test_retrieval_state_rejects_scale_or_execution_recipe_different_from_joint(changed):
    identity = {"study": "sanw_practical_v10", "mode": "full", "family": "retrieval_only", "encoder": "vit_b32",
                "config": {"seed": 17}, "protocol_sha256": "h", "fixed_composition_multiplier": 0.,
                "candidate_selection_allowed": False, "explanatory_control_only": True, "contract": rec("contract"),
                "source_sha256": {}, "control_source_sha256": {}, "protocol": rec("protocol"),
                "fresh_confirmation_inclusion_allowed": False, "heldout_used_in_fitting_or_checkpoint_selection": False,
                "fit_gallery_image_count": 6000, "fit_gallery_text_count": 30000, "training_provenance": {},
                "retrieval_logit_scale": 10., "streaming": {"query_block_size": 64}}
    joint = {"training_provenance": {}, "retrieval_logit_scale": 10., "streaming": {"query_block_size": 64}}
    identity[changed] = 20. if changed == "retrieval_logit_scale" else {"query_block_size": 32}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    completion = {**identity, "ledger_sha256": ledger_sha, "development_or_test_used": False}
    content = {"run/ledger.json": {"identity": identity, "ledger_sha256": ledger_sha},
               "run/completion.json": completion, "joint-ledger": {"identity": joint}}
    core_lock = {"protocol": {"sha256": "h"}, "states": {"vit_b32__joint__17": {"ledger": rec("joint-ledger")}}}
    with patch.object(supplement, "root_path", side_effect=Path), \
         patch.object(supplement, "read", side_effect=lambda p: content[str(p)]), \
         patch.object(supplement, "digest", return_value="h"), \
         patch.object(supplement, "verify_record", side_effect=lambda item: item["path"]):
        with pytest.raises(ValueError, match="scale|streaming|recipe|normalization|provenance"):
            supplement.verify_state("run", "contract", {"new_source_sha256": {}}, {"source_sha256": {}, "fit_config": {}}, core_lock)
