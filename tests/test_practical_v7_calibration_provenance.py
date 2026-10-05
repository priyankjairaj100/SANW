"""Prevent re-labeling an old architecture-compatible checkpoint as v7."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_practical_text_development_v7 as calibration


def test_v7_ledger_requires_exact_study_protocol_and_fixed_fit_configuration():
    protocol_record = {"path": "protocol.json", "sha256": "example"}
    protocol = {"fit_config": {"epochs": 4, "epsilon": .01}, "source_hashes": {"new.py": "hash"}}
    ledger = {"identity": {"schema": "sanw_practical_source_pair_token_v7", "protocol": protocol_record,
                           "source_factory": "gcr.practical_source_pair_v7.SourcePairTrainingExamples",
                           "config": {"encoder": "vit_b32", "seed": 17, "epochs": 4, "epsilon": .01},
                           "source_sha256": protocol["source_hashes"]}}
    calibration.validate_v7_ledger(ledger, protocol, protocol_record, "vit_b32")
    for field, value in (("schema", "sanw_practical_text_last_block_v1"),
                         ("protocol", {"path": "v6.json", "sha256": "example"}),
                         ("source_sha256", {"old.py": "hash"})):
        old = copy.deepcopy(ledger); old["identity"][field] = value
        with pytest.raises(ValueError):
            calibration.validate_v7_ledger(old, protocol, protocol_record, "vit_b32")
    old = copy.deepcopy(ledger); old["identity"]["config"]["epsilon"] = .02
    with pytest.raises(ValueError):
        calibration.validate_v7_ledger(old, protocol, protocol_record, "vit_b32")


def test_checkpoint_study_marker_required_even_with_matching_ledger(tmp_path, monkeypatch):
    filename = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(calibration, "verify", lambda record: filename)
    ledger = {"ledger_sha256": "example"}
    torch.save({"ledger_sha256": "example"}, filename)
    with pytest.raises(ValueError):
        calibration.validate_v7_checkpoint_study([{}], ledger)
    torch.save({"ledger_sha256": "example", "study_schema": "sanw_practical_source_pair_token_v7",
                "source_factory": "gcr.practical_source_pair_v7.SourcePairTrainingExamples"}, filename)
    calibration.validate_v7_checkpoint_study([{}], ledger)
