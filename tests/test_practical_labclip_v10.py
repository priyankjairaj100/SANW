"""Synthetic-only checks for the separately declared expanded comparator."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from gcr.practical_labclip_v9 import LABCLIPConfig, NormalizedFullRankAlignment, fit_alignment, hard_negative_batch_loss
from gcr.practical_labclip_v10 import (
    audit_bank_record, fixed_audit_batches, fixed_training_objective,
    fit_expanded_alignment, functional_witness, select_epoch,
)


def fixture(number=4, dimension=7):
    rng = np.random.default_rng(202)
    images = rng.normal(size=(number, dimension))
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    texts = rng.normal(size=(number * 7, dimension))
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    sources = [list(range(i * 5, (i + 1) * 5)) for i in range(number)]
    rows = np.arange(number * 5, dtype=np.int64)
    owner = np.repeat(np.arange(number), 5)
    negatives = [[number * 5 + i * 2, number * 5 + i * 2 + 1] for i in range(number)]
    return images, texts, rows, owner, sources, negatives


def test_fixed_audit_has_unique_owners_all_sources_and_valid_negatives():
    _, _, _, _, sources, negatives = fixture(260)
    eligible = np.arange(0, 260, 2)
    first = fixed_audit_batches(eligible, sources, negatives)
    second = fixed_audit_batches(eligible, sources, negatives)
    assert len(first) == 10
    visited = []
    for batch in first:
        ii, pp, nn = batch
        assert len(set(ii)) == len(ii) and len(ii) <= 128
        assert set(ii) <= set(eligible)
        for i, p, n in zip(ii, pp, nn):
            assert p in sources[i] and n in negatives[i]
        assert not ii.flags.writeable
        visited.extend(pp.tolist())
    assert sorted(visited) == sorted(p for i in eligible for p in sources[i])
    assert audit_bank_record(first) == audit_bank_record(second)
    assert audit_bank_record(first)["source_rows"] == 650


def test_fixed_audit_objective_uses_row_weights_not_equal_batch_weights():
    images, texts, _, _, sources, negatives = fixture(130, 3)
    batches = fixed_audit_batches(np.arange(130), sources, negatives)
    model = NormalizedFullRankAlignment(3, native_logit_scale=3., learned_scale=False)
    image_tensor, text_tensor = torch.tensor(images, dtype=torch.float32), torch.tensor(texts, dtype=torch.float32)
    values = []
    with torch.no_grad():
        for ii, pp, nn in batches:
            loss, _ = hard_negative_batch_loss(image_tensor[ii.copy()], model(text_tensor[pp.copy()]),
                                               model(text_tensor[nn.copy()]), model.logit_scale)
            values.append((len(ii), float(loss)))
    actual = fixed_training_objective(model, image_tensor, text_tensor, batches)
    assert actual == sum(n * loss for n, loss in values) / sum(n for n, _ in values)
    assert abs(actual - np.mean([loss for _, loss in values])) > .01
    assert actual == fixed_training_objective(model, image_tensor, text_tensor, batches)
    assert model.training


def test_functional_witness_rejects_positive_scale_and_uses_fixed_owned_pairs():
    images, texts, rows, owners, _, _ = fixture(30, 5)
    identity = functional_witness(images, texts, rows, owners, np.arange(30), np.eye(5))
    scaled = functional_witness(images, texts, rows, owners, np.arange(30), 2 * np.eye(5))
    assert identity["nonzero_functional_update"] is False
    assert scaled["nonzero_functional_update"] is False
    changed = np.eye(5); changed[0, 1] = .1
    witness = functional_witness(images, texts, rows, owners, np.arange(30), changed)
    assert witness["nonzero_functional_update"] is True
    assert witness["source_rows"] == list(range(128)) and witness["pair_count"] == 128
    assert witness["owner_rows"] == owners[:128].tolist()


def test_selection_ignores_identity_and_nonfinite_and_breaks_ties_earliest():
    rows = [{"epoch": 0, "training_objective": 0., "nonzero_functional_update": True},
            {"epoch": 1, "training_objective": 1., "nonzero_functional_update": False},
            {"epoch": 3, "training_objective": 2., "nonzero_functional_update": True},
            {"epoch": 2, "training_objective": 2., "nonzero_functional_update": True},
            {"epoch": 4, "training_objective": float("nan"), "nonzero_functional_update": True}]
    assert select_epoch(rows) == 2
    with pytest.raises(ValueError, match="No finite"):
        select_epoch(rows[:2])


def test_toy_optimizer_trajectory_is_unchanged_from_inherited_recipe():
    images, texts, rows, owners, _, negatives = fixture(3, 4)
    config = LABCLIPConfig(seed=17)
    inherited, old_result = fit_alignment(images, texts, rows, owners, negatives, np.arange(3), config, 3.)
    weights = {}
    def save(row, model):
        weights[row["epoch"]] = model.linear.weight.detach().numpy().copy()
    result = fit_expanded_alignment(images, texts, rows, owners, negatives, np.arange(3), config, 3., save)
    np.testing.assert_array_equal(weights[32], inherited.linear.weight.detach().numpy())
    assert sorted(weights) == list(range(33))
    assert result["optimizer_steps"] == old_result["optimizer_steps"] == 160
    for old, new in zip(old_result["history"], result["history"]):
        assert old["loss"] == new["online_loss"]
        assert old["caption_rows"] == new["caption_rows"] == 15
    assert len({row["audit_bank_sha256"] for row in result["checkpoint_history"]}) == 1
    assert 1 <= result["selected_epoch"] <= 32


def test_contract_scope_rejects_candidate_promotion_and_wrong_family_size():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_practical_labclip_v10 import CONTRACT_STUDY, SELECTION, INFERENCE, ENDPOINTS, config_record, validate_contract_payload
    protocol = {"study": "sanw_practical_v10", "training_inputs": {}, "fresh_confirmation_contract": {"path": "fresh", "sha256": "f"}}
    contract = {"study": CONTRACT_STUDY, "family": "labclip", "status": "draft_not_executable",
                "inherited_protocol": {"sha256": "p"}, "config": config_record(), "selection": SELECTION,
                "encoders": ["vit_b32", "rn50"], "seeds": [17, 29, 43], "training_inputs": {},
                "candidate_selection_allowed": False, "fresh_confirmation_inclusion_allowed": False,
                "explanatory_control_only": True, "required_gate_study": "sanw_practical_v10_fixed_seed_development_gate",
                "required_gate_family": "joint", "required_gate_status": "both_encoders_passing_fixed_three_seed_aggregate",
                "contrasts": [["joint_minus_labclip", "joint", "labclip"]], "endpoints": ENDPOINTS, "inference": INFERENCE,
                "all_six_states_locked_before_any_practical_benchmark_outcome": True,
                "fresh_confirmation_contract": protocol["fresh_confirmation_contract"]}
    validate_contract_payload(contract, protocol, "p")
    bad = copy.deepcopy(contract); bad["candidate_selection_allowed"] = True
    with pytest.raises(ValueError): validate_contract_payload(bad, protocol, "p")
    bad = copy.deepcopy(contract); bad["inference"]["family_size"] = 60
    with pytest.raises(ValueError): validate_contract_payload(bad, protocol, "p")
    bad = copy.deepcopy(contract); bad["selection"]["audit_seed"] += 1
    with pytest.raises(ValueError): validate_contract_payload(bad, protocol, "p")


def test_draft_is_rejected_before_sources_or_feature_loading(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_practical_labclip_v10 as runner
    from unittest.mock import patch
    protocol = tmp_path / "protocol.json"; protocol.write_text('{}')
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"status": "draft_not_executable", "inherited_protocol": {"path": str(protocol), "sha256": runner.digest(protocol)}}))
    with patch.object(runner, "validate_contract_payload"), pytest.raises(ValueError, match="Draft"):
        runner.verify_contract(tmp_path, contract)
