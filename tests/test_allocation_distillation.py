"""Independent loss identities and extension execution/selection invariants."""
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gcr.adapters import ResidualAdapter
from gcr.allocation_distillation import (
    ALPHAS, BASELINE_POLICIES, FAMILIES, POLICIES, allocation_distillation_loss,
    fit_candidate, fit_matched_controls, policy_parameters, protocol_template,
    select_and_export, verify_protocol,
)
from gcr.retention_losses import retention_loss
from gcr.review_controls import make_promotion_assignment
from gcr.review_training import candidate_path
from gcr.strengthen_retention import fit_candidate as old_fit_candidate
from gcr.training import StudyConfig, atomic_json, sha256_file
from test_strengthen_retention import fixture_features, tiny_pool
from test_training import fixture_dataset

REPOSITORY = Path(__file__).resolve().parents[1]


def independent_loss(images, texts, rel, frozen_images, frozen_texts, mix, beta, scale=7.):
    """Write the target distributions explicitly, without project loss helpers."""
    logits = (images @ texts.T) * scale
    source = rel == 1
    supported = source | (rel == 2)
    def directional(target):
        image_targets = target.to(logits.dtype) / target.sum(1, keepdim=True)
        eligible = target.any(0)
        reverse_targets = target.T[eligible].to(logits.dtype)
        reverse_targets = reverse_targets / reverse_targets.sum(1, keepdim=True)
        return .5 * (-(image_targets * F.log_softmax(logits, 1)).sum(1).mean()
                     - (reverse_targets * F.log_softmax(logits.T[eligible], 1)).sum(1).mean())
    supervised = mix * directional(source) + (1-mix) * directional(supported)
    columns = source.any(0)
    teacher = ((F.normalize(frozen_images.detach(), dim=-1)
                @ F.normalize(frozen_texts.detach(), dim=-1).T) * scale)[:, columns] / 2
    student = logits[:, columns] / 2
    def divergence(t, s):
        q = F.softmax(t, 1)
        return (q * (F.log_softmax(t, 1) - F.log_softmax(s, 1))).sum(1).mean()
    return supervised + beta * 4 * .5 * (divergence(teacher, student) + divergence(teacher.T, student.T))


@pytest.mark.parametrize("mix", [.5, .8, 1.])
@pytest.mark.parametrize("beta", [1., 4., 16.])
def test_bidirectional_composition_and_gradients_match_explicit_distributions(mix, beta):
    images, texts, rel = fixture_features()
    frozen_images, frozen_texts = images.clone().requires_grad_(), texts.clone().requires_grad_()
    # Unequal source/support counts test the reverse eligible-anchor average.
    rel[0, 3] = 2
    images = F.normalize(images + .03 * torch.arange(8), dim=-1).requires_grad_()
    texts = F.normalize(texts - .02 * torch.arange(8), dim=-1).requires_grad_()
    actual = allocation_distillation_loss(images, texts, rel, 7., source_mix=mix, beta=beta,
        frozen_images=frozen_images, frozen_texts=frozen_texts)
    expected = independent_loss(images, texts, rel, frozen_images, frozen_texts, mix, beta)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(actual, (images, texts), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (images, texts))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    assert frozen_images.grad is None and frozen_texts.grad is None


@pytest.mark.parametrize("beta", [1., 4., 16.])
def test_zero_allocation_exactly_reproduces_old_distillation_and_gradients(beta):
    images, texts, rel = fixture_features()
    images.requires_grad_(); texts.requires_grad_()
    actual = allocation_distillation_loss(images, texts, rel, 7., source_mix=0., beta=beta,
                                         frozen_images=images, frozen_texts=texts)
    expected = retention_loss(images, texts, rel, f"distilled_{beta:g}", 7.,
                              frozen_images=images, frozen_texts=texts)
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual, (images, texts), retain_graph=True)
    b = torch.autograd.grad(expected, (images, texts))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("mix", [.5, .8])
def test_zero_distillation_exactly_reproduces_old_allocation_and_gradients(mix):
    images, texts, rel = fixture_features()
    images.requires_grad_(); texts.requires_grad_()
    actual = allocation_distillation_loss(images, texts, rel, 7., source_mix=mix, beta=0.)
    expected = retention_loss(images, texts, rel, f"allocation_{mix:g}", 7.)
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual, (images, texts), retain_graph=True)
    b = torch.autograd.grad(expected, (images, texts))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_protocol_requires_frozen_hash_and_keeps_parent_unchanged(tmp_path):
    before = sha256_file(REPOSITORY / "results/strengthen_retention/protocol_v3.json")
    path = tmp_path / "protocol.json"
    protocol = protocol_template()
    protocol["evaluation"]["status"] = "draft"
    atomic_json(path, protocol)
    with pytest.raises(ValueError, match="Freeze"):
        verify_protocol(REPOSITORY, path, sha256_file(path), StudyConfig())
    protocol["evaluation"]["status"] = "frozen"
    atomic_json(path, protocol)
    assert verify_protocol(REPOSITORY, path, sha256_file(path), StudyConfig()) == protocol
    with pytest.raises(ValueError, match="SHA256"):
        verify_protocol(REPOSITORY, path, "0" * 64, StudyConfig())
    protocol["training"]["epochs"] = 9
    atomic_json(path, protocol)
    with pytest.raises(ValueError, match="training"):
        verify_protocol(REPOSITORY, path, sha256_file(path), StudyConfig())
    assert before == sha256_file(REPOSITORY / "results/strengthen_retention/protocol_v3.json")


def tiny_ledger(data):
    return {"ledger_sha256": "synthetic-only", "identity": {
        "encoder": "vit_b32", "protocol": {"sha256": "synthetic-protocol"},
        "inputs": {"logit_scale": data.logit_scale}}}


@pytest.mark.parametrize("policy", BASELINE_POLICIES)
def test_fresh_baselines_reproduce_old_optimizer_trajectory_exactly(tmp_path, policy):
    data = fixture_dataset(tmp_path / "data")
    config = StudyConfig(feature_dim=8, epochs=2, image_batch_size=1, threads=1)
    ledger = tiny_ledger(data)
    old_fit_candidate(tmp_path, tmp_path / "old", config, data, tiny_pool(), ledger, policy, 1e-3, 17)
    fit_candidate(tmp_path, tmp_path / "new", config, data, tiny_pool(), ledger, policy, 1e-3, 17)
    for epoch in range(3):
        old_dir = candidate_path(tmp_path / "old", policy, 1e-3, 17)
        new_dir = candidate_path(tmp_path / "new", policy, 1e-3, 17)
        old = torch.load(old_dir / f"epochs/epoch_{epoch:02d}.pt", weights_only=True)
        new = torch.load(new_dir / f"epochs/epoch_{epoch:02d}.pt", weights_only=True)
        assert all(torch.equal(value, new["state_dict"][key]) for key, value in old["state_dict"].items())
        a = json.loads((old_dir / "history.json").read_text())["history"][epoch]
        b = json.loads((new_dir / "history.json").read_text())["history"][epoch]
        assert a["mean_training_loss"] == b["mean_training_loss"]
        assert a["validation"] == b["validation"]


def test_complete_grid_selection_decomposition_wise_and_matched_schedule(tmp_path):
    data = fixture_dataset(tmp_path / "data")
    config = StudyConfig(feature_dim=8, epochs=1, image_batch_size=2, threads=1,
                        learning_rates=(1e-4,))
    ledger = tiny_ledger(data)
    output = tmp_path / "new"
    for policy in POLICIES:
        for seed in config.seeds:
            fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, policy, 1e-4, seed)
    manifest = select_and_export(tmp_path, output, config, ledger)
    assert len(manifest["selections"]) == 36
    assert manifest["test_outcomes_used_for_selection"] is False
    selection = json.loads((output / "selection_primary.json").read_text())
    assert tuple(selection["families"]) == tuple(sorted(FAMILIES))
    joint = selection["families"]["allocation_distillation"]
    # All synthetic development accuracies tie; the lowest joint parameters win.
    assert joint["parameter"] == [.5, 1.]
    for family in FAMILIES:
        assert len(selection["families"][family]["runs"]) == 3
    states = {state["state_id"]: state for state in manifest["states"]}
    for entry in manifest["decomposition_selections"]:
        chosen = next(run for run in joint["runs"] if run["seed"] == entry["seed"])
        state = states[entry["state_id"]]
        assert state["epoch"] == chosen["epoch"]
        assert state["learning_rate"] == chosen["learning_rate"]
        assert state["source_mix"] == entry["source_mix"]
        assert state["beta"] == entry["beta"]
    for run in selection["families"]["wise_ft"]["runs"]:
        checkpoint = torch.load(tmp_path / run["checkpoint"], weights_only=True)
        assert checkpoint["parameters_already_scaled"]
        assert checkpoint["alpha"] in ALPHAS
    assignments = {draw: make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
        data.pairs, data.split_indices["train"], "score_stratified", seed,
        {key: "a" * 64 for key in ("manifest_sha256", "features_sha256", "protocol_sha256")},
        text_strings=[row["text"] for row in data.manifest["texts"]])
        for draw, seed in enumerate((101, 211, 307))}
    final = fit_matched_controls(tmp_path, output, config, data, tiny_pool(), ledger, assignments)
    assert final["matched_controls_complete"]
    matched = [state for state in final["states"] if state["family"] == "matched_allocation_distillation"]
    assert len(matched) == 9
    for state in matched:
        run = next(run for run in joint["runs"] if run["seed"] == state["seed"])
        for key in ("epoch", "learning_rate", "source_mix", "beta"):
            assert state[key] == run[key]
    assert select_and_export(tmp_path, output, config, ledger) == final
    victim = candidate_path(output, "allocation_distillation_0.5_1", 1e-4, 17) / "epochs/epoch_01.pt"
    victim.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or modified"):
        fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, "allocation_distillation_0.5_1", 1e-4, 17)


def test_matched_controls_apply_assignments_through_each_nonzero_selected_epoch(tmp_path, monkeypatch):
    import gcr.allocation_distillation as extension
    data = fixture_dataset(tmp_path / "data")
    config = StudyConfig(feature_dim=8, epochs=2, image_batch_size=2, threads=1)
    ledger = tiny_ledger(data)
    output = tmp_path / "nonzero"
    assignments = {draw: make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
        data.pairs, data.split_indices["train"], "score_stratified", seed,
        {key: "a" * 64 for key in ("manifest_sha256", "features_sha256", "protocol_sha256")},
        text_strings=[row["text"] for row in data.manifest["texts"]])
        for draw, seed in enumerate((101, 211, 307))}
    runs = [{"seed": seed, "epoch": epoch} for seed, epoch in zip(config.seeds, (1, 2, 1))]
    selection = {"ledger_sha256": ledger["ledger_sha256"], "families": {
        "allocation_distillation": {"policy": "allocation_distillation_0.8_4", "learning_rate": 1e-3, "runs": runs}}}
    atomic_json(output / "selection_primary.json", selection)
    atomic_json(output / "state_manifest.json", {"ledger_sha256": ledger["ledger_sha256"],
        "selection_sha256": {"primary": sha256_file(output / "selection_primary.json")},
        "states": [], "selections": []})
    original_apply = extension.apply_promotion_assignment
    calls = []
    def observed_apply(relations, indices, text_indices, assignment):
        calls.append((tuple(indices), tuple(text_indices)))
        result = original_apply(relations, indices, text_indices, assignment)
        assert torch.equal(result == 1, relations == 1)
        assert result.shape == relations.shape
        return result
    monkeypatch.setattr(extension, "apply_promotion_assignment", observed_apply)
    manifest = fit_matched_controls(tmp_path, output, config, data, tiny_pool(), ledger, assignments)
    assert len(calls) == 3 * (1 + 2 + 1)
    assert len(manifest["states"]) == 9
    for state in manifest["states"]:
        expected_epoch = next(run["epoch"] for run in runs if run["seed"] == state["seed"])
        assert state["epoch"] == expected_epoch
        assert state["source_mix"] == .8 and state["beta"] == 4.
        assert state["update_norm"] > 0
        checkpoint = torch.load(tmp_path / state["checkpoint"], weights_only=True)
        assert checkpoint["selection_sha256"] == sha256_file(output / "selection_primary.json")
        directory = candidate_path(output, state["method"], state["learning_rate"], state["seed"])
        history = json.loads((directory / "history.json").read_text())["history"]
        assert sum(row["gradient_steps"] for row in history) == expected_epoch
