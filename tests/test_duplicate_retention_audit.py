"""Ensure duplicate-fit evidence ignores metadata and detects tensor changes."""
from pathlib import Path
import copy
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_duplicate_retention_fits import (compare_tensors, tensor_digest,
                                           compare_replication_history)


def test_canonical_hash_is_order_independent_and_compares_exact_tensor_bytes():
    left = {"image.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "text.weight": torch.zeros(2, 2)}
    right = {key: value.clone() for key, value in reversed(list(left.items()))}
    assert compare_tensors(left, right) == tensor_digest(left)
    right["image.weight"][0, 0] = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
    with pytest.raises(ValueError, match="tensor differs"):
        compare_tensors(left, right)


def test_exact_byte_hash_rejects_signed_zero_difference():
    left = {"x": torch.tensor([0.])}
    right = {"x": torch.tensor([-0.])}
    assert torch.equal(left["x"], right["x"])
    assert tensor_digest(left) != tensor_digest(right)
    with pytest.raises(ValueError, match="parameter bytes"):
        compare_tensors(left, right)


def matching_histories():
    retention, replication = [], []
    for epoch in range(11):
        common = {"epoch": epoch, "mean_training_loss": None if not epoch else 1. / epoch,
                  "training_images": 1200 if epoch else 0, "gradient_steps": 38 if epoch else 0}
        metrics = {"native": {"score": .5, "image_count": 100},
                   "source_retrieval": {"i2t_r1": .7, "t2i_r1": .5}}
        # Interpolated variants need not match the separately trained replication.
        retention.append({**common, "validation": [
            {"alpha": .5, "native": {"score": .9}, "source_retrieval": {}},
            {"alpha": 1., **copy.deepcopy(metrics), "update_norm": float(epoch)}]})
        replication.append({**common, "validation": copy.deepcopy(metrics)})
        replication[-1]["validation"]["native"]["per_image"] = [{"score": .5}]
    return retention, replication


def test_replication_comparison_uses_only_unscaled_state_and_all_common_metrics():
    retention, replication = matching_histories()
    compare_replication_history(retention, replication)
    replication[4]["validation"]["native"]["image_count"] = 99
    with pytest.raises(ValueError, match="validation differs: epoch 4 native"):
        compare_replication_history(retention, replication)


@pytest.mark.parametrize("field,value", [("mean_training_loss", 99.), ("gradient_steps", 37),
                                         ("training_images", 1199)])
def test_replication_comparison_rejects_training_difference(field, value):
    retention, replication = matching_histories()
    replication[5][field] = value
    with pytest.raises(ValueError, match="trajectory differs: epoch 5"):
        compare_replication_history(retention, replication)


def test_replication_comparison_rejects_missing_extra_or_ambiguous_summary_fields():
    retention, replication = matching_histories()
    del replication[2]["validation"]["source_retrieval"]["i2t_r1"]
    with pytest.raises(ValueError, match="validation differs"):
        compare_replication_history(retention, replication)
    retention, replication = matching_histories()
    replication[3]["validation"]["source_retrieval"]["new_metric"] = 1.
    with pytest.raises(ValueError, match="validation differs"):
        compare_replication_history(retention, replication)
    retention, replication = matching_histories()
    retention[1]["validation"].append(copy.deepcopy(retention[1]["validation"][-1]))
    with pytest.raises(ValueError, match="one unscaled state"):
        compare_replication_history(retention, replication)


def test_replication_comparison_rejects_missing_or_reordered_epochs():
    retention, replication = matching_histories()
    with pytest.raises(ValueError, match="all epochs"):
        compare_replication_history(retention[:-1], replication)
    replication[4], replication[5] = replication[5], replication[4]
    with pytest.raises(ValueError, match="epoch ordering"):
        compare_replication_history(retention, replication)
