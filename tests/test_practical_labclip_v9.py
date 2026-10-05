"""Focused implementation checks; synthetic arrays only, no real fitting."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gcr.practical_labclip_v9 import (
    LABCLIPConfig, NormalizedFullRankAlignment, canonical_transform,
    epoch_batches, hard_negative_batch_loss, sources_by_owner, validate_owner_split,
)


def test_hnb_matches_official_formula_and_gradients():
    generator = torch.Generator().manual_seed(4)
    image = F.normalize(torch.randn(4, 7, generator=generator, dtype=torch.float64), dim=1)
    pos = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    neg = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    actual, _ = hard_negative_batch_loss(image, pos, neg, scale)
    logits = scale * (image @ torch.cat((pos, neg)).T)
    labels = torch.arange(4)
    official = (F.cross_entropy(logits, torch.eye(4, 8, dtype=torch.float64))
                + F.cross_entropy(logits.T[:4], labels)) / 2
    torch.testing.assert_close(actual, official, atol=1e-14, rtol=1e-14)
    for a, b in zip(torch.autograd.grad(actual, (pos, neg, scale), retain_graph=True),
                    torch.autograd.grad(official, (pos, neg, scale))):
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)


def test_full_rank_identity_normalization_scale_and_checkpoint(tmp_path):
    text = torch.tensor([[3., 4., 0.], [1., 2., 3.]], dtype=torch.float32)
    model = NormalizedFullRankAlignment(3, native_logit_scale=100., learned_scale=True)
    assert model.linear.bias is None
    assert model.linear.weight.shape == (3, 3)
    assert float(model.logit_scale) == 1.0
    torch.testing.assert_close(model(text), F.normalize(text, dim=1))
    fixed = NormalizedFullRankAlignment(3, native_logit_scale=71., learned_scale=False)
    assert not fixed.log_scale.requires_grad
    assert float(fixed.logit_scale) == pytest.approx(71., rel=1e-14)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1., .1, .2], [.3, 1., .1], [.2, .2, 1.]]))
    output = tmp_path / "state.npz"
    model.save(output)
    with np.load(output, allow_pickle=False) as saved:
        assert str(saved["schema"]) == "sanw_labclip_inner_v9"
        np.testing.assert_array_equal(saved["weight"], model.linear.weight.detach().numpy())


def test_canonical_transform_is_row_partition_invariant_and_scale_invariant():
    rng = np.random.default_rng(41)
    text, weight = rng.normal(size=(19, 17)), rng.normal(size=(17, 17))
    combined = canonical_transform(text, weight)
    separated = np.concatenate([canonical_transform(row[None], weight) for row in text])
    np.testing.assert_array_equal(combined, separated)
    np.testing.assert_allclose(np.linalg.norm(combined, axis=1), 1, atol=3e-16)
    identity = canonical_transform(text, np.eye(17))
    scaled = canonical_transform(text, 2 * np.eye(17))
    np.testing.assert_array_equal(identity, scaled)
    with pytest.raises(FloatingPointError):
        canonical_transform(text, np.zeros((17, 17)))


def test_owner_split_rejects_overlap_and_nontraining_ids():
    split = {"schema": "sanw_inner_training_owner_split_v9", "train_image_manifest_indices": [10, 30],
             "validation_image_manifest_indices": [20, 40]}
    train, validation = validate_owner_split(split, [10, 20, 30, 40])
    np.testing.assert_array_equal(train, [0, 2]); np.testing.assert_array_equal(validation, [1, 3])
    with pytest.raises(ValueError):
        validate_owner_split({**split, "validation_image_manifest_indices": [30, 40]}, [10, 20, 30, 40])
    with pytest.raises(ValueError):
        validate_owner_split({**split, "validation_image_manifest_indices": [20, 999]}, [10, 20, 30, 40])


def test_adapted_sampler_is_unique_owner_complete_source_and_deterministic():
    number = 12
    sources = [list(range(i * 5, i * 5 + 5)) for i in range(number)]
    negatives = [[1000 + 2*i, 1001 + 2*i] for i in range(number)]
    train = np.asarray([0, 2, 4, 6, 8, 10])
    config = LABCLIPConfig()
    a = list(epoch_batches(train, sources, negatives, config, np.random.default_rng(17)))
    b = list(epoch_batches(train, sources, negatives, config, np.random.default_rng(17)))
    assert len(a) == 5
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            np.testing.assert_array_equal(x, y)
    visited = []
    for image, positive, negative in a:
        assert len(set(image)) == len(image)
        assert set(image) == set(train)
        visited.extend(positive.tolist())
        for i, p, n in zip(image, positive, negative):
            assert p in sources[i] and n in negatives[i]
    assert sorted(visited) == sorted(t for i in train for t in sources[i])


def test_published_sampler_keeps_caption_rows_fixed_negatives_and_repeated_owners():
    sources = [list(range(i * 5, i * 5 + 5)) for i in range(8)]
    negatives = [[100 + i, 200 + i] for i in range(8)]
    config = LABCLIPConfig.published_control()
    first = list(epoch_batches([0, 2, 4, 6], sources, negatives, config, np.random.default_rng(17)))
    second = list(epoch_batches([0, 2, 4, 6], sources, negatives, config, np.random.default_rng(19)))
    assert len(first) == 1
    ii, pp, nn = first[0]
    assert len(ii) == 20 and len(set(ii)) == 4
    assert dict(zip(pp, nn)) == dict(zip(second[0][1], second[0][2]))
    config.validate()
    with pytest.raises(ValueError):
        replace(config, batch_size=128).validate()
    with pytest.raises(ValueError):
        replace(LABCLIPConfig(), source_passes_per_epoch=1).validate()


def test_source_ownership_rejects_duplicate_source_rows():
    assert sources_by_owner([4, 9, 12, 14], [0, 0, 1, 1], 2) == [[4, 9], [12, 14]]
    with pytest.raises(ValueError):
        sources_by_owner([4, 4], [0, 1], 2)


def test_runner_refuses_unlocked_fitting_before_loading_inputs(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_practical_labclip_v9 import verify_protocol
    protocol = tmp_path / "bad.json"
    protocol.write_text(json.dumps({"study": "wrong"}))
    with pytest.raises(ValueError, match="immutable v9 protocol"):
        verify_protocol(tmp_path, protocol, "vit_b32", "small_data_fixed_scale", 17, "inner")
