"""Isolation and immutable-protocol regressions for the expanded-data path."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gcr.practical_joint_v9 import JointFitConfig
from gcr.practical_training_data_v10 import _read_features, digest, validate_encoder_metadata, validate_manifest
from run_practical_streaming_v10 import REQUIRED_SOURCES, verify_protocol, verify_replication_gate


@pytest.fixture(scope="module")
def manifests():
    images = [{"id": f"image:{i}", "split": "train", "path": f"{i}.jpg"} for i in range(6000)]
    texts, pairs = [], []
    for i in range(6000):
        for j in range(5):
            tid = f"text:{i}:{j}"
            texts.append({"id": tid, "text": f"source {i} {j}"})
            pairs.append({"image_id": images[i]["id"], "text_id": tid, "relation": "source"})
    expanded = {"images": images, "texts": texts, "pairs": pairs}
    original = {"images": images[:1200] + [{"id": "reserved", "split": "validation"}],
                "texts": texts[:6000], "pairs": pairs[:6000]}
    sample = {"schema": "sanw-expanded-train-fixed-sample-v1", "selection_uses_outcomes": False,
              "selection_before_image_download_or_decoding": True, "total_train_image_count": 6000,
              "existing_train_image_count": 1200, "additional_image_count": 4800,
              "additional_image_ids": [x["id"] for x in images[1200:]]}
    return expanded, original, sample


def test_strict_manifest_retains_only_old_training_rows(manifests):
    manifest, original, sample = manifests
    pairs, rows, texts = validate_manifest(manifest, original, sample)
    assert rows == list(range(1200)) and texts == list(range(6000))
    assert len(pairs) == 6000 and all(len(p) == 5 for p in pairs)


@pytest.mark.parametrize("corruption", ["heldout", "wrong_owner", "old_label", "caption_shared", "missing_source", "outcome_selected"])
def test_rejects_data_leakage_or_changed_training_identity(manifests, corruption):
    manifest, original, sample = copy.deepcopy(manifests)
    if corruption == "heldout":
        manifest["images"][-1]["split"] = "validation"
    elif corruption == "wrong_owner":
        sample["additional_image_ids"][-1] = "reserved"
    elif corruption == "old_label":
        manifest["pairs"][0]["relation"] = "contradicted"
    elif corruption == "caption_shared":
        manifest["pairs"][-1]["text_id"] = manifest["pairs"][-6]["text_id"]
    elif corruption == "missing_source":
        manifest["pairs"][-1]["relation"] = "neutral"
    else:
        sample["selection_uses_outcomes"] = True
    with pytest.raises(ValueError):
        validate_manifest(manifest, original, sample)


def test_feature_order_dtype_and_finiteness_are_enforced(tmp_path):
    path = tmp_path / "cache.npz"
    def save(images, texts, image_ids=("a", "b")):
        np.savez(path, image_features=images, text_features=texts, image_ids=image_ids, text_ids=["x"])
    save(np.eye(2, dtype=np.float32), np.array([[1, 0]], dtype=np.float32))
    _read_features(path, ["a", "b"], ["x"], 2)
    for images, ids in ((np.eye(2, dtype=np.float64), ("a", "b")),
                        (np.eye(2, dtype=np.float32), ("b", "a")),
                        (np.array([[np.nan, 0], [0, 1]], dtype=np.float32), ("a", "b"))):
        save(images, np.array([[1, 0]], dtype=np.float32), ids)
        with pytest.raises(ValueError):
            _read_features(path, ["a", "b"], ["x"], 2)


def protocol_fixture(tmp_path, monkeypatch):
    sources = {}
    for name in (*REQUIRED_SOURCES, "extra_declared_source.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = digest(path)
    config = asdict(JointFitConfig(radius=1, composition_weight=.25)); config.pop("seed")
    protocol = {"study": "sanw_practical_v10", "encoders": ["vit_b32", "rn50"], "seeds": [17, 29, 43],
                "fit_config": config, "source_sha256": sources,
                "streaming": {"fit_threads": 3, "cache_block_size": 128, "query_block_size": 64}}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "3")
    return protocol


def test_every_declared_source_is_verified(tmp_path, monkeypatch):
    protocol = protocol_fixture(tmp_path, monkeypatch)
    config, _ = verify_protocol(tmp_path, protocol, "vit_b32", 17)
    assert config.radius == 1 and config.composition_weight == .25
    (tmp_path / "extra_declared_source.py").write_text("changed")
    with pytest.raises(ValueError, match="Declared source changed"):
        verify_protocol(tmp_path, protocol, "vit_b32", 17)


def test_new_grid_or_unbound_threads_rejected(tmp_path, monkeypatch):
    protocol = protocol_fixture(tmp_path, monkeypatch)
    protocol["fit_config"]["radius"] = .3
    with pytest.raises(ValueError, match="inherited"):
        verify_protocol(tmp_path, protocol, "rn50", 17)
    protocol["fit_config"]["radius"] = 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    with pytest.raises(ValueError, match="thread environment"):
        verify_protocol(tmp_path, protocol, "rn50", 17)


def test_replication_cannot_use_an_unbound_success_boolean(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"study": "sanw_practical_v10_replication_gate", "family": "joint", "passed": True,
                               "protocol_sha256": "p", "encoders_passed": ["rn50", "vit_b32"], "seed": 17}))
    with pytest.raises(ValueError, match="pilot results"):
        verify_replication_gate(tmp_path, path, "p")


def test_encoder_metadata_rejects_changed_crops_or_tokenization():
    prep = {"size": 224, "mode": "RGB", "interpolation": "bicubic", "resize_mode": "shortest", "crop": "center",
            "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}
    metadata = {key: key for key in ("model_repository", "model_revision", "weights_sha256", "logit_scale", "normalization",
                                    "dtype", "dimension", "open_clip_version", "torch_version")}
    metadata.update(text_encoding="causally_trimmed_after_last_eot", preprocess_config=prep)
    old = copy.deepcopy(metadata); old["preprocess_config"] = {key: prep[key] for key in ("mean", "std")}
    validate_encoder_metadata(metadata, old)
    changed = copy.deepcopy(metadata); changed["preprocess_config"]["crop"] = "random"
    with pytest.raises(ValueError, match="preprocessing"):
        validate_encoder_metadata(changed, old)
    changed = copy.deepcopy(metadata); changed["text_encoding"] = "untrimmed_different_tokenizer"
    with pytest.raises(ValueError, match="text_encoding"):
        validate_encoder_metadata(changed, old)
