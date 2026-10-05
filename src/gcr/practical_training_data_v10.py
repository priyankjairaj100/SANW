"""Strict training-only expanded cache; old training rows retain byte identity.

This loader never constructs a validation dataset. All labels, feature rows,
means, and PCA covariates returned to fitting belong to the locked train owners.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np


RELATIONS = {"source": 1, "supported": 2, "contradicted": 3, "neutral": 4}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def verify_record(repository, record):
    if set(record) != {"path", "sha256"}:
        raise ValueError("Require an exact path/SHA256 input record")
    path = (Path(repository) / record["path"]).resolve()
    if digest(path) != record["sha256"]:
        raise ValueError(f"Input hash differs: {record['path']}")
    return path


def caption_key(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate_manifest(manifest, original, sample):
    """Validate owner isolation and exact old training annotations without fitting."""
    images, texts = manifest["images"], manifest["texts"]
    image_ids, text_ids = [x["id"] for x in images], [x["id"] for x in texts]
    if (len(images) != 6000 or len(set(image_ids)) != len(images)
            or len(set(text_ids)) != len(texts) or any(x["split"] != "train" for x in images)):
        raise ValueError("Expanded manifest must contain exactly 6000 distinct train-only owners")
    old_image_rows = [i for i, item in enumerate(original["images"]) if item["split"] == "train"]
    old_image_ids = {original["images"][i]["id"] for i in old_image_rows}
    old_pairs = [p for p in original["pairs"] if p["image_id"] in old_image_ids]
    old_text_ids = {p["text_id"] for p in old_pairs}
    old_text_rows = [i for i, item in enumerate(original["texts"]) if item["id"] in old_text_ids]
    additions = sample.get("additional_image_ids", [])
    if (sample.get("schema") != "sanw-expanded-train-fixed-sample-v1"
            or sample.get("selection_uses_outcomes") is not False
            or sample.get("selection_before_image_download_or_decoding") is not True
            or sample.get("total_train_image_count") != 6000
            or sample.get("existing_train_image_count") != 1200
            or sample.get("additional_image_count") != 4800
            or len(old_image_rows) != 1200 or len(additions) != 4800
            or len(set(additions)) != 4800 or old_image_ids & set(additions)
            or set(image_ids) != old_image_ids | set(additions)):
        raise ValueError("Expanded owners differ from the pre-acquisition fixed sample")
    if (images[:1200] != [original["images"][i] for i in old_image_rows]
            or texts[:len(old_text_rows)] != [original["texts"][i] for i in old_text_rows]):
        raise ValueError("Original training manifest rows were changed or reordered")
    canonical_pair = lambda p: (p["image_id"], p["text_id"], p["relation"])
    actual_old_pairs = [p for p in manifest["pairs"] if p["image_id"] in old_image_ids]
    if sorted(map(canonical_pair, actual_old_pairs)) != sorted(map(canonical_pair, old_pairs)):
        raise ValueError("Original training relation annotations changed")
    ilook, tlook = {x: i for i, x in enumerate(image_ids)}, {x: i for i, x in enumerate(text_ids)}
    pairs = [dict() for _ in images]
    text_owner = np.full(len(texts), -1, dtype=np.int64)
    for pair in manifest["pairs"]:
        if pair["image_id"] not in ilook or pair["text_id"] not in tlook or pair["relation"] not in RELATIONS:
            raise ValueError("Invalid relation ID or label")
        i, j = ilook[pair["image_id"]], tlook[pair["text_id"]]
        if j in pairs[i] or text_owner[j] >= 0:
            raise ValueError("Duplicate pair or caption ID with multiple owners")
        pairs[i][j] = RELATIONS[pair["relation"]]
        text_owner[j] = i
    if np.any(text_owner < 0) or any(sum(code == 1 for code in p.values()) != 5 for p in pairs):
        raise ValueError("Require all texts owner-associated and exactly five source captions per owner")
    return pairs, old_image_rows, old_text_rows


def _read_features(path, expected_images, expected_texts, dimension):
    with np.load(path, allow_pickle=False) as archive:
        if ([str(x) for x in archive["image_ids"]] != expected_images
                or [str(x) for x in archive["text_ids"]] != expected_texts):
            raise ValueError("Feature order differs from manifest")
        images, texts = archive["image_features"], archive["text_features"]
    for name, value, size in (("image", images, len(expected_images)), ("text", texts, len(expected_texts))):
        if value.dtype != np.float32 or value.shape != (size, dimension) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} feature dtype/shape/values")
        norms = np.sqrt(np.sum(value.astype(np.float64) ** 2, axis=1))
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
            raise ValueError(f"Unnormalized {name} features")
    return images, texts


def validate_encoder_metadata(metadata, old_metadata):
    for key in ("model_repository", "model_revision", "weights_sha256", "logit_scale", "normalization", "dtype", "dimension",
                "text_encoding", "open_clip_version", "torch_version"):
        if key not in metadata or metadata[key] != old_metadata.get(key):
            raise ValueError(f"Expanded encoder identity differs on {key}")
    expected = {"size": 224, "mode": "RGB", "interpolation": "bicubic", "resize_mode": "shortest", "crop": "center",
                "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}
    if (metadata.get("preprocess_config") != expected
            or any(expected.get(key) != value for key, value in old_metadata.get("preprocess_config", {}).items())
            or metadata["text_encoding"] != "causally_trimmed_after_last_eot"):
        raise ValueError("Image preprocessing or causal text encoding changed")


def load_training(repository, encoder, protocol):
    """One float32 Torch normalization pass, followed by canonical float64 math."""
    import torch
    import torch.nn.functional as F

    if encoder not in ("vit_b32", "rn50"):
        raise ValueError("Unknown encoder")
    repository = Path(repository).resolve()
    records = protocol["training_inputs"][encoder]
    old_records = protocol["original_training_inputs"][encoder]
    if set(records) != {"manifest", "features", "metadata"} or set(old_records) != set(records):
        raise ValueError("Require all three new and original input records")
    paths = {key: verify_record(repository, value) for key, value in records.items()}
    old_paths = {key: verify_record(repository, value) for key, value in old_records.items()}
    sample_path = verify_record(repository, protocol["owner_sample"])
    confirmation_path = verify_record(repository, protocol["confirmation_owner_lock"])
    manifest, original = (json.loads(p.read_text()) for p in (paths["manifest"], old_paths["manifest"]))
    sample = json.loads(sample_path.read_text())
    pairs, old_image_rows, old_text_rows = validate_manifest(manifest, original, sample)
    confirmation = json.loads(confirmation_path.read_text())
    reserved = confirmation.get("image_ids", [])
    if (confirmation.get("schema") != "sanw-fresh-confirmation-owner-sample-v1"
            or confirmation.get("fit_sample_sha256") != protocol["owner_sample"]["sha256"]
            or confirmation.get("fit_or_selection_allowed") is not False
            or len(reserved) != 1500 or len(set(reserved)) != 1500
            or set(reserved) & {x["id"] for x in manifest["images"]}):
        raise ValueError("Fresh confirmation IDs must remain disjoint and excluded from fitting")
    metadata, old_metadata = (json.loads(p.read_text()) for p in (paths["metadata"], old_paths["metadata"]))
    dimension = 512 if encoder == "vit_b32" else 1024
    validate_encoder_metadata(metadata, old_metadata)
    if (metadata["dimension"] != dimension or metadata["dtype"] != "float32" or metadata["normalization"] != "L2"
            or metadata.get("manifest_sha256") != records["manifest"]["sha256"]
            or metadata.get("features_sha256") != records["features"]["sha256"]
            or metadata.get("image_count") != len(manifest["images"])
            or metadata.get("text_count") != len(manifest["texts"])
            or not math.isfinite(float(metadata["logit_scale"])) or float(metadata["logit_scale"]) <= 0):
        raise ValueError("Expanded feature metadata is inconsistent")
    pilot_record, integrity_record = protocol["encoding_pilots"][encoder], protocol["image_integrity_audit"]
    pilot = json.loads(verify_record(repository, pilot_record).read_text())
    integrity = json.loads(verify_record(repository, integrity_record).read_text())
    if (metadata.get("pilot_receipt_sha256") != pilot_record["sha256"]
            or metadata.get("image_integrity_audit_sha256") != integrity_record["sha256"]
            or pilot.get("schema") != "sanw-expansion-cold-encoding-pilot-v1" or pilot.get("encoder") != encoder
            or pilot.get("weights_sha256") != metadata["weights_sha256"]
            or pilot.get("sample_sha256") != protocol["owner_sample"]["sha256"]
            or pilot.get("anchor_parity_passed") is not True or pilot.get("anchor_tolerance") != 2e-6
            or pilot.get("anchor_images", 0) <= 0 or pilot.get("anchor_texts", 0) <= 0
            or not all(0 <= pilot.get(key, float("inf")) <= 2e-6 for key in
                       ("old_image_anchor_max_abs_error", "old_text_anchor_max_abs_error", "padding_parity_max_error"))
            or pilot.get("confirmation_images_encoded") != 0 or pilot.get("benchmark_outcomes_read") is not False
            or integrity.get("sample_sha256") != protocol["owner_sample"]["sha256"]
            or integrity.get("confirmation_sha256") != protocol["confirmation_owner_lock"]["sha256"]
            or integrity.get("passed_no_reserved_content_overlap") is not True):
        raise ValueError("Encoder anchor parity or image integrity audit differs")
    if (metadata.get("original_manifest_sha256") != old_records["manifest"]["sha256"]
            or metadata.get("original_features_sha256") != old_records["features"]["sha256"]
            or metadata.get("original_metadata_sha256") != old_records["metadata"]["sha256"]
            or metadata.get("sample_lock_sha256") != protocol["owner_sample"]["sha256"]
            or metadata.get("fresh_confirmation_lock_sha256") != protocol["confirmation_owner_lock"]["sha256"]
            or metadata.get("all_rows_training_only") is not True or metadata.get("fresh_confirmation_rows") != 0):
        raise ValueError("Expanded feature lineage differs from the locked inputs")
    image_ids, text_ids = [x["id"] for x in manifest["images"]], [x["id"] for x in manifest["texts"]]
    images32, texts32 = _read_features(paths["features"], image_ids, text_ids, dimension)
    # Only these explicit original training rows participate in the parity audit.
    with np.load(old_paths["features"], allow_pickle=False) as old:
        if ([str(x) for x in old["image_ids"][old_image_rows]] != image_ids[:len(old_image_rows)]
                or [str(x) for x in old["text_ids"][old_text_rows]] != text_ids[:len(old_text_rows)]
                or not np.array_equal(old["image_features"][old_image_rows].view(np.uint32), images32[:len(old_image_rows)].view(np.uint32))
                or not np.array_equal(old["text_features"][old_text_rows].view(np.uint32), texts32[:len(old_text_rows)].view(np.uint32))):
            raise ValueError("Original training feature bytes changed")
    images = F.normalize(torch.from_numpy(images32), dim=-1).numpy().astype(np.float64)
    texts = F.normalize(torch.from_numpy(texts32), dim=-1).numpy().astype(np.float64)
    if (not np.array_equal(images[:len(old_image_rows)], F.normalize(torch.from_numpy(images32[:len(old_image_rows)]), dim=-1).numpy())
            or not np.array_equal(texts[:len(old_text_rows)], F.normalize(torch.from_numpy(texts32[:len(old_text_rows)]), dim=-1).numpy())):
        raise ValueError("Expanded batch shape changed the original one-pass feature normalization")
    sources, supported, contra, excluded = [], [], [], []
    source_rows = np.asarray(sorted(j for p in pairs for j, code in p.items() if code == 1), dtype=np.int64)
    source_lookup = {j: k for k, j in enumerate(source_rows)}
    owner = np.full(len(source_rows), -1, dtype=np.int64)
    for i, relations in enumerate(pairs):
        positives = {caption_key(manifest["texts"][j]["text"]) for j, code in relations.items() if code in (1, 2)}
        sources.append(sorted(j for j, code in relations.items() if code == 1))
        supported.append(sorted(j for j, code in relations.items() if code == 2))
        negative = []
        for j, code in relations.items():
            if code == 1:
                owner[source_lookup[j]] = i
            elif code == 3:
                if caption_key(manifest["texts"][j]["text"]) in positives:
                    excluded.append({"training_image_index": i, "global_text_index": j})
                else:
                    negative.append(j)
        contra.append(sorted(negative))
    provenance = {"inputs": records, "original_training_inputs": old_records,
                  "owner_sample": protocol["owner_sample"], "confirmation_owner_lock": protocol["confirmation_owner_lock"],
                  "training_image_manifest_indices": list(range(len(images))),
                  "training_text_manifest_indices": list(range(len(texts))),
                  "training_source_text_manifest_indices": source_rows.tolist(),
                  "original_image_manifest_indices": old_image_rows, "original_text_manifest_indices": old_text_rows,
                  "original_training_feature_bytes_identical": True,
                  "original_normalized_training_features_identical": True,
                  "encoding_pilot": pilot_record, "image_integrity_audit": integrity_record,
                  "excluded_exact_normalized_contradiction_conflicts": excluded,
                  "basis_text_scope": "all distinct training-associated captions including neutral rows as unlabelled PCA covariates",
                  "feature_preprocessing": "float32_cache_then_Torch_float32_F_normalize_once_then_float64",
                  "torch_version": torch.__version__, "fit_split": "train", "heldout_used": False,
                  "caption_equivalence_assumed": False, "neutral_as_negative": False}
    return images, texts, source_rows, owner, sources, supported, contra, provenance, float(metadata["logit_scale"])
