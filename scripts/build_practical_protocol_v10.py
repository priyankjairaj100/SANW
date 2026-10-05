#!/usr/bin/env python3
"""Bind the already chosen expanded-data recipe after feature export is complete."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_joint_v9 import JointFitConfig


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name):
    p = ROOT / name
    if not p.is_file():
        raise FileNotFoundError(f"Required completed artifact is absent: {name}")
    return {"path": name, "sha256": sha(p)}


def main():
    output = ROOT / "results/practical_v10/expanded_training_protocol_v1.json"
    if output.exists():
        raise FileExistsError("Refusing to replace an existing protocol")
    old = json.loads((ROOT / "results/practical_v9/alignment_protocol_v1.json").read_text())
    config = asdict(JointFitConfig(radius=1.0, composition_weight=.25))
    config.pop("seed")
    added_sources = [
        "scripts/build_practical_protocol_v10.py", "scripts/run_practical_streaming_v10.py",
        "src/gcr/practical_training_data_v10.py", "src/gcr/practical_streaming_v10.py",
        "src/gcr/practical_no_retention_v10.py",
        "scripts/extract_features.py", "scripts/strengthen_extract_second_encoder.py",
        "recovery/training_expansion_private/encode_expanded_train6000.py",
        "recovery/training_expansion_private/encode_expansion_pilot.py",
        "recovery/training_expansion_private/encode_expansion_pilot_v1.py",
        "recovery/training_expansion_private/audit_expansion_images.py",
        "recovery/training_expansion_private/build_additional_train_manifest.py",
        "recovery/training_expansion_private/acquire_expansion_assets.py",
        "tests/test_practical_streaming_v10.py",
        "tests/test_practical_streaming_independent_v10.py",
        "tests/test_practical_training_data_v10.py",
        "tests/test_practical_no_retention_v10.py",
    ]
    names = sorted(set(old["source_sha256"]) | set(added_sources))
    sources = {name: record(name)["sha256"] for name in names}
    for name, digest in old["source_sha256"].items():
        if sources[name] != digest:
            raise ValueError(f"Original v9-bound source changed: {name}")
    encoders = ["vit_b32", "rn50"]
    training, completions, export_audits = {}, {}, {}
    for encoder in encoders:
        folder = f"results/official_train_expansion/{encoder}/train_6000"
        records = {"manifest": record("data/official_train_expansion/train_6000/manifest.json"),
                   "features": record(folder + "/features.npz"),
                   "metadata": record(folder + "/metadata.json")}
        metadata = json.loads((ROOT / records["metadata"]["path"]).read_text())
        if (metadata.get("features_sha256") != records["features"]["sha256"]
                or metadata.get("manifest_sha256") != records["manifest"]["sha256"]
                or metadata.get("image_count") != 6000 or metadata.get("text_count") != 110516):
            raise ValueError("Feature export is incomplete or not the prescribed pool")
        completions[encoder] = record(folder + "/completion.json")
        completed = json.loads((ROOT / completions[encoder]["path"]).read_text())
        if (completed.get("complete") is not True
                or completed.get("features_sha256") != records["features"]["sha256"]
                or completed.get("metadata_sha256") != records["metadata"]["sha256"]):
            raise ValueError("Final feature completion marker does not bind the exported bytes")
        export_audits[encoder] = record(f"results/practical_v10/feature_export_audit_{encoder}.json")
        audited = json.loads((ROOT / export_audits[encoder]["path"]).read_text())
        if (audited.get("passed") is not True or audited.get("encoder") != encoder
                or audited.get("image_count") != 6000 or audited.get("text_count") != 110516
                or audited.get("inputs") != records or audited.get("completion") != completions[encoder]
                or audited.get("loader_source_sha256") != sources["src/gcr/practical_training_data_v10.py"]):
            raise ValueError("Independent final feature-export audit is missing or stale")
        training[encoder] = records
    protocol = {
        "study": "sanw_practical_v10", "schema_version": 1,
        "encoders": encoders, "seeds": [17, 29, 43], "fit_config": config,
        "streaming": {"cache_block_size": 128, "query_block_size": 64, "fit_threads": 3},
        "execution": "Sequential encoder fits after feature extraction completes; no overlapping large PCA jobs.",
        "training_inputs": training, "original_training_inputs": old["training_inputs"],
        "training_feature_completions": completions, "independent_feature_export_audits": export_audits,
        "owner_sample": record("recovery/training_expansion_private/FIXED_6000_TRAIN_SAMPLE.json"),
        "confirmation_owner_lock": record("recovery/training_expansion_private/FRESH_CONFIRMATION_1500_OWNER_LOCK.json"),
        "fresh_confirmation_contract": record("results/practical_v10/fresh_confirmation_contract_v1.json"),
        "encoding_pilots": {e: record(f"results/official_train_expansion/{e}/pilot100/receipt.json") for e in encoders},
        "image_integrity_audit": record("recovery/training_expansion_private/IMAGE_DECODING_AND_CONTENT_AUDIT.json"),
        "independent_sample_image_audit": record("recovery/training_expansion_private/INDEPENDENT_SAMPLE_IMAGE_AUDIT_V1.json"),
        "caption_inventory": record("recovery/training_expansion_private/TRAIN6000_METADATA_SUMMARY.json"),
        "source_sha256": sources,
        "inherited_recipe": {
            "protocol": record("results/practical_v9/alignment_protocol_v1.json"),
            "training_only_selection": record("results/practical_v9/inner_selection_joint/selection.json"),
            "failed_development_stop": record("results/practical_v9/pilot_stop_decision.json"),
            "choice": "One inherited rank128/radius1/weight0.25/32epoch recipe, fixed before expanded feature outcomes; no new inner grid.",
        },
        "development_inputs": old["development_inputs"],
        "development_contract": old["development_contract"], "practical_gate": old["practical_gate"],
        "controls": {"no_retention": {"execution": "only_after_joint_seed17_development_pass",
                                       "config": "same_inherited_joint_config", "seeds": [17, 29, 43]}},
        "control_contrast": "Remove active retention penalty and feasibility repair only; preserve data, geometry, objective, initialization, paired query orders, radius, optimizer budget, and minimum training-objective checkpoint selection. Measure actual training losses without enforcing retention.",
        "continuation": {
            "pilot": "Both joint seed17 encoders must pass the unchanged official-development contract before replication.",
            "replication": "Prespecified fixed three-seed mean must pass the same development contract before benchmark evaluation.",
            "replication_rule": "same_fixed_3seed_mean_development_contract_before_benchmark_lock",
            "test_lock": "Freeze and preserve all final joint and matched control states and evaluation sources before test or fresh-confirmation outcomes; no family selection on those outcomes.",
            "failure": "Record the failed stage and stop this recipe; no checkpoint or amplitude search on official development.",
        },
        "selection": "minimum_nonzero_feasible_training_objective_then_earliest_epoch_for_joint; minimum_nonzero_training_objective_then_earliest_epoch_for_control",
        "data_and_compute_scope": "6000 owners and 3008 optimizer updates versus v9's 1200 owners and 608 updates. This tests an expanded data-and-compute recipe, not isolated data-size causality or sample efficiency.",
        "source_caption_duplicates": "Retain inherited source-owner relevance and CE. Report 22 shared source-caption strings across owners; possible false negatives and tie ambiguity are not silently filtered. Caption equivalence is not assumed.",
        "inference": old["inference"],
        "guarantee_scope": "joint only: finite training-gallery inequalities within 1e-12 plus zero canonical frozen-correct ranking losses. No unseen-data guarantee; no-retention control has no such requirement.",
        "evaluation_status": old["evaluation_status"],
        "persistence": "Preserve completed feature inputs, protocol, sources and prefit audit before fitting; preserve fitted states before official development and before test.",
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha(output), "source_count": len(sources)}))


if __name__ == "__main__":
    main()
