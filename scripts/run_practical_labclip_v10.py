#!/usr/bin/env python3
"""Prespecify and fit the expanded-data explanatory LABCLIP-style comparator.

No inherited protocol/source is edited. Draft contracts cannot execute fits.
The family is never a practical candidate and never enters fresh confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from run_practical_constrained_v8 import atomic_json
from run_practical_streaming_v10 import verify_protocol
from evaluate_practical_benchmark_v10 import verify_development_gate as strict_development_gate, EXTRA_SOURCES
from evaluate_practical_official_development_v10 import require_evaluation_threads
from gcr.practical_labclip_v9 import LABCLIPConfig, OFFICIAL_FILES, sources_by_owner
from gcr.practical_labclip_v10 import AUDIT_SEED, WITNESS_SOURCE_COUNT, FUNCTIONAL_TOLERANCE, fit_expanded_alignment
from gcr.practical_training_data_v10 import digest, load_training

STUDY = "sanw_practical_v10_expanded_labclip"
CONTRACT_STUDY = STUDY + "_contract"
NEW_SOURCES = tuple(dict.fromkeys(("src/gcr/practical_labclip_v10.py", "scripts/run_practical_labclip_v10.py",
                                  "tests/test_practical_labclip_v10.py") + EXTRA_SOURCES))
ENDPOINTS = {"e_vil_test1000": ["i2t.r1", "t2i.r1"], "coco_karpathy": ["i2t.r1", "t2i.r1"],
             "sugarcrepe_pp": ["both_accuracy"]}
SELECTION = {
    "rule": "minimum_fixed_training_HNB_objective_then_earliest_epoch_among_nonzero_witness_states",
    "candidate_epochs": list(range(1, 33)), "audit_seed": AUDIT_SEED,
    "audit_source_scope": "all_five_source_rows_per_eligible_owner_once",
    "audit_candidates": "fixed_unique_owner_B128_batch_pools_one_fixed_valid_contradiction_per_source",
    "audit_aggregation": "image_row_weighted_batch_HNB_CE",
    "audit_numerics": "float32_model_forward_and_CE_then_ordered_float64_scalar_accumulation",
    "changing_online_loss_is_selection_objective": False,
    "full_gallery_or_all_negative_objective": False,
    "witness_source_count": WITNESS_SOURCE_COUNT,
    "witness_scope": "first_128_eligible_owned_source_rows_in_immutable_manifest_order",
    "witness_reference": "identity_normalized_text_map_not_original_frozen_cosine",
    "witness_requirements": "canonical_float64_direction_and_owned_pair_score_movement_strictly_above_tolerance",
    "witness_tolerance": FUNCTIONAL_TOLERANCE,
}
INFERENCE = {"seeds": [17, 29, 43], "seed_aggregation": "fixed_mean_per_paired_item_no_seed_resampling",
             "resampling": "paired_image_clusters_with_ratio_of_item_totals_for_unequal_cluster_sizes",
             "bootstrap_replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80,
             "familywise_alpha": .05, "interval_quantiles": "exact_rational_linear_percentiles",
             "tail_probability": {"numerator": 1, "denominator": 3200},
             "supplementary_effects": 10, "inherited_main_effects": 30,
             "retrieval_only_effects": 20, "combined_effects": 60,
             "baseline_absolute_metrics": "descriptive_only_no_additional_intervals_or_gate",
             "new_candidate_or_confirmatory_gate": False}


def path_record(path, repository=ROOT):
    path, repository = Path(path).resolve(), Path(repository).resolve()
    return {"path": str(path.relative_to(repository)) if path.is_relative_to(repository) else str(path),
            "sha256": digest(path)}


def verified_record(repository, record):
    path = (Path(repository) / record["path"]).resolve()
    if digest(path) != record["sha256"]:
        raise ValueError("Bound LABCLIP prerequisite artifact changed")
    return path


def config_record():
    config = asdict(LABCLIPConfig()); config.pop("seed")
    config["checkpoint_epochs"] = list(config["checkpoint_epochs"])
    return config


def validate_contract_payload(contract, protocol, protocol_sha):
    if (contract.get("study") != CONTRACT_STUDY or contract.get("family") != "labclip"
            or contract.get("status") not in ("draft_not_executable", "frozen")
            or protocol.get("study") != "sanw_practical_v10"
            or contract.get("inherited_protocol", {}).get("sha256") != protocol_sha
            or contract.get("config") != config_record() or contract.get("selection") != SELECTION
            or contract.get("encoders") != ["vit_b32", "rn50"] or contract.get("seeds") != [17, 29, 43]
            or contract.get("training_inputs") != protocol.get("training_inputs")
            or contract.get("candidate_selection_allowed") is not False
            or contract.get("fresh_confirmation_inclusion_allowed") is not False
            or contract.get("explanatory_control_only") is not True
            or contract.get("required_gate_study") != "sanw_practical_v10_fixed_seed_development_gate"
            or contract.get("required_gate_family") != "joint"
            or contract.get("required_gate_status") != "both_encoders_passing_fixed_three_seed_aggregate"
            or contract.get("contrasts") != [["joint_minus_labclip", "joint", "labclip"]]
            or contract.get("endpoints") != ENDPOINTS or contract.get("inference") != INFERENCE
            or contract.get("all_six_states_locked_before_any_practical_benchmark_outcome") is not True
            or contract.get("fresh_confirmation_contract") != protocol.get("fresh_confirmation_contract")):
        raise ValueError("Expanded LABCLIP contract differs from its finite explanatory recipe")


def build_contract(repository, protocol_path, *, freeze=False, source_audit=None):
    repository, protocol_path = Path(repository).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    for path, sha in protocol["source_sha256"].items():
        if digest(repository / path) != sha:
            raise ValueError("An inherited bound source changed")
    verified_record(repository, protocol["fresh_confirmation_contract"])
    sources = {path: digest(repository / path) for path in NEW_SOURCES}
    audit_record = None
    if freeze:
        if source_audit is None:
            raise ValueError("Freezing requires the independent source-audit receipt")
        audit = json.loads(Path(source_audit).read_text())
        if (audit.get("study") != STUDY + "_source_audit" or audit.get("passed") is not True
                or audit.get("protocol_sha256") != digest(protocol_path)
                or audit.get("new_source_sha256") != sources):
            raise ValueError("Source audit does not cover the exact comparator sources")
        audit_record = path_record(source_audit, repository)
    contract = {
        "study": CONTRACT_STUDY, "schema_version": 1, "family": "labclip",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen" if freeze else "draft_not_executable",
        "inherited_protocol": path_record(protocol_path, repository),
        "config": config_record(), "seeds": [17, 29, 43], "encoders": ["vit_b32", "rn50"],
        "training_inputs": protocol["training_inputs"], "selection": SELECTION,
        "objective": "inherited_HNB:half_Bx2B_I2T_CE_plus_half_positive_only_BxB_T2I_CE",
        "optimizer": {"name": "Adam", "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8,
                      "weight_decay": 0, "scheduler": None, "gradient_clip": None},
        "score": "pure_image_dot_L2_normalized_full_rank_transformed_text",
        "initialization": "identity_W_no_bias_native_fixed_logit_scale",
        "canonical_inference": "unchanged_v9_float64_rowwise_einsum_then_fixed_norm_then_pair_sum",
        "frozen_comparator": "unchanged_original_float32_normalized_feature_cosine",
        "identity_diagnostic": "Report_identity_W_vs_original_frozen_score_error_winner_and_correctness_parity_on_each_evaluated_gallery_no_selection_or_recalibration",
        "source_role": "five_owned_source_captions_only_as_positive_loss_targets",
        "supported_label_use": "same_owner_negative_conflict_exclusion_only_not_direct_loss_targets",
        "negative_role": "sample_valid_annotated_owner_contradictions_not_published_synthetic_word_shuffles",
        "negative_conflict_rule": "same_expanded_loader_NFKC_casefold_word_tokens_keep_digits_against_any_source_or_supported_same_owner",
        "omission_rule": "omit_owners_without_any_valid_contradiction_report_exact_owner_source_row_and_step_counts",
        "source_duplicate_rule": "retain_inherited_owner_labels_including_shared_strings_no_cross_owner_relabelling",
        "data_comparison_scope": "same_locked6000_owner_pool_and_features_with_disclosed_eligible_subset_and_supervision_differences",
        "compute_comparison_scope": "32epochs_five_source_passes_B128_up_to7520updates_versus_joint3008_not_equal_steps_capacity_or_compute",
        "checkpoint_adaptation": "all32epochs_evaluated_on_one_fixed_complete_eligible_source_batch_bank_distinct_from_stochastic_training_loss",
        "official_repository": "https://github.com/kdariina/CLIP-not-BoW-unimodally",
        "official_file_git_blob_sha": OFFICIAL_FILES,
        "published_fidelity_scope": "architecture_HNB_and_Adam_reused_no_faithful_published_word_shuffle_training_claim",
        "explanatory_control_only": True, "candidate_selection_allowed": False,
        "required_gate_study": "sanw_practical_v10_fixed_seed_development_gate", "required_gate_family": "joint",
        "required_gate_status": "both_encoders_passing_fixed_three_seed_aggregate",
        "execution": "only_after_joint_three_seed_development_gate_sequential_CPU_fits_three_threads",
        "all_six_states_locked_before_any_practical_benchmark_outcome": True,
        "no_outcome_based_tuning": True, "new_grid_or_schedule_changes": False,
        "contrasts": [["joint_minus_labclip", "joint", "labclip"]], "endpoints": ENDPOINTS, "inference": INFERENCE,
        "fresh_confirmation_inclusion_allowed": False,
        "fresh_confirmation_contract": protocol["fresh_confirmation_contract"],
        "fresh_confirmation_scope": "unchanged_joint_and_no_retention_only",
        "new_source_sha256": sources, "inherited_source_sha256": protocol["source_sha256"],
        "source_audit": audit_record,
        "prespecification_context": "prepared_during_joint_seed17_training_without_v10_heldout_outcomes",
        "benchmark_execution_dependency": "separate_source_audited_additional_model_lock_and_evaluator_must_be_bound_before_any_benchmark_outcome",
    }
    validate_contract_payload(contract, protocol, digest(protocol_path))
    return contract


def verify_contract(repository, contract_path, *, allow_draft=False):
    contract = json.loads(Path(contract_path).read_text())
    protocol_path = verified_record(repository, contract["inherited_protocol"])
    protocol = json.loads(protocol_path.read_text())
    validate_contract_payload(contract, protocol, digest(protocol_path))
    if not allow_draft and contract["status"] != "frozen":
        raise ValueError("Draft LABCLIP contracts cannot execute")
    if (contract["inherited_source_sha256"] != protocol["source_sha256"]
            or set(contract["new_source_sha256"]) != set(NEW_SOURCES)):
        raise ValueError("Comparator source dependency set changed")
    for name, sha in {**protocol["source_sha256"], **contract["new_source_sha256"]}.items():
        if digest(Path(repository) / name) != sha:
            raise ValueError("A bound inherited or comparator source changed")
    verified_record(repository, contract["fresh_confirmation_contract"])
    if contract["status"] == "frozen":
        audit = json.loads(verified_record(repository, contract["source_audit"]).read_text())
        if (audit.get("study") != STUDY + "_source_audit" or audit.get("passed") is not True
                or audit.get("protocol_sha256") != digest(protocol_path)
                or audit.get("new_source_sha256") != contract["new_source_sha256"]):
            raise ValueError("Frozen source audit no longer matches")
    return contract, protocol, protocol_path


def verify_gate(gate_path, protocol_sha):
    states = strict_development_gate(Path(gate_path), protocol_sha)
    if set(states) != {(enc, seed) for enc in ("vit_b32", "rn50") for seed in (17, 29, 43)}:
        raise ValueError("Passing joint aggregate must bind all six fixed states")
    return {"record": path_record(gate_path), "qualified_joint_states": [states[key] for key in sorted(states)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build-contract")
    build.add_argument("--protocol", type=Path, required=True); build.add_argument("--output", type=Path, required=True)
    build.add_argument("--freeze", action="store_true"); build.add_argument("--source-audit", type=Path)
    pre = sub.add_parser("verify-prerequisites")
    pre.add_argument("--contract", type=Path, required=True); pre.add_argument("--development-gate", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--contract", type=Path, required=True); fit.add_argument("--development-gate", type=Path, required=True)
    fit.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    fit.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "build-contract":
        if args.output.exists(): raise FileExistsError("Refusing to overwrite comparator contract")
        atomic_json(args.output, build_contract(ROOT, args.protocol, freeze=args.freeze, source_audit=args.source_audit))
        print(json.dumps(path_record(args.output)), flush=True); return
    if args.mode == "verify-prerequisites":
        require_evaluation_threads()
        if args.output.exists(): raise FileExistsError("Refusing to overwrite prerequisite receipt")
        contract, protocol, protocol_path = verify_contract(ROOT, args.contract)
        gate = verify_gate(args.development_gate, digest(protocol_path))
        atomic_json(args.output, {"study": STUDY + "_prerequisites", "passed": True,
                                 "contract": path_record(args.contract), "protocol_sha256": digest(protocol_path),
                                 "development_gate": gate, "benchmark_outcomes_read": False})
        print(json.dumps(path_record(args.output)), flush=True); return
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError("Refusing to overwrite comparator fit")
    contract, protocol, protocol_path = verify_contract(ROOT, args.contract)
    _, thread_values = verify_protocol(ROOT, protocol, args.encoder, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    prerequisite_path = args.output / "prerequisite_validation.json"
    env = {**os.environ, **{name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify-prerequisites",
                    "--contract", str(args.contract.resolve()), "--development-gate", str(args.development_gate.resolve()),
                    "--output", str(prerequisite_path.resolve())], env=env, check=True)
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    images, texts, source_rows, owner, sources, supported, negatives, provenance, scale = load_training(ROOT, args.encoder, protocol)
    if len(images) != 6000 or len(source_rows) != 30000 or any(len(rows) != 5 for rows in sources):
        raise ValueError("Expanded comparator requires the frozen6000owner/five-source pool")
    if sources_by_owner(source_rows, owner, len(images)) != sources:
        raise ValueError("Loader and source-ownership reconstruction differ")
    eligible = np.asarray([i for i, rows in enumerate(negatives) if rows], dtype=np.int64)
    omitted = [i for i, rows in enumerate(negatives) if not rows]
    if not len(eligible): raise ValueError("No owner has a valid contradiction")
    config = LABCLIPConfig(seed=args.seed)
    steps_per_epoch = 5 * ((len(eligible) + 127) // 128)
    identity = {"study": STUDY, "family": "labclip", "encoder": args.encoder, "config": asdict(config),
                "protocol": path_record(protocol_path), "protocol_sha256": digest(protocol_path),
                "contract": path_record(args.contract), "source_audit": contract["source_audit"],
                "prerequisite_validation": path_record(prerequisite_path), "training_provenance": provenance,
                "new_source_sha256": contract["new_source_sha256"], "source_sha256": protocol["source_sha256"],
                "eligible_owner_indices": eligible.tolist(), "eligible_owner_count": len(eligible),
                "omitted_no_valid_contradiction_owner_indices": omitted, "requested_owner_count": len(images),
                "eligible_source_caption_count": 5 * len(eligible), "source_rows_per_epoch": 5 * len(eligible),
                "updates_per_epoch": steps_per_epoch, "optimizer_update_budget": 32 * steps_per_epoch,
                "native_logit_scale": scale, "selection": SELECTION,
                "supported_captions_used_as_loss_targets": False, "supported_labels_used_for_negative_conflict_exclusion": True,
                "candidate_selection_allowed": False, "fresh_confirmation_inclusion_allowed": False,
                "heldout_used_in_fitting_or_selection": False,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                                "blas_threads": thread_values, "torch_fit_threads": 3, "normalization_threads": 1}}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_json(args.output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_sha})
    checkpoints = []
    def on_epoch(row, model):
        path = args.output / "checkpoints" / f"epoch_{row['epoch']:03d}.npz"
        model.save(path)
        record = {**row, "checkpoint": {"path": str(path.relative_to(args.output)), "sha256": digest(path),
                                        "ledger_sha256": ledger_sha}}
        checkpoints.append(record); atomic_json(args.output / "checkpoint_history.json", checkpoints)
        print(json.dumps({"encoder": args.encoder, "seed": args.seed, "epoch": row["epoch"],
                          "fixed_training_objective": row["training_objective"],
                          "nonzero_functional_update": row["nonzero_functional_update"]}), flush=True)
    result = fit_expanded_alignment(images, texts, source_rows, owner, negatives, eligible, config, scale, on_epoch)
    if result["optimizer_steps"] != 32 * steps_per_epoch:
        raise ValueError("Expanded LABCLIP optimizer budget differs")
    selected = next(row for row in checkpoints if row["epoch"] == result["selected_epoch"])
    result.update({"study": STUDY, "family": "labclip", "encoder": args.encoder,
                   "protocol_sha256": digest(protocol_path), "contract": path_record(args.contract),
                   "ledger_sha256": ledger_sha, "checkpoint_history": checkpoints,
                   "selected_checkpoint": selected["checkpoint"], "nonzero": True,
                   "eligible_owner_count": len(eligible), "omitted_no_valid_contradiction_owner_indices": omitted,
                   "candidate_selection_allowed": False, "fresh_confirmation_inclusion_allowed": False,
                   "elapsed_seconds": time.monotonic() - started})
    atomic_json(args.output / "completion.json", result)
    print(json.dumps({"complete": True, "encoder": args.encoder, "seed": args.seed,
                      "selected_epoch": result["selected_epoch"], "optimizer_steps": result["optimizer_steps"]}), flush=True)


if __name__ == "__main__":
    main()
