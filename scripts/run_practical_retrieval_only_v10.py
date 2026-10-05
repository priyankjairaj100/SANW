#!/usr/bin/env python3
"""Separately prespecify and run the explanatory retrieval-only v10 control.

The inherited protocol and all its bound sources remain unchanged. This family
cannot become the practical candidate and is excluded from fresh confirmation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
from run_practical_constrained_v8 import atomic_json
from run_practical_streaming_v10 import verify_protocol
from evaluate_practical_benchmark_v10 import (verify_development_gate as strict_development_gate,
                                              verify_control as strict_control, EXTRA_SOURCES)
from evaluate_practical_official_development_v10 import require_evaluation_threads
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_joint_v9 import JointCompositionExamples, JointFitConfig
from gcr.practical_retrieval_only_v10 import fit_retrieval_only
from gcr.practical_streaming_v10 import FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss
from gcr.practical_training_data_v10 import digest, load_training

NEW_SOURCES = tuple(dict.fromkeys(("src/gcr/practical_retrieval_only_v10.py", "scripts/run_practical_retrieval_only_v10.py",
                                  "tests/test_practical_retrieval_only_v10.py") + EXTRA_SOURCES))
CONTRACT_SCHEMA = "sanw_practical_v10_retrieval_only_control_contract"
GATE_STUDY = "sanw_practical_v10_fixed_seed_development_gate"
CONTRASTS = [["joint_minus_retrieval_only","joint","retrieval_only"],
             ["retrieval_only_minus_frozen","retrieval_only","frozen"]]
ENDPOINTS = {"e_vil_test1000":["i2t.r1","t2i.r1"], "coco_karpathy":["i2t.r1","t2i.r1"],
             "sugarcrepe_pp":["both_accuracy"]}
INFERENCE = {"seeds":[17,29,43],"seed_aggregation":"fixed_mean_per_paired_item_no_seed_resampling",
             "resampling":"paired_image_clusters_with_ratio_of_item_totals_for_unequal_cluster_sizes",
             "bootstrap_replicates":100000,"bootstrap_seed":20261007,"family_size":80,"familywise_alpha":.05,
             "interval_quantiles":"exact_rational_linear_percentiles","tail_probability":{"numerator":1,"denominator":3200},
             "supplementary_effects":20,"inherited_main_effects":30,"combined_effects":50,
             "new_candidate_or_confirmatory_gate":False}


def path_record(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": digest(path)}


def verified_record(repository, item):
    path = (Path(repository)/item["path"]).resolve()
    if digest(path) != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ValueError("Control prerequisite artifact changed")
    return path


def validate_contract_payload(contract, protocol, protocol_sha):
    expected = asdict(JointFitConfig(radius=1., composition_weight=.25)); expected.pop("seed")
    if (contract.get("study") != CONTRACT_SCHEMA or contract.get("family") != "retrieval_only"
            or contract.get("inherited_protocol", {}).get("sha256") != protocol_sha
            or protocol.get("study") != "sanw_practical_v10" or protocol.get("fit_config") != expected
            or contract.get("inherited_fit_config") != expected or contract.get("seeds") != [17,29,43]
            or contract.get("fixed_composition_multiplier") != 0.0
            or contract.get("candidate_selection_allowed") is not False
            or contract.get("explanatory_control_only") is not True
            or contract.get("fresh_confirmation_inclusion_allowed") is not False
            or contract.get("fresh_confirmation_contract") != protocol.get("fresh_confirmation_contract")
            or contract.get("fresh_confirmation_contract_unchanged") is not True
            or contract.get("required_gate_study") != GATE_STUDY
            or contract.get("required_gate_family") != "joint"
            or contract.get("require_completed_matched_no_retention_seeds") != [17,29,43]
            or contract.get("contrasts") != CONTRASTS or contract.get("endpoints") != ENDPOINTS
            or contract.get("inference") != INFERENCE or contract.get("encoders") != ["vit_b32","rn50"]
            or contract.get("selection") != "minimum_feasible_nonzero_training_objective_then_earliest_epoch"):
        raise ValueError("Retrieval-only control contract does not match its fixed explanatory scope")


def build_contract(repository, protocol_path):
    repository, protocol_path = Path(repository).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    for name, sha in protocol["source_sha256"].items():
        if digest(repository/name) != sha:
            raise ValueError("An inherited frozen source changed")
    verified_record(repository, protocol["fresh_confirmation_contract"])
    result = {"study": CONTRACT_SCHEMA, "family": "retrieval_only", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "inherited_protocol": path_record(protocol_path),
              "inherited_fit_config": protocol["fit_config"], "seeds": [17,29,43],
              "fixed_composition_multiplier": 0.0, "explanatory_control_only": True, "candidate_selection_allowed": False,
              "objective": "full_gallery_source_CE + ridge/2 * squared_Frobenius_norm; active_retention_penalty_during_updates",
              "objective_change": "Remove the joint source-supported-contradiction objective only; its inherited composition_weight config field is unused.",
              "unchanged": "All6000 owner data, training-only PCA, universal scorer, rank128, radius1, gamma0.5, full galleries, exact retention repair, seeds, paired query orders, batch64,32epochs,3008updates.",
              "selection": "minimum_feasible_nonzero_training_objective_then_earliest_epoch",
              "hypothesis": "Does joint composition supervision add beyond retrieval alignment when geometry and finite-training retention are fixed?",
              "required_gate_study": GATE_STUDY, "required_gate_family": "joint",
              "required_gate_status": "both_encoders_passing_fixed_seed17_29_43_aggregate_development",
              "require_completed_matched_no_retention_seeds": [17,29,43],
              "execution": "Only after passing joint three-seed development and all six matched no-retention fits; run sequentially if resources permit.",
              "no_outcome_based_tuning": True, "new_grid_or_schedule_changes": False,
              "endpoint_scope": "Exactly20 supplementary effects; with30 inherited main effects,50 fit within unchanged family80. No new gate.",
              "contrasts": CONTRASTS, "endpoints": ENDPOINTS, "inference": INFERENCE, "encoders":["vit_b32","rn50"],
              "fresh_confirmation_contract": protocol["fresh_confirmation_contract"],
              "fresh_confirmation_contract_unchanged": True, "fresh_confirmation_inclusion_allowed": False,
              "fresh_confirmation_scope": "The fresh1500 contract remains joint/no-retention only; this control adds no confirmatory family or endpoint.",
              "new_source_sha256": {name: digest(repository/name) for name in NEW_SOURCES},
              "inherited_source_sha256": protocol["source_sha256"],
              "prespecification_context": "Created during joint seed17 training, before v10 heldout evaluation; contains no heldout outcomes."}
    validate_contract_payload(result, protocol, digest(protocol_path))
    return result


def verify_contract(repository, contract_path):
    contract = json.loads(Path(contract_path).read_text())
    protocol_path = verified_record(repository, contract["inherited_protocol"])
    protocol = json.loads(protocol_path.read_text())
    validate_contract_payload(contract, protocol, digest(protocol_path))
    if (contract.get("inherited_source_sha256") != protocol["source_sha256"]
            or set(contract.get("new_source_sha256", {})) != set(NEW_SOURCES)):
        raise ValueError("Control source dependencies differ")
    for name, sha in contract["new_source_sha256"].items():
        if digest(Path(repository)/name) != sha:
            raise ValueError("A separately bound control source changed")
    for name, sha in protocol["source_sha256"].items():
        if digest(Path(repository)/name) != sha:
            raise ValueError("An inherited frozen source changed")
    verified_record(repository, contract["fresh_confirmation_contract"])
    return contract, protocol, protocol_path


def verify_development_gate(repository, gate_path, protocol, protocol_sha):
    gate = json.loads(Path(gate_path).read_text())
    if (gate.get("study") != GATE_STUDY or gate.get("family") != "joint" or gate.get("passed") is not True
            or gate.get("protocol_sha256") != protocol_sha or set(gate.get("encoders", {})) != {"vit_b32","rn50"}):
        raise ValueError("Joint fixed-three-seed aggregate development must pass first")
    # Reconstruct all six raw development results, deterministic bootstrap
    # arrays, item-wise fixed-seed averages, exact changes and aggregate gate.
    states = strict_development_gate(gate_path,protocol_sha)
    if set(states) != {(encoder,seed) for encoder in ("vit_b32","rn50") for seed in (17,29,43)}:
        raise ValueError("Strict prerequisite reconstruction did not return six qualified states")
    return {"record":path_record(gate_path),"qualified_joint_states":[states[key] for key in sorted(states)]}


def verify_matched_controls(repository, paths, protocol_sha, protocol):
    found, records = set(), []
    for path in paths:
        path = Path(path).resolve(); value = json.loads(path.read_text())
        state = strict_control(path.parent,protocol,protocol_sha)
        encoder, seed = value.get("encoder"), value.get("config", {}).get("seed")
        identity = encoder, seed
        if (value.get("study") != "sanw_practical_v10" or value.get("family") != "no_retention"
                or value.get("protocol_sha256") != protocol_sha or identity in found
                or encoder not in ("vit_b32","rn50") or seed not in (17,29,43)
                or value.get("config") != {**protocol["fit_config"],"seed":seed}
                or value.get("optimizer_steps") != 3008 or value.get("retention_enforced") is not False
                or value.get("final_training_retention_diagnostic", {}).get("ranking_checked_canonically") is not True):
            raise ValueError("All six inherited matched no-retention controls must finish first")
        history=value["history"];rows=value["checkpoint_history"]
        ledger=json.loads((path.parent/"ledger.json").read_text())
        if ([row["epoch"] for row in history] != list(range(1,33))
                or any(row["optimizer_steps"] != row["epoch"]*94 for row in history)):
            raise ValueError("Matched control did not complete the fixed32epoch/3008update budget")
        selected=ConstrainedBilinearScorer.load(path.parent/value["selected_checkpoint"]["path"])
        if selected.coefficient.shape != (protocol["fit_config"]["rank"],)*2:
            raise ValueError("Matched control rank differs")
        for row in rows:
            cp=path.parent/row["checkpoint"]["path"]
            if (digest(cp) != row["checkpoint"]["sha256"]
                    or row["checkpoint"]["ledger_sha256"] != ledger["ledger_sha256"]):
                raise ValueError("Matched control epoch artifact changed")
            saved=ConstrainedBilinearScorer.load(cp)
            if any(not np.array_equal(getattr(saved,key),getattr(selected,key)) for key in
                   ("image_mean","text_mean","image_basis","text_basis")):
                raise ValueError("Matched control geometry changed between epochs")
            norm2=float(np.sum(saved.coefficient*saved.coefficient,dtype=np.float64))
            if (not np.isfinite(norm2) or norm2>protocol["fit_config"]["radius"]**2*(1+128*np.finfo(np.float64).eps)
                    or row["nonzero"] != bool(np.any(saved.coefficient != 0))):
                raise ValueError("Matched control epoch coefficient or nonzero marker differs")
        cp = path.parent/value["selected_checkpoint"]["path"]
        if digest(cp) != value["selected_checkpoint"]["sha256"]:
            raise ValueError("Matched no-retention checkpoint changed")
        found.add(identity); records.append({"completion":path_record(path),"state":state})
    if found != {(encoder,seed) for encoder in ("vit_b32","rn50") for seed in (17,29,43)}:
        raise ValueError("Exactly six completed matched no-retention controls required")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build-contract")
    build.add_argument("--protocol",type=Path,required=True); build.add_argument("--output",type=Path,required=True)
    pre = sub.add_parser("verify-prerequisites")
    pre.add_argument("--contract",type=Path,required=True);pre.add_argument("--development-gate",type=Path,required=True)
    pre.add_argument("--matched-control-completions",type=Path,nargs=6,required=True)
    pre.add_argument("--output",type=Path,required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--contract",type=Path,required=True); fit.add_argument("--encoder",choices=("vit_b32","rn50"),required=True)
    fit.add_argument("--seed",type=int,choices=(17,29,43),required=True)
    fit.add_argument("--development-gate",type=Path,required=True)
    fit.add_argument("--matched-control-completions",type=Path,nargs=6,required=True)
    fit.add_argument("--output",type=Path,required=True); fit.add_argument("--cache",type=Path,required=True)
    args = parser.parse_args()
    if args.mode == "build-contract":
        if args.output.exists(): raise FileExistsError("Refusing to overwrite a control contract")
        atomic_json(args.output, build_contract(ROOT,args.protocol))
        print(json.dumps(path_record(args.output)),flush=True); return
    if args.mode == "verify-prerequisites":
        require_evaluation_threads()
        if args.output.exists(): raise FileExistsError("Refusing to overwrite prerequisite validation")
        contract,protocol,protocol_path=verify_contract(ROOT,args.contract)
        protocol_sha=digest(protocol_path)
        gate=verify_development_gate(ROOT,args.development_gate,protocol,protocol_sha)
        completed=verify_matched_controls(ROOT,args.matched_control_completions,protocol_sha,protocol)
        result={"study":"sanw_practical_v10_retrieval_only_prerequisites","passed":True,"contract":path_record(args.contract),
                "protocol_sha256":protocol_sha,"source_sha256":contract["new_source_sha256"],"validation_threads":1,
                "development_gate":gate,"completed_matched_no_retention":completed,
                "raw_development_evidence_reconstructed":True,"all_control_epochs_verified":True,"benchmark_outcomes_read":False}
        atomic_json(args.output,result);print(json.dumps(path_record(args.output)),flush=True);return
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError("Refusing to overwrite a control fit")
    contract, protocol, protocol_path = verify_contract(ROOT,args.contract)
    protocol_sha = digest(protocol_path)
    config, threads = verify_protocol(ROOT,protocol,args.encoder,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    prerequisite_path=args.output/"prerequisite_validation.json"
    env={**os.environ,**{key:"1" for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}}
    command=[sys.executable,str(Path(__file__).resolve()),"verify-prerequisites","--contract",str(args.contract.resolve()),
             "--development-gate",str(args.development_gate.resolve()),"--matched-control-completions",
             *[str(path.resolve()) for path in args.matched_control_completions],"--output",str(prerequisite_path.resolve())]
    subprocess.run(command,env=env,check=True)
    prerequisites=json.loads(prerequisite_path.read_text())
    gate=prerequisites["development_gate"];completed_controls=prerequisites["completed_matched_no_retention"]
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    images,texts,source_rows,owner,sources,supported,contra,provenance,scale = load_training(ROOT,args.encoder,protocol)
    model = ConstrainedBilinearScorer.from_training(images,texts,config.rank)
    settings = protocol["streaming"]
    cache = FrozenScoreCache.create(args.cache,images,texts[source_rows],owner,settings["cache_block_size"])
    constraints = StreamingFullGalleryConstraints(images,texts[source_rows],owner,model,cache,config.retention_fraction,settings["cache_block_size"])
    composition = JointCompositionExamples(images,texts,sources,supported,contra,model)
    retrieval = StreamingFullGallerySourceLoss(constraints,scale,settings["query_block_size"])
    identity = {"study":"sanw_practical_v10","family":"retrieval_only","mode":"full","encoder":args.encoder,
                "config":asdict(config),"protocol_sha256":protocol_sha,"protocol":path_record(protocol_path),
                "contract":path_record(args.contract),"fixed_composition_multiplier":0.0,
                "source_sha256":protocol["source_sha256"],"control_source_sha256":contract["new_source_sha256"],
                "explanatory_control_only":True,"candidate_selection_allowed":False,
                "fresh_confirmation_inclusion_allowed":False,"development_gate":gate,
                "prerequisite_validation":path_record(prerequisite_path),
                "completed_matched_no_retention":completed_controls,"training_provenance":provenance,
                "retrieval_logit_scale":scale,"fit_gallery_image_count":len(images),"fit_gallery_text_count":len(source_rows),
                "frozen_score_cache":{"path":str(args.cache.resolve()),"metadata_sha256":digest(args.cache/"metadata.json")},
                "streaming":settings,"environment":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,
                                                     "blas_threads":threads,"torch_normalization_threads":1},
                "heldout_used_in_fitting_or_checkpoint_selection":False}
    ledger_sha = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"checkpoints").mkdir()
    atomic_json(args.output/"ledger.json",{"identity":identity,"ledger_sha256":ledger_sha})
    rows = []
    def on_epoch(row,coefficient):
        model.coefficient = coefficient
        path = args.output/"checkpoints"/f"epoch_{row['epoch']:03d}.npz"; model.save(path)
        row = dict(row); row["checkpoint"]={"path":str(path.relative_to(args.output)),"sha256":digest(path),"ledger_sha256":ledger_sha}
        rows.append(row); atomic_json(args.output/"history.json",rows)
        print(json.dumps({key:row[key] for key in ("epoch","training_objective","retrieval_loss","fixed_composition_multiplier")}),flush=True)
    result = fit_retrieval_only(model,composition,retrieval,constraints,config,on_epoch)
    cert = result["final_certificate"]
    if not all(cert[key] is True for key in ("ranking_checked_canonically","ranking_preserved","feasible_with_tolerance")):
        raise ValueError("Retrieval-only selected state failed finite-training retention")
    model.save(args.output/"selected.npz")
    result.update({"study":"sanw_practical_v10","family":"retrieval_only","mode":"full","encoder":args.encoder,
                   "ledger_sha256":ledger_sha,"protocol_sha256":protocol_sha,"contract":path_record(args.contract),
                   "selected_checkpoint":{"path":"selected.npz","sha256":digest(args.output/"selected.npz")},
                   "checkpoint_history":rows,"elapsed_seconds":time.monotonic()-started})
    atomic_json(args.output/"completion.json",result)
    print(json.dumps({"complete":True,"family":"retrieval_only","selected_epoch":result["selected_epoch"]}),flush=True)


if __name__ == "__main__": main()
