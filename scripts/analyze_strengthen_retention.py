#!/usr/bin/env python3
"""Restore all original retention-v3 contrasts from newly saved raw predictions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from gcr.allocation_distillation_analysis import (ENDPOINTS, equal_seed_draw_values, effect,
                                                 require_aligned, practical_gate)
from analyze_study import independent_metrics, metric_at
from analyze_allocation_distillation import describe_strategy
from evaluate_study import sha256, write_json
from evaluate_strengthen_retention import (ENCODERS, FAMILIES, SEEDS, RATES, DATASETS,
    FACTORIAL_CELLS, EVIDENCE_TYPE, PROTOCOL_SHA256, verify_protocol, path_at_root,
    planned_states, factorial_selections, validate_manifest)

PRIMARY_FAMILY_SIZE = 80
FACTORIAL_FAMILY_SIZE = 36
REPLICATES = 100000
PRIMARY_BOOTSTRAP_SEED = 20261005
FACTORIAL_BOOTSTRAP_SEED = 20261006
SOURCE_PATHS = ("scripts/analyze_strengthen_retention.py", "scripts/evaluate_strengthen_retention.py",
    "scripts/analyze_allocation_distillation.py", "scripts/analyze_study.py",
    "scripts/evaluate_allocation_distillation.py", "scripts/evaluate_study.py",
    "src/gcr/strengthen_retention.py", "src/gcr/allocation_distillation_analysis.py", "src/gcr/evaluation.py")


def load_index(repository, path, protocol_hash):
    index = json.loads(path.read_text())
    if index.get("status") != "complete" or index.get("protocol_sha256") != protocol_hash:
        raise ValueError("Analysis requires a complete protocol-bound evaluation index")
    if index.get("evidence_type") != EVIDENCE_TYPE or index.get("encoder") not in ENCODERS or set(index["datasets"]) != set(DATASETS):
        raise ValueError("Unexpected evidence type, encoder or dataset set")
    for name, digest in index["source_hashes"].items():
        if sha256(repository / name) != digest:
            raise ValueError(f"Scoring implementation changed: {name}")
    manifest_path = path_at_root(repository, index["manifest"])
    if sha256(manifest_path) != index["manifest_sha256"]:
        raise ValueError("Scored manifest content changed")
    manifest = validate_manifest(repository, manifest_path)
    receipt_path = path_at_root(repository, index["prescore_receipt"])
    if sha256(receipt_path) != index["prescore_receipt_sha256"]:
        raise ValueError("Pre-score receipt content changed")
    receipt = json.loads(receipt_path.read_text())
    planned = planned_states(manifest)
    actual = [{key: value for key,value in row.items() if key != "datasets"} for row in index["runs"]]
    if actual != planned or receipt["runs"] != planned:
        raise ValueError("Scored states differ from the locked pre-score plan")
    for key in ("protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256", "source_hashes", "selections", "factorial_selections", "evidence_type"):
        if index[key] != receipt[key]:
            raise ValueError(f"Index differs from immutable receipt: {key}")
    predictions, metrics, raw_rows, discrepancies = {}, {}, [], []
    for run in index["runs"]:
        if set(run["datasets"]) != set(DATASETS):
            raise ValueError("Every planned state must have every benchmark")
        metrics[run["state_id"]] = {}
        for dataset, record in run["datasets"].items():
            archive = path_at_root(repository, record["predictions"])
            metadata_path = path_at_root(repository, record["metadata"])
            if sha256(archive) != record["predictions_sha256"] or sha256(metadata_path) != record["metadata_sha256"]:
                raise ValueError("Prediction artifact content hash mismatch")
            metadata = json.loads(metadata_path.read_text())
            state = {key:value for key,value in run.items() if key != "datasets"}
            expected_provenance = {"run": state, "dataset": dataset,
                                   "prescore_receipt_sha256": index["prescore_receipt_sha256"]}
            if metadata["provenance"] != expected_provenance or metadata["metrics"] != record["metrics"] or metadata["predictions_sha256"] != record["predictions_sha256"]:
                raise ValueError("Prediction metadata provenance mismatch")
            with np.load(archive, allow_pickle=False) as raw:
                prediction = {key: raw[key] for key in raw.files}
            predictions[(run["state_id"], dataset)] = prediction
            recomputed = independent_metrics(dataset, prediction)
            if dataset in ("e_vil_test1000", "coco_karpathy"):
                recomputed["mean_bidirectional_r1"] = .5 * (recomputed["i2t.r1"] + recomputed["t2i.r1"])
            for key,value in recomputed.items():
                discrepancy = abs(value - metric_at(record["metrics"], key))
                discrepancies.append(discrepancy)
                if discrepancy > 1e-12:
                    raise ValueError(f"Raw aggregate differs from scored metric: {run['state_id']} {dataset} {key}")
                metrics[run["state_id"]][f"{dataset}.{key}"] = value
                raw_rows.append({"encoder": index["encoder"], "state_id": run["state_id"],
                    "method": run["method"], "seed": run["seed"], "draw_id": run["draw_id"],
                    "learning_rate": run["learning_rate"], "epoch": run["epoch"], "source_mix": run["source_mix"],
                    "beta": run["beta"], "alpha": run["alpha"], "dataset": dataset, "metric": key, "value": value})
    return index, predictions, metrics, raw_rows, discrepancies


def selected_runs(index,family,tolerance=1.):
    states = {row["state_id"]:row for row in index["runs"]}
    if family == "frozen":
        return [states["frozen"]]
    selections = [row for row in index["selections"] if row["family"] == family and row["tolerance_pp"] == tolerance]
    if len(selections) != (9 if family == "matched_distilled" else 3):
        raise ValueError("Incomplete selected strategy")
    result = [states[row["state_id"]] for row in selections]
    for row,state in zip(selections,result):
        if row["seed"] != state["seed"] or row.get("draw_id") != state.get("draw_id"):
            raise ValueError("Selection identities differ from state metadata")
    return sorted(result,key=lambda row:(row["seed"],-1 if row["draw_id"] is None else row["draw_id"]))


def strategy(index,predictions,family,dataset,metric,tolerance=1.):
    return equal_seed_draw_values(selected_runs(index,family,tolerance),predictions,dataset,metric,
        frozen=family == "frozen",expected_draws=(0,1,2) if family == "matched_distilled" else None)


def factorial_runs(index,rate):
    states = {row["state_id"]:row for row in index["runs"]}
    cells = {cell:[] for cell in FACTORIAL_CELLS}
    for row in index["factorial_selections"]:
        if row["learning_rate"] != rate:
            continue
        cell,state = row["cell"],states[row["state_id"]]
        if (cell not in cells or state["method"] != cell or state["learning_rate"] != rate or
            state["epoch"] != 10 or state["alpha"] != 1. or state["draw_id"] is not None or state["seed"] != row["seed"]):
            raise ValueError("Factorial cell differs from fixed terminal schedule")
        cells[cell].append(state)
    if any(len(runs) != 3 or {r["seed"] for r in runs} != set(SEEDS) for runs in cells.values()):
        raise ValueError("Each factorial cell needs all three training seeds")
    return cells


def factorial_arrays(source,supported,image_source_only,reverse_source_only):
    require_aligned(source,supported,image_source_only,reverse_source_only)
    s,u,i,r = source[0],supported[0],image_source_only[0],reverse_source_only[0]
    values = {"image_main_effect":.5*(s+i-u-r),"reverse_main_effect":.5*(s+r-u-i),
              "interaction":s-i-r+u}
    return {name:(value,source[1],source[2]) for name,value in values.items()}


def validate_matched_controls(index):
    distilled = {row["seed"]:row for row in selected_runs(index,"distilled")}
    controls = selected_runs(index,"matched_distilled")
    for row in controls:
        if any(row[key] != distilled[row["seed"]][key] for key in ("learning_rate","epoch","beta")):
            raise ValueError("Matched controls must use the exact primary distilled schedule and beta")
    if {(row["seed"],row["draw_id"]) for row in controls} != {(seed,draw) for seed in SEEDS for draw in range(3)}:
        raise ValueError("All nine matched controls must receive equal weight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices",nargs=2,type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.protocol = path_at_root(ROOT,args.protocol)
    args.output = path_at_root(ROOT,args.output)
    args.indices = [path_at_root(ROOT,path) for path in args.indices]
    protocol = verify_protocol(args.protocol,args.protocol_sha256)
    args.output.mkdir(parents=True,exist_ok=True)
    receipt = {"schema_version":1,"protocol_sha256":PROTOCOL_SHA256,
        "indices":{str(path.relative_to(ROOT)):sha256(path) for path in args.indices},
        "source_hashes":{name:sha256(ROOT/name) for name in SOURCE_PATHS},"evidence_type":EVIDENCE_TYPE,
        "bootstrap_replicates":REPLICATES,"primary_bootstrap_seed":PRIMARY_BOOTSTRAP_SEED,
        "factorial_bootstrap_seed":FACTORIAL_BOOTSTRAP_SEED,"primary_family_size":PRIMARY_FAMILY_SIZE,
        "factorial_family_size":FACTORIAL_FAMILY_SIZE,"runtime":{"python":platform.python_version(),"numpy":np.__version__}}
    receipt_path = args.output/"preanalysis_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing analysis receipt is immutable")
    if not receipt_path.exists():
        write_json(receipt_path,receipt)
    data,raw_rows,discrepancies = {},[],[]
    for path in args.indices:
        index,predictions,metrics,raw,diffs = load_index(ROOT,path,PROTOCOL_SHA256)
        encoder = index["encoder"]
        if encoder in data:
            raise ValueError("Duplicate encoder index")
        data[encoder] = (index,predictions,metrics)
        raw_rows.extend(raw)
        discrepancies.extend(diffs)
        validate_matched_controls(index)
    if set(data) != set(ENCODERS) or len({entry[0]["selection_lock_sha256"] for entry in data.values()}) != 1:
        raise ValueError("Both encoders must share one locked selection plan")
    with (args.output/"all_state_metrics.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    primary,factorial,samples,descriptions,gates = [],[],{},[],[]
    for encoder in ENCODERS:
        index,predictions,metrics = data[encoder]
        for tolerance in (1.,0.):
            families = FAMILIES + (("matched_distilled",) if tolerance == 1. else ())
            for family in families:
                descriptions.append({"encoder":encoder,"family":family,"tolerance_pp":tolerance,
                    "inference":"descriptive",**describe_strategy(selected_runs(index,family,tolerance),metrics)})
        descriptions.append({"encoder":encoder,"family":"frozen","tolerance_pp":None,
            "inference":"descriptive",**describe_strategy(selected_runs(index,"frozen"),metrics)})
        comparisons = [(family,"frozen") for family in FAMILIES]
        comparisons += [("distilled",family) for family in ("allocation","wise_ft","matched_distilled")]
        for left,right in comparisons:
            for dataset,metric in ENDPOINTS:
                result,draws = effect(strategy(index,predictions,left,dataset,metric),
                    strategy(index,predictions,right,dataset,metric),family_size=PRIMARY_FAMILY_SIZE,
                    replicates=REPLICATES,seed=PRIMARY_BOOTSTRAP_SEED)
                key = f"{encoder}__{left}_minus_{right}__{dataset}__{metric}"
                result.update({"effect_id":key,"encoder":encoder,"left":left,"right":right,
                               "dataset":dataset,"metric":metric,"tolerance_pp":1.})
                primary.append(result)
                samples[key] = draws
                print(json.dumps({"completed_effect":key,"difference":result["difference"]}),flush=True)
        for rate in RATES:
            cells = factorial_runs(index,rate)
            for dataset,metric in ENDPOINTS[:2]:
                arrays = {cell:equal_seed_draw_values(runs,predictions,dataset,metric) for cell,runs in cells.items()}
                effects = factorial_arrays(*(arrays[cell] for cell in FACTORIAL_CELLS))
                for name,values in effects.items():
                    zero = (np.zeros_like(values[0]),values[1],values[2])
                    result,draws = effect(values,zero,family_size=FACTORIAL_FAMILY_SIZE,
                        replicates=REPLICATES,seed=FACTORIAL_BOOTSTRAP_SEED)
                    key = f"{encoder}__{name}__lr_{rate:.8g}__{dataset}__{metric}"
                    result.update({"effect_id":key,"encoder":encoder,"effect":name,"learning_rate":rate,
                        "epoch":10,"dataset":dataset,"metric":metric,
                        "conditioning":"fixed terminal schedules; fixed training seeds averaged before image-cluster resampling",
                        "cells":{cell:[r["state_id"] for r in runs] for cell,runs in cells.items()}})
                    factorial.append(result)
                    samples[key] = draws
                    print(json.dumps({"completed_effect":key,"difference":result["difference"]}),flush=True)
        for family in FAMILIES:
            rows = [row for row in primary if row["encoder"] == encoder and row["left"] == family and row["right"] == "frozen"]
            gate = practical_gate(selected_runs(index,family),rows)
            gate["evidence_type"] = EVIDENCE_TYPE
            gates.append({"encoder":encoder,"family":family,**gate})
    if len(primary) != PRIMARY_FAMILY_SIZE or len(factorial) != FACTORIAL_FAMILY_SIZE:
        raise ValueError("Inference family sizes differ from the original frozen protocol")
    common = {"protocol_sha256":PROTOCOL_SHA256,"evidence_type":EVIDENCE_TYPE,
        "scale":"fractions; multiply differences and intervals by100 for percentage points",
        "test_based_selection":False,"prior_test_exposure":True,"historical_aggregates_used":False,
        "fresh_confirmatory_claim":False}
    write_json(args.output/"primary_contrasts.json",{**common,"family_size":PRIMARY_FAMILY_SIZE,"contrasts":primary})
    write_json(args.output/"factorial_contrasts.json",{**common,"family_size":FACTORIAL_FAMILY_SIZE,
        "interpretation":protocol["evaluation"]["mechanism_factorial"]["interpretation"],"contrasts":factorial})
    write_json(args.output/"selected_strategy_metrics.json",{**common,"strategies":descriptions})
    cross_encoder = [{"family":family,"passed_both_encoders":all(row["passed"] for row in gates if row["family"] == family)} for family in FAMILIES]
    write_json(args.output/"practical_success_gates.json",{**common,"gates":gates,"cross_encoder":cross_encoder})
    np.savez_compressed(args.output/"bootstrap_samples.npz",**samples)
    audit = {"status":"passed",**common,"preanalysis_receipt_sha256":sha256(receipt_path),
        "encoders":list(ENCODERS),"unique_states":sum(len(value[0]["runs"]) for value in data.values()),
        "prediction_archives":sum(len(value[1]) for value in data.values()),"aggregate_checks":len(discrepancies),
        "maximum_recomputation_difference":max(discrepancies),"primary_effects":len(primary),
        "factorial_effects":len(factorial),"bootstrap_replicates":REPLICATES,
        "all_random_draws_equally_weighted":True,"certificate_diagnostics":"separate post-fit diagnostic artifact; this analysis does not compute certificates",
        "scope":"raw-score correctness and aggregate audit; does not independently recompute feature dot products or retrieval ranks"}
    write_json(args.output/"analysis_audit.json",audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == "__main__":
    main()
