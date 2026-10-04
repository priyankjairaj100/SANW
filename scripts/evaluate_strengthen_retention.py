#!/usr/bin/env python3
"""Restore raw evaluation of the unchanged retention-v3 protocol.

This is a new execution after the previous local raw evidence was lost. Both
encoder selections must be locked before any held-out scoring. No model choice
uses this evaluator's outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from gcr.adapters import ResidualAdapter
from gcr.evaluation import PAIR_TIE_POLICY, RETRIEVAL_TIE_POLICY
from gcr.strengthen_retention import FAMILIES, POLICIES, PROTOCOL_SHA256
from gcr.training import canonical_json
from evaluate_study import sha256, write_json
from evaluate_allocation_distillation import (load_dataset_config, score_state, path_at_root,
                                             SOURCE_PATHS as SHARED_SOURCES)

ENCODERS = ("vit_b32", "rn50")
SEEDS = (17,29,43)
RATES = (1e-4,3e-4,1e-3)
DATASETS = ("e_vil_test1000", "visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
FACTORIAL_CELLS = ("source", "supported", "image_source_only", "reverse_source_only")
EVIDENCE_TYPE = "retention_v3_reconstructed_new_execution_after_local_evidence_loss"
SOURCE_PATHS = tuple(dict.fromkeys(("scripts/evaluate_strengthen_retention.py",
                                   "src/gcr/strengthen_retention.py") + SHARED_SOURCES))


def verify_protocol(path, digest):
    if digest != PROTOCOL_SHA256 or sha256(path) != PROTOCOL_SHA256:
        raise ValueError("The unchanged retention-v3 protocol hash is required")
    protocol = json.loads(path.read_text())
    if (protocol["evaluation"]["primary_family_size"] != 80 or
        protocol["evaluation"]["bootstrap_seed"] != 20261005 or
        protocol["evaluation"]["mechanism_factorial"]["family_size"] != 36 or
        protocol["evaluation"]["mechanism_factorial"]["bootstrap_seed"] != 20261006):
        raise ValueError("Unexpected frozen inference contract")
    return protocol


def factorial_selections(manifest):
    by_key = {}
    for row in manifest["states"]:
        if row["method"] not in FACTORIAL_CELLS:
            continue
        key = (row["method"],row["learning_rate"],row["seed"],row["epoch"])
        if key in by_key:
            raise ValueError("Duplicate state schedule")
        by_key[key] = row
    result = []
    for cell in FACTORIAL_CELLS:
        for rate in RATES:
            for seed in SEEDS:
                row = by_key[(cell,rate,seed,10)]
                if row["alpha"] != 1. or row["draw_id"] is not None:
                    raise ValueError("Factorial requires unscaled, nonrandom terminal states")
                result.append({"state_id":row["state_id"],"cell":cell,"learning_rate":rate,"seed":seed,"epoch":10})
    return result


def planned_states(manifest):
    states = {row["state_id"]:row for row in manifest["states"]}
    if len(states) != len(manifest["states"]):
        raise ValueError("Duplicate state IDs")
    expected_grid = {(method,rate,seed,epoch) for method in POLICIES for rate in RATES for seed in SEEDS for epoch in range(11)}
    grid = [(row["method"],row["learning_rate"],row["seed"],row["epoch"]) for row in states.values() if row["method"] in POLICIES]
    if len(grid) != len(expected_grid) or set(grid) != expected_grid:
        raise ValueError("Retention manifest must contain the complete 891-state original grid")
    selections = manifest["selections"]
    for tolerance in (1.,0.):
        actual = [(row["family"],row["seed"]) for row in selections if row["tolerance_pp"] == tolerance and row["family"] in FAMILIES]
        expected = {(family,seed) for family in FAMILIES for seed in SEEDS}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Incomplete seven-family development selection")
    controls = [row for row in selections if row["family"] == "matched_distilled"]
    if len(controls) != 9 or {(r["seed"],r["draw_id"]) for r in controls} != {(seed,draw) for seed in SEEDS for draw in range(3)}:
        raise ValueError("Matched distillation requires all nine seed-by-draw controls")
    distilled = {row["seed"]:states[row["state_id"]] for row in selections if row["tolerance_pp"] == 1. and row["family"] == "distilled"}
    for row in selections:
        if row["state_id"] not in states:
            raise ValueError("Selected state is absent")
        state = states[row["state_id"]]
        if state["seed"] != row["seed"] or state.get("draw_id") != row.get("draw_id"):
            raise ValueError("Selection identities disagree with their states")
    for row in controls:
        state,reference = states[row["state_id"]],distilled[row["seed"]]
        if (any(state[key] != reference[key] for key in ("seed","learning_rate","epoch","beta")) or
            state["draw_id"] != row["draw_id"] or state["family"] != "matched_distilled" or row["tolerance_pp"] != 1.):
            raise ValueError("Matched distillation must use the primary selected schedule and beta")
    selected_ids = {row["state_id"] for row in selections}
    selected_ids.update(row["state_id"] for row in factorial_selections(manifest))
    frozen = {"state_id":"frozen","method":"frozen","family":"frozen","seed":None,
              "epoch":0,"learning_rate":None,"source_mix":None,"beta":None,"alpha":0.,
              "draw_id":None,"update_norm":0.,"checkpoint":None,"checkpoint_sha256":None,
              "encoder":manifest["encoder"],"architecture":"linear"}
    return [frozen] + [states[sid] for sid in sorted(selected_ids)]


def validate_manifest(repository, path):
    manifest = json.loads(path.read_text())
    if (manifest.get("protocol_sha256") != PROTOCOL_SHA256 or manifest.get("encoder") not in ENCODERS or
        manifest.get("test_outcomes_used_for_selection") is not False or manifest.get("matched_controls_complete") is not True):
        raise ValueError("Scoring requires both development selections and complete matched controls")
    states = {row["state_id"]:row for row in manifest["states"]}
    for label,tolerance in (("primary",1.),("sensitivity",0.)):
        selection_path = path.parent / f"selection_{label}.json"
        if sha256(selection_path) != manifest["selection_sha256"][label]:
            raise ValueError("Development selection content changed")
        selection = json.loads(selection_path.read_text())
        if (selection.get("test_outcomes_used") is not False or selection["ledger_sha256"] != manifest["ledger_sha256"] or
            selection["protocol_sha256"] != PROTOCOL_SHA256 or set(selection["families"]) != set(FAMILIES)):
            raise ValueError("Invalid development selection provenance")
        expected = set()
        for family,record in selection["families"].items():
            for run in record["runs"]:
                state = states.get(run["state_id"])
                if state is None or any(run.get(key) != state.get(key) for key in
                    ("method","seed","epoch","learning_rate","checkpoint_sha256","source_mix","beta","alpha","update_norm")):
                    raise ValueError("Manifest state differs from the development-selection file")
                expected.add((run["state_id"],family,run["seed"]))
        actual = [(row["state_id"],row["family"],row["seed"]) for row in manifest["selections"]
                  if row["tolerance_pp"] == tolerance and row["family"] in FAMILIES]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Manifest selections differ from the locked selection files")
    planned_states(manifest)
    return manifest


def make_selection_lock(repository,paths):
    records = {}
    for value in paths:
        path = path_at_root(repository,value)
        manifest = validate_manifest(repository,path)
        encoder = manifest["encoder"]
        if encoder in records:
            raise ValueError("Duplicate encoder in selection lock")
        records[encoder] = {"manifest":str(path.relative_to(repository)),"manifest_sha256":sha256(path),
                           "selection_sha256":manifest["selection_sha256"],"ledger_sha256":manifest["ledger_sha256"]}
    if set(records) != set(ENCODERS):
        raise ValueError("Both encoders must be locked before held-out scoring")
    return {"schema_version":1,"protocol_sha256":PROTOCOL_SHA256,"encoders":records,
            "purpose":"lock all development selections before held-out scoring","evidence_type":EVIDENCE_TYPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--selection-lock",type=Path,required=True)
    parser.add_argument("--selection-lock-sha256")
    parser.add_argument("--lock-manifests",nargs=2,type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--dataset-config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--block-size",type=int,default=128)
    parser.add_argument("--torch-threads",type=int,default=2)
    args = parser.parse_args()
    args.protocol = path_at_root(ROOT,args.protocol)
    args.selection_lock = path_at_root(ROOT,args.selection_lock)
    verify_protocol(args.protocol,args.protocol_sha256)
    if args.lock_manifests:
        lock = make_selection_lock(ROOT,args.lock_manifests)
        if args.selection_lock.exists() and json.loads(args.selection_lock.read_text()) != lock:
            raise ValueError("Existing cross-encoder selection lock is immutable")
        write_json(args.selection_lock,lock)
        print(json.dumps({"selection_lock":str(args.selection_lock),"sha256":sha256(args.selection_lock)}))
        return
    if any(value is None for value in (args.manifest,args.dataset_config,args.output,args.selection_lock_sha256)):
        parser.error("Scoring requires --manifest, --dataset-config, --output and --selection-lock-sha256")
    for name in ("manifest","dataset_config","output"):
        setattr(args,name,path_at_root(ROOT,getattr(args,name)))
    if sha256(args.selection_lock) != args.selection_lock_sha256:
        raise ValueError("Selection lock content hash mismatch")
    lock = json.loads(args.selection_lock.read_text())
    if lock != make_selection_lock(ROOT,[row["manifest"] for row in lock["encoders"].values()]):
        raise ValueError("A selected state manifest changed after the cross-encoder lock")
    manifest = validate_manifest(ROOT,args.manifest)
    if sha256(args.manifest) != lock["encoders"][manifest["encoder"]]["manifest_sha256"]:
        raise ValueError("Requested manifest is not the locked encoder manifest")
    ledger_path = args.manifest.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if (hashlib.sha256(canonical_json(ledger["identity"])).hexdigest() != ledger["ledger_sha256"] or
        ledger["ledger_sha256"] != manifest["ledger_sha256"]):
        raise ValueError("Execution ledger content hash mismatch")
    for name,digest in ledger["identity"]["source_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Frozen training implementation changed: {name}")
    runs = planned_states(manifest)
    for run in runs:
        if Path(run["state_id"]).name != run["state_id"] or run["state_id"] in (".",".."):
            raise ValueError("Unsafe state ID")
        if run["checkpoint"] is not None and sha256(path_at_root(ROOT,run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint content hash mismatch")
    datasets = load_dataset_config(ROOT,args.dataset_config,manifest["encoder"],ledger)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    sources = {name:sha256(ROOT/name) for name in SOURCE_PATHS}
    environment = {"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
        "torch_threads":torch.get_num_threads(),"blas_thread_environment":{name:os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}}
    receipt = {"schema_version":1,"protocol_sha256":PROTOCOL_SHA256,"encoder":manifest["encoder"],
        "manifest_sha256":sha256(args.manifest),"ledger_file_sha256":sha256(ledger_path),
        "selection_lock_sha256":args.selection_lock_sha256,"dataset_config_sha256":sha256(args.dataset_config),
        "source_hashes":sources,"input_hashes":{name:data["hashes"] for name,data in datasets.items()},
        "runs":runs,"selections":manifest["selections"],"factorial_selections":factorial_selections(manifest),
        "environment":environment,"block_size":args.block_size,"evidence_type":EVIDENCE_TYPE,
        "prior_test_exposure":True,"historical_aggregates_used":False,
        "certificate_diagnostics":"deferred; no certificate outputs are claimed"}
    args.output.mkdir(parents=True,exist_ok=True)
    receipt_path = args.output / "prescore_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing pre-score receipt is immutable")
    if not receipt_path.exists():
        write_json(receipt_path,receipt)
    index = {**{key:receipt[key] for key in ("schema_version","protocol_sha256","encoder","manifest_sha256",
        "selection_lock_sha256","source_hashes","selections","factorial_selections","evidence_type")},
        "status":"running","manifest":str(args.manifest.relative_to(ROOT)),"datasets":list(DATASETS),
        "prescore_receipt":str(receipt_path.relative_to(ROOT)),"prescore_receipt_sha256":sha256(receipt_path),
        "normalization":"same final float32 L2 normalization for frozen and adapted features",
        "score_precision":"unscaled float64 dot products","retrieval_tie_policy":RETRIEVAL_TIE_POLICY,
        "pair_tie_policy":PAIR_TIE_POLICY,"prior_test_exposure":True,"runs":[]}
    for run in runs:
        adapter = None
        if run["checkpoint"]:
            payload = torch.load(ROOT/run["checkpoint"],map_location="cpu",weights_only=True)
            if any(payload[key] != run[key] for key in ("method","seed","epoch","learning_rate")):
                raise ValueError("Checkpoint state metadata mismatch")
            if payload.get("protocol_sha256") != PROTOCOL_SHA256 or payload.get("ledger_sha256") != manifest["ledger_sha256"]:
                raise ValueError("Checkpoint protocol or ledger mismatch")
            adapter = ResidualAdapter(512 if manifest["encoder"] == "vit_b32" else 1024)
            adapter.load_state_dict(payload["state_dict"],strict=True)
            adapter.eval()
            norm = sum(float(value.double().square().sum()) for value in adapter.state_dict().values()) ** .5
            if not np.isclose(norm,run["update_norm"],rtol=1e-12,atol=1e-12):
                raise ValueError("Recorded update norm differs from checkpoint weights")
        output_run = dict(run,datasets={})
        for name,dataset in datasets.items():
            started = time.monotonic()
            directory = args.output / "predictions" / run["state_id"]
            directory.mkdir(parents=True,exist_ok=True)
            archive,metadata_path = directory/f"{name}.npz",directory/f"{name}.json"
            provenance = {"run":run,"dataset":name,"prescore_receipt_sha256":sha256(receipt_path)}
            reused = False
            if archive.exists() or metadata_path.exists():
                if not archive.exists() or not metadata_path.exists():
                    raise ValueError("Incomplete saved prediction pair")
                metadata = json.loads(metadata_path.read_text())
                if metadata["provenance"] != provenance or metadata["predictions_sha256"] != sha256(archive):
                    raise ValueError("Saved prediction provenance or content changed")
                metrics,reused = metadata["metrics"],True
            else:
                metrics,predictions = score_state(adapter,dataset,args.block_size)
                temporary = archive.with_suffix(".tmp.npz")
                np.savez_compressed(temporary,**predictions)
                temporary.replace(archive)
                write_json(metadata_path,{"metrics":metrics,"provenance":provenance,
                    "predictions_sha256":sha256(archive),"seconds":time.monotonic()-started})
            output_run["datasets"][name] = {"metrics":metrics,"predictions":str(archive.relative_to(ROOT)),
                "predictions_sha256":sha256(archive),"metadata":str(metadata_path.relative_to(ROOT)),
                "metadata_sha256":sha256(metadata_path)}
            print(json.dumps({"state":run["state_id"],"dataset":name,"reused":reused,
                              "seconds":round(time.monotonic()-started,3)}),flush=True)
        index["runs"].append(output_run)
        write_json(args.output/"index.json",index)
    index["status"] = "complete"
    index["run_count"] = len(index["runs"])
    write_json(args.output/"index.json",index)


if __name__ == "__main__":
    main()
