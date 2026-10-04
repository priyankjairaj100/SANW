#!/usr/bin/env python3
"""Restore full-gallery rank certificates for original retention-v3 selections.

These are post-fit diagnostics. They do not change stable-index benchmark ranks,
checkpoint selection, or practical-success gates. A certificate refers only to
its exact scored gallery. No training-minibatch guarantee is extrapolated.
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
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from gcr.adapters import ResidualAdapter
from gcr.rank_retention import rank_retention_diagnostics
from gcr.training import canonical_json
from evaluate_study import adapted_features, sha256, write_json
from evaluate_allocation_distillation import load_dataset_config, path_at_root, SOURCE_PATHS as SHARED_SOURCES
from evaluate_strengthen_retention import (FAMILIES, PROTOCOL_SHA256, EVIDENCE_TYPE,
    make_selection_lock, validate_manifest, verify_protocol)

TEMPERATURE = 2.
K = 1
ATOL = 1e-12
RTOL = 1e-10
RETRIEVAL_DATASETS = ("e_vil_test1000","coco_karpathy")
SOURCE_PATHS = tuple(dict.fromkeys(("scripts/diagnose_strengthen_retention.py",
    "scripts/evaluate_strengthen_retention.py","src/gcr/rank_retention.py") + SHARED_SOURCES))
MASK_KEYS = ("certified","logit_drift_certified","pairwise_certified","combined_certified")


def stable_ranks(scores,relevance):
    """Original stable-index ranks from unscaled scores, separate from diagnostics."""
    scores = np.asarray(scores,np.float64)
    if scores.ndim != 2 or len(relevance) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid stable-rank inputs")
    ranks = np.empty(len(scores),dtype=np.int64)
    candidate_indices = np.arange(scores.shape[1])
    for index,row in enumerate(scores):
        relevant = np.unique(np.asarray(relevance[index],np.int64))
        if not len(relevant) or relevant[0] < 0 or relevant[-1] >= scores.shape[1]:
            raise ValueError("Invalid relevance indices")
        best = relevant[np.argmax(row[relevant])]
        value = row[best]
        ranks[index] = 1 + np.count_nonzero(row > value) + np.count_nonzero((row == value) & (candidate_indices < best))
    return ranks


def diagnose_direction(teacher_queries,teacher_candidates,student_queries,student_candidates,
                       relevance,logit_scale,block_size=128):
    """Stream complete gallery rows through the intact positive-set R1 diagnostic."""
    features = [np.asarray(value,np.float64) for value in
        (teacher_queries,teacher_candidates,student_queries,student_candidates)]
    tq,tc,sq,sc = features
    if (tq.shape != sq.shape or tc.shape != sc.shape or tq.ndim != 2 or tc.ndim != 2 or
        tq.shape[1] != tc.shape[1] or not len(tq) or not len(tc) or len(relevance) != len(tq)):
        raise ValueError("Teacher and student must use identical query and gallery shapes")
    if not all(np.isfinite(value).all() for value in features):
        raise ValueError("Nonfinite diagnostic features")
    if not np.isfinite(logit_scale) or logit_scale <= 0 or block_size < 1:
        raise ValueError("Native logit scale and block size must be positive")
    pieces = {}
    for start in range(0,len(tq),block_size):
        stop = min(start+block_size,len(tq))
        teacher_scores = tq[start:stop] @ tc.T
        student_scores = sq[start:stop] @ sc.T
        relevant = relevance[start:stop]
        diagnostic = rank_retention_diagnostics(teacher_scores*logit_scale,student_scores*logit_scale,
            relevant,temperature=TEMPERATURE,k=K,atol=ATOL,rtol=RTOL)
        diagnostic["benchmark_teacher_ranks"] = stable_ranks(teacher_scores,relevant)
        diagnostic["benchmark_student_ranks"] = stable_ranks(student_scores,relevant)
        for key,values in diagnostic.items():
            pieces.setdefault(key,[]).append(values)
    result = {key:np.concatenate(values) for key,values in pieces.items()}
    for key in MASK_KEYS:
        if np.any(result[key] & ~(result["teacher_correct"] & result["student_correct"])):
            raise ValueError(f"A full-gallery {key} certificate contradicts actual diagnostic ranks")
    return result


def relevance_indices(dataset):
    features,manifest = dataset["features"],dataset["manifest"]
    image_ids,text_ids = features["image_ids"],features["text_ids"]
    images = {str(value):i for i,value in enumerate(image_ids)}
    texts = {str(value):i for i,value in enumerate(text_ids)}
    image_relevance = [[] for _ in image_ids]
    text_relevance = [[] for _ in text_ids]
    for row in manifest["pairs"]:
        if row["relation"] != "source":
            raise ValueError("Held-out retrieval diagnostics require original source ownership")
        image,text = images[str(row["image_id"])],texts[str(row["text_id"])]
        image_relevance[image].append(text)
        text_relevance[text].append(image)
    if any(not row for row in image_relevance) or any(len(row) != 1 for row in text_relevance):
        raise ValueError("Every query must retain its complete original relevance set")
    return image_relevance,text_relevance


def coverage_summary(values):
    teacher,student = values["teacher_correct"],values["student_correct"]
    retained = teacher & student
    for key in MASK_KEYS:
        if np.any(values[key] & ~retained):
            raise ValueError(f"Raw {key} mask contains a false full-gallery certificate")
    counts = {"queries":len(teacher),"teacher_correct":int(teacher.sum()),
              "student_correct":int(student.sum()),"actually_retained":int(retained.sum()),
              "benchmark_teacher_correct":int(np.count_nonzero(values["benchmark_teacher_ranks"]<=1)),
              "benchmark_student_correct":int(np.count_nonzero(values["benchmark_student_ranks"]<=1))}
    summary = {"counts":counts,"correctness_tie_policy":"pessimistic: any irrelevant tie at the best relevant score is a failure",
        "benchmark_tie_policy":"descending unscaled float64 score, then ascending candidate manifest index",
        "certificate_coverage":{},"numerics":{}}
    for key in MASK_KEYS:
        mask = values[key]
        certified = int(mask.sum())
        summary["certificate_coverage"][key] = {"certified_queries":certified,
            "false_certificates":int(np.count_nonzero(mask & ~retained)),
            "fraction_of_all_queries":certified/counts["queries"],
            "fraction_of_teacher_correct":certified/counts["teacher_correct"] if counts["teacher_correct"] else None,
            "fraction_of_actually_retained":certified/counts["actually_retained"] if counts["actually_retained"] else None,
            "teacher_correct_denominator":counts["teacher_correct"],"actually_retained_denominator":counts["actually_retained"]}
    summary["tie_disagreements"] = {
        "teacher_stable_vs_pessimistic":int(np.count_nonzero((values["benchmark_teacher_ranks"]<=1)!=teacher)),
        "student_stable_vs_pessimistic":int(np.count_nonzero((values["benchmark_student_ranks"]<=1)!=student))}
    for key in ("threshold","divergence","logit_margin","logit_drift_oscillation","logit_drift_linf_centered","pairwise_threshold"):
        finite = np.asarray(values[key])[np.isfinite(values[key])]
        summary["numerics"][key] = {"finite_count":len(finite),"infinite_count":int(np.isinf(values[key]).sum()),
            "mean":float(finite.mean()) if len(finite) else None,"maximum":float(finite.max()) if len(finite) else None,
            "median":float(np.median(finite)) if len(finite) else None}
    return summary


def selected_diagnostic_states(manifest):
    states = {row["state_id"]:row for row in manifest["states"]}
    selections = [row for row in manifest["selections"] if row["family"] in FAMILIES and row["tolerance_pp"] in (0.,1.)]
    for tolerance in (0.,1.):
        chosen = [row for row in selections if row["tolerance_pp"]==tolerance]
        if len(chosen)!=21 or {(row["family"],row["seed"]) for row in chosen} != {(family,seed) for family in FAMILIES for seed in (17,29,43)}:
            raise ValueError("Diagnostics require every primary and sensitivity selected checkpoint")
    result = []
    for sid in sorted({row["state_id"] for row in selections}):
        run = dict(states[sid])
        run["selection_roles"] = [{"family":row["family"],"tolerance_pp":row["tolerance_pp"],"seed":row["seed"]}
                                  for row in selections if row["state_id"]==sid]
        result.append(run)
    return result


def assert_benchmark_alignment(raw,prediction,teacher_prediction):
    for direction in ("i2t","t2i"):
        if not np.array_equal(raw[f"{direction}_benchmark_teacher_ranks"],teacher_prediction[f"{direction}_ranks"]):
            raise ValueError("Recomputed teacher ranks differ from unchanged benchmark evaluation")
        if not np.array_equal(raw[f"{direction}_benchmark_student_ranks"],prediction[f"{direction}_ranks"]):
            raise ValueError("Recomputed student ranks differ from unchanged benchmark evaluation")
    for key in ("image_ids","text_ids","text_source_image_ids"):
        if not np.array_equal(raw[key],prediction[key]) or not np.array_equal(raw[key],teacher_prediction[key]):
            raise ValueError("Diagnostic and benchmark gallery or query identities differ")


def read_prediction(record):
    path = path_at_root(ROOT,record["predictions"])
    if sha256(path)!=record["predictions_sha256"]:
        raise ValueError("Benchmark prediction archive content changed")
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--selection-lock",type=Path,required=True)
    parser.add_argument("--selection-lock-sha256",required=True)
    parser.add_argument("--evaluation-index",type=Path,required=True)
    parser.add_argument("--dataset-config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--block-size",type=int,default=128)
    parser.add_argument("--torch-threads",type=int,default=2)
    args = parser.parse_args()
    for name in ("protocol","manifest","selection_lock","evaluation_index","dataset_config","output"):
        setattr(args,name,path_at_root(ROOT,getattr(args,name)))
    verify_protocol(args.protocol,args.protocol_sha256)
    if sha256(args.selection_lock)!=args.selection_lock_sha256:
        raise ValueError("Cross-encoder selection-lock digest mismatch")
    lock = json.loads(args.selection_lock.read_text())
    if lock!=make_selection_lock(ROOT,[record["manifest"] for record in lock["encoders"].values()]):
        raise ValueError("Locked encoder selections changed")
    manifest = validate_manifest(ROOT,args.manifest)
    if sha256(args.manifest)!=lock["encoders"][manifest["encoder"]]["manifest_sha256"]:
        raise ValueError("Requested manifest is not in the cross-encoder lock")
    index = json.loads(args.evaluation_index.read_text())
    if (index.get("status")!="complete" or index.get("protocol_sha256")!=PROTOCOL_SHA256 or
        index.get("manifest_sha256")!=sha256(args.manifest) or index.get("selection_lock_sha256")!=args.selection_lock_sha256 or
        index.get("encoder")!=manifest["encoder"]):
        raise ValueError("Diagnostics require the complete evaluation for these locked selections")
    for name,digest in index["source_hashes"].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError("The benchmark scorer changed after evaluation")
    prescore_path = path_at_root(ROOT,index["prescore_receipt"])
    if sha256(prescore_path)!=index["prescore_receipt_sha256"]:
        raise ValueError("Benchmark pre-score receipt changed")
    prescore = json.loads(prescore_path.read_text())
    ledger_path = args.manifest.parent/"ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if (hashlib.sha256(canonical_json(ledger["identity"])).hexdigest()!=ledger["ledger_sha256"] or
        ledger["ledger_sha256"]!=manifest["ledger_sha256"]):
        raise ValueError("Execution ledger content hash mismatch")
    logit_scale = float(ledger["identity"]["inputs"]["logit_scale"])
    if logit_scale!=manifest["logit_scale"]:
        raise ValueError("Teacher native logit scale differs from manifest")
    datasets = load_dataset_config(ROOT,args.dataset_config,manifest["encoder"],ledger)
    for name in RETRIEVAL_DATASETS:
        if datasets[name]["hashes"]!=prescore["input_hashes"][name]:
            raise ValueError("Diagnostic inputs differ from the evaluated candidate gallery")
    runs = selected_diagnostic_states(manifest)
    evaluated = {row["state_id"]:row for row in index["runs"]}
    if len(evaluated) != len(index["runs"]):
        raise ValueError("Duplicate state IDs in the benchmark evaluation index")
    for run in runs:
        if run["state_id"] not in evaluated or any(run[key]!=evaluated[run["state_id"]][key]
                for key in ("checkpoint_sha256","epoch","seed","method","alpha")):
            raise ValueError("Selected diagnostic checkpoint differs from benchmark checkpoint")
        if sha256(path_at_root(ROOT,run["checkpoint"]))!=run["checkpoint_sha256"]:
            raise ValueError("Selected diagnostic checkpoint content changed")
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    source_hashes = {name:sha256(ROOT/name) for name in SOURCE_PATHS}
    environment = {"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
        "torch_threads":torch.get_num_threads(),"blas_thread_environment":{key:os.environ.get(key)
        for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}}
    receipt = {"schema_version":1,"protocol_sha256":PROTOCOL_SHA256,"encoder":manifest["encoder"],
        "manifest_sha256":sha256(args.manifest),"selection_lock_sha256":args.selection_lock_sha256,
        "evaluation_index":str(args.evaluation_index.relative_to(ROOT)),"evaluation_index_sha256":sha256(args.evaluation_index),
        "dataset_config_sha256":sha256(args.dataset_config),"source_hashes":source_hashes,
        "input_hashes":{name:datasets[name]["hashes"] for name in RETRIEVAL_DATASETS},"states":runs,
        "temperature":TEMPERATURE,"k":K,"native_logit_scale":logit_scale,"atol":ATOL,"rtol":RTOL,
        "block_size":args.block_size,"environment":environment,"evidence_type":EVIDENCE_TYPE,
        "diagnostic_kind":"post-fit exact positive-set R1 threshold at the same full gallery",
        "kl_direction":"teacher_to_student","kl_scale":"KL at T=2 without a T-squared multiplier",
        "drift_scale":"native positive logit scale times the unscaled float64 cosine score",
        "scope":"only the exact evaluated candidate gallery; no training-minibatch extrapolation",
        "benchmark_ranks":"unchanged stable-index ranks, saved separately from pessimistic diagnostic correctness",
        "selection_modified":False,"success_gates_modified":False,"floating_point_caveat":"not an interval-arithmetic proof"}
    args.output.mkdir(parents=True,exist_ok=True)
    receipt_path=args.output/"prediagnostic_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text())!=receipt:
        raise ValueError("Existing pre-diagnostic receipt is immutable")
    if not receipt_path.exists():
        write_json(receipt_path,receipt)
    receipt_hash=sha256(receipt_path)
    teacher_features={name:adapted_features(datasets[name]["features"],None) for name in RETRIEVAL_DATASETS}
    relevance={name:relevance_indices(datasets[name]) for name in RETRIEVAL_DATASETS}
    teacher_predictions={name:read_prediction(evaluated["frozen"]["datasets"][name]) for name in RETRIEVAL_DATASETS}
    result={"schema_version":1,"status":"running","protocol_sha256":PROTOCOL_SHA256,"encoder":manifest["encoder"],
        "evidence_type":EVIDENCE_TYPE,"prediagnostic_receipt":str(receipt_path.relative_to(ROOT)),
        "prediagnostic_receipt_sha256":receipt_hash,"source_hashes":source_hashes,"temperature":TEMPERATURE,
        "native_logit_scale":logit_scale,"k":K,"scope":receipt["scope"],"runs":[]}
    for run in runs:
        payload=torch.load(ROOT/run["checkpoint"],map_location="cpu",weights_only=True)
        if payload.get("ledger_sha256")!=manifest["ledger_sha256"] or payload.get("protocol_sha256")!=PROTOCOL_SHA256:
            raise ValueError("Checkpoint provenance differs from the selected execution")
        adapter=ResidualAdapter(512 if manifest["encoder"]=="vit_b32" else 1024)
        adapter.load_state_dict(payload["state_dict"],strict=True)
        adapter.eval()
        output_run=dict(run,datasets={})
        for name in RETRIEVAL_DATASETS:
            started=time.monotonic()
            directory=args.output/"predictions"/run["state_id"]
            directory.mkdir(parents=True,exist_ok=True)
            archive_path,metadata_path=directory/f"{name}.npz",directory/f"{name}.json"
            provenance={"state":run,"dataset":name,"prediagnostic_receipt_sha256":receipt_hash}
            benchmark=read_prediction(evaluated[run["state_id"]]["datasets"][name])
            reused=False
            if archive_path.exists() or metadata_path.exists():
                if not archive_path.exists() or not metadata_path.exists():
                    raise ValueError("Incomplete existing diagnostic artifact pair")
                metadata=json.loads(metadata_path.read_text())
                if metadata["provenance"]!=provenance or metadata["predictions_sha256"]!=sha256(archive_path):
                    raise ValueError("Saved diagnostic content or provenance changed")
                with np.load(archive_path,allow_pickle=False) as archive:
                    raw={key:archive[key] for key in archive.files}
                summary=metadata["summary"]
                reused=True
            else:
                student_images,student_texts=adapted_features(datasets[name]["features"],adapter)
                teacher_images,teacher_texts=teacher_features[name]
                image_relevance,text_relevance=relevance[name]
                features=datasets[name]["features"]
                raw={"image_ids":np.asarray(features["image_ids"],str),"text_ids":np.asarray(features["text_ids"],str),
                     "text_source_image_ids":np.asarray([features["image_ids"][row[0]] for row in text_relevance],str)}
                summary={}
                for direction,inputs,labels in (("i2t",(teacher_images,teacher_texts,student_images,student_texts),image_relevance),
                                                ("t2i",(teacher_texts,teacher_images,student_texts,student_images),text_relevance)):
                    values=diagnose_direction(*inputs,labels,logit_scale,args.block_size)
                    raw.update({f"{direction}_{key}":value for key,value in values.items()})
                    summary[direction]=coverage_summary(values)
                assert_benchmark_alignment(raw,benchmark,teacher_predictions[name])
                temporary=archive_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary,**raw)
                temporary.replace(archive_path)
                write_json(metadata_path,{"provenance":provenance,"summary":summary,
                    "predictions_sha256":sha256(archive_path),"seconds":time.monotonic()-started})
            assert_benchmark_alignment(raw,benchmark,teacher_predictions[name])
            for direction in ("i2t","t2i"):
                reconstructed=coverage_summary({key.removeprefix(direction+"_"):value for key,value in raw.items()
                                                if key.startswith(direction+"_")})
                if reconstructed!=summary[direction]:
                    raise ValueError("Saved coverage summary disagrees with raw diagnostic arrays")
            output_run["datasets"][name]={"predictions":str(archive_path.relative_to(ROOT)),"predictions_sha256":sha256(archive_path),
                "metadata":str(metadata_path.relative_to(ROOT)),"metadata_sha256":sha256(metadata_path),"summary":summary}
            print(json.dumps({"state":run["state_id"],"dataset":name,"reused":reused,
                              "seconds":round(time.monotonic()-started,3)}),flush=True)
        result["runs"].append(output_run)
        write_json(args.output/"index.json",result)
    result["status"]="complete"
    result["state_count"]=len(result["runs"])
    result["archive_count"]=2*len(result["runs"])
    result["false_exact_kl_certificates"]=sum(record["summary"][direction]["certificate_coverage"]["certified"]["false_certificates"]
        for run in result["runs"] for record in run["datasets"].values() for direction in ("i2t","t2i"))
    write_json(args.output/"index.json",result)
    print(json.dumps({key:result[key] for key in ("status","state_count","archive_count","false_exact_kl_certificates")}),flush=True)


if __name__=="__main__":
    main()
