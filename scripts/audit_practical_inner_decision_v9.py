#!/usr/bin/env python3
"""Independent raw-outcome, checkpoint and finite-selection audit for v9.

Does not repeat gallery inference. It verifies winner correctness, reconstructs
composition labels/counts from every saved pair score, and selects using exact
rational arithmetic independently of the production metric/selector functions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_practical_inner_v9 import verified_inner_run

METRICS = ("i2t", "t2i", "original", "source_pair")
HASHES = {}
COUNTS = {"indices": 0, "states": 0, "learned_states": 0, "winner_queries": 0, "composition_triples": 0, "raw_pair_scores": 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked(entry):
    path = ROOT / entry["path"]
    actual = sha(path)
    assert actual == entry["sha256"], path
    if "bytes" in entry:
        assert path.stat().st_size == entry["bytes"]
    HASHES[str(path.relative_to(ROOT))] = actual
    return path


def read(path):
    return json.loads(Path(path).read_text())


def archive(entry):
    with np.load(checked(entry), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def same(a, b):
    assert set(a) == set(b)
    assert all(np.array_equal(a[key], b[key]) for key in a)


def raw_outcomes(raw, manifest):
    owner = raw["gallery_owner"]
    assert len(raw["gallery_image_ids"]) == 1200 and len(owner) == 6000
    assert len(raw["i2t_query_indices"]) == 240 and len(raw["t2i_query_indices"]) == 1200
    for direction in ("i2t", "t2i"):
        query = raw[f"{direction}_query_indices"]
        winner = raw[f"{direction}_top_indices"]
        correct = owner[winner] == query if direction == "i2t" else winner == owner[query]
        np.testing.assert_array_equal(correct, raw[f"{direction}_correct"])
        clusters = raw["gallery_image_ids"][query if direction == "i2t" else owner[query]]
        np.testing.assert_array_equal(clusters, raw[f"{direction}_cluster_ids"])
        saved_q = raw[f"{direction}_rescored_query_indices"]
        saved_g = raw[f"{direction}_rescored_gallery_indices"]
        saved_s = raw[f"{direction}_rescored_scores"]
        for i, q in enumerate(query):
            mask = saved_q == q
            gs, ss = saved_g[mask], saved_s[mask]
            assert len(gs) == raw[f"{direction}_candidate_counts"][i] and len(gs) > 0
            assert np.all(np.diff(gs) > 0)
            best = int(np.argmax(ss))
            assert gs[best] == winner[i] and ss[best] == raw[f"{direction}_top_scores"][i]
        COUNTS["winner_queries"] += len(query)
    original, source_pair = {}, {}
    raw_i, raw_t, raw_r, raw_s = (raw[k] for k in ("raw_image_index", "raw_text_index", "raw_relation", "raw_score"))
    COUNTS["raw_pair_scores"] += len(raw_i)
    for image in np.unique(raw_i):
        use = raw_i == image
        indices, labels, scores = raw_t[use], raw_r[use], raw_s[use]
        src, sup, neg = scores[labels == 1], scores[labels == 2], scores[labels == 3]
        image_id = str(manifest["images"][int(image)]["id"])
        if len(src) and len(sup) and len(neg):
            good = total = 0
            for a in src:
                for b in sup:
                    for c in neg:
                        good += int(a > c and b > c); total += 1
            original[image_id] = (good, total)
        keys = {int(t): tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", manifest["texts"][int(t)]["text"]).casefold())) for t in indices}
        positives = {keys[int(t)] for t, label in zip(indices, labels) if label in (1, 2)}
        negative_rows = [int(t) for t, label in zip(indices, labels) if label == 3 and keys[int(t)] not in positives]
        pairs = [(int(a), int(b)) for a, b in combinations(indices[labels == 1], 2) if keys[int(a)] != keys[int(b)]]
        lookup = dict(zip(indices.tolist(), scores.tolist()))
        if pairs and negative_rows:
            good = total = 0
            for a, b in pairs:
                for c in negative_rows:
                    good += int(lookup[a] > lookup[c] and lookup[b] > lookup[c]); total += 1
            source_pair[image_id] = (good, total)
    for name, actual in (("original", original), ("source_pair", source_pair)):
        ids = raw[f"{name}_image_ids"]
        assert len(ids) == len(actual)
        correct_counts = np.array([actual[str(image)][0] for image in ids])
        totals = np.array([actual[str(image)][1] for image in ids])
        np.testing.assert_array_equal(totals, raw[f"{name}_triplet_counts"])
        np.testing.assert_array_equal(correct_counts/totals, raw[f"{name}_correct"])
        np.testing.assert_array_equal(correct_counts/totals, raw[f"{name}_joint_accuracy"])
        np.testing.assert_array_equal(ids, raw[f"{name}_cluster_ids"])
        COUNTS["composition_triples"] += int(totals.sum())


def paired_exact(frozen, current, paired_record, declared, declared_float):
    stored = archive(paired_record)
    result = {}
    for name in METRICS:
        baseline, trained = frozen[f"{name}_correct"], current[f"{name}_correct"]
        n = np.ones(len(baseline), dtype=np.int64) if name in ("i2t", "t2i") else frozen[f"{name}_triplet_counts"]
        a, b = np.rint(baseline*n).astype(np.int64), np.rint(trained*n).astype(np.int64)
        numerator = b-a
        exact = sum((Fraction(int(x), int(d)) for x, d in zip(numerator, n)), Fraction()) / len(n)
        assert declared[name] == {"numerator": exact.numerator, "denominator": exact.denominator}
        assert declared_float[name] == float(exact)
        np.testing.assert_array_equal(stored[f"{name}_difference_numerators"], numerator)
        np.testing.assert_array_equal(stored[f"{name}_difference_denominators"], n)
        np.testing.assert_array_equal(stored[f"{name}_difference"], numerator.astype(float)/n)
        np.testing.assert_array_equal(stored[f"{name}_cluster_ids"], frozen[f"{name}_cluster_ids"])
        result[name] = exact
    return result


def audit_index(index_record, protocol_hash, manifest, frozen_by_encoder):
    index = read(checked(index_record))
    start = read(checked(index["start_receipt"]))
    completion_path = checked(start["completion"])
    checked(start["ledger"])
    identity, completion, protocol, split, expected = verified_inner_run(completion_path.parent, protocol_hash)
    assert start["encoder"] == index["encoder"] == identity["encoder"]
    assert index["protocol_sha256"] == protocol_hash and start["split"] == protocol["split"]
    assert [(x["epoch"],x["family"]) for x in index["states"]] == [(x["epoch"],x["family"]) for x in expected]
    encoder = identity["encoder"]
    if identity["study"] == "sanw_labclip_inner_v9":
        updates = identity["updates_per_epoch"]
        eligible = len(identity["fit_loader_owner_indices"])
        wanted = ((5*eligible + identity["config"]["batch_size"]-1)//identity["config"]["batch_size"]
                  if identity["config"]["arm"] == "published_control"
                  else 5*((eligible + identity["config"]["batch_size"]-1)//identity["config"]["batch_size"]))
        assert updates == wanted
        for row in completion["history"]:
            assert row["optimizer_steps"] == row["epoch"]*updates and row["batches"] == updates
            assert row["caption_rows"] == 5*eligible
        assert completion["optimizer_steps"] == updates*identity["config"]["epochs"]
    rows, identity_raw, parity = [], None, None
    for entry, state in zip(index["states"], expected):
        result = read(checked(entry["result"]))
        for key, value in state.items():
            if key != "scorer":
                assert result[key] == value
        assert result["start_receipt"] == index["start_receipt"] and result["protocol_sha256"] == protocol_hash
        frozen, trained = archive(result["frozen"]), archive(result["predictions"])
        if encoder not in frozen_by_encoder:
            raw_outcomes(frozen, manifest); frozen_by_encoder[encoder] = frozen
        else:
            same(frozen_by_encoder[encoder], frozen)
        raw_outcomes(trained, manifest)
        changes = paired_exact(frozen, trained, result["paired_predictions"], result["exact_paired_changes"], result["paired_changes"])
        checkpoint = checked(result["checkpoint"])
        with np.load(checkpoint, allow_pickle=False) as z:
            if result["family"].startswith("labclip"):
                weight = z["weight"].astype(float)
                distance = float(np.linalg.norm(weight-np.eye(len(weight))))
                if result["epoch"] == 0:
                    assert distance == 0
                else:
                    assert distance > 0 and result["nonzero_functional_update"]
        COUNTS["states"] += 1
        if result["epoch"] == 0:
            identity_raw = trained
            parity = {"changes": {k:float(v) for k,v in changes.items()},
                      "i2t_winner_changes": int(np.sum(frozen["i2t_top_indices"] != trained["i2t_top_indices"])),
                      "t2i_winner_changes": int(np.sum(frozen["t2i_top_indices"] != trained["t2i_top_indices"])),
                      "max_pair_score_difference": float(np.max(np.abs(frozen["raw_score"]-trained["raw_score"]))) }
            continue
        COUNTS["learned_states"] += 1
        identity_changes = None
        if result["family"].startswith("labclip"):
            assert identity_raw is not None
            identity_changes = paired_exact(identity_raw, trained, result["identity_paired_predictions"], result["exact_identity_paired_changes"], result["identity_paired_changes"])
            assert np.max(np.abs(identity_raw["raw_score"]-trained["raw_score"])) > 1e-12
        eligible_state = bool(result["nonzero_functional_update"] and changes["original"] > 0 and all(changes[k] >= 0 for k in METRICS))
        if identity_changes is not None:
            eligible_state = eligible_state and identity_changes["original"] > 0
        rows.append({"encoder": encoder, "family": result["family"], "epoch": result["epoch"],
                     "radius": result.get("radius"), "composition_weight": result.get("composition_weight"),
                     "eligible": eligible_state, "gains": changes, "result": entry["result"]})
    COUNTS["indices"] += 1
    return rows, {"encoder":encoder,"family":expected[-1]["family"],"identity_parity":parity,
                  "optimizer_steps":completion["optimizer_steps"],"state_count":len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--controls", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = read(args.selection)
    start = read(checked(selection["start_receipt"]))
    protocol_path = checked(start["protocol"])
    protocol = read(protocol_path); protocol_hash=sha(protocol_path)
    assert selection["protocol_sha256"] == protocol_hash
    manifest = read(ROOT/protocol["training_inputs"]["vit_b32"]["manifest"]["path"])
    frozen_by_encoder, rows, index_summaries = {}, [], []
    for entry in start["indices"]:
        actual, summary = audit_index(entry, protocol_hash, manifest, frozen_by_encoder)
        rows.extend(actual); index_summaries.append(summary)
    assert len(rows) == selection["candidate_count"] == len(selection["candidates"])
    assert {r["result"]["sha256"] for r in rows} == {r["result"]["sha256"] for r in selection["candidates"]}
    family = selection["family"]
    expected_keys = ([(epoch,) for epoch in protocol["labclip"]["epochs"]] if family == "labclip"
                     else [(r,w) for r in protocol["joint"]["radii"] for w in protocol["joint"]["composition_weights"]])
    lookup={}
    for row in rows:
        key=(row["epoch"],) if family=="labclip" else (row["radius"],row["composition_weight"])
        assert (row["encoder"],key) not in lookup
        lookup[row["encoder"],key]=row
    assert set(lookup)=={(encoder,key) for encoder in protocol["encoders"] for key in expected_keys}
    eligible=[]
    for key in expected_keys:
        states=[lookup[encoder,key] for encoder in protocol["encoders"]]
        if all(state["eligible"] for state in states):
            gains=[state["gains"]["original"] for state in states]
            tail=(-key[0],) if family=="labclip" else (-key[0],-abs(math.log(key[1])),-key[1])
            eligible.append(((min(gains),sum(gains),*tail),key))
    assert selection["eligible_count"]==len(eligible)
    assert selection["passed"]==bool(eligible)
    selected=None if not eligible else max(eligible)[1]
    if selected is None:
        assert selection["selected_config"] is None and selection["selected_epoch"] is None
        assert selection["status"]=="stop_before_full_fit_and_official_development"
    elif family=="labclip":assert selection["selected_epoch"]==selected[0]
    else:assert selection["selected_config"]=={"radius":selected[0],"composition_weight":selected[1]}
    controls=[]
    for filename in args.controls:
        record={"path":str(filename),"sha256":sha(filename),"bytes":filename.stat().st_size}
        control_rows, summary=audit_index(record,protocol_hash,manifest,frozen_by_encoder)
        assert all(r["family"]=="labclip_published_control" for r in control_rows)
        controls.extend(control_rows);index_summaries.append(summary)
    serial_rows=[]
    for row in rows+controls:
        serial_rows.append({**row,"gains":{k:{"numerator":v.numerator,"denominator":v.denominator,"value":float(v)} for k,v in row["gains"].items()}})
    output={"schema":"sanw_v9_independent_inner_decision_audit_v1","passed":True,
            "meaning_of_passed":"record_integrity_and_correct_decision_not_empirical_success",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"family":family,
            "selection":{"path":str(args.selection),"sha256":sha(args.selection)},"protocol_sha256":protocol_hash,
            "decision_passed":bool(eligible),"eligible_common_candidates":len(eligible),"selected_key":selected,
            "counts":COUNTS,"indices":index_summaries,"states":serial_rows,"verified_artifact_sha256":HASHES,
            "audit_source_sha256":sha(Path(__file__)),
            "scope":"Checkpoint/provenance validation; independent saved-winner correctness and rescored-candidate argmax; independent all-triple composition reconstruction; exact rational finite selection. Full gallery model inference not rerun.",
            "heldout_benchmark_access":False,"official_development_access":False,"new_model_fits":0}
    with args.output.open('x') as stream:json.dump(output,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({"audit":str(args.output),"sha256":sha(args.output),"passed":True,"decision_passed":bool(eligible),"counts":COUNTS,"indices":index_summaries}),flush=True)


if __name__=='__main__':main()
