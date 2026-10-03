#!/usr/bin/env python3
"""Audit fixed dataset manifests and actual image bytes without model scores.

This script never edits, filters, normalizes, or relabels dataset contents.
Text comparisons are exact Python string equality. File identity is SHA256.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path

DATASETS = ("visual_entailment", "coco_karpathy", "sugarcrepe", "sugarcrepe_pp")
RELATIONS = {"source", "supported", "contradicted", "neutral"}
POLARITY = {"source": "positive", "supported": "positive", "contradicted": "contradicted", "neutral": "neutral"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def duplicate_summary(groups):
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    conflicts = [rows for rows in groups.values()
                 if len({POLARITY[r["relation"]] for r in rows}) > 1]
    examples = []
    for rows in conflicts:
        examples.append({"image_id": rows[0]["image_id"], "text": rows[0]["text"],
                         "relations": dict(Counter(r["relation"] for r in rows)),
                         "text_ids": [r["text_id"] for r in rows]})
    return {
        "unique_image_text_groups": len(groups),
        "repeated_groups": len(repeated),
        "rows_in_repeated_groups": sum(map(len, repeated)),
        "duplicate_rows_beyond_first": sum(len(x) - 1 for x in repeated),
        "relation_conflict_groups": len(conflicts),
        "rows_in_relation_conflict_groups": sum(map(len, conflicts)),
        "relation_conflict_images": len({rows[0]["image_id"] for rows in conflicts}),
        "conflict_relation_patterns": dict(Counter(
            "+".join(sorted({r["relation"] for r in rows})) for rows in conflicts)),
        "conflicts": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/input_audit.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifests = {name: json.loads((repo / "data" / name / "manifest.json").read_text()) for name in DATASETS}
    paths = sorted({row["path"] for data in manifests.values() for row in data["images"]})
    for path in paths:
        resolved = (repo / path).resolve()
        if not resolved.is_relative_to(repo) or not resolved.is_file():
            raise ValueError(f"Missing or out-of-repository image path: {path}")
    with ThreadPoolExecutor(args.workers) as executor:
        actual_hashes = dict(zip(paths, executor.map(lambda p: sha256(repo / p), paths)))
    ve_receipts = json.loads((repo / "data/visual_entailment/provenance.json").read_text())["images"]
    errors, audits, global_groups = [], {}, defaultdict(list)
    image_sets, hash_sets, split_sets = {}, {}, {}
    for name, data in manifests.items():
        images = {row["id"]: row for row in data["images"]}
        texts = {row["id"]: row for row in data["texts"]}
        if len(images) != len(data["images"]) or len(texts) != len(data["texts"]):
            errors.append(f"{name}: duplicate image/text IDs")
        if any(not isinstance(row["text"], str) or not row["text"].strip() for row in data["texts"]):
            errors.append(f"{name}: empty/non-string text")
        image_sets[name] = set(images)
        hash_sets[name] = {actual_hashes[row["path"]] for row in images.values()}
        split_sets[name] = defaultdict(set)
        hashes_to_ids = defaultdict(list)
        for image_id, row in images.items():
            digest = actual_hashes[row["path"]]
            hashes_to_ids[digest].append(image_id)
            split_sets[name][row["split"]].add(digest)
            expected = row.get("sha256")
            expected_bytes = row.get("bytes")
            if name == "visual_entailment":
                receipt = ve_receipts[Path(row["path"]).name]
                expected, expected_bytes = receipt["sha256"], receipt["size_bytes"]
            if digest != expected or (repo / row["path"]).stat().st_size != expected_bytes:
                errors.append(f"{name}: byte receipt mismatch for {image_id}")
        counts, source_counts = Counter(), Counter()
        groups, hyp_groups = defaultdict(list), defaultdict(list)
        pairs_lookup = defaultdict(set)
        referenced_texts, referenced_images = set(), set()
        for pair in data["pairs"]:
            image_id, text_id, relation = pair["image_id"], pair["text_id"], pair["relation"]
            if image_id not in images or text_id not in texts or relation not in RELATIONS:
                errors.append(f"{name}: invalid pair reference/relation: {pair}")
                continue
            text = texts[text_id]["text"]
            row = {"image_id": image_id, "text_id": text_id, "text": text, "relation": relation}
            groups[(image_id, text)].append(row)
            if relation != "source":
                hyp_groups[(image_id, text)].append(row)
            global_groups[(actual_hashes[images[image_id]["path"]], text)].append({"dataset": name, **row})
            counts[relation] += 1
            source_counts[image_id] += int(relation == "source")
            pairs_lookup[(image_id, text_id)].add(relation)
            referenced_texts.add(text_id)
            referenced_images.add(image_id)
        five_source_check = None
        if name in {"visual_entailment", "coco_karpathy"}:
            five_source_check = all(source_counts[i] == 5 for i in images)
            if not five_source_check:
                errors.append(f"{name}: an image does not have five source-caption rows")
        triplet_ids, triplet_text_equalities = set(), Counter()
        for triplet in data["triplets"]:
            image_id = triplet["image_id"]
            if triplet["id"] in triplet_ids or image_id not in images:
                errors.append(f"{name}: invalid triplet ID/image: {triplet['id']}")
            triplet_ids.add(triplet["id"])
            fields = ["positive1_id", "negative_id"]
            if name == "sugarcrepe_pp":
                fields.append("positive2_id")
            for field in fields:
                text_id = triplet.get(field)
                wanted = "contradicted" if field == "negative_id" else "supported"
                if text_id not in texts or wanted not in pairs_lookup[(image_id, text_id)]:
                    errors.append(f"{name}: bad triplet reference/relation: {triplet['id']} {field}")
            for a, b in itertools.combinations(fields, 2):
                if triplet.get(a) in texts and triplet.get(b) in texts:
                    triplet_text_equalities[f"{a}={b}"] += int(texts[triplet[a]]["text"] == texts[triplet[b]]["text"])
        if set(texts) != referenced_texts or set(images) != referenced_images:
            errors.append(f"{name}: unreferenced manifest texts/images")
        audits[name] = {
            "manifest_sha256": sha256(repo / "data" / name / "manifest.json"),
            "images": len(images), "texts": len(texts), "pairs": len(data["pairs"]),
            "triplets": len(data["triplets"]), "relations": dict(counts),
            "split_image_counts": dict(Counter(r["split"] for r in images.values())),
            "image_byte_receipts_checked": len(images),
            "identical_file_groups": [ids for ids in hashes_to_ids.values() if len(ids) > 1],
            "five_source_captions_per_image": five_source_check,
            "all_pair_and_triplet_references_valid": not any(error.startswith(name + ":") for error in errors),
            "exact_image_text_duplicates": duplicate_summary(groups),
            "exact_image_hypothesis_duplicates": duplicate_summary(hyp_groups),
            "triplet_exact_text_equalities": dict(triplet_text_equalities),
        }
    cross = {}
    for a, b in itertools.combinations(DATASETS, 2):
        cross[f"{a}:{b}"] = {"image_id_overlap": len(image_sets[a] & image_sets[b]),
                              "actual_image_sha256_overlap": len(hash_sets[a] & hash_sets[b])}
    ve_split_overlap = {}
    for a, b in itertools.combinations(sorted(split_sets["visual_entailment"]), 2):
        overlap = len(split_sets["visual_entailment"][a] & split_sets["visual_entailment"][b])
        ve_split_overlap[f"{a}:{b}"] = overlap
        if overlap:
            errors.append(f"visual_entailment: actual image bytes overlap across {a}:{b}")
    for name in DATASETS[1:]:
        if cross[f"visual_entailment:{name}"]["actual_image_sha256_overlap"]:
            errors.append(f"training-source images share exact bytes with {name}")
    cross_groups = [rows for rows in global_groups.values() if len({r["dataset"] for r in rows}) > 1]
    cross_conflicts = [rows for rows in cross_groups if len({POLARITY[r["relation"]] for r in rows}) > 1]
    result = {
        "status": "passed" if not errors else "failed", "errors": errors,
        "scope": "Input-only audit. No model scores read. No examples filtered or labels modified.",
        "comparison_rules": {"image_identity": "SHA256 of actual file bytes",
                             "text_identity": "Exact string equality, without normalization",
                             "conflict": "Distinct positive/neutral/contradicted categories; source and supported are both positive."},
        "unique_image_files_hashed": len(paths),
        "actual_image_bytes_hashed": sum((repo / path).stat().st_size for path in paths),
        "datasets": audits, "cross_dataset_image_overlap": cross,
        "visual_entailment_cross_split_actual_sha256_overlap": ve_split_overlap,
        "cross_dataset_exact_image_text": {
            "shared_groups": len(cross_groups), "relation_conflict_groups": len(cross_conflicts),
            "conflicts": [{"image_id": rows[0]["image_id"], "text": rows[0]["text"],
                           "rows": [{k: v for k, v in r.items() if k != "text"} for r in rows]}
                          for rows in cross_conflicts]},
        "script_sha256": sha256(Path(__file__)),
    }
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "unique_image_files_hashed": len(paths),
                      "cross_dataset_image_overlap": cross,
                      "within_dataset_hypothesis_conflicts": {k: v["exact_image_hypothesis_duplicates"]["relation_conflict_groups"] for k, v in audits.items()},
                      "cross_dataset_image_text_conflicts": len(cross_conflicts), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
