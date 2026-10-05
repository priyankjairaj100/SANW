#!/usr/bin/env python3
"""Plan or restore only the benchmark inputs named by the recovered archive index.

Restoration checks complete archive SHA256 before copying allowlisted bytes. It
does not parse benchmark labels/features or compute any model outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = "recovery/v8_input_recovery/SCIENTIFIC_INPUTS_ORIGINAL_INDEX.json"
INDEX_SHA256 = "898f875d0acc6cb6d3286f9d3c33ea9cd75fe48ca848bbb7b75044b8a19540a6"
ENCODERS = ("vit_b32", "rn50")
DATASETS = ("e_vil_test1000", "coco_karpathy", "sugarcrepe_pp")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def required_paths():
    output = {}
    for encoder in ENCODERS:
        output[encoder] = {}
        for dataset in DATASETS:
            relative = f"review_followup/{dataset}" if dataset == "e_vil_test1000" else dataset
            manifest = f"data/{relative}/manifest.json"
            if encoder == "rn50":
                folder = f"results/strengthen_second_encoder/features/{relative}"
            else:
                folder = (f"results/review_followup/features/{dataset}" if dataset == "e_vil_test1000"
                          else f"results/resume_features/{dataset}")
            output[encoder][dataset] = {"manifest": manifest, "features": folder + "/features.npz", "metadata": folder + "/metadata.json"}
    return output


def inventory():
    index_path = ROOT / INDEX_PATH
    if digest(index_path) != INDEX_SHA256:
        raise ValueError("Recovered scientific archive index changed")
    index = json.loads(index_path.read_text())
    required = sorted({name for datasets in required_paths().values() for entries in datasets.values() for name in entries.values()})
    packages = []
    seen = []
    for package in index["packages"]:
        names = sorted(set(package["paths"]) & set(required))
        if names:
            packages.append({key: package[key] for key in ("part", "name", "path", "sha256", "bytes")} | {"restore_members": names})
            seen.extend(names)
    if sorted(seen) != required:
        raise ValueError("Expected input files do not have unique recovered archive membership")
    return {"study": "sanw_practical_v10_benchmark_recovery_plan",
            "scientific_index": {"path": INDEX_PATH, "sha256": INDEX_SHA256}, "packages": packages,
            "datasets": required_paths(), "required_unique_file_count": len(required),
            "no_model_scoring": True, "excluded": "Original SugarCrepe, raw images, and unrelated artifacts are not restored."}


def restore(archives, output):
    plan = inventory()
    files = {}
    for package in plan["packages"]:
        found = list(Path(archives).rglob(Path(package["path"]).name))
        if len(found) != 1:
            raise ValueError(f"Expected one archive for part {package['part']}")
        filename = found[0]
        if filename.stat().st_size != package["bytes"] or digest(filename) != package["sha256"]:
            raise ValueError("Recovery archive identity mismatch")
        with zipfile.ZipFile(filename) as archive:
            for name in package["restore_members"]:
                content = archive.read(name)
                sha = hashlib.sha256(content).hexdigest()
                destination = ROOT / name
                if destination.exists() and digest(destination) != sha:
                    raise ValueError(f"Refusing to replace a different existing input: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(content)
                files[name] = {"path": name, "sha256": sha}
    receipt = {"study": "sanw_practical_v10_benchmark_inputs", "scientific_index": plan["scientific_index"],
               "recovery_packages": plan["packages"], "restoration_source_sha256": digest(__file__),
               "datasets": {encoder: {dataset: {key: files[name] for key, name in entries.items()}
                                       for dataset, entries in datasets.items()} for encoder, datasets in required_paths().items()},
               "benchmark_labels_or_features_parsed": False, "model_outcomes_computed": False}
    save(output, receipt)


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a recovery record")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "restore")); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        save(args.output, inventory())
    elif args.archives is None:
        parser.error("restore requires --archives")
    else:
        restore(args.archives, args.output)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "model_outcomes_computed": False}))


if __name__ == "__main__":
    main()
