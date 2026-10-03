#!/usr/bin/env python3
"""Acquire complete official SugarCrepe, SugarCrepe++, and COCO test pools.

All source annotations are retained. Source revisions and content hashes are
recorded, every selected image is downloaded, and no failed item is dropped.
Image files are shared by their official COCO ID across the three manifests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

SC_REVISION = "0047054b243992f0fad63d6f64f7544862daf846"
SCPP_REVISION = "c083bbf039a5ca215acaf25e7ee736d2ea6f3864"
SC_CATEGORIES = ("add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj")
SCPP_CATEGORIES = SC_CATEGORIES[2:]
KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(url: str, path: Path, expected_hash: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        last = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "grounded-contrastive-reproducibility/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target, 1024 * 1024)
                temporary.replace(path)
                break
            except Exception as error:
                last = error
                temporary.unlink(missing_ok=True)
                if attempt == 4:
                    raise RuntimeError(f"Download failed: {url}: {last}") from error
                time.sleep(min(2 ** attempt, 8))
    actual_hash = sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"Source hash mismatch for {path}: {actual_hash} != {expected_hash}")
    return {"url": url, "path": path.relative_to(ROOT).as_posix(), "sha256": actual_hash, "bytes": path.stat().st_size}


def source(url: str, directory: Path, filename: str, revision: str | None = None) -> tuple[Path, dict]:
    path = directory / "source" / filename
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    pins = json.loads((ROOT / "data" / "benchmark_source_pins.json").read_text())
    expected = next(row["sha256"] for row in pins["sources"] if row["url"] == url)
    if receipt_path.exists():
        assert json.loads(receipt_path.read_text())["sha256"] == expected
    receipt = download(url, path, expected)
    if revision:
        receipt["revision"] = revision
    dump(receipt_path, receipt)
    return path, receipt


def make_manifest(name: str) -> dict:
    return {"schema_version": 1, "dataset": name, "images": [], "texts": [], "pairs": [], "triplets": []}


def image_id(value: str | int) -> str:
    return f"coco:{int(value)}"


def image_row(coco_id: int) -> dict:
    return {"id": image_id(coco_id), "coco_id": coco_id, "path": f"data/coco_images/{coco_id:012d}.jpg", "split": "test"}


def build_sugar(name: str, repository: str, revision: str, categories: tuple[str, ...], second_positive: bool) -> tuple[dict, list[dict]]:
    directory = ROOT / "data" / name
    manifest = make_manifest(name)
    sources = []
    images = {}
    for category in categories:
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/data/{category}.json"
        path, receipt = source(url, directory, f"{category}.json", revision)
        sources.append(receipt)
        annotation = json.loads(path.read_text())
        records = annotation.items() if isinstance(annotation, dict) else ((str(x["id"]), x) for x in annotation)
        for original_id, item in records:
            coco_id = int(Path(item["filename"]).stem)
            images.setdefault(coco_id, image_row(coco_id))
            uid = f"{name}:{category}:{original_id}"
            triple = {"id": uid, "official_id": original_id, "image_id": image_id(coco_id), "category": category,
                      "positive1_id": uid + ":positive1", "negative_id": uid + ":negative"}
            fields = [("positive1_id", "caption", "supported"), ("negative_id", "negative_caption", "contradicted")]
            if second_positive:
                triple["positive2_id"] = uid + ":positive2"
                fields.insert(1, ("positive2_id", "caption2", "supported"))
            for id_field, caption_field, relation in fields:
                text = item[caption_field]
                assert isinstance(text, str) and text.strip(), (uid, caption_field)
                manifest["texts"].append({"id": triple[id_field], "text": text})
                manifest["pairs"].append({"image_id": image_id(coco_id), "text_id": triple[id_field], "relation": relation})
            manifest["triplets"].append(triple)
    manifest["images"] = list(images.values())
    assert len({t["id"] for t in manifest["triplets"]}) == len(manifest["triplets"])
    expected_items, expected_images = (4757, 1542) if second_positive else (7511, 1560)
    assert len(manifest["triplets"]) == expected_items, (name, len(manifest["triplets"]))
    assert len(manifest["images"]) == expected_images, (name, len(manifest["images"]))
    return manifest, sources


def build_coco() -> tuple[dict, list[dict]]:
    directory = ROOT / "data" / "coco_karpathy"
    archive, receipt = source(KARPATHY_URL, directory, "caption_datasets.zip")
    json_path = directory / "source" / "dataset_coco.json"
    if not json_path.exists():
        with zipfile.ZipFile(archive) as z:
            assert "dataset_coco.json" in z.namelist()
            with z.open("dataset_coco.json") as src, json_path.open("wb") as target:
                shutil.copyfileobj(src, target)
    pins = json.loads((ROOT / "data" / "benchmark_source_pins.json").read_text())
    assert sha256(json_path) == pins["derived_files"][json_path.relative_to(ROOT).as_posix()]
    manifest = make_manifest("coco_karpathy")
    dataset = json.loads(json_path.read_text())
    selected = [row for row in dataset["images"] if row["split"] == "test"]
    assert len(selected) == 5000
    omitted_captions = []
    for row in selected:
        coco_id = int(row["cocoid"])
        assert row["filepath"] == "val2014"
        assert row["filename"] == f"COCO_val2014_{coco_id:012d}.jpg"
        image = image_row(coco_id)
        image["official_filename"] = row["filename"]
        image["official_imgid"] = row["imgid"]
        manifest["images"].append(image)
        assert len(row["sentences"]) >= 5
        omitted_captions.extend({"coco_id": coco_id, "official_sentid": sentence["sentid"], "source_position": position}
                                for position, sentence in enumerate(row["sentences"][5:], 5))
        for position, sentence in enumerate(row["sentences"][:5]):
            tid = f"coco_karpathy:{coco_id}:{position}"
            manifest["texts"].append({"id": tid, "text": sentence["raw"], "official_sentid": sentence["sentid"]})
            manifest["pairs"].append({"image_id": image_id(coco_id), "text_id": tid, "relation": "source"})
    assert len(manifest["texts"]) == 25000
    assert len({r["id"] for r in manifest["images"]}) == 5000
    assert len(omitted_captions) == 10
    return manifest, [receipt, {"path": json_path.relative_to(ROOT).as_posix(), "archive_member": "dataset_coco.json", "sha256": sha256(json_path), "bytes": json_path.stat().st_size,
                                "available_test_captions": 25010, "selected_test_captions": 25000,
                                "omitted_captions": omitted_captions,
                                "omission_reason": "Exactly the first five source-order captions are used for every image."}]


def audit_annotation_alignment() -> None:
    counts = []
    for category in SCPP_CATEGORIES:
        sc = json.loads((ROOT / "data" / "sugarcrepe" / "source" / f"{category}.json").read_text())
        scpp = json.loads((ROOT / "data" / "sugarcrepe_pp" / "source" / f"{category}.json").read_text())
        assert len(sc) == len(scpp)
        row = {"category": category, "rows": len(scpp), "caption_differences": 0,
               "negative_differences": 0, "image_differences": 0, "official_id_differences": 0}
        for (original_id, a), b in zip(sc.items(), scpp):
            row["official_id_differences"] += str(b["id"]) != original_id
            for field, label in (("caption", "caption_differences"), ("negative_caption", "negative_differences"), ("filename", "image_differences")):
                row[label] += a[field] != b[field]
        assert row["image_differences"] == 0
        counts.append(row)
    dump(ROOT / "data" / "benchmark_annotation_alignment.json", {
        "alignment": "Original annotation order per category. Images match in every aligned pair; strings are compared without normalization. Official IDs need not match across datasets.",
        "counts": counts,
        "warning": "SugarCrepe++ changes original positive and negative text for some rows. Its P1 accuracy is not a substitute for SugarCrepe replace/swap accuracy."})


def audit_coco_versions(shared_ids: set[int], image_receipts: dict[int, dict], workers: int) -> None:
    """Verify that using one file for shared COCO2014/2017 IDs is exact."""
    path = ROOT / "data" / "coco_source_version_identity.json"
    if path.exists():
        audit = json.loads(path.read_text())
    else:
        def compare(coco_id: int) -> dict:
            url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{coco_id:012d}.jpg"
            temporary = ROOT / "data" / "coco_images" / "version_verification" / f"{coco_id:012d}.jpg"
            second = download(url, temporary)
            validate_image(temporary)
            first_hash = image_receipts[coco_id]["sha256"]
            row = {"coco_id": coco_id, "val2014_sha256": first_hash, "val2017_sha256": second["sha256"],
                   "identical": first_hash == second["sha256"]}
            temporary.unlink()
            return row
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(compare, sorted(shared_ids)))
        audit = {"compared_shared_images": len(rows), "all_byte_identical": all(row["identical"] for row in rows), "rows": rows}
        dump(path, audit)
    assert {row["coco_id"] for row in audit["rows"]} == shared_ids
    assert audit["all_byte_identical"] and audit["compared_shared_images"] == len(shared_ids)
    for row in audit["rows"]:
        assert row["identical"]
        assert row["val2014_sha256"] == row["val2017_sha256"] == image_receipts[row["coco_id"]]["sha256"]


def validate_image(path: Path) -> None:
    with path.open("rb") as f:
        assert f.read(2) == b"\xff\xd8", f"Not a JPEG: {path}"
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required to validate every downloaded image.")
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        img.convert("RGB").load()


def get_image(coco_id: int, is_karpathy: bool) -> dict:
    path = ROOT / "data" / "coco_images" / f"{coco_id:012d}.jpg"
    receipt_path = path.with_suffix(".json")
    if is_karpathy:
        relative = f"val2014/COCO_val2014_{coco_id:012d}.jpg"
    else:
        relative = f"val2017/{coco_id:012d}.jpg"
    # Official COCO bucket, using its valid HTTPS S3 hostname.
    url = f"https://s3.amazonaws.com/images.cocodataset.org/{relative}"
    expected = json.loads(receipt_path.read_text())["sha256"] if receipt_path.exists() else None
    receipt = download(url, path, expected)
    validate_image(path)
    receipt.update({"coco_id": coco_id, "image_id": image_id(coco_id), "official_dataset_url": "https://cocodataset.org/#download"})
    dump(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--metadata-only", action="store_true", help="Write pending manifests and annotation provenance without image acquisition.")
    args = parser.parse_args()
    datasets = {}
    datasets["sugarcrepe"] = build_sugar("sugarcrepe", "RAIVNLab/sugar-crepe", SC_REVISION, SC_CATEGORIES, False)
    datasets["sugarcrepe_pp"] = build_sugar("sugarcrepe_pp", "Sri-Harsha/scpp", SCPP_REVISION, SCPP_CATEGORIES, True)
    datasets["coco_karpathy"] = build_coco()
    audit_annotation_alignment()
    image_sets = {name: {r["coco_id"] for r in manifest["images"]} for name, (manifest, _) in datasets.items()}
    all_ids = sorted(set.union(*image_sets.values()))
    overlaps = {f"{a}__{b}": {"count": len(image_sets[a] & image_sets[b]), "coco_ids": sorted(image_sets[a] & image_sets[b])}
                for i, a in enumerate(datasets) for b in list(datasets)[i + 1:]}
    image_receipts = {}
    if not args.metadata_only:
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(get_image, cid, cid in image_sets["coco_karpathy"]): cid for cid in all_ids}
            for done, future in enumerate(as_completed(futures), 1):
                cid = futures[future]
                try:
                    image_receipts[cid] = future.result()
                except Exception as error:
                    failures.append({"coco_id": cid, "error": str(error)})
                if done % 250 == 0 or done == len(all_ids):
                    print(f"Images verified {done}/{len(all_ids)}; failures {len(failures)}", flush=True)
        dump(ROOT / "data" / "benchmark_download_failures.json", failures)
        if failures:
            raise RuntimeError(f"{len(failures)} image downloads failed. No complete manifest will be emitted; rerun resumes verified files.")
        audit_coco_versions(image_sets["sugarcrepe"] & image_sets["coco_karpathy"], image_receipts, args.workers)
    for name, (manifest, sources) in datasets.items():
        if image_receipts:
            for row in manifest["images"]:
                receipt = image_receipts[row["coco_id"]]
                row.update({"sha256": receipt["sha256"], "bytes": receipt["bytes"]})
        directory = ROOT / "data" / name
        manifest_path = directory / ("manifest.pending.json" if args.metadata_only else "manifest.json")
        dump(manifest_path, manifest)
        provenance = {"schema_version": 1, "dataset": name, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                      "sources": sources, "image_downloads_complete": not args.metadata_only,
                      "counts": {"images": len(manifest["images"]), "texts": len(manifest["texts"]), "pairs": len(manifest["pairs"]), "triplets": len(manifest["triplets"])},
                      "categories": dict(Counter(t["category"] for t in manifest["triplets"])),
                      "manifest_sha256": sha256(manifest_path), "acquisition_script_sha256": sha256(Path(__file__)),
                      "selection": "All official benchmark items, original annotation order; category order fixed by script." if name != "coco_karpathy" else "All 5000 Stanford Karpathy test images; first five raw captions per image in original annotation order. No caption normalization.",
                      "image_storage": "Shared official COCO-ID cache; bytes are hashed and decoded without modification. Karpathy members use val2014; remaining members use val2017. Individual URL/hash receipts are adjacent to JPEGs.",
                      "overlaps": {k: v for k, v in overlaps.items() if name in k.split("__")},
                      "exclusions": [], "failed_images": []}
        dump(directory / "provenance.json", provenance)
        print(name, provenance["counts"], flush=True)
    dump(ROOT / "data" / "benchmark_overlaps.json", {"unique_coco_images": len(all_ids), "overlaps": overlaps})


if __name__ == "__main__":
    main()
