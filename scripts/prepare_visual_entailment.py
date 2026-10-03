#!/usr/bin/env python3
"""Reconstruct the pinned e-SNLI-VE / Flickr30k natural-image study sample.

This makes a NEW deterministic sample, not a claim to recover missing old IDs.
No example is skipped if acquisition or validation fails. Every selected image
retains its five source captions and every annotation in the official split.
Only selected images are fetched from a public, immutable ZIP by byte ranges.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib

ANNOTATION_REVISION = "94f17f5c189573c9d1087f82eb27db37690fc810"
FLICKR_REVISION = "2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
CAPTION_REVISION = "68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b"
ANNOTATION_BASE = f"https://raw.githubusercontent.com/maximek3/e-ViL/{ANNOTATION_REVISION}/data/"
FLICKR_BASE = f"https://huggingface.co/datasets/nlphuji/flickr30k/resolve/{FLICKR_REVISION}/"
ZIP_URL = FLICKR_BASE + "flickr30k-images.zip"
EXPECTED_ZIP_SIZE = 4390240817
EXPECTED_ZIP_ETAG = "f78a015e45ea38d4367367223b2bb63cec0d549d5da13c6211663e5439a93216"
SOURCE_FILES = {
    "esnlive_train.csv": (ANNOTATION_BASE + "esnlive_train.csv", 62565577),
    "esnlive_dev.csv": (ANNOTATION_BASE + "esnlive_dev.csv", 2266354),
    "esnlive_test.csv": (ANNOTATION_BASE + "esnlive_test.csv", 2320988),
    "flickr30k_entities_annotations.zip": (f"https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/{CAPTION_REVISION}/annotations.zip", 29284070),
}
SOURCE_SHA256 = {
    "esnlive_train.csv": "61ad4a26ec8e2b9c5fc69e52a4870d972f7fab92f448b8d069eb9793dba3aeb1",
    "esnlive_dev.csv": "30ee4b326dc1c9276f82a9e3ada73639c67c2d54ee61e90795c405c8809a29a3",
    "esnlive_test.csv": "56cc45b747a2bcec759e5b2cf16fb9731625e2697594892392e76c84226fa19d",
    "flickr30k_entities_annotations.zip": "1bdde439e41fa936e31e6d72898f1886d49bb4298f2abcdb50771de4f516b026",
}
LABELS = {"entailment": "supported", "contradiction": "contradicted", "neutral": "neutral"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def download(url: str, dest: Path, expected_size: int) -> dict:
    if dest.exists() and dest.stat().st_size == expected_size:
        digest = sha256(dest)
        if digest != SOURCE_SHA256[dest.name]:
            raise ValueError(f"Cached source checksum mismatch: {dest}")
        return {"url": url, "size_bytes": expected_size, "sha256": digest}
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            if part.stat().st_size != expected_size:
                raise ValueError(f"Wrong size for {dest.name}: {part.stat().st_size} != {expected_size}")
            if sha256(part) != SOURCE_SHA256[dest.name]:
                raise ValueError(f"Downloaded source checksum mismatch: {dest.name}")
            part.replace(dest)
            print(f"Downloaded {dest.name}: {expected_size:,} bytes", flush=True)
            return {"url": url, "size_bytes": expected_size, "sha256": sha256(dest)}
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def get_range(start: int, end: int) -> bytes:
    assert 0 <= start <= end < EXPECTED_ZIP_SIZE
    expected = f"bytes {start}-{end}/{EXPECTED_ZIP_SIZE}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(ZIP_URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"Incorrect range response for {expected}: {response.status}, {response.headers.get('Content-Range')}")
                etag = response.headers.get("ETag", "").strip('"')
                if etag != EXPECTED_ZIP_ETAG:
                    raise ValueError(f"Archive ETag changed: {etag}")
                data = response.read()
            if len(data) != end - start + 1:
                raise ValueError("Incomplete range response")
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else EXPECTED_ZIP_SIZE + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = EXPECTED_ZIP_SIZE - self.position
        size = min(size, EXPECTED_ZIP_SIZE - self.position)
        if size <= 0:
            return b""
        data = get_range(self.position, self.position + size - 1)
        self.position += len(data)
        return data


def archive_index(raw_dir: Path) -> dict:
    path = raw_dir / "flickr30k_zip_selected_index.json"
    # Small index is retained to allow a completely offline rebuild from images.
    if path.exists():
        index = json.loads(path.read_text())
        if index["archive_etag"] != EXPECTED_ZIP_ETAG:
            raise ValueError("Cached archive index is for a different source")
        return index
    with zipfile.ZipFile(RangeReader()) as archive:
        items = {}
        for info in archive.infolist():
            if info.filename.startswith("flickr30k-images/") and info.filename.lower().endswith(".jpg"):
                filename = Path(info.filename).name
                if filename in items:
                    raise ValueError(f"Duplicate archive image: {filename}")
                items[filename] = {
                    "member": info.filename, "offset": info.header_offset,
                    "compressed_size": info.compress_size, "size": info.file_size,
                    "compression": info.compress_type, "crc32": info.CRC,
                }
    index = {"archive_url": ZIP_URL, "archive_size": EXPECTED_ZIP_SIZE,
             "archive_etag": EXPECTED_ZIP_ETAG, "images": items}
    write_json(path, index)
    return index


def acquire_image(filename: str, info: dict, image_dir: Path, repo: Path) -> dict:
    dest = image_dir / filename
    if dest.exists():
        data = dest.read_bytes()
    else:
        start = info["offset"]
        # Fetch the payload and ordinary local header together. The additional
        # request only occurs for an unusually large local extra field.
        packet = get_range(start, min(start + info["compressed_size"] + 2047, EXPECTED_ZIP_SIZE - 1))
        header = packet[:30]
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP local header for {filename}")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        local_size = 30 + name_len + extra_len
        if local_size + info["compressed_size"] <= len(packet):
            raw = packet[local_size:local_size + info["compressed_size"]]
        else:
            payload_start = start + local_size
            raw = get_range(payload_start, payload_start + info["compressed_size"] - 1)
        if info["compression"] == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(raw, -15)
        elif info["compression"] == zipfile.ZIP_STORED:
            data = raw
        else:
            raise ValueError(f"Unsupported ZIP compression: {info['compression']}")
        part = dest.with_suffix(".jpg.part")
        part.write_bytes(data)
        part.replace(dest)
    if len(data) != info["size"] or zlib.crc32(data) & 0xffffffff != info["crc32"]:
        raise ValueError(f"Image CRC or size mismatch: {filename}")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"Image is not JPEG: {filename}")
    # Pillow verifies encoded structure and opens the exact retained bytes.
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
    return {"path": str(dest.relative_to(repo)), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "width": width,
            "height": height, "archive_member": info["member"], "crc32": info["crc32"]}


def read_annotations(path: Path) -> dict:
    grouped = {}
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed annotation at {path.name}:{row_number}")
            label = row["gold_label"]
            if label not in LABELS:
                raise ValueError(f"Unrecognized label {label!r} at {path.name}:{row_number}")
            if row["pairID"] in seen:
                raise ValueError(f"Duplicate pair ID in {path.name}: {row['pairID']}")
            seen.add(row["pairID"])
            row["source_line"] = row_number
            grouped.setdefault(row["Flickr30kID"], []).append(row)
    return grouped


def strip_entity_markup(line: str) -> str:
    """Preserve sentence words and punctuation, removing only entity markers."""
    words = []
    in_phrase = False
    for word in line.split():
        if word.startswith("[/EN#"):
            if in_phrase:
                raise ValueError("Nested entity markup in source caption")
            in_phrase = True
            continue
        if in_phrase and word.endswith("]"):
            word = word[:-1]
            in_phrase = False
        words.append(word)
    if in_phrase:
        raise ValueError("Unterminated entity markup in source caption")
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    data_dir = repo / "data/visual_entailment"
    raw_dir = data_dir / "raw"
    image_dir = data_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        pending = {name: executor.submit(download, url, raw_dir / name, size)
                   for name, (url, size) in SOURCE_FILES.items()}
        sources = {name: future.result() for name, future in pending.items()}
    pools = {split: read_annotations(raw_dir / f"esnlive_{split}.csv")
             for split in ("train", "dev", "test")}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = set(pools[a]) & set(pools[b])
        if overlap:
            raise ValueError(f"Official image splits overlap: {a}/{b}: {len(overlap)}")
    rng = random.Random(42)
    # Sorting precedes sampling so source row ordering cannot change selection.
    train_ids = rng.sample(sorted(pools["train"]), 1200)
    dev_ids = rng.sample(sorted(pools["dev"]), 200)
    test_ids = rng.sample(sorted(pools["test"]), 400)
    chosen = {"train": sorted(train_ids), "calibration": sorted(dev_ids[:100]),
              "validation": sorted(dev_ids[100:]), "test": sorted(test_ids)}
    selected_sources = {"train": "train", "calibration": "dev", "validation": "dev", "test": "test"}
    captions = {}
    with zipfile.ZipFile(raw_dir / "flickr30k_entities_annotations.zip") as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("Sentences/") or not member.endswith(".txt"):
                continue
            filename = Path(member).stem + ".jpg"
            lines = archive.read(member).decode("utf-8").splitlines()
            caps = [strip_entity_markup(line) for line in lines if line.strip()]
            if len(caps) != 5:
                raise ValueError(f"Image has {len(caps)} source captions, expected five: {filename}")
            if filename in captions:
                raise ValueError(f"Duplicate caption image: {filename}")
            captions[filename] = caps
    selected_rows = {}
    for split, filenames in chosen.items():
        selected_rows[split] = {name: pools[selected_sources[split]][name] for name in filenames}
        for name in filenames:
            if name not in captions:
                raise ValueError(f"Missing source captions: {name}")
    write_json(data_dir / "selection.json", {
        "reconstruction": True, "seed": 42, "rng": "Python random.Random, sample without replacement",
        "selection_order": ["train1200", "dev200: first100 calibration, next100 validation", "test400"],
        "pool_sizes": {name: len(pool) for name, pool in pools.items()},
        "official_source_split": selected_sources, "images": chosen,
    })
    write_json(raw_dir / "selected_annotations.json", selected_rows)
    print("Selection complete:", {k: len(v) for k, v in chosen.items()}, flush=True)
    if args.annotations_only:
        write_json(data_dir / "sources.json", sources)
        return
    index = archive_index(raw_dir)
    filenames = [name for split in chosen.values() for name in split]
    missing = sorted(set(filenames) - set(index["images"]))
    if missing:
        raise ValueError(f"Selected images absent from public archive: {missing}")
    image_receipts = {}
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(acquire_image, name, index["images"][name], image_dir, repo): name for name in filenames}
        for i, task in enumerate(futures.as_completed(tasks), start=1):
            filename = tasks[task]
            image_receipts[filename] = task.result()
            if i % 100 == 0:
                print(f"Verified {i}/{len(filenames)} images", flush=True)
    images, texts, pairs = [], [], []
    stats = {}
    for split, filenames in chosen.items():
        counts = {"source": 0, "supported": 0, "contradicted": 0, "neutral": 0}
        for filename in filenames:
            stem = Path(filename).stem
            image_id = f"flickr:{stem}"
            images.append({"id": image_id, "path": str((image_dir / filename).relative_to(repo)), "split": split})
            for c, caption in enumerate(captions[filename]):
                text_id = f"source:{stem}:{c}"
                texts.append({"id": text_id, "text": caption})
                pairs.append({"image_id": image_id, "text_id": text_id, "relation": "source"})
                counts["source"] += 1
            for row in selected_rows[split][filename]:
                text_id = f"hypothesis:{selected_sources[split]}:{row['pairID']}"
                texts.append({"id": text_id, "text": row["hypothesis"]})
                relation = LABELS[row["gold_label"]]
                pairs.append({"image_id": image_id, "text_id": text_id, "relation": relation})
                counts[relation] += 1
        stats[split] = {"images": len(filenames), "pairs": counts, "texts": sum(counts.values())}
    if len({row["id"] for row in texts}) != len(texts):
        raise ValueError("Text IDs are not unique")
    manifest = {"schema_version": 1, "dataset": "visual_entailment", "images": images,
                "texts": texts, "pairs": pairs, "triplets": []}
    write_json(data_dir / "manifest.json", manifest)
    write_json(data_dir / "provenance.json", {
        "reconstruction": "New seed-42 sample; historical exact image IDs were not recoverable.",
        "annotation_repository": "https://github.com/maximek3/e-ViL",
        "annotation_revision": ANNOTATION_REVISION,
        "image_mirror": "https://huggingface.co/datasets/nlphuji/flickr30k",
        "image_revision": FLICKR_REVISION,
        "source_caption_repository": "https://github.com/BryanPlummer/flickr30k_entities",
        "source_caption_revision": CAPTION_REVISION,
        "sources": sources, "selection": json.loads((data_dir / "selection.json").read_text()),
        "zip_url": ZIP_URL, "zip_size_bytes": EXPECTED_ZIP_SIZE, "zip_etag": EXPECTED_ZIP_ETAG,
        "archive_validation": "Every selected member checked against ZIP CRC32, size, JPEG signature, Pillow.verify(), and full image decoding with Pillow.load().",
        "images": {name: image_receipts[name] for name in sorted(image_receipts)},
        "statistics": stats, "manifest_sha256": sha256(data_dir / "manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "retention_policy": "All rows from the chosen official split retained, in source order; no label-based filtering or silent image skipping.",
        "annotation_release": "Kayser et al., ICCV 2021 official e-SNLI-VE CSV release. Deprecated 2020 release was audited and rejected before model evaluation because it has dev/test overlap.",
        "source_caption_order": "The five lines of each official Entities Sentences/ID.txt in stored order. Entity markup removed; source words, case and punctuation retained, with single spaces between stored tokens.",
    })
    print(json.dumps({"status": "complete", "manifest_sha256": sha256(data_dir / "manifest.json"), "statistics": stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
