#!/usr/bin/env python3
"""Verify and restore the exact stage14 prediction bundles (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = "results/review_followup/evaluation/predictions/"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError(f"Unsafe archive path: {name}")
    if not name.startswith(PREFIX) or path.suffix not in (".npz", ".json"):
        raise ValueError(f"Unexpected prediction path: {name}")
    return path


def target_at(root, name):
    target = root.joinpath(*safe_relative(name).parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes root through a symlink: {name}")
    return target


def check(content, row, label):
    if len(content) != row["size"] or sha256(content) != row["sha256"]:
        raise ValueError(f"Size/SHA256 mismatch: {label}")


def archive_at(directory, row):
    name = Path(row["path"]).name
    candidates = (directory / name, directory / ("checkpoint-14-" + name))
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"Missing bundle: {name} in {directory}")
    return found[0]


def verify_archives(manifest, directory):
    if manifest["pair_count"] != 1080 or manifest["file_count"] != 2160:
        raise ValueError("Unexpected stage14 inventory counts")
    rows = manifest["files"]
    files = {row["path"]: row for row in rows}
    if len(files) != len(rows) or len(files) != manifest["file_count"]:
        raise ValueError("Duplicate or missing file inventory entries")
    if sum(row["size"] for row in rows) != manifest["raw_bytes"]:
        raise ValueError("Global raw-byte count differs")
    for name in files:
        safe_relative(name)
        partner = str(PurePosixPath(name).with_suffix(".json" if name.endswith(".npz") else ".npz"))
        if partner not in files or files[partner]["archive"] != files[name]["archive"]:
            raise ValueError(f"Prediction/metadata pairing differs: {name}")
    archives = manifest["archives"]
    if len(archives) != manifest["part_count"] or {row["part_number"] for row in archives} != set(range(1, manifest["part_count"] + 1)):
        raise ValueError("Missing or duplicate part inventory")
    verified, seen = [], set()
    for record in archives:
        path = archive_at(directory, record)
        check(path.read_bytes(), record, path)
        expected = {name: {k: row[k] for k in ("path", "size", "sha256")}
                    for name, row in files.items() if row["archive"] == record["path"]}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or set(names) != set(expected) | {"BUNDLE_MANIFEST.json"}:
                raise ValueError(f"Unlisted, duplicate or missing archive entries: {path}")
            bundle_bytes = archive.read("BUNDLE_MANIFEST.json")
            if sha256(bundle_bytes) != record["manifest_sha256"]:
                raise ValueError(f"Bundle manifest SHA256 mismatch: {path}")
            bundle = json.loads(bundle_bytes)
            for key in ("protocol_sha256", "source_index_and_prescore_hashes", "part_count"):
                if bundle[key] != manifest[key]:
                    raise ValueError(f"Bundle provenance mismatch ({key}): {path}")
            if bundle["part_number"] != record["part_number"] or bundle["file_count"] != len(expected):
                raise ValueError(f"Bundle part/file count mismatch: {path}")
            actual = {row["path"]: row for row in bundle["files"]}
            if actual != expected or len(bundle["files"]) != len(expected):
                raise ValueError(f"Bundle/global file inventories differ: {path}")
            for name, row in expected.items():
                check(archive.read(name), row, name)  # ZipFile also verifies CRC.
        seen.update(expected)
        verified.append((path, expected))
    if seen != set(files):
        raise ValueError("Not every listed raw file is covered by a bundle")
    return verified


def restore_file(target, content):
    """Publish an already verified file atomically; never replace an existing path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".stage14-", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.chmod(temporary, 0o644)
        os.link(temporary, target)  # Fails if another writer created the destination.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository destination; default is this script's repository.")
    parser.add_argument("--archives-dir", type=Path, default=HERE, help="Bundle directory; also accepts checkpoint-14-* download names.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify all archive bytes and members without restoring files.")
    mode.add_argument("--verify-restored", action="store_true", help="Also require all destination files to exist and match; write nothing.")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((HERE / "PREDICTION_ARCHIVES.json").read_text())
    archives = verify_archives(manifest, args.archives_dir.resolve())
    existing = missing = restored = 0
    if not args.verify_only:
        # Check every existing destination before making any writes.
        for _, rows in archives:
            for name, row in rows.items():
                target = target_at(root, name)
                if target.exists():
                    check(target.read_bytes(), row, target)
                    existing += 1
                else:
                    missing += 1
        if args.verify_restored and missing:
            raise FileNotFoundError(f"{missing} expected destination files are missing")
        if not args.verify_restored:
            for path, rows in archives:
                with zipfile.ZipFile(path) as archive:
                    for name, row in rows.items():
                        target = target_at(root, name)
                        if not target.exists():
                            content = archive.read(name)
                            check(content, row, name)
                            restore_file(target, content)
                            restored += 1
                        check(target.read_bytes(), row, target)
    print(json.dumps({"status": "verified", "archive_count": len(archives),
                      "prediction_pairs": manifest["pair_count"], "payload_files": manifest["file_count"],
                      "raw_bytes": manifest["raw_bytes"], "archive_bytes": manifest["archive_bytes"],
                      "existing_files_verified": existing, "files_restored": restored,
                      "destination": str(root), "archive_only": args.verify_only,
                      "protocol_sha256": manifest["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
