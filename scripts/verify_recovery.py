#!/usr/bin/env python3
"""Verify a recovery ZIP or extracted files using only Python's standard library.

Accepted manifests have files:[{path,bytes,sha256}]. The final archive uses
RECOVERY_MANIFEST.json; incremental code ZIPs use CHECKPOINT_MANIFEST.json.
Every listed file is streamed and checked. Unexpected files fail verification
unless --allow-unlisted is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in value.split("/")):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path.as_posix()


def stream_hash(source) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def validate_manifest(value: dict) -> dict[str, dict]:
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError("Manifest must contain a files list")
    files = {}
    for row in value["files"]:
        if not isinstance(row, dict):
            raise ValueError("Each manifest file must be an object")
        name = safe_name(row.get("path"))
        if name in files:
            raise ValueError(f"Duplicate manifest path: {name}")
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError(f"Invalid manifest byte count: {name}")
        if not isinstance(row.get("sha256"), str) or not HEX_SHA.fullmatch(row["sha256"]):
            raise ValueError(f"Invalid SHA256: {name}")
        files[name] = row
    if not files:
        raise ValueError("Manifest contains no payload files")
    return files


def choose_manifest(names: set[str], requested: str | None) -> str:
    if requested:
        name = safe_name(requested)
        if name not in names:
            raise ValueError(f"Requested manifest is missing: {name}")
        return name
    for name in ("RECOVERY_MANIFEST.json", "CHECKPOINT_MANIFEST.json"):
        if name in names:
            return name
    raise ValueError("No RECOVERY_MANIFEST.json or CHECKPOINT_MANIFEST.json was found")


def verify(archive_path: Path | None = None, root: Path | None = None,
           manifest_name: str | None = None, allow_unlisted: bool = False) -> tuple[dict, int]:
    archive = None
    try:
        if archive_path is not None:
            archive = zipfile.ZipFile(archive_path)
            members = [member for member in archive.infolist() if not member.is_dir()]
            counts = Counter(member.filename for member in members)
            duplicates = sorted(name for name, count in counts.items() if count > 1)
            if duplicates:
                raise ValueError(f"Archive has duplicate member names: {duplicates}")
            names = {safe_name(member.filename) for member in members}
            selected_manifest = choose_manifest(names, manifest_name)
            manifest = json.loads(archive.read(selected_manifest))
            open_member = archive.open
        else:
            root = (root or Path.cwd()).resolve()
            names = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            selected_manifest = choose_manifest(names, manifest_name)
            manifest_path = root / selected_manifest
            if not manifest_path.resolve().is_relative_to(root):
                raise ValueError("Manifest escapes the recovery root")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            def open_member(name):
                path = root / safe_name(name)
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"File escapes recovery root: {name}")
                return path.open("rb")

        expected = validate_manifest(manifest)
        if selected_manifest in expected:
            raise ValueError("Manifest cannot include its own hash")
        permitted_metadata = {selected_manifest}
        if selected_manifest == "CHECKPOINT_MANIFEST.json":
            permitted_metadata.add("RESTORE.txt")
        unlisted = sorted(names - set(expected) - permitted_metadata)
        missing = sorted(set(expected) - names)
        failures = []
        checked_files = 0
        checked_bytes = 0
        # ZIP readers validate each member CRC when the stream reaches EOF.
        # Stream unlisted ZIP members too, so CRC coverage spans the archive.
        to_check = sorted((names - {selected_manifest}) if archive else (set(expected) & names))
        for name in to_check:
            try:
                with open_member(name) as source:
                    size, digest = stream_hash(source)
                if name in expected:
                    wanted = expected[name]
                    if (size, digest) != (wanted["bytes"], wanted["sha256"]):
                        failures.append({"path": name, "kind": "content_mismatch", "actual_bytes": size, "actual_sha256": digest})
                    checked_files += 1
                    checked_bytes += size
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
                failures.append({"path": name, "kind": "read_failure", "message": str(error)})
        complete = not missing and not failures and (allow_unlisted or not unlisted)
        report = {"schema_version": 1, "complete": complete, "manifest": selected_manifest,
                  "scope": "All manifest-listed files; unlisted files permitted explicitly" if allow_unlisted else "All archive payload files must be listed and verified",
                  "expected_files": len(expected), "checked_files": checked_files, "checked_bytes": checked_bytes,
                  "missing_files": missing, "failures": failures, "unlisted_files": unlisted}
        if archive is not None:
            report["zip_crc_verified"] = not any(row["kind"] == "read_failure" for row in failures)
            report["archive_bytes"] = archive_path.stat().st_size
        return report, 0 if complete else 1
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        return {"schema_version": 1, "complete": False, "error": str(error)}, 1
    finally:
        if archive is not None:
            archive.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Verify directly inside one ZIP without extracting it.")
    source.add_argument("--root", type=Path, help="Verify an extracted directory; defaults to the current directory.")
    parser.add_argument("--manifest", help="Relative manifest path; defaults to RECOVERY_MANIFEST.json, then CHECKPOINT_MANIFEST.json.")
    parser.add_argument("--allow-unlisted", action="store_true", help="Explicitly verify only listed files, allowing additional restored assets or later checkpoint files.")
    args = parser.parse_args()
    report, status = verify(args.archive, args.root, args.manifest, args.allow_unlisted)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
