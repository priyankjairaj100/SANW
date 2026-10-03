#!/usr/bin/env python3
"""Restore exact project files from independently delivered asset ZIP parts.

Extract every received ZIP into one common directory, alongside the project/
directory from the code ZIP, then run: python restore_assets.py --root .
Missing parts produce an incomplete report and exit status 2. Invalid or corrupt
input and unresolved conflicts produce exit status 1. Only complete recovery
produces exit status 0. Existing different files require --overwrite.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

FORMAT = "gcr-exact-asset-parts-v1"
MAX_PART_BYTES = 35 * 1024 * 1024
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def contained_path(base: Path, relative: str) -> Path:
    path = base.joinpath(*safe_relative(relative).parts)
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes the recovery directory: {relative}")
    return path


def make_asset_id(target: str, size: int, digest: str, chunk_bytes: int) -> str:
    identity = json.dumps([FORMAT, target, size, digest, chunk_bytes], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def validate_descriptor(value: dict, filename: str) -> dict:
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("schema_version") != 1:
        raise ValueError("Unsupported asset descriptor")
    asset_id = value.get("asset_id", "")
    if not isinstance(asset_id, str) or not HEX_SHA.fullmatch(asset_id) or filename != asset_id + ".json":
        raise ValueError("Descriptor filename and asset ID do not agree")
    target = value.get("relative_target")
    safe_relative(target)
    size, digest, chunk_bytes = value.get("bytes"), value.get("sha256"), value.get("chunk_bytes")
    if type(size) is not int or size < 0 or type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_PART_BYTES:
        raise ValueError("Invalid asset byte counts")
    if not isinstance(digest, str) or not HEX_SHA.fullmatch(digest):
        raise ValueError("Invalid whole-file SHA256")
    if make_asset_id(target, size, digest, chunk_bytes) != asset_id:
        raise ValueError("Descriptor identity does not match its contents")
    chunks = value.get("chunks")
    expected_count = max(1, (size + chunk_bytes - 1) // chunk_bytes)
    if not isinstance(chunks, list) or len(chunks) != expected_count:
        raise ValueError("Wrong number of chunk descriptors")
    total = 0
    for index, chunk in enumerate(chunks):
        expected_size = min(chunk_bytes, max(0, size - index * chunk_bytes))
        if not isinstance(chunk, dict) or chunk.get("index") != index or type(chunk.get("bytes")) is not int:
            raise ValueError("Invalid chunk index or size")
        if chunk["bytes"] != expected_size or chunk.get("path") != f"ASSET_PARTS/{asset_id}/{index:06d}.bin":
            raise ValueError("Chunk path or byte count does not match its position")
        if not isinstance(chunk.get("sha256"), str) or not HEX_SHA.fullmatch(chunk["sha256"]):
            raise ValueError("Invalid chunk SHA256")
        total += chunk["bytes"]
    if total != size:
        raise ValueError("Chunk sizes do not sum to whole-file size")
    return value


def restore_one(root: Path, descriptor: dict, overwrite: bool) -> dict:
    asset_id = descriptor["asset_id"]
    result = {"asset_id": asset_id, "relative_target": descriptor["relative_target"], "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]}
    temporary_path = None
    try:
        project = root / "project"
        if not project.resolve().is_relative_to(root.resolve()):
            raise ValueError("The project directory points outside the recovery directory")
        target = contained_path(project, descriptor["relative_target"])
        if target.is_symlink():
            raise ValueError("Refusing to replace a symbolic-link target")
        if target.exists():
            if not target.is_file():
                raise ValueError("The asset target exists and is not a regular file")
            if hash_file(target) == (descriptor["bytes"], descriptor["sha256"]):
                return dict(result, status="already_present")
            if not overwrite:
                return dict(result, status="existing_conflict", message="Existing target differs. Preserve it or explicitly rerun with --overwrite.")
        missing = []
        chunk_paths = []
        for chunk in descriptor["chunks"]:
            path = contained_path(root, chunk["path"])
            if not path.exists():
                missing.append(chunk["index"])
            elif not path.is_file():
                raise ValueError(f"Chunk is not a regular file: {chunk['path']}")
            chunk_paths.append(path)
        if missing:
            return dict(result, status="incomplete", missing_part_indices=missing, required_parts=len(descriptor["chunks"]))
        # Verify every part before opening a replacement, then hash again while
        # copying to catch accidental mutation between verification and use.
        for path, chunk in zip(chunk_paths, descriptor["chunks"]):
            if hash_file(path) != (chunk["bytes"], chunk["sha256"]):
                return dict(result, status="corrupt_chunk", part_index=chunk["index"], message="Chunk byte count or SHA256 does not match.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target = contained_path(project, descriptor["relative_target"])
        whole = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(prefix=".restore-asset-", dir=target.parent, delete=False) as output:
            temporary_path = Path(output.name)
            for path, chunk in zip(chunk_paths, descriptor["chunks"]):
                piece = hashlib.sha256()
                piece_size = 0
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        piece.update(block)
                        whole.update(block)
                        piece_size += len(block)
                        total += len(block)
                if (piece_size, piece.hexdigest()) != (chunk["bytes"], chunk["sha256"]):
                    return dict(result, status="corrupt_chunk", part_index=chunk["index"], message="Chunk changed during restoration.")
            output.flush()
            os.fsync(output.fileno())
        if (total, whole.hexdigest()) != (descriptor["bytes"], descriptor["sha256"]):
            return dict(result, status="whole_hash_mismatch", message="Assembled file does not match the descriptor.")
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError("The target changed into a nonregular file during restoration")
            if hash_file(target) == (descriptor["bytes"], descriptor["sha256"]):
                return dict(result, status="already_present")
            if not overwrite:
                return dict(result, status="existing_conflict", message="A different target appeared during restoration; it was preserved.")
        # Publication happens only after all chunk and whole-file checks. The
        # no-overwrite route remains safe if a target appears after our check.
        if overwrite:
            os.replace(temporary_path, target)
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                if target.is_file() and not target.is_symlink() and hash_file(target) == (descriptor["bytes"], descriptor["sha256"]):
                    return dict(result, status="already_present")
                return dict(result, status="existing_conflict", message="A different target appeared during publication; it was preserved.")
            temporary_path.unlink()
        temporary_path = None
        return dict(result, status="restored")
    except (OSError, ValueError, TypeError) as error:
        return dict(result, status="error", message=str(error))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def restore_assets(root: Path, selected_ids: list[str] | None = None, overwrite: bool = False) -> tuple[dict, int]:
    root = root.resolve()
    requested = set(selected_ids or [])
    results = []
    descriptors = []
    found = set()
    directory = root / "ASSET_DESCRIPTORS"
    if directory.exists() and not directory.resolve().is_relative_to(root):
        return {"complete": False, "assets": [], "errors": ["Descriptor directory escapes recovery root"]}, 1
    for path in sorted(directory.glob("*.json")):
        if requested and path.stem not in requested:
            continue
        found.add(path.stem)
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Descriptor resolves outside recovery root")
            descriptor = validate_descriptor(json.loads(path.read_text(encoding="utf-8")), path.name)
            descriptors.append(descriptor)
        except (OSError, ValueError, TypeError, KeyError) as error:
            results.append({"asset_id": path.stem, "status": "invalid_descriptor", "message": str(error)})
    for asset_id in sorted(requested - found):
        results.append({"asset_id": asset_id, "status": "missing_descriptor"})
    targets = defaultdict(list)
    for descriptor in descriptors:
        targets[descriptor["relative_target"]].append(descriptor)
    for target, versions in sorted(targets.items()):
        if len({(row["bytes"], row["sha256"]) for row in versions}) > 1:
            results.extend({"asset_id": row["asset_id"], "relative_target": target, "status": "conflicting_descriptors",
                            "message": "Multiple versions target the same file. Select the intended version with --asset-id."} for row in versions)
        else:
            # Different chunk sizes may describe the same exact file. Choose
            # one with all parts present, then report the redundant descriptor.
            versions.sort(key=lambda row: sum(not (root / part["path"]).is_file() for part in row["chunks"]))
            for descriptor in versions:
                results.append(restore_one(root, descriptor, overwrite))
    successful = {"restored", "already_present"}
    complete = bool(results) and all(row["status"] in successful for row in results)
    report = {"schema_version": 1, "complete": complete, "assets": results,
              "counts": {status: sum(row["status"] == status for row in results) for status in sorted({row["status"] for row in results})}}
    if not results:
        report["message"] = "No asset descriptors were found. Extract the received asset ZIPs into this directory."
    hard_errors = any(row["status"] not in successful | {"incomplete", "missing_descriptor"} for row in results)
    return report, 0 if complete else (1 if hard_errors else 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Common directory containing project/, ASSET_DESCRIPTORS/, and ASSET_PARTS/.")
    parser.add_argument("--asset-id", action="append", help="Restore only this asset ID; repeat to select multiple assets.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacement of a different existing target after complete verification.")
    args = parser.parse_args()
    report, status = restore_assets(args.root, args.asset_id, args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
