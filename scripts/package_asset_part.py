#!/usr/bin/env python3
"""Create one immutable, exact-byte recovery ZIP for a project asset.

Example:
  python scripts/package_asset_part.py --file results/features/example/features.npz \
      --part-index 0 --chunk-mib 32 --output ../output/example-part-000.zip

Each ZIP contains one chunk, the full asset descriptor, and standalone
restore_assets.py. Extract all parts into one directory with project/ from the
code ZIP, then run python restore_assets.py. Originals are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from restore_assets import FORMAT, MAX_PART_BYTES, make_asset_id, safe_relative

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
DENIED_PARTS = {".git", ".aws", ".ssh", ".gnupg", ".azure", ".config", ".venv", "__pycache__", "credentials", "secrets"}
DENIED_NAMES = {".env", ".envrc", ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "token", "token.json", "tokens.json", "id_rsa", "id_ed25519", "service-account.json", "service_account.json", "api_key.json", "api_keys.json", "hf_token"}
README = """EXACT ASSET RECOVERY

1. Extract the code ZIP and every asset-part ZIP into the SAME directory.
   Keep project/, ASSET_DESCRIPTORS/, and ASSET_PARTS/ together.
2. Run: python restore_assets.py --root .
3. Exit status 0 and complete:true mean every selected asset is restored and
   verified. Exit status 2 means parts/descriptors are missing. Exit status 1
   means corrupt input or an unresolved conflict. The JSON report names them.
4. Existing different project files are preserved. After deciding which copy
   to retain, --overwrite explicitly permits verified replacement. If several
   versions target one file, select one using --asset-id ID.

Each descriptor specifies every required part, its exact byte count and SHA256,
and the original file's exact byte count and SHA256. File bytes are concatenated
without reserialization, so NPZ container bytes and checkpoint bytes survive
exactly. Extraction may repeat identical descriptor/helper files across ZIPs.
Parts alone are incomplete unless all descriptor-listed chunks are available.
"""


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def selected_file(root: Path, relative: str) -> tuple[str, Path]:
    canonical = safe_relative(relative).as_posix()
    parts = canonical.split("/")
    lower = [part.lower() for part in parts]
    if any(part in DENIED_PARTS for part in lower) or lower[-1] in DENIED_NAMES:
        raise ValueError(f"Credential or runtime paths cannot be packaged: {relative}")
    if any(part.startswith(".env.") or part.endswith((".pem", ".key", ".p12", ".pfx")) for part in lower):
        raise ValueError(f"Credential-like paths cannot be packaged: {relative}")
    if any(part.split(".", 1)[0] in {"credentials", "secret", "secrets", "token", "tokens", "api_key", "api_keys"} for part in lower):
        raise ValueError(f"Credential-like paths cannot be packaged: {relative}")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Selected source escapes the project root: {relative}")
    if not path.is_file():
        raise ValueError(f"Selected source is not a regular file: {relative}")
    return canonical, path


def fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def describe_asset(root: Path, relative: str, chunk_bytes: int) -> tuple[dict, Path, tuple]:
    relative, path = selected_file(root, relative)
    if not 0 < chunk_bytes <= MAX_PART_BYTES:
        raise ValueError("Chunk size must be positive and at most 35 MiB")
    before = fingerprint(path)
    full = hashlib.sha256()
    chunks = []
    total = 0
    with path.open("rb") as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            full.update(block)
            total += len(block)
            chunks.append({"index": len(chunks), "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    if not chunks:
        chunks = [{"index": 0, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}]
    if before != fingerprint(path) or total != before[2]:
        raise ValueError("Source changed while hashing; no part was created")
    asset_id = make_asset_id(relative, total, full.hexdigest(), chunk_bytes)
    for chunk in chunks:
        chunk["path"] = f"ASSET_PARTS/{asset_id}/{chunk['index']:06d}.bin"
    descriptor = {"format": FORMAT, "schema_version": 1, "asset_id": asset_id,
                  "relative_target": relative, "bytes": total, "sha256": full.hexdigest(),
                  "chunk_bytes": chunk_bytes, "chunks": chunks}
    return descriptor, path, before


def zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.create_system = 3
    member.external_attr = 0o100644 << 16
    member.compress_type = zipfile.ZIP_STORED
    archive.writestr(member, data)


def package_part(root: Path, relative: str, part_index: int, chunk_bytes: int, output: Path, includes: list[str] | None = None) -> dict:
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}")
    descriptor, source_path, source_fingerprint = describe_asset(root, relative, chunk_bytes)
    if type(part_index) is not int or not 0 <= part_index < len(descriptor["chunks"]):
        raise ValueError(f"part-index must be in 0..{len(descriptor['chunks']) - 1}")
    chunk = descriptor["chunks"][part_index]
    with source_path.open("rb") as source:
        source.seek(part_index * chunk_bytes)
        payload = source.read(chunk["bytes"])
    if fingerprint(source_path) != source_fingerprint or hashlib.sha256(payload).hexdigest() != chunk["sha256"] or len(payload) != chunk["bytes"]:
        raise ValueError("Source changed between hashing and packaging; no part was created")
    helper_path = Path(__file__).with_name("restore_assets.py")
    members = [("restore_assets.py", helper_path.read_bytes()), ("ASSET_PART_README.txt", README.encode("utf-8")),
               (f"ASSET_DESCRIPTORS/{descriptor['asset_id']}.json", json_bytes(descriptor)), (chunk["path"], payload)]
    included = []
    seen = set()
    for item in includes or []:
        relative_include, path = selected_file(root, item)
        if relative_include == descriptor["relative_target"]:
            raise ValueError("The chunked asset cannot also be included as a whole file")
        if relative_include in seen:
            continue
        seen.add(relative_include)
        if path.stat().st_size > MAX_PART_BYTES:
            raise ValueError(f"Optional included file is too large: {item}")
        before = fingerprint(path)
        data = path.read_bytes()
        if before != fingerprint(path) or len(data) != before[2]:
            raise ValueError(f"Included file changed while reading: {item}")
        archive_path = "project/" + relative_include
        members.append((archive_path, data))
        included.append({"path": archive_path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {"schema_version": 1, "asset_id": descriptor["asset_id"], "part_index": part_index,
               "part_count": len(descriptor["chunks"]), "chunk": chunk, "whole_file_bytes": descriptor["bytes"],
               "whole_file_sha256": descriptor["sha256"], "relative_target": descriptor["relative_target"],
               "included_files": included, "descriptor_sha256": hashlib.sha256(json_bytes(descriptor)).hexdigest()}
    members.append((f"ASSET_PART_RECEIPTS/{descriptor['asset_id']}/{part_index:06d}.json", json_bytes(receipt)))
    raw_bytes = sum(len(data) for _, data in members)
    if raw_bytes > MAX_PART_BYTES:
        raise ValueError(f"ZIP raw payload is {raw_bytes} bytes, exceeding 35 MiB. Reduce chunk-mib or optional includes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".asset-part-", suffix=".zip", dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, data in members:
                zip_member(archive, name, data)
        if temporary.stat().st_size > MAX_PART_BYTES:
            raise ValueError("ZIP including headers exceeds 35 MiB. Reduce chunk-mib or optional includes.")
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP integrity check failed for {bad}")
        archive_bytes = temporary.read_bytes()
        receipt.update({"output": str(output), "zip_bytes": len(archive_bytes), "zip_sha256": hashlib.sha256(archive_bytes).hexdigest(), "raw_payload_bytes": raw_bytes})
        if fingerprint(source_path) != source_fingerprint:
            raise ValueError("Source changed before ZIP publication; no part was created")
        # Hard-link publication is atomic and fails if output appeared meanwhile.
        # The temporary file and output share one directory/filesystem.
        os.link(temporary, output)
        return receipt
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Asset path relative to the project root.")
    parser.add_argument("--part-index", required=True, type=int, help="Zero-based chunk index.")
    parser.add_argument("--chunk-mib", type=int, default=32, help="Exact chunk size in MiB; default 32.")
    parser.add_argument("--output", required=True, type=Path, help="New immutable ZIP path; must not already exist.")
    parser.add_argument("--include", action="append", default=[], help="Optional small project-relative file copied into this part; repeat for multiple files.")
    args = parser.parse_args()
    try:
        receipt = package_part(PROJECT_ROOT, args.file, args.part_index, args.chunk_mib * MIB, args.output, args.include)
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Asset part was not created: {error}\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
