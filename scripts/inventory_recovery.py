#!/usr/bin/env python3
"""Inventory every project file and explain exact final-archive exclusions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".cache", ".git", ".venv", "pytest-of-root"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def exclusion_reason(relative: Path, bank_redundant: bool) -> str | None:
    parts = relative.parts
    if any(part in CACHE_DIRS for part in parts):
        return "Runtime cache, bytecode, package cache, or temporary test output"
    if parts[:2] in (("manuscript", "build"), ("manuscript", "tex-runtime")):
        return "Rebuildable TeX compilation/cache files; manuscript sources and delivered PDFs are retained"
    if parts and re.fullmatch(r"tmp[a-zA-Z0-9_]{6,}", parts[0]):
        return "Temporary test/work directory at the project root"
    if relative.name.endswith((".part", ".tmp", ".incomplete")):
        return "Incomplete download or temporary write, superseded by the verified complete file"
    if parts[:2] == ("data", "model") and relative.name.startswith(".open_clip_model.safetensors."):
        return "Incomplete model download; complete pinned open_clip_model.safetensors is retained"
    if bank_redundant and relative.as_posix() in {"results/features/bank.sqlite", "results/features/bank.sqlite-wal", "results/features/bank.sqlite-shm"}:
        return "Duplicate feature runtime cache: every bank key/vector is preserved byte-for-byte in retained NPZs; see results/feature_bank_redundancy.json"
    return None


def inventory(root: Path, output: Path) -> dict:
    root = root.resolve()
    proof_path = root / "results/feature_bank_redundancy.json"
    bank = root / "results/features/bank.sqlite"
    bank_redundant = False
    if proof_path.exists() and bank.exists():
        proof = json.loads(proof_path.read_text())
        bank_redundant = bool(proof.get("fully_redundant") and proof.get("bank_sha256") == digest(bank))
    eligible = []
    exclusions = []
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path in sorted(root.rglob("*")):
        if path.resolve() == output.resolve() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            exclusions.append({"path": relative.as_posix(), "bytes": 0, "reason": "Symbolic link resolves outside this project; external files are not part of the project archive"})
            continue
        size = path.stat().st_size
        reason = exclusion_reason(relative, bank_redundant)
        row = {"path": relative.as_posix(), "bytes": size}
        if reason:
            exclusions.append(dict(row, reason=reason))
        else:
            eligible.append(row)
            key = "/".join(relative.parts[:2])
            groups[key]["files"] += 1
            groups[key]["bytes"] += size
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Current project files. Rerun this inventory after scientific outputs are finalized. The inventory file itself is included in the final archive but omitted from its own recursive byte totals.",
              "retention": "Keep every raw annotation, actual selected image, complete pinned model weight/config, exact feature NPZ, all 108 candidate best states, all 36 selected states, item-level results, historical context, code, manuscript source, and delivered PDF.",
              "feature_bank_exclusion_verified": bank_redundant,
              "eligible_file_count": len(eligible), "eligible_bytes": sum(row["bytes"] for row in eligible),
              "excluded_file_count": len(exclusions), "excluded_bytes": sum(row["bytes"] for row in exclusions),
              "eligible_groups": dict(sorted(groups.items(), key=lambda item: -item[1]["bytes"])),
              "eligible_files": eligible, "excluded_files": exclusions}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/archive_inventory.json")
    args = parser.parse_args()
    result = inventory(ROOT, args.output)
    print(json.dumps({key: result[key] for key in ["eligible_file_count", "eligible_bytes", "excluded_file_count", "excluded_bytes", "feature_bank_exclusion_verified"]}, indent=2))


if __name__ == "__main__":
    main()
