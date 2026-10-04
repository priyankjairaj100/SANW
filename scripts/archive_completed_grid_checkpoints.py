#!/usr/bin/env python3
"""Lossless archival of completed, unreferenced epoch weights.

Archive never removes originals. Prune is a separate, explicit operation after
successful private uploads. No scientific modules are imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

MAX_PART_BYTES = 128 * 1024 * 1024
EPOCH = re.compile(r"epochs/epoch_\d{2,}\.pt\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(root, name, *, exists=True):
    require(isinstance(name, str) and name, "Missing relative path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and
            "\\" not in name and str(relative) == name, f"Unsafe path: {name}")
    path = root.joinpath(*relative.parts)
    for node in [path, *path.parents]:
        if node == root:
            break
        require(not node.is_symlink(), f"Symlink refused: {node}")
    require(path.resolve().is_relative_to(root), f"Path escapes root: {name}")
    if exists:
        require(path.is_file(), f"Missing regular file: {name}")
    return path


def relative_path(root, path):
    return path.relative_to(root).as_posix()


def record(root, path, expected=None):
    name = relative_path(root, path)
    safe_path(root, name)
    actual = digest(path)
    require(expected is None or actual == expected, f"SHA256 mismatch: {name}")
    return {"path": name, "bytes": path.stat().st_size, "sha256": actual}


def read_json(path):
    return json.loads(path.read_text())


def exact_records(records):
    result = {}
    for item in records:
        name = item["path"]
        require(name not in result or result[name] == item, f"Conflicting record: {name}")
        result[name] = item
    return [result[name] for name in sorted(result)]


def bind_reference(repository, reference, protected):
    item = record(repository, safe_path(repository, reference["path"]), reference["sha256"])
    require("bytes" not in reference or reference["bytes"] == item["bytes"], "Reference size mismatch")
    protected.append(item)
    return read_json(repository / item["path"])


def inspect_grid(repository, grid_name, keep_name, keep_sha256, expected_completions):
    """Validate the full grid before deciding which receipt-bound files qualify."""
    grid = safe_path(repository, grid_name, exists=False)
    require(grid.is_dir(), "Grid directory is missing")
    protected = []
    keep_record = record(repository, safe_path(repository, keep_name), keep_sha256)
    protected.append(keep_record)
    keep = read_json(repository / keep_name)
    require(keep_name == relative_path(repository, grid / "state_manifest.json") or "full_state_manifest" in keep,
            "A separate retained manifest must bind the full state manifest")
    ledger_path = grid / "ledger.json"
    protected.append(record(repository, ledger_path))
    ledger = read_json(ledger_path)
    ledger_hash = ledger["ledger_sha256"]
    require(hashlib.sha256(canonical(ledger["identity"])).hexdigest() == ledger_hash, "Ledger identity hash mismatch")
    require(keep["ledger_sha256"] == ledger_hash, "Retained manifest belongs to another ledger")
    protocol = bind_reference(repository, ledger["identity"]["protocol"], protected)
    training = protocol["training"]
    require(keep.get("protocol_sha256", ledger["identity"]["protocol"]["sha256"]) ==
            ledger["identity"]["protocol"]["sha256"], "Retained protocol hash mismatch")
    states = keep["states"]
    require(isinstance(states, list) and states, "Retained states must be a nonempty list")
    ids = [state["state_id"] for state in states]
    require(len(ids) == len(set(ids)), "Duplicate retained state IDs")
    retained = []
    for state in states:
        name, sha = state.get("checkpoint"), state.get("checkpoint_sha256")
        if name is None:
            require(sha is None, "Null checkpoint has a digest")
            continue
        require(isinstance(sha, str) and len(sha) == 64, "Missing checkpoint SHA256")
        retained.append(record(repository, safe_path(repository, name), sha))
    retained = exact_records(retained)
    protected.extend(retained)
    if "full_state_manifest" in keep:
        full = bind_reference(repository, keep["full_state_manifest"], protected)
        require(full["ledger_sha256"] == ledger_hash, "Full manifest ledger mismatch")
        full_ids = {state["state_id"]: state for state in full["states"]}
        require(len(full_ids) == len(full["states"]), "Duplicate full state IDs")
        for state in states:
            require(state["state_id"] in full_ids, "Retained state absent from full manifest")
            original = full_ids[state["state_id"]]
            require(all(state.get(key) == original.get(key) for key in ("checkpoint", "checkpoint_sha256")),
                    "Retained checkpoint differs from full manifest")
    else:
        full = keep
    selection_hashes = keep.get("selection_sha256", {})
    require(selection_hashes == full.get("selection_sha256", selection_hashes), "Selection binding differs from full manifest")
    for name, sha in selection_hashes.items():
        require(name in ("primary", "sensitivity"), "Unknown selection file")
        selection = bind_reference(repository, {"path": relative_path(repository, grid / f"selection_{name}.json"),
                                                "sha256": sha}, protected)
        require(selection["ledger_sha256"] == ledger_hash, "Selection ledger mismatch")
    # Selection references must be represented in the explicit retained manifest.
    for selection in full.get("selections", []) + full.get("decomposition_selections", []):
        require(selection["state_id"] in ids, "A selected/decomposition state is missing from retained states")

    epochs = training["epochs"]
    expected = {}
    for policy in training["policies"]:
        for rate in training["learning_rates"]:
            for seed in training["seeds"]:
                name = f"candidates/{policy}/lr_{rate:.8g}/seed_{seed}"
                expected[name] = {"method": policy, "learning_rate": rate, "seed": seed,
                                  "epochs_completed": epochs, "selection_sha256": None,
                                  "loss_policy": policy, "draw_id": None}
    control = None
    if "matched_controls" in training:
        control = ("allocation_distillation", "matched_allocation_distillation_draw_", training["matched_controls"]["fits_per_encoder"])
    elif "matched_distilled_control" in training:
        control = ("distilled", "matched_distilled_draw_", training["matched_distilled_control"]["additional_runs_per_encoder"])
    if control:
        require(set(selection_hashes) == {"primary", "sensitivity"}, "Matched controls require both selection hashes")
        require(full.get("matched_controls_complete") is True, "Matched controls are not marked complete")
        family, prefix, count = control
        chosen = read_json(grid / "selection_primary.json")["families"][family]
        require(sorted(run["seed"] for run in chosen["runs"]) == sorted(training["seeds"]), "Matched seed schedule mismatch")
        added = 0
        for run in chosen["runs"]:
            require(isinstance(run["epoch"], int) and 0 <= run["epoch"] <= epochs, "Invalid matched epoch")
            for draw in range(3):
                method, rate, seed = prefix + str(draw), chosen["learning_rate"], run["seed"]
                name = f"candidates/{method}/lr_{rate:.8g}/seed_{seed}"
                expected[name] = {"method": method, "learning_rate": rate, "seed": seed,
                    "epochs_completed": run["epoch"], "selection_sha256": selection_hashes["primary"],
                    "loss_policy": chosen["policy"], "draw_id": draw}
                added += 1
        require(added == count, "Protocol matched count mismatch")
    require(len(expected) == expected_completions, "Expected completion count differs from frozen grid")
    discovered = {relative_path(grid, p.parent) for p in (grid / "candidates").rglob("completion.json")}
    require(discovered == set(expected), f"Incomplete/unexpected candidates: missing={sorted(set(expected)-discovered)} extra={sorted(discovered-set(expected))}")
    artifacts = []
    bound_epoch_paths = set()
    for candidate_name, metadata in sorted(expected.items()):
        candidate = grid / candidate_name
        receipt_path = candidate / "completion.json"
        protected.append(record(repository, receipt_path))
        receipt = read_json(receipt_path)
        require(receipt["ledger_sha256"] == ledger_hash, f"Completion ledger mismatch: {candidate_name}")
        for key, value in metadata.items():
            # Replication receipts have a distinct condition/draw schema.
            if "checkpoint_sha256" in receipt and key in ("loss_policy", "selection_sha256", "draw_id"):
                continue
            require(receipt.get(key) == value, f"Completion {key} mismatch: {candidate_name}")
        n = metadata["epochs_completed"]
        weights = {f"epochs/epoch_{i:02d}.pt" for i in range(n + 1)}
        validations = {f"validation/epoch_{i:02d}.npz" for i in range(n + 1)}
        hashes = receipt["artifact_sha256"]
        if "checkpoint_sha256" in receipt:
            require(set(receipt["checkpoint_sha256"]) == weights and receipt["checkpoint_digest_count"] == n + 1,
                    "Incomplete replication checkpoint digest ledger")
            required_weights = set(receipt["retained_checkpoints"])
            require(required_weights <= weights and required_weights, "Invalid replication retained weights")
            require(all(hashes.get(name) == receipt["checkpoint_sha256"][name] for name in required_weights),
                    "Replication digest mismatch")
        else:
            require(receipt["checkpoint_count"] == n + 1, "Incomplete checkpoint count")
            required_weights = weights
        require(set(hashes) == validations | required_weights | {"history.json"}, f"Unexpected/incomplete artifact set: {candidate_name}")
        for name, sha in sorted(hashes.items()):
            path = safe_path(candidate, name)
            item = record(repository, path, sha)
            artifacts.append(item)
            if EPOCH.fullmatch(name):
                bound_epoch_paths.add(item["path"])
            else:
                protected.append(item)
    present_weights = {relative_path(repository, path) for path in (grid / "candidates").rglob("*.pt")}
    require(present_weights == bound_epoch_paths, "Unbound or missing epoch checkpoint files")
    retained_paths = {item["path"] for item in retained}
    eligible = [item for item in artifacts if item["path"] in bound_epoch_paths - retained_paths]
    return {"grid_root": grid_name, "retained_manifest": keep_record, "expected_completions": expected_completions,
            "ledger_sha256": ledger_hash, "retained_checkpoints": retained,
            "protected_files": exact_records(protected), "members": exact_records(eligible)}


def manifest_bytes(members):
    return canonical({"schema_version": 1, "path_base": "repository", "members": members}) + b"\n"


def partition(members, limit):
    groups, current = [], []
    for item in members:
        proposed = current + [item]
        if sum(row["bytes"] for row in proposed) + len(manifest_bytes(proposed)) > limit:
            require(current, f"Single member exceeds part limit: {item['path']}")
            groups.append(current)
            current = [item]
        else:
            current = proposed
        require(sum(row["bytes"] for row in current) + len(manifest_bytes(current)) <= limit, "Single member exceeds part limit")
    if current:
        groups.append(current)
    return groups


def verify_part(path, members, limit):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(set(names) == {"MANIFEST.json", *[item["path"] for item in members]}, "ZIP membership mismatch")
        total = sum(info.file_size for info in archive.infolist())
        require(total <= limit <= MAX_PART_BYTES, "ZIP uncompressed size limit exceeded")
        require(archive.testzip() is None, "ZIP CRC failure")
        require(archive.read("MANIFEST.json") == manifest_bytes(members), "ZIP MANIFEST mismatch")
        for item in members:
            value, size = hashlib.sha256(), 0
            with archive.open(item["path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
                    size += len(block)
            require(size == item["bytes"] and value.hexdigest() == item["sha256"], f"ZIP member digest mismatch: {item['path']}")
    return total


def archive_grid(args):
    repository = args.repository.resolve()
    snapshot = inspect_grid(repository, args.grid_root, args.retained_manifest,
                            args.retained_manifest_sha256, args.expected_completions)
    output = args.output_root.resolve()
    require(not output.is_relative_to(repository / args.grid_root), "Archive output must be outside the grid")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix), "Unsafe archive prefix")
    require(0 < args.part_limit_bytes <= MAX_PART_BYTES, "Invalid part limit")
    output.mkdir(parents=True, exist_ok=True)
    require(not list(output.glob(args.prefix + "*")), "Archive prefix already exists; use verify or a fresh prefix")
    parts = []
    for number, members in enumerate(partition(snapshot["members"], args.part_limit_bytes), 1):
        path = output / f"{args.prefix}_part_{number:04d}.zip"
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            archive.writestr("MANIFEST.json", manifest_bytes(members))
            for item in members:
                archive.write(safe_path(repository, item["path"]), arcname=item["path"])
        total = verify_part(path, members, args.part_limit_bytes)
        parts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                      "uncompressed_bytes": total, "members": members})
        print(json.dumps({"verified_part": path.name, "members": len(members), "bytes": path.stat().st_size}), file=sys.stderr, flush=True)
    # All source and retained bytes must still match after writing archives.
    for item in snapshot["protected_files"] + snapshot["members"]:
        require(record(repository, safe_path(repository, item["path"]), item["sha256"]) == item, "Source changed during archive")
    index = {"schema_version": 1, "purpose": "lossless_completed_grid_epoch_archive", "path_base": "repository",
             "part_limit_bytes": args.part_limit_bytes, "archive_prefix": args.prefix,
             "originals_deleted": False, "snapshot": snapshot, "parts": parts}
    path = output / f"{args.prefix}_MANIFEST.json"
    with path.open("xb") as stream:
        stream.write(canonical(index) + b"\n")
    return public_result(path, index)


def public_result(path, index):
    return {"manifest": {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)},
            "parts": [{key: part[key] for key in ("name", "bytes", "sha256", "uncompressed_bytes")} for part in index["parts"]],
            "archived_members": len(index["snapshot"]["members"]),
            "original_bytes": sum(item["bytes"] for item in index["snapshot"]["members"]),
            "retained_checkpoint_paths": len(index["snapshot"]["retained_checkpoints"]), "originals_deleted": False}


def verify_index(path, expected_sha):
    path = path.resolve()
    require(digest(path) == expected_sha, "Archive manifest SHA256 mismatch")
    index = read_json(path)
    require(index["schema_version"] == 1 and index["purpose"] == "lossless_completed_grid_epoch_archive", "Unsupported archive manifest")
    all_members, names = [], []
    for part in index["parts"]:
        require(Path(part["name"]).name == part["name"], "Unsafe ZIP filename")
        target = safe_path(path.parent, part["name"])
        require(target.stat().st_size == part["bytes"] and digest(target) == part["sha256"], "ZIP file hash/size mismatch")
        require(verify_part(target, part["members"], index["part_limit_bytes"]) == part["uncompressed_bytes"], "ZIP size receipt mismatch")
        all_members.extend(part["members"])
        names.append(part["name"])
    require(len(names) == len(set(names)), "Duplicate ZIP part names")
    require(len(all_members) == len({item["path"] for item in all_members}), "Overlapping ZIP part members")
    require(exact_records(all_members) == index["snapshot"]["members"], "Part membership differs from archive manifest")
    return index


def prune_grid(args):
    require(args.confirm_uploaded_and_authorize_deletion, "Prune needs explicit deletion authorization")
    path = args.manifest.resolve()
    index = verify_index(path, args.manifest_sha256)
    receipt_path = path.parent / f"{index['archive_prefix']}_LOCAL_PRUNE_RECEIPT.json"
    require(not receipt_path.exists(), "Prune receipt already exists; inspect prior status before proceeding")
    receipt = read_json(args.upload_receipt)
    results = receipt["results"]
    expected_paths = {str(path), *[str(path.parent / part["name"]) for part in index["parts"]]}
    upload_sources = {str(path): {"sha256": args.manifest_sha256, "bytes": path.stat().st_size},
                      **{str(path.parent / part["name"]): part for part in index["parts"]}}
    require(len(results) == len(expected_paths), "Upload receipt must cover exactly the manifest and all ZIP parts")
    actual_paths = []
    for result in results:
        require(result.get("status") == "succeeded" and result.get("purpose") == "create_library_file",
                "Every archive upload must have succeeded")
        require(result.get("file_id") and result.get("library_file_id"), "Upload success lacks finalized IDs")
        uploaded_path = str(Path(result["local_path"]).resolve())
        require(uploaded_path in upload_sources, "Unexpected successful upload path")
        expected_source = upload_sources[uploaded_path]
        require(all(result.get(key) == expected_source[key] for key in ("sha256", "bytes")),
                "Upload result lacks the exact uploaded source digest/size")
        actual_paths.append(uploaded_path)
    require(set(actual_paths) == expected_paths and len(actual_paths) == len(set(actual_paths)), "Successful upload paths differ from archive files")
    repository = args.repository.resolve()
    saved = index["snapshot"]
    current = inspect_grid(repository, saved["grid_root"], saved["retained_manifest"]["path"],
                           saved["retained_manifest"]["sha256"], saved["expected_completions"])
    require(current == saved, "Current eligibility, retained set, or artifact bytes differ from archived snapshot")
    # All checks above complete before the first removal. Never remove archive files.
    deleted = []
    # Reserve a writable receipt before deletion. An interrupted prune leaves an
    # explicit in-progress record. Its archived bytes remain independently valid.
    with receipt_path.open("x") as stream:
        json.dump({"status": "in_progress", "archive_manifest_sha256": args.manifest_sha256,
                   "planned_members": saved["members"]}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        for item in saved["members"]:
            target = safe_path(repository, item["path"])
            require(digest(target) == item["sha256"], "Source changed immediately before removal")
            target.unlink()
            deleted.append(item)
    for item in saved["protected_files"]:
        require(record(repository, safe_path(repository, item["path"]), item["sha256"]) == item, "Retained file changed")
    result = {"archive_manifest": path.name, "archive_manifest_sha256": args.manifest_sha256,
              "upload_receipt_sha256": digest(args.upload_receipt), "deleted_members": deleted,
              "deleted_bytes": sum(item["bytes"] for item in deleted), "retained_files_verified": True}
    result["status"] = "completed"
    with receipt_path.open("w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"prune_receipt": receipt_path.name, "deleted_count": len(deleted), "deleted_bytes": result["deleted_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    archive = sub.add_parser("archive")
    archive.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    archive.add_argument("--grid-root", required=True)
    archive.add_argument("--retained-manifest", required=True)
    archive.add_argument("--retained-manifest-sha256", required=True)
    archive.add_argument("--expected-completions", type=int, required=True)
    archive.add_argument("--output-root", type=Path, required=True)
    archive.add_argument("--prefix", required=True)
    archive.add_argument("--part-limit-bytes", type=int, default=MAX_PART_BYTES)
    for name in ("verify", "prune"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        if name == "prune":
            command.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
            command.add_argument("--upload-receipt", type=Path, required=True)
            command.add_argument("--confirm-uploaded-and-authorize-deletion", action="store_true")
    args = parser.parse_args()
    if args.command == "archive":
        result = archive_grid(args)
    elif args.command == "verify":
        result = public_result(args.manifest, verify_index(args.manifest, args.manifest_sha256))
    else:
        result = prune_grid(args)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Archive validation failed: {error}") from error
