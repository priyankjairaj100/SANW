#!/usr/bin/env python3
"""Bounded, lossless batches for completed candidates in an unfinished grid.

This storage helper never imports or changes scientific code. Each batch is an
independent recoverable archive. Uploads and deletions require separate commands.
Frozen fit/select commands still require original checkpoint bytes on disk.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

from archive_completed_grid_checkpoints import (
    canonical, digest, exact_records, manifest_bytes, read_json, record,
    relative_path, require, safe_path, verify_part,
)

PURPOSE = "lossless_completed_candidate_batch"
PART_LIMIT = 100 * 1024 * 1024
RESIDENT_LIMIT = 128 * 1024 * 1024


def write_new(path, value):
    """Exclusive durable receipt; never overwrite an earlier decision."""
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def candidate_locks(repository, grid_name, names):
    with ExitStack() as stack:
        for name in sorted(names):
            directory = safe_path(repository, grid_name + "/" + name, exists=False)
            require(directory.is_dir(), "Candidate directory is missing")
            path = safe_path(repository, grid_name + "/" + name + "/.candidate.lock", exists=False)
            stream = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Candidate is busy: {name}") from error
        yield


def retained_states(repository, grid_name, ledger, training, keep_name, keep_sha):
    """Bind a complete grid's immutable selection and explicit retained states."""
    require(keep_name and keep_sha, "A selected grid requires a pinned retained manifest")
    grid = repository / grid_name
    keep_record = record(repository, safe_path(repository, keep_name), keep_sha)
    keep = read_json(repository / keep_name)
    protected = [keep_record]
    require(keep["ledger_sha256"] == ledger["ledger_sha256"], "Retained ledger mismatch")
    full_ref = keep.get("full_state_manifest")
    require(full_ref and full_ref["path"] == grid_name + "/state_manifest.json",
            "Retained manifest must bind this full state manifest")
    protected.append(record(repository, safe_path(repository, full_ref["path"]), full_ref["sha256"]))
    full = read_json(repository / full_ref["path"])
    require(full["ledger_sha256"] == ledger["ledger_sha256"], "Full state ledger mismatch")
    require(keep.get("protocol_sha256", ledger["identity"]["protocol"]["sha256"]) ==
            ledger["identity"]["protocol"]["sha256"], "Retained protocol mismatch")
    selections = keep["selection_sha256"]
    require(set(selections) == {"primary", "sensitivity"} and selections == full["selection_sha256"],
            "Both frozen selection bindings are required")
    loaded = {}
    for name, sha in selections.items():
        path = safe_path(repository, grid_name + f"/selection_{name}.json")
        protected.append(record(repository, path, sha))
        loaded[name] = read_json(path)
        require(loaded[name]["ledger_sha256"] == ledger["ledger_sha256"], "Selection ledger mismatch")
    full_states = {state["state_id"]: state for state in full["states"]}
    require(len(full_states) == len(full["states"]), "Duplicate full state IDs")
    states = {state["state_id"]: state for state in keep["states"]}
    require(states and len(states) == len(keep["states"]), "Missing or duplicate retained states")
    selected = (full.get("selections", []) + full.get("decomposition_selections", []) +
                full.get("factorial_selections", []) + keep.get("factorial_selections", []))
    require(selected and all(s["state_id"] in states for s in selected),
            "A selected/decomposition/factorial state is missing from retained states")
    if "directional_factorial" in training:
        protocol = read_json(repository / ledger["identity"]["protocol"]["path"])
        factorial = protocol["evaluation"]["mechanism_factorial"]
        cells = set(training["directional_factorial"]) - {"all_candidates_fixed"}
        require(cells and cells <= set(training["policies"]), "Unknown directional factorial cell")
        for cell in sorted(cells):
            for rate in factorial["learning_rates"]:
                for seed in factorial["seeds"]:
                    matches = [state for state in full_states.values() if
                        (state.get("method"), state.get("learning_rate"), state.get("seed"), state.get("epoch")) ==
                        (cell, rate, seed, factorial["epoch"])]
                    require(len(matches) == 1 and matches[0]["state_id"] in states and
                            matches[0].get("alpha") == 1. and matches[0].get("draw_id") is None,
                            "A frozen directional factorial state is missing or altered")
    retained = []
    for sid, state in states.items():
        require(sid in full_states and all(state.get(k) == full_states[sid].get(k)
                for k in ("checkpoint", "checkpoint_sha256")), "Retained state differs from full manifest")
        if state.get("checkpoint") is None:
            require(state.get("checkpoint_sha256") is None, "Null retained state has a digest")
        else:
            retained.append(record(repository, safe_path(repository, state["checkpoint"]), state["checkpoint_sha256"]))
    retained = exact_records(retained)
    expected = {f"candidates/{policy}/lr_{rate:.8g}/seed_{seed}": {
        "method": policy, "learning_rate": rate, "seed": seed, "epochs_completed": training["epochs"],
        "loss_policy": policy, "draw_id": None, "selection_sha256": None}
        for policy in training["policies"] for rate in training["learning_rates"] for seed in training["seeds"]}
    if "matched_controls" in training:
        family, prefix, count = "allocation_distillation", "matched_allocation_distillation_draw_", training["matched_controls"]["fits_per_encoder"]
    elif "matched_distilled_control" in training:
        family, prefix, count = "distilled", "matched_distilled_draw_", training["matched_distilled_control"]["additional_runs_per_encoder"]
    else:
        family = None
    if family:
        require(full.get("matched_controls_complete") is True, "Matched controls are incomplete")
        chosen = loaded["primary"]["families"][family]
        require(sorted(run["seed"] for run in chosen["runs"]) == sorted(training["seeds"]), "Matched seed mismatch")
        require(count == 3 * len(chosen["runs"]), "Matched control count mismatch")
        for run in chosen["runs"]:
            for draw in range(3):
                method, rate, seed = prefix + str(draw), chosen["learning_rate"], run["seed"]
                expected[f"candidates/{method}/lr_{rate:.8g}/seed_{seed}"] = {
                    "method": method, "learning_rate": rate, "seed": seed, "epochs_completed": run["epoch"],
                    "loss_policy": chosen["policy"], "draw_id": draw, "selection_sha256": selections["primary"]}
    discovered = {relative_path(grid, p.parent) for p in (grid / "candidates").rglob("completion.json")}
    require(discovered == set(expected), "A retained manifest requires the complete frozen grid")
    retained_map = {x["path"]: x for x in retained}
    for name, metadata in expected.items():
        path = safe_path(repository, grid_name + "/" + name + "/completion.json")
        protected.append(record(repository, path))
        receipt = read_json(path)
        require(receipt["ledger_sha256"] == ledger["ledger_sha256"] and
                all(receipt.get(k) == v for k, v in metadata.items()), "Full-grid completion metadata mismatch")
        terminal = f"epochs/epoch_{metadata['epochs_completed']:02d}.pt"
        full_name = grid_name + "/" + name + "/" + terminal
        require(full_name in retained_map and retained_map[full_name]["sha256"] == receipt["artifact_sha256"][terminal],
                "Every terminal checkpoint must be retained")
    protected.extend(retained)
    return keep_record, protected, retained, expected


def inspect_candidates(repository, grid_name, names, keep_name=None, keep_sha=None):
    """Check whole base candidates and optional complete-grid retained states."""
    require(names and len(names) == len(set(names)), "Supply unique candidate directories")
    grid = safe_path(repository, grid_name, exists=False)
    require(grid.is_dir(), "Grid directory is missing")
    selected_grid = any((grid / name).exists() for name in
                       ("selection_primary.json", "selection_sensitivity.json", "state_manifest.json"))
    require((keep_name is None) == (keep_sha is None), "Supply both retained manifest path and digest")
    require(not selected_grid or keep_name is not None,
            "A grid after selection requires a pinned retained manifest; omit it only before selection")
    protected = [record(repository, safe_path(repository, grid_name + "/ledger.json"))]
    ledger = read_json(grid / "ledger.json")
    require(hashlib.sha256(canonical(ledger["identity"])).hexdigest() == ledger["ledger_sha256"],
            "Ledger identity hash mismatch")
    ref = ledger["identity"]["protocol"]
    protected.append(record(repository, safe_path(repository, ref["path"]), ref["sha256"]))
    training = read_json(repository / ref["path"])["training"]
    expected = {f"candidates/{policy}/lr_{rate:.8g}/seed_{seed}": {
                "method": policy, "learning_rate": rate, "seed": seed, "epochs_completed": training["epochs"],
                "selection_sha256": None, "loss_policy": policy, "draw_id": None}
                for policy in training["policies"] for rate in training["learning_rates"]
                for seed in training["seeds"]}
    epochs = training["epochs"]
    require(isinstance(epochs, int) and epochs > 0, "Invalid frozen epoch count")
    keep_record, retained = None, []
    if keep_name is not None:
        keep_record, keep_protected, retained, expected = retained_states(repository, grid_name, ledger, training, keep_name, keep_sha)
        protected.extend(keep_protected)
    keep_paths = {x["path"] for x in retained}
    members, candidates = [], []
    for name in sorted(names):
        require(name in expected, f"Candidate is outside the frozen grid: {name}")
        directory = safe_path(repository, grid_name + "/" + name, exists=False)
        completion = safe_path(repository, grid_name + "/" + name + "/completion.json")
        protected.append(record(repository, completion))
        receipt = read_json(completion)
        metadata = expected[name]
        epochs = metadata["epochs_completed"]
        require(isinstance(epochs, int) and 0 <= epochs <= training["epochs"], "Invalid matched epoch count")
        values = {"ledger_sha256": ledger["ledger_sha256"], "checkpoint_count": epochs + 1, **metadata}
        require(all(receipt.get(key) == value for key, value in values.items()),
                f"Completion metadata mismatch: {name}")
        weights = {f"epochs/epoch_{i:02d}.pt" for i in range(epochs + 1)}
        validations = {f"validation/epoch_{i:02d}.npz" for i in range(epochs + 1)}
        hashes = receipt["artifact_sha256"]
        require(set(hashes) == weights | validations | {"history.json"},
                f"Incomplete or unexpected completion artifacts: {name}")
        actual_weights = {relative_path(directory, p) for p in directory.rglob("*.pt")}
        require(actual_weights == weights, f"Unbound or missing candidate weights: {name}")
        candidate_members = []
        for artifact, sha in sorted(hashes.items()):
            item = record(repository, safe_path(repository, grid_name + "/" + name + "/" + artifact), sha)
            if artifact in weights and artifact != f"epochs/epoch_{epochs:02d}.pt" and item["path"] not in keep_paths:
                members.append(item)
                candidate_members.append(item)
            else:
                protected.append(item)
                if artifact in weights:
                    retained.append(item)
        candidates.append({"path": name, "members": candidate_members})
    return {"grid_root": grid_name, "candidate_names": sorted(names),
            "ledger_sha256": ledger["ledger_sha256"], "candidates": candidates,
            "members": exact_records(members), "protected_files": exact_records(protected),
            "retained_manifest": keep_record, "retained_checkpoints": exact_records(retained)}


def check_sources(repository, snapshot, *, allow_missing_members=False):
    for item in snapshot["protected_files"]:
        require(record(repository, safe_path(repository, item["path"]), item["sha256"]) == item,
                "Protected source changed")
    for item in snapshot["members"]:
        path = safe_path(repository, item["path"], exists=False)
        if allow_missing_members and not path.exists():
            continue
        require(record(repository, path, item["sha256"]) == item, "Checkpoint source changed")


def zip_upper_bound(members):
    sizes = [("MANIFEST.json", len(manifest_bytes(members)))] + [(x["path"], x["bytes"]) for x in members]
    # zlib's compressBound plus generous ZIP64/local/central header allowance.
    return 1024 + sum(n + (n >> 12) + (n >> 14) + (n >> 25) + 64 +
                      2 * len(name.encode("utf-8")) + 256 for name, n in sizes)


def resident_bytes(output):
    paths = [p for p in output.rglob("*") if p.is_file() and
             (p.name.endswith(".zip") or p.name.endswith(".partial"))]
    require(all(not p.is_symlink() for p in paths), "Symlink archive cache refused")
    return sum(p.stat().st_size for p in paths)


def create(args):
    repository, output = args.repository.resolve(), args.output_root.resolve()
    require(0 < args.part_limit_bytes <= PART_LIMIT, "Part limit must be at most 100 MiB")
    require(args.resident_limit_bytes >= args.part_limit_bytes, "Resident bound is smaller than the part bound")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix), "Unsafe archive prefix")
    require(not output.is_relative_to(repository / args.grid_root), "Archive output must be outside the grid")
    output.mkdir(parents=True, exist_ok=True)
    cleanup = getattr(args, "pre_allocation_cleanup", None)
    creation_cleanup = getattr(args, "pre_creation_cleanup", None)
    require(cleanup is None or callable(cleanup), "Pre-allocation cleanup must be callable")
    require(creation_cleanup is None or callable(creation_cleanup), "Pre-creation cleanup must be callable")
    def current_resident_bytes():
        # The orchestration layer supplies only previously verified completed
        # batches. No current-batch temporary file is eligible for this callback.
        if cleanup is not None:
            cleanup()
        return resident_bytes(output)
    with candidate_locks(repository, args.grid_root, args.candidate):
        snapshot = inspect_candidates(repository, args.grid_root, args.candidate,
                                      getattr(args, "retained_manifest", None),
                                      getattr(args, "retained_manifest_sha256", None))
        # This one-shot hook may discard only a previously certified abandoned
        # creation. It runs under candidate locks, before this attempt writes ZIPs.
        if creation_cleanup is not None:
            creation_cleanup(snapshot)
        require(not list(output.glob(args.prefix + "*")), "Prefix already exists; inspect its receipts before recovery")
        plans = [candidate["members"] for candidate in snapshot["candidates"]]
        require(all(plans), "A candidate has no unretained epoch checkpoints")
        require(all(sum(x["bytes"] for x in members) + len(manifest_bytes(members)) <=
                    args.part_limit_bytes for members in plans), "One candidate exceeds the part limit")
        require(current_resident_bytes() + sum(zip_upper_bound(members) for members in plans) <=
                args.resident_limit_bytes, "Resident archive bound exceeded; upload the existing batch before continuing")
        parts = []
        for number, candidate in enumerate(snapshot["candidates"], 1):
            members = candidate["members"]
            size = sum(x["bytes"] for x in members) + len(manifest_bytes(members))
            require(size <= args.part_limit_bytes, "One candidate exceeds the part limit")
            require(current_resident_bytes() + zip_upper_bound(members) <= args.resident_limit_bytes,
                    "Resident archive bound exceeded; upload the existing batch before continuing")
            path = output / f"{args.prefix}_part_{number:04d}.zip"
            partial = path.with_suffix(".zip.partial")
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                archive.writestr("MANIFEST.json", manifest_bytes(members))
                for item in members:
                    archive.write(safe_path(repository, item["path"]), arcname=item["path"])
            total = verify_part(partial, members, args.part_limit_bytes)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            partial.rename(path)
            require(current_resident_bytes() <= args.resident_limit_bytes, "Actual archive cache exceeded bound")
            parts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                          "uncompressed_bytes": total, "members": members})
        check_sources(repository, snapshot)
        index = {"schema_version": 1, "purpose": PURPOSE, "path_base": "repository",
                 "prefix": args.prefix, "part_limit_bytes": args.part_limit_bytes,
                 "resident_limit_bytes": args.resident_limit_bytes, "snapshot": snapshot, "parts": parts}
        manifest = output / f"{args.prefix}_MANIFEST.json"
        write_new(manifest, index)
    return {"manifest": file_receipt(manifest), "parts": [file_receipt(output / x["name"]) for x in parts],
            "candidate_count": len(snapshot["candidate_names"]), "member_count": len(snapshot["members"]),
            "original_bytes": sum(x["bytes"] for x in snapshot["members"]), "originals_deleted": False}


def file_receipt(path):
    return {"local_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def load_index(path, sha, *, verify_zips=False):
    require(digest(path) == sha, "Batch manifest digest mismatch")
    index = read_json(path)
    require(index.get("schema_version") == 1 and index.get("purpose") == PURPOSE, "Unsupported batch manifest")
    require(0 < index["part_limit_bytes"] <= PART_LIMIT, "Invalid manifest part limit")
    parts, all_members = index["parts"], []
    require(parts and len(parts) == len({x["name"] for x in parts}), "Missing or duplicate parts")
    for part in parts:
        require(Path(part["name"]).name == part["name"] and part["name"].endswith(".zip"), "Unsafe ZIP name")
        require(part["members"], "Empty part")
        all_members.extend(part["members"])
        if verify_zips:
            verify_bound_part(path.parent / part["name"], part, index)
    require(len(all_members) == len({x["path"] for x in all_members}), "Overlapping part members")
    require(exact_records(all_members) == index["snapshot"]["members"], "Missing or extra part members")
    require(not ({x["path"] for x in all_members} & {x["path"] for x in index["snapshot"]["protected_files"]}),
            "Protected files cannot be archive members")
    return index


def verify_bound_part(path, part, index):
    require(not path.is_symlink() and path.is_file(), "Missing regular ZIP part")
    require(path.stat().st_size == part["bytes"] and digest(path) == part["sha256"], "ZIP digest/size mismatch")
    require(verify_part(path, part["members"], index["part_limit_bytes"]) == part["uncompressed_bytes"],
            "ZIP uncompressed byte count mismatch")


def validate_upload_results(results, sources):
    require(len(results) == len(sources), "Upload results must cover exactly this manifest and its parts")
    seen = set()
    for result in results:
        name = result.get("local_path")
        require(name in sources and name not in seen, "Unexpected or duplicate upload path")
        require(result.get("status") == "succeeded" and result.get("purpose") == "create_library_file",
                "Every upload must succeed")
        require(result.get("file_id") and result.get("library_file_id"), "Upload lacks finalized remote IDs")
        require(all(result.get(key) == sources[name][key] for key in ("bytes", "sha256")),
                "Upload source digest/size mismatch")
        seen.add(name)


def upload_sources(manifest, index):
    return {str(manifest): file_receipt(manifest),
            **{str(manifest.parent / x["name"]): x for x in index["parts"]}}


def record_uploads(args):
    manifest = args.manifest.resolve()
    index = load_index(manifest, args.manifest_sha256, verify_zips=True)
    results = read_json(args.upload_receipt)["results"]
    sources = upload_sources(manifest, index)
    validate_upload_results(results, sources)
    receipt = {"schema_version": 1, "purpose": PURPOSE + "_verified_uploads",
               "archive_manifest_sha256": args.manifest_sha256, "archive_manifest_bytes": manifest.stat().st_size,
               "sources": sources, "results": results, "all_parts_verified_locally": True,
               "input_upload_receipt_sha256": digest(args.upload_receipt)}
    write_new(args.output_receipt, receipt)
    return {"verified_upload_record": file_receipt(args.output_receipt), "originals_deleted": False}


def prune(args):
    require(args.confirm_uploaded_and_authorize_deletion, "Explicit deletion authorization is required")
    repository, manifest = args.repository.resolve(), args.manifest.resolve()
    index = load_index(manifest, args.manifest_sha256)
    require(digest(args.upload_record) == args.upload_record_sha256, "Verified upload record digest mismatch")
    upload = read_json(args.upload_record)
    require(upload.get("purpose") == PURPOSE + "_verified_uploads" and upload.get("all_parts_verified_locally") is True,
            "A verified upload record is required")
    require(upload.get("archive_manifest_sha256") == args.manifest_sha256 and
            upload.get("archive_manifest_bytes") == manifest.stat().st_size, "Upload record binds another manifest")
    sources = upload_sources(manifest, index)
    require(upload["sources"] == sources, "Verified source records changed")
    validate_upload_results(upload["results"], sources)
    snapshot = index["snapshot"]
    prune_path = manifest.with_name(index["prefix"] + "_LOCAL_PRUNE_RECEIPT.json")
    require(not prune_path.exists(), "Prune receipt exists; inspect interrupted/completed status first")
    with candidate_locks(repository, snapshot["grid_root"], snapshot["candidate_names"]):
        keep = snapshot.get("retained_manifest")
        current = inspect_candidates(repository, snapshot["grid_root"], snapshot["candidate_names"],
                                     keep["path"] if keep else None, keep["sha256"] if keep else None)
        require(current == snapshot, "Current candidate snapshot differs from the archive")
        write_new(prune_path, {"status": "in_progress", "archive_manifest_sha256": args.manifest_sha256,
                              "verified_upload_record_sha256": args.upload_record_sha256,
                              "planned_members": snapshot["members"]})
        deleted = []
        for item in snapshot["members"]:
            path = safe_path(repository, item["path"])
            require(record(repository, path, item["sha256"]) == item, "Source changed before removal")
            path.unlink()
            deleted.append(item)
        check_sources(repository, snapshot, allow_missing_members=True)
        completed = {"status": "completed", "archive_manifest_sha256": args.manifest_sha256,
                     "verified_upload_record_sha256": args.upload_record_sha256,
                     "deleted_members": deleted, "deleted_bytes": sum(x["bytes"] for x in deleted)}
        final = prune_path.with_suffix(".json.completed")
        write_new(final, completed)
        final.replace(prune_path)
    return {"prune_receipt": file_receipt(prune_path), "deleted_count": len(deleted),
            "deleted_bytes": completed["deleted_bytes"]}


def restore(args):
    """Restore one downloaded ZIP at a time, without changing completion receipts."""
    repository, manifest = args.repository.resolve(), args.manifest.resolve()
    index = load_index(manifest, args.manifest_sha256)
    matches = [p for p in index["parts"] if p["name"] == args.part.name]
    require(len(matches) == 1, "ZIP is absent from this manifest")
    part = matches[0]
    verify_bound_part(args.part, part, index)
    snapshot = index["snapshot"]
    restored = []
    with candidate_locks(repository, snapshot["grid_root"], snapshot["candidate_names"]):
        check_sources(repository, snapshot, allow_missing_members=True)
        with zipfile.ZipFile(args.part) as archive:
            for item in part["members"]:
                path = safe_path(repository, item["path"], exists=False)
                if path.exists():
                    require(record(repository, path, item["sha256"]) == item, "Existing restored bytes differ")
                    continue
                temporary = path.with_suffix(path.suffix + ".archive-restore-partial")
                require(not temporary.exists(), "Interrupted restore file exists; inspect before continuing")
                with archive.open(item["path"]) as source, temporary.open("xb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                require(temporary.stat().st_size == item["bytes"] and digest(temporary) == item["sha256"],
                        "Restored member digest mismatch")
                temporary.replace(path)
                restored.append(item["path"])
    return {"restored_count": len(restored), "part": args.part.name, "restored_paths": restored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--grid-root", required=True)
    make.add_argument("--candidate", action="append", required=True, help="Grid-relative completed base candidate directory")
    make.add_argument("--output-root", type=Path, required=True)
    make.add_argument("--prefix", required=True)
    make.add_argument("--retained-manifest", help="Repository-relative retained states for a completed, selected grid")
    make.add_argument("--retained-manifest-sha256")
    make.add_argument("--part-limit-bytes", type=int, default=PART_LIMIT)
    make.add_argument("--resident-limit-bytes", type=int, default=RESIDENT_LIMIT)
    upload = sub.add_parser("record-uploads")
    upload.add_argument("--upload-receipt", type=Path, required=True)
    upload.add_argument("--output-receipt", type=Path, required=True)
    remove = sub.add_parser("prune")
    remove.add_argument("--upload-record", type=Path, required=True)
    remove.add_argument("--upload-record-sha256", required=True)
    remove.add_argument("--confirm-uploaded-and-authorize-deletion", action="store_true")
    recover = sub.add_parser("restore")
    recover.add_argument("--part", type=Path, required=True)
    for command in (make, remove, recover):
        command.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    for command in (upload, remove, recover):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    result = {"create": create, "record-uploads": record_uploads, "prune": prune, "restore": restore}[args.command](args)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Storage validation failed: {error}") from error
