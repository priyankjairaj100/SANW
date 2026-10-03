#!/usr/bin/env python3
"""Acquire or verify the exact pretrained model used by the frozen protocol.

--verify reads only local files and never contacts Hugging Face. Existing files
with a wrong size or checksum are errors and are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
FILES = {
    "open_clip_config.json": {"bytes": 604, "sha256": "4302a891b495ad0966d1e7eeff102b5f34a3f1e647e57a9198933f862d5f8ba7"},
    "README.md": {"bytes": 7464, "sha256": "d0600101ba66849bd84eeae14e0b1ccd56cd962ea94aec5d60c415d9ce4e5f28"},
    "open_clip_model.safetensors": {"bytes": 605143316, "sha256": "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"},
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check(path: Path, pin: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required pinned model file missing: {path}")
    observed = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if observed != pin:
        raise ValueError(f"Pinned file mismatch; existing file left untouched: {path}; expected {pin}, observed {observed}")
    return observed


def publish_without_overwrite(part: Path, destination: Path) -> None:
    # Linking a completed file within the same directory is atomic and fails
    # rather than replacing a concurrently created destination.
    try:
        os.link(part, destination)
    finally:
        part.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Offline: require and verify all three existing local files.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/model")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface_hub")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/RERUN_PROTOCOL.json")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    protocol = json.loads(args.protocol.read_text())
    encoder = protocol["encoder"]
    expected_encoder = {"repository": REPOSITORY, "revision": REVISION,
                        "weight_sha256": FILES["open_clip_model.safetensors"]["sha256"]}
    if any(encoder.get(key) != value for key, value in expected_encoder.items()):
        raise ValueError("Model pins disagree with the supplied frozen protocol")

    # Validate every present file before any download or write takes place.
    for name, pin in FILES.items():
        destination = model_dir / name
        if destination.exists():
            check(destination, pin)
        elif args.verify:
            raise FileNotFoundError(f"Offline verification cannot download missing file: {destination}")
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, pin in FILES.items():
        destination = model_dir / name
        if destination.exists():
            continue
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(repo_id=REPOSITORY, revision=REVISION,
                                     filename=name, cache_dir=args.cache_dir,
                                     token=False, local_files_only=False))
        check(cached, pin)
        fd, filename = tempfile.mkstemp(prefix=f"{name}.", suffix=".part", dir=model_dir)
        part = Path(filename)
        try:
            with os.fdopen(fd, "wb") as out, cached.open("rb") as source:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            check(part, pin)
            try:
                publish_without_overwrite(part, destination)
            except FileExistsError:
                check(destination, pin)
            downloaded.append(name)
        finally:
            part.unlink(missing_ok=True)
    verified = {name: check(model_dir / name, pin) for name, pin in FILES.items()}
    config = json.loads((model_dir / "open_clip_config.json").read_text())
    if config["model_cfg"]["embed_dim"] != encoder["feature_dim"]:
        raise ValueError("Pinned configuration dimension disagrees with protocol")
    identity = {"repository": REPOSITORY, "revision": REVISION,
                "protocol_sha256": sha256(args.protocol), "files": verified}
    receipt_path = model_dir / "acquisition_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity:
            raise ValueError("Existing model receipt does not match verified pins; receipt left untouched")
    else:
        receipt = {
            "schema_version": 1, "identity": identity,
            "source_urls": {name: f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}" for name in FILES},
            "first_receipt_operation": "verify_existing_files_offline" if args.verify else "acquire_and_verify",
            "downloaded_by_first_receipt_operation": downloaded,
            "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
            "script_sha256": sha256(Path(__file__)),
            "preservation": "Existing mismatched files are never overwritten. Downloaded bytes and atomic local copies both require pinned size and SHA256.",
        }
        fd, filename = tempfile.mkstemp(prefix="receipt.", suffix=".part", dir=model_dir)
        part = Path(filename)
        with os.fdopen(fd, "w") as out:
            json.dump(receipt, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        try:
            publish_without_overwrite(part, receipt_path)
        except FileExistsError:
            if json.loads(receipt_path.read_text()).get("identity") != identity:
                raise ValueError("Concurrent receipt identity mismatch")
    print(json.dumps({"status": "verified", "offline": args.verify,
                      "downloaded": downloaded, "model_dir": str(model_dir),
                      "receipt": str(receipt_path), "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
