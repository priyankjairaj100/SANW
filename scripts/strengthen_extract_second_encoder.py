#!/usr/bin/env python3
"""Pinned OpenAI RN50 features; transactional, content-addressed extraction only.

Reconstructed from the preserved second-encoder protocol and original extractor.
No fitting, checkpoint selection, or held-out scoring is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
WEIGHT_BYTES = 255827503
WEIGHT_URL = f"https://openaipublic.azureedge.net/clip/models/{WEIGHT_SHA}/RN50.pt"
DIMENSION = 1024
PREPROCESS = {
    "size": 224, "mode": "RGB", "interpolation": "bicubic",
    "resize_mode": "shortest", "crop": "center",
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inside_root(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Input path escapes the project: {path}")
    return path


def read_manifest(dataset):
    if Path(dataset).is_absolute() or ".." in Path(dataset).parts:
        raise ValueError(f"Invalid dataset path: {dataset}")
    path = ROOT / "data" / dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    for kind in ["images", "texts"]:
        items = manifest[kind]
        ids = [item["id"] for item in items]
        if not items or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or non-unique {kind} IDs in {dataset}")
    return path, manifest


def image_key(item):
    path = inside_root(item["path"])
    actual = digest(path)
    if item.get("sha256") and item["sha256"] != actual:
        raise RuntimeError(f"Manifest image hash mismatch: {path}")
    if item.get("bytes") is not None and item["bytes"] != path.stat().st_size:
        raise RuntimeError(f"Manifest image byte-count mismatch: {path}")
    return actual


def short_text(model, tokens):
    """Omit only batch-wide trailing positions; verify against stock encoding."""
    import torch
    import torch.nn.functional as F
    end = tokens.argmax(dim=-1)
    length = int(end.max()) + 1
    tokens = tokens[:, :length]
    dtype = model.transformer.get_cast_dtype()
    value = model.token_embedding(tokens).to(dtype)
    value = value + model.positional_embedding[:length].to(dtype)
    value = model.transformer(value, attn_mask=model.attn_mask[:length, :length])
    value = model.ln_final(value)
    value = value[torch.arange(value.shape[0], device=value.device), end]
    if model.text_projection is not None:
        if isinstance(model.text_projection, torch.nn.Linear):
            value = model.text_projection(value)
        else:
            value = value @ model.text_projection
    return F.normalize(value, dim=-1)


def validate_vectors(values):
    import numpy as np
    if values.ndim != 2 or values.shape[1] != DIMENSION:
        raise RuntimeError(f"Invalid feature dimensions: {values.shape}")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError("Features must be finite float32")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5, rtol=0):
        raise RuntimeError("Features are not L2-normalized")


def verify_export(output, expected, manifest):
    import numpy as np
    metadata_path, feature_path = output / "metadata.json", output / "features.npz"
    if not metadata_path.exists():
        return False
    if not feature_path.exists():
        raise RuntimeError(f"Completed metadata exists without features: {output}")
    saved = json.loads(metadata_path.read_text())
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Existing feature export identity mismatch ({key}): {output}")
    if digest(feature_path) != saved.get("features_sha256"):
        raise RuntimeError(f"Completed feature-file hash mismatch: {output}")
    with np.load(feature_path, allow_pickle=False) as archive:
        for kind, collection in [("image", "images"), ("text", "texts")]:
            values = archive[f"{kind}_features"]
            validate_vectors(values)
            expected_ids = np.asarray([item["id"] for item in manifest[collection]])
            if len(values) != len(expected_ids) or not np.array_equal(archive[f"{kind}_ids"], expected_ids):
                raise RuntimeError(f"Completed row IDs disagree with manifest: {output}")
    return True


def preflight(datasets):
    rows = []
    for dataset in datasets:
        path, manifest = read_manifest(dataset)
        missing = [item["path"] for item in manifest["images"] if not inside_root(item["path"]).is_file()]
        rows.append({"dataset": dataset, "manifest_sha256": digest(path),
                     "images": len(manifest["images"]), "texts": len(manifest["texts"]),
                     "missing_image_count": len(missing), "first_missing_images": missing[:5]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--text-batch", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stock-text", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="Commit text vectors while images are being acquired; no NPZ exports")
    parser.add_argument("--preflight", action="store_true", help="Report missing inputs without loading the model")
    args = parser.parse_args()
    if min(args.threads, args.image_batch, args.text_batch) < 1:
        parser.error("Thread and batch counts must be positive")
    if len(args.dataset) != len(set(args.dataset)):
        parser.error("Dataset arguments must be unique")
    status = preflight(args.dataset)
    if args.preflight:
        print(json.dumps(status, indent=2))
        return
    if not args.text_only and any(row["missing_image_count"] for row in status):
        raise FileNotFoundError("Source images are absent; restore them before full extraction. " + canonical(status))

    # Verify the exact official bytes BEFORE any checkpoint deserialization.
    weight = ROOT / "data/model_rn50_openai/RN50.pt"
    if weight.stat().st_size != WEIGHT_BYTES or digest(weight) != WEIGHT_SHA:
        raise RuntimeError("Pinned RN50 checkpoint byte-count or SHA256 mismatch")
    import numpy as np
    import torch
    import open_clip
    from PIL import Image
    from open_clip.openai import load_openai_model
    from open_clip.transform import image_transform

    if open_clip.__version__ != "2.32.0":
        raise RuntimeError("Protocol requires OpenCLIP 2.32.0")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model = load_openai_model(str(weight), precision="fp32", device=args.device)
    model.eval().requires_grad_(False)
    preprocess = image_transform(224, is_train=False, mean=PREPROCESS["mean"],
                                 std=PREPROCESS["std"], interpolation="bicubic", resize_mode="shortest")
    tokenizer = open_clip.get_tokenizer("RN50")
    scale = float(model.logit_scale.exp())
    if not np.isclose(scale, 100.00000762939453, rtol=0, atol=1e-5):
        raise RuntimeError(f"Unexpected frozen RN50 logit scale: {scale}")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "open_clip_torch", "numpy", "Pillow", "ftfy", "regex"]}
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": versions, "device": args.device, "threads": args.threads,
               "torch_interop_threads": 1, "deterministic_algorithms": True,
               "cuda_available": torch.cuda.is_available()}
    encoder_identity = {"encoder_id": "rn50", "weights_sha256": WEIGHT_SHA,
                        "preprocess_config": PREPROCESS, "dimension": DIMENSION,
                        "dtype": "float32", "normalization": "L2", "runtime": runtime,
                        "text_encoding": "stock" if args.stock_text else "causally_trimmed_after_last_eot",
                        "source_sha256": digest(__file__)}
    signature = canonical(encoder_identity)
    signature_hash = hashlib.sha256(signature.encode()).hexdigest()
    feature_root = ROOT / "results/strengthen_second_encoder/features"
    feature_root.mkdir(parents=True, exist_ok=True)
    atomic_json(feature_root.parent / "runtime.json", runtime)
    db = sqlite3.connect(feature_root / "bank.sqlite", timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS features(kind TEXT, key TEXT, value BLOB, PRIMARY KEY(kind,key))")
    previous = db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()
    if previous and previous[0] != signature:
        raise RuntimeError("Feature-bank encoder identity mismatch; preserve the old bank and use a separate tree")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('encoder',?)", (signature,))
    db.commit()
    print("RUNTIME " + canonical(runtime), flush=True)
    with torch.inference_mode():
        for dataset in args.dataset:
            start = time.monotonic()
            manifest_path, manifest = read_manifest(dataset)
            texts, images = manifest["texts"], manifest["images"]
            output = feature_root / dataset
            keyed = {"text": [(hashlib.sha256(item["text"].encode("utf-8")).hexdigest(), item) for item in texts]}
            if not args.text_only:
                keyed["image"] = [(image_key(item), item) for item in images]
            receipt = {kind: [{"id": item["id"], "content_sha256": key} for key, item in entries]
                       for kind, entries in keyed.items()}
            receipt_hash = hashlib.sha256(canonical(receipt).encode()).hexdigest()
            expected = {"manifest_sha256": digest(manifest_path), "encoder_signature_sha256": signature_hash,
                        "input_content_sha256": receipt_hash, "weights_sha256": WEIGHT_SHA}
            if not args.text_only and verify_export(output, expected, manifest):
                print(f"REUSE verified completed export {dataset}", flush=True)
                continue
            indices = sorted(set([0, len(texts) // 2, len(texts) - 1] + list(range(min(20, len(texts))))))
            tokens = tokenizer([texts[i]["text"] for i in indices]).to(args.device)
            stock = model.encode_text(tokens, normalize=True)
            shortened = short_text(model, tokens)
            error = float((stock - shortened).abs().max())
            parity = {"dataset": dataset, "examples": len(indices), "indices": indices,
                      "max_abs_error": error, "tolerance": 2e-6, "passed": error <= 2e-6}
            if not parity["passed"]:
                raise RuntimeError(f"Causal text-truncation parity failed for {dataset}: {error}")
            atomic_json(output / "text_parity.json", parity)
            encoded_arrays = {}
            for kind in (["text"] if args.text_only else ["image", "text"]):
                batch_size = args.image_batch if kind == "image" else args.text_batch
                pending, seen = [], set()
                for key, item in keyed[kind]:
                    if key not in seen and not db.execute("SELECT 1 FROM features WHERE kind=? AND key=?", (kind, key)).fetchone():
                        pending.append((key, item))
                    seen.add(key)
                if kind == "text":
                    pending.sort(key=lambda pair: len(pair[1]["text"]))
                print(f"{dataset} {kind}: {len(pending)} new / {len(keyed[kind])} total", flush=True)
                for position in range(0, len(pending), batch_size):
                    batch = pending[position:position + batch_size]
                    if kind == "image":
                        tensors = []
                        for _, item in batch:
                            with Image.open(inside_root(item["path"])) as image:
                                tensors.append(preprocess(image.convert("RGB")))
                        encoded = model.encode_image(torch.stack(tensors).to(args.device), normalize=True)
                    else:
                        tokens = tokenizer([item["text"] for _, item in batch]).to(args.device)
                        encoded = model.encode_text(tokens, normalize=True) if args.stock_text else short_text(model, tokens)
                    values = encoded.cpu().numpy().astype(np.float32)
                    validate_vectors(values)
                    db.executemany("INSERT OR IGNORE INTO features VALUES (?,?,?)",
                                   [(kind, key, value.tobytes()) for (key, _), value in zip(batch, values)])
                    for (key, _), value in zip(batch, values):
                        stored = np.frombuffer(db.execute("SELECT value FROM features WHERE kind=? AND key=?", (kind, key)).fetchone()[0], dtype=np.float32)
                        if not np.allclose(stored, value, atol=2e-6, rtol=0):
                            db.rollback()
                            raise RuntimeError("Concurrent feature-bank entry failed numerical parity")
                    db.commit()
                    if position // batch_size % 10 == 0 or position + batch_size >= len(pending):
                        print(f"{dataset} {kind}: encoded {min(position + batch_size, len(pending))}/{len(pending)} elapsed {time.monotonic() - start:.1f}s", flush=True)
                rows = [np.frombuffer(db.execute("SELECT value FROM features WHERE kind=? AND key=?", (kind, key)).fetchone()[0], dtype=np.float32).copy() for key, _ in keyed[kind]]
                encoded_arrays[kind] = np.stack(rows)
                validate_vectors(encoded_arrays[kind])
            if args.text_only:
                print(f"TEXT BANK COMPLETE {dataset}; image extraction/export pending", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / "features.tmp.npz"
            np.savez_compressed(temporary, image_features=encoded_arrays["image"], text_features=encoded_arrays["text"],
                                image_ids=np.asarray([item["id"] for item in images]),
                                text_ids=np.asarray([item["id"] for item in texts]))
            temporary.replace(output / "features.npz")
            atomic_json(output / "input_content_receipt.json", receipt)
            atomic_json(output / "metadata.json", {
                "schema_version": 2, "dataset": dataset, **expected, **encoder_identity,
                "model_repository": "openai/CLIP", "model_revision": WEIGHT_SHA,
                "model_name": "RN50", "model_url": WEIGHT_URL,
                "logit_scale": scale, "image_count": len(images), "text_count": len(texts),
                "open_clip_version": open_clip.__version__, "torch_version": torch.__version__,
                "preprocess_repr": re.sub(r" at 0x[0-9a-fA-F]+", "", str(preprocess)),
                "causal_padding_parity": [parity], "features_sha256": digest(output / "features.npz"),
                "elapsed_seconds": time.monotonic() - start,
                "reconstruction": "2026-10-04 from preserved protocol; no prior RN50 result assumed",
            })
            if not verify_export(output, expected, manifest):
                raise RuntimeError("Export verification unexpectedly failed")
            print(f"COMPLETE {dataset}: {output}", flush=True)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()


if __name__ == "__main__":
    main()
