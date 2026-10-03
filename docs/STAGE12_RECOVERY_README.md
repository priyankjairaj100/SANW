# Stage 12: frozen follow-up inputs, implementation, and feature caches

This incremental checkpoint preserves the implementation and inputs for the
relation-assignment follow-up. It is an addition to the reconstructed project,
not a complete standalone project archive or a completed-results release.

The controlling specification is `docs/REVIEW_FOLLOWUP_PROTOCOL.json`, SHA256
`3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34`.
Its freeze receipt is included. Scientific choices, interpretation boundaries,
and full commands are in `docs/REVIEW_FOLLOWUP_README.md`.

## Included scope

- New follow-up source, tests, protocol/freeze, and scientific/reproduction docs.
- All six immutable randomized-assignment JSONs and their hash ledger.
- The frozen execution ledger, full execution plan, and completed training/control
  test receipt; active candidate files and fit logs are excluded.
- The source-only e-ViL development pool of 900 images and 4,500 captions, with
  original calibration100 excluded, and the official e-ViL test pool of 1,000
  images and 5,000 captions.
- Both pools' exact frozen-encoder feature NPZ files, metadata, manifests,
  acquisition provenance, and split/construction/bank-reuse audit receipts.

The development pool contains original validation100 plus additional800. The
test pool contains original test400 plus additional600. Each image records its
`review_subgroup`; any later subgroup retrieval summary must keep the complete
1,000-image/5,000-caption test candidate pool. These are e-ViL-defined splits
and Flickr30k Entities captions, not the standard Karpathy Flickr1k task.

All forbidden fitting/calibration/development/test image-ID and exact-image-byte
intersections passed the recorded audit. Original feature rows were reused
bit-for-bit; additional images and captions were encoded with the pinned
OpenCLIP model and preprocessing. No adapter has been applied to these feature
files. The checkpoint does not contain follow-up retrieval scores.

## Excluded scope and prerequisites

The checkpoint excludes the 1,400 newly downloaded raw JPEGs, the original
baseline project assets, model weights, SQLite bank, and any live follow-up
training weights, histories, selections, or evaluation results. Original base
files remain prerequisites. In particular, follow-up fitting requires the
original training manifest and exact feature NPZ/metadata, original protocol,
and original study records used by the replication checks. Original benchmark
feature caches are required for their later evaluations.

The new feature NPZs are sufficient for working with the two new pools without
their raw image files or re-encoding. Rebuilding these exact saved files is
unnecessary; retain their paired metadata. Reacquisition and re-encoding are
separate operations that require the public sources, original pinned model,
and matching environment. See `scripts/review_followup_prepare_flickr_pools.py`.

## Extract and verify

The ZIP uses the existing `project/` layout. Extract into the same recovery
parent used for the earlier reconstructed project, so the new project files
join the base project. The ZIP's `RECOVERY_MANIFEST.json` describes only this
checkpoint's scope.

```bash
python -m zipfile -e checkpoint-12-review-followup-inputs-source-features.zip recovery
python recovery/verify_recovery.py --archive checkpoint-12-review-followup-inputs-source-features.zip
python recovery/verify_recovery.py --root recovery --allow-unlisted
```

The archive check verifies the immutable checkpoint without writing files.
The extracted-tree check permits earlier base files outside this checkpoint's
manifest. Save the ZIP unchanged when later live source or result files evolve.

## Run from the restored project root

Use the original pinned Python 3.12/CPU PyTorch environment described in
`docs/REPRODUCE.md`. Inspect the plan before fitting; the assignment files in
this checkpoint are already prepared and hash-bound.

```bash
FOLLOWUP_PROTOCOL_SHA=3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/run_review_followup.py plan --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py fit --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py select --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
```

Fit all 72 declared candidates and freeze both selection strategies before any
follow-up held-out scoring. The companion README describes the terminal,
selected, and trajectory evaluation suites and the separate 18-effect primary
statistical family. Do not substitute partial live outputs for completed and
verified study artifacts.
