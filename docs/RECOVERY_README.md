# Recover and reproduce this project

This archive preserves the execution reconstructed on 2026-10-03. Its protocol,
inputs, feature files, adapter checkpoints, item-level predictions, analysis,
and manuscript source are separate, inspectable records. Historical recovered
text is retained in `historical_context/`.

## Recover the complete ZIP

Use Python 3.10 or newer to extract and verify the archive. These steps require
only the Python standard library. Replace `COMPLETE_PROJECT.zip` with the actual
downloaded filename.

```bash
python -m zipfile -e COMPLETE_PROJECT.zip recovery
python recovery/project/scripts/verify_recovery.py --root recovery
```

The verifier reads `RECOVERY_MANIFEST.json`, checks every listed byte count and
SHA256, and reports missing, changed, or unlisted files. Success means exit
status `0` and `"complete": true`. To verify a ZIP directly without extracting
another copy:

```bash
python recovery/project/scripts/verify_recovery.py --archive COMPLETE_PROJECT.zip
```

The archive manifest is the exact contents record. `project/results/archive_inventory.json`
explains each excluded file. Exclusions cover incomplete downloads, runtime and
test caches, TeX compilation caches, and the redundant SQLite feature bank.
Every bank vector was compared byte-for-byte with the retained NPZ files;
`project/results/feature_bank_redundancy.json` records that audit. All candidate
and selected checkpoints are retained, including repeated states.

## Recover incremental checkpoints and asset parts

Extract the newest code checkpoint into an empty common directory. Extract
additional required data/feature/model checkpoints into that same directory.
Every ZIP uses the same `project/` prefix. Later code checkpoints update the
working files while earlier ZIPs remain available as historical snapshots.

The feature parts named `checkpoint-05-...-part-...zip` contain exact binary
chunks. They place `ASSET_DESCRIPTORS/`, `ASSET_PARTS/`, and `restore_assets.py`
beside `project/`. After extracting every received part, run:

```bash
python recovery/restore_assets.py --root recovery
```

The report identifies every asset and any missing zero-based part indices.
Exit status `2` means required parts or descriptors are missing; exit status
`1` means corruption or a conflict. Exit status `0` means all selected assets
are present and match their complete-file hashes. Existing different files are
preserved. Use `--overwrite` only after choosing to replace such a file. If two
versions target one path, choose the intended descriptor with `--asset-id ID`.

Restoration concatenates verified chunks without loading or rewriting the NPZ
or checkpoint format. It preserves the complete original file bytes, including
NPZ container metadata. The five real checkpoint-05 archives were extracted
and restored in a separate directory; both reconstructed feature files matched
their originals and metadata hashes exactly. The receipt is
`project/results/asset_roundtrip_checkpoint05.json`.

An incremental code manifest covers its listed files. A combined recovery
directory also contains assets and files from other checkpoints. To verify
only that code checkpoint's scope, explicitly permit these additional files:

```bash
python recovery/project/scripts/verify_recovery.py --root recovery \
  --manifest CHECKPOINT_MANIFEST.json --allow-unlisted
```

## Project contents

| Path inside `project/` | Contents |
|---|---|
| `docs/RERUN_PROTOCOL.json` | Fixed data, model, objective, training, selection, and evaluation decisions |
| `docs/LOSS_RECONSTRUCTION.md` | Exact definitions of all twelve training objectives |
| `data/*/manifest.json`, `provenance.json` | Ordered evaluation/training records and pinned acquisition sources |
| `data/coco_images/` | Shared official COCO image bytes and per-image receipts |
| `data/visual_entailment/` | Selected Flickr images, annotations, and source history |
| `data/model/` | Complete pinned OpenCLIP weights and configuration |
| `results/features/*/features.npz` | Exact saved normalized image/text features and row IDs |
| `results/study/` | All 108 candidate runs, histories, 36 selected states, and the selection ledger |
| `results/evaluation/` | Fresh item-level scores, rankings, and evaluation index |
| `results/analysis/` | Audited aggregates, contrasts, and uncertainty estimates |
| `results/relation_diagnostic_iter10000/` | Separate relation-classifier diagnostic under its recorded iteration amendment |
| `manuscript/` | Flat ACL/Overleaf source, references, figures, and result receipts |
| `output/pdf/` | Delivered compiled PDFs and build receipts |
| `historical_context/` | Recovered prior text and research context |

SugarCrepe and SugarCrepe++ retain their own official caption strings. Their
first-positive scores are not interchangeable: the latter revises some
original positives and negatives. COCO retrieval uses the first five source
captions per image, giving exactly 25,000 captions for 5,000 test images.
The ten omitted sixth captions are explicitly recorded in provenance.

## Install the research environment

The recorded execution used Python 3.12 on Linux and CPU computation. No GPU or
paid service is required. `requirements-reproduction.txt` pins the 46 relevant
installed distributions; `results/environment.txt` preserves the broader
observed environment. From the recovered `project/` directory:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Package installation needs internet access. Complete archived scientific
inputs and computed assets can be used offline afterward. The model file's
SHA256 is `ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`;
the feature extractor checks it before use.

## Recompute analysis from the preserved predictions

This route independently rebuilds the analysis without retraining or encoding
images. It writes to a new directory so the preserved analysis remains intact.

```bash
python scripts/analyze_study.py \
  --index results/evaluation/index.json \
  --output results/reproduction-analysis \
  --replicates 10000 --bootstrap-seed 20261003
```

The analysis verifies prediction hashes and requires the complete selected
study: one frozen reference and 36 selected adapters on all four benchmarks.

## Rerun training and held-out evaluation

The preserved feature files avoid repeating encoder work. Run the fixed grid
into new output directories. The frozen protocol SHA256 is
`066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847`.

```bash
python scripts/run_study.py fit --output results/reproduction-study \
  --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 \
  --threads 2
python scripts/run_study.py select --output results/reproduction-study \
  --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847
python scripts/evaluate_study.py \
  --selection results/reproduction-study/selection.json \
  --output results/reproduction-evaluation --torch-threads 2
python scripts/analyze_study.py \
  --index results/reproduction-evaluation/index.json \
  --output results/reproduction-analysis-new-training \
  --replicates 10000 --bootstrap-seed 20261003
```

Fit and selection precede held-out scoring. Selection is determined from the
validation rule in the protocol. Saved binary assets recover exactly; new
floating-point executions should be compared through their recorded numerical
outputs and environment receipts.

To reproduce the separate diagnostic with its documented 10,000-iteration
solver ceiling:

```bash
python scripts/relation_diagnostic.py \
  --output results/reproduction-relation-diagnostic --threads 2
```

## Reacquire inputs or recompute features

The complete recovery archive already supplies the selected image bytes and
model weights. If an input is absent, the acquisition scripts resume downloads
from the pinned public sources and verify their content. Retain
`data/benchmark_source_pins.json` with the benchmark downloader.

```bash
python scripts/prepare_visual_entailment.py
python scripts/prepare_benchmarks.py --workers 16
python scripts/audit_inputs.py --output results/reproduction-input-audit.json
```

Run the encoder in a separate project copy with `results/features/` absent if
new encoding is required. Existing validated feature files are deliberately
reused by the extractor.

```bash
python scripts/extract_features.py \
  --dataset visual_entailment --dataset sugarcrepe \
  --dataset sugarcrepe_pp --dataset coco_karpathy --threads 4
```

No SQLite bank is needed initially. The extractor creates one as a temporary
cross-dataset cache and writes complete standalone NPZ files. Model identity,
manifest hashes, preprocessing, normalization, and text-encoder parity checks
are recorded in each feature metadata file.

## Rebuild the paper

Upload the flat files in `manuscript/` to Overleaf and compile `main.tex` with
pdfLaTeX. For a local build, install a standard TeX Live environment providing
pdfLaTeX, BibTeX, the required LaTeX packages, and Type 1 fonts. Then run:

```bash
python scripts/build_paper.py
```

The final build verifies official ACL style hashes and the fresh-result input
receipt. `--draft` is reserved for intentional drafts containing pending
results. Generated PDF files are placed in `output/pdf/`.

To regenerate the tables and figures from the supplied audited execution
before compiling, run:

```bash
python scripts/generate_paper_results.py \
  --analysis-dir results/analysis \
  --diagnostic results/relation_diagnostic_iter10000/summary.json \
  --selection results/study/selection.json --output manuscript
python scripts/build_paper.py
```

See `docs/REPRODUCE.md` for the full fresh-run pipeline and its input receipts.

## Make another small asset checkpoint

Each invocation creates one immutable ZIP, capped at 35 MiB including ZIP
headers. The default payload chunk is 32 MiB. Its receipt reports the total
number of parts, the complete asset hash, and the individual part hash.

```bash
python scripts/package_asset_part.py \
  --file results/features/visual_entailment/features.npz \
  --part-index 0 --chunk-mib 32 --output ../feature-part-000.zip \
  --include results/features/visual_entailment/metadata.json
```

Repeat with each zero-based part index and a new output filename. Existing
archives are never overwritten. Original scientific files are never removed.
