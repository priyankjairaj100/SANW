# GitHub source snapshot and recovery

This repository contains the scientific source and compact results of the
2026-10-03 reconstructed execution of **What Do Relation Labels Add to
Image–Text Retrieval?** The original scientific files in this snapshot are
copied byte-for-byte from `acl27_relation_labels_complete_project_v2.zip`.
Historical recovered material is identified in `historical_context/`; it is
distinct from the newly computed execution.

## Included

- Python source, tests, acquisition/training/evaluation scripts, pinned
  requirements, protocols, reproduction instructions, and historical context.
- Flat ACL/Overleaf sources in `manuscript/`, the final PDF at
  `output/pdf/acl27_relation_labels_final_rewrite_v2.pdf`, and the upload-ready
  `output/acl27-overleaf-final-rewrite-v2.zip`.
- Ordered dataset manifests, provenance, public source pins, acquisition
  receipts, training histories and selection ledgers, aggregate results and
  audits, compact discrimination predictions, bootstrap samples, and the
  complete retained relation-classifier diagnostic outputs.

## Heavy assets outside this Git snapshot

The repository omits raw images and source annotation downloads, the pretrained
model weights, feature NPZ matrices and redundant SQLite cache, all adapter
checkpoint `.pt` files, and the full COCO item-level prediction/ranking NPZs.
Build/runtime caches, test scratch output, download fragments, and lockfiles
are also excluded. The complete recovery ZIP retains the scientific heavy
assets, but that ZIP is **not included or published as a GitHub release by this
snapshot**. This repository alone cannot restore every original computed byte.

Existing hash receipts refer to the original complete execution and can name
assets absent here. They are preserved as evidence, not rewritten to claim a
complete local recovery. `GIT_SNAPSHOT_MANIFEST.json` is the exact inventory of
the files selected for Git. Its entries record original relative paths, byte
counts, SHA-256, and Git blob SHA-1; the manifest excludes its own hashes to
avoid a circular definition. Original execution logs may contain historical
local filesystem paths; these are provenance records, not setup instructions.

## Rebuild the paper

Upload `output/acl27-overleaf-final-rewrite-v2.zip` to Overleaf and use
`main.tex`, or compile `manuscript/main.tex` with a local LaTeX installation.
The numerical tables and figures are already included. The Python paper tools
are `scripts/generate_paper_results.py` and `scripts/build_paper.py`.

## Reacquire inputs and rerun the computation

Follow `docs/REPRODUCE.md` and `docs/RECOVERY_README.md` for the pinned environment,
full commands, and verification details. Run from the repository root. Public
input acquisition requires internet access; acquisition scripts check pinned
sources and recorded identities.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e .
python scripts/prepare_visual_entailment.py
python scripts/prepare_benchmarks.py --workers 16
python scripts/prepare_model.py
python scripts/extract_features.py --dataset visual_entailment --dataset sugarcrepe --dataset sugarcrepe_pp --dataset coco_karpathy --threads 4
```

Use new result directories to preserve the recorded compact evidence:

```bash
python scripts/run_study.py fit --output results/reproduction-study --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 --threads 2
python scripts/run_study.py select --output results/reproduction-study --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847
python scripts/evaluate_study.py --selection results/reproduction-study/selection.json --output results/reproduction-evaluation --torch-threads 2
python scripts/analyze_study.py --index results/reproduction-evaluation/index.json --output results/reproduction-analysis --replicates 10000 --bootstrap-seed 20261003
python scripts/relation_diagnostic.py --output results/reproduction-relation-diagnostic --threads 2
```

Recomputation produces a new execution; compare numerical results using the
recorded environment and audits. Exact recovery of omitted original model,
feature, checkpoint, or prediction files requires those archived assets.
