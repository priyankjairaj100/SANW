# What Do Relation Labels Add to Image–Text Retrieval?

This project rebuilds the lost study as a new, auditable execution beginning
2026-10-03. The historical paper and tables in `historical_context/` are
recovered context. All current findings were computed from newly acquired
data, new features and new training runs.

## Current stage

All public data and frozen features have been acquired and verified. All
108 adapter candidates have completed training, and validation has selected
36 checkpoints. All 37 states, including the frozen reference, have completed
held-out evaluation on four benchmarks. The separate relation-classifier
diagnostic also completed. Independent audits verified all selected validation
scores, 629 evaluation aggregates, 60,000 bootstrap replicates, and sampled
full-pool retrieval rankings. The pretraining suite passed 69 tests.

Extra positive labels improved the supervised relation test by 5.81 percentage
points over source-positive adaptation, but SugarCrepe improved by 0.14 points
and COCO image-to-text R@1 fell by 8.56 points. The combined relation policy's
adjusted intervals against positive expansion include zero on both primary
tasks. Full results, seed values and intervals are under `results/analysis/`.
These are newly computed results, not the lost execution's reported values.

The final editorial rewrite is
`output/pdf/acl27_relation_labels_final_rewrite_v2.pdf`, with four main pages
followed by limitations, references, and the complete appendix. Upload
`output/acl27-overleaf-final-rewrite-v2.zip` directly to Overleaf; its source files are
flat and `main.tex` is the entry point. Numerical tables and figures are
generated from the audited results and bound by a provenance receipt.

This version centers on what each use of relation labels buys for search.
The constant-weight control is promoted into the main comparison, the
positive-target analysis follows the empirical evidence, and the complete
negative-weighting theory remains in the appendix. The protocol and scientific
results are unchanged. `docs/EDITORIAL_REWRITE_V2.md` records the changes;
`docs/CONTEXT_INVENTORY_FINAL_REWRITE.md` maps every accessible old and current
theorem, experiment, figure, and unused idea. The preceding source/PDF snapshot
is preserved in `historical_context/pre_editorial_rewrite_20261003_source.zip`.

`docs/RERUN_PROTOCOL.json` fixes choices before fitting or held-out scoring.
`docs/LOSS_RECONSTRUCTION.md` specifies every objective. `docs/INTERFACES.md`
defines data and feature contracts. Dataset source revisions and content
hashes are saved with each manifest.

The main protocol SHA256 is
`066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847`.
The classifier diagnostic has a separately recorded iteration-limit amendment
in `docs/RELATION_DIAGNOSTIC_ITERATION_AMENDMENT.json`; this does not change
the adapter protocol, data, training, or selection.

The deprecated early e-SNLI-VE release has overlapping development/test rows.
The new execution uses the authors' later e-ViL release with image-disjoint
official development/test splits. This source correction was chosen before
feature scoring or training. Exact selections and all rows are retained.

## Environment

Use Python 3.12 and CPU PyTorch 2.7.1 / torchvision 0.22.1. Install from the
official CPU wheel index, then `pip install -e .`. The observed full package
list is in `results/environment.txt`. This environment does not need a GPU.

## Workflow

1. `python scripts/prepare_visual_entailment.py` and
   `python scripts/prepare_benchmarks.py` acquire the public inputs.
2. Download the pinned model named in the protocol to `data/model/`.
3. `python scripts/extract_features.py --dataset visual_entailment` encodes
   training/development/test images and texts. Repeat for `sugarcrepe`,
   `sugarcrepe_pp`, `coco_karpathy`. Exact contents share a resumable feature
   bank while each benchmark keeps its own official strings.
4. `python scripts/run_study.py fit --protocol-sha256 HASH` fits the frozen
   108-candidate grid. The required hash is SHA256 of the protocol file.
5. `python scripts/run_study.py select --protocol-sha256 HASH` freezes the
   36 selected states before held-out evaluation.
6. `python scripts/evaluate_study.py` writes item-level scores and ranks.
7. `python scripts/analyze_study.py` independently audits raw scores and
   constructs the planned primary contrasts and descriptive secondary tables.
8. `python scripts/generate_paper_results.py` regenerates numerical paper
   assets; `python scripts/build_paper.py` compiles and verifies the paper.

Exact invocation options and execution receipts are retained in `docs/` and
`results/`.

## Recovery ZIPs

Small immutable ZIPs are delivered at completed stages. Each contains a
file manifest with SHA256 checks and passes ZIP CRC verification. Extract
subsequent stage checkpoints into the same directory. Heavy data and feature
packs are separately identified; a code checkpoint does not contain raw data
or model weights unless its manifest lists them.

Use `python scripts/make_checkpoint.py --name STAGE` to snapshot the current
code, protocols, manifests, and compact result files. Public raw data and
model pins allow reacquisition. The complete project ZIP also retains raw
images, model weights, feature matrices, all candidate and selected states,
and item-level predictions. `docs/RECOVERY_README.md` explains exact-byte
verification and reproduction. The compact companion omits those heavy assets
and includes a precise `PACKAGE_SCOPE.txt`.
