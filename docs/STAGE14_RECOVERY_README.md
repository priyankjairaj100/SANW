# Stage 14: completed evaluation and analysis recovery

The summary ZIP contains the completed reports, figures, analysis (including bootstrap samples), all three evaluation indices and pre-score receipts, audit records, scientific scripts, and this recovery guide. It is an incremental checkpoint on top of the original project and stages 12–13. The separately stored prediction parts preserve all **1,080 NPZ files and 1,080 paired metadata JSON files**, byte for byte. The new v3 manuscript is a later checkpoint and is excluded here.

## Recover from Git

```bash
git clone --branch review-followup-controls-20261003 https://github.com/priyankjairaj100/SANW.git
cd SANW
python recovery/stage14/restore_predictions.py
python recovery/stage14/restore_predictions.py --verify-restored
```

The restore utility uses only the Python standard library (Python 3.9+). It validates every archive, member path, byte count, SHA-256 and ZIP CRC against `recovery/stage14/PREDICTION_ARCHIVES.json`, then restores exact repository-relative paths. Existing matching files are retained; differing files cause an error and are never overwritten. `--verify-only` checks all archives without restoring. The 24 archives total 189,700,260 bytes; each is below 8 MiB. No model, image, feature cache or checkpoint is required for restoration.

For individual downloads, overlay the summary ZIP's `project/` directory onto the recovered project. Keep all 24 prediction ZIPs together in a directory and run:

```bash
python recovery/stage14/restore_predictions.py --archives-dir /path/to/downloads
```

Both repository names `predictions-part-XX.zip` and download names `checkpoint-14-predictions-part-XX.zip` are accepted. Each part is an independent ZIP; do not concatenate them. Its top-level `BUNDLE_MANIFEST.json` describes only that part and is not a project file. The utility handles this automatically.

## Recompute the analysis without weights

Install the project's declared Python dependencies; the analysis imports NumPy and PyTorch helpers but performs no model inference. Run from the restored repository:

```bash
python scripts/analyze_review_followup.py \
  --indices results/review_followup/evaluation/terminal_index.json \
            results/review_followup/evaluation/selected_index.json \
            results/review_followup/evaluation/trajectory_index.json \
  --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json \
  --protocol-sha256 3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34 \
  --output results/review_followup/analysis_recovered
```

Use a fresh output directory. This reads the restored raw NPZ files, three indices/pre-score receipts, frozen protocol and Python source helpers. It also reads the original `data/visual_entailment/manifest.json` (SHA-256 `85f948579c3ec2365b9e7d1ab60807c0a7ac7a012ec339da5e91e9ba7c3cf495`) to label original400 versus additional600 queries. That manifest and original helper scripts are already in Git/the original project; they are not duplicated in this incremental summary ZIP. Analysis requires no image files, model or adapter weights, feature NPZs, or full state checkpoint set.

The published report is `results/review_followup/COMPLETE_RESULTS.md`; plots, captions, plotted CSV data and provenance are in `results/review_followup/figures/`. `docs/REVIEW_FOLLOWUP_STORY_ASSESSMENT.md` interprets the completed evidence; `docs/REVIEW_FOLLOWUP_ALLOCATION_DESIGN_NOT_RUN.md` describes a proposed experiment that has **not** been run. To regenerate plots in a separate working copy, run `python scripts/plot_review_followup_results.py`. This script reads the canonical analysis directory and rewrites canonical figures/report/digest, so do not run it on a preservation copy.

## What each rerun needs

| Operation | Required artifacts |
| --- | --- |
| Inspect reports and figures | Summary ZIP or normal Git files; no raw prediction restore needed |
| Verify/restore predictions | All 24 parts, archive manifest and standard-library restore script |
| Recompute statistics | Restored raw NPZs, all three indices/receipts, frozen protocol, original visual-entailment manifest and code/dependencies |
| Recreate plots/report | Canonical analysis/audits, protocol, state manifest, indices and figure source/hash dependencies; NumPy, PyTorch and Matplotlib |
| Score adapters again | Appropriate original/follow-up dataset manifests and feature caches, protocol/ledger/selector records and **all 792 checkpoint files** required by the unchanged strict evaluator |
| Refit omitted checkpoints | Original training features/model configuration, stage12 assignments and source, protocol/environment; fit the full grid in a separate clean reproduction copy |
| Re-encode features | Pinned model plus raw dataset images/text; see the preparation commands in `docs/REVIEW_FOLLOWUP_README.md` |

Stage13 preserves the exact 32 unique selected checkpoints for 48 selections. It omits 760 other epoch weights and 792 validation-cache NPZs. The strict frozen evaluator validates all 792 checkpoint hashes before selecting a scoring suite; selected-only checkpoint recovery therefore does not by itself enable that CLI. This limitation does not affect raw-prediction analysis. See `docs/STAGE13_RECOVERY_README.md` for reproduction and omitted-file hashes.

The compact summary itself can be verified or rebuilt from Git with `python recovery/stage14/rebuild_summary.py --verify-only` or `--output /path/to/new-summary.zip`. `recovery/stage14/receipt.json` records its SHA-256 and deterministic ZIP recipe. Raw prediction files are backed by the archive parts, not duplicated as 2,160 flat Git blobs.

## Individual prediction parts

Hashes and exact member lists are in `recovery/stage14/PREDICTION_ARCHIVES.json`.

| Part | Download/open in Git | Bytes |
| --- | --- | ---: |
| 01 | [predictions-part-01.zip](../recovery/stage14/predictions-part-01.zip) | 8,201,437 |
| 02 | [predictions-part-02.zip](../recovery/stage14/predictions-part-02.zip) | 8,249,147 |
| 03 | [predictions-part-03.zip](../recovery/stage14/predictions-part-03.zip) | 8,252,143 |
| 04 | [predictions-part-04.zip](../recovery/stage14/predictions-part-04.zip) | 8,105,101 |
| 05 | [predictions-part-05.zip](../recovery/stage14/predictions-part-05.zip) | 8,256,013 |
| 06 | [predictions-part-06.zip](../recovery/stage14/predictions-part-06.zip) | 8,146,328 |
| 07 | [predictions-part-07.zip](../recovery/stage14/predictions-part-07.zip) | 8,208,112 |
| 08 | [predictions-part-08.zip](../recovery/stage14/predictions-part-08.zip) | 8,278,329 |
| 09 | [predictions-part-09.zip](../recovery/stage14/predictions-part-09.zip) | 8,082,877 |
| 10 | [predictions-part-10.zip](../recovery/stage14/predictions-part-10.zip) | 8,236,447 |
| 11 | [predictions-part-11.zip](../recovery/stage14/predictions-part-11.zip) | 8,252,350 |
| 12 | [predictions-part-12.zip](../recovery/stage14/predictions-part-12.zip) | 8,137,704 |
| 13 | [predictions-part-13.zip](../recovery/stage14/predictions-part-13.zip) | 8,212,392 |
| 14 | [predictions-part-14.zip](../recovery/stage14/predictions-part-14.zip) | 8,276,149 |
| 15 | [predictions-part-15.zip](../recovery/stage14/predictions-part-15.zip) | 8,079,499 |
| 16 | [predictions-part-16.zip](../recovery/stage14/predictions-part-16.zip) | 8,139,133 |
| 17 | [predictions-part-17.zip](../recovery/stage14/predictions-part-17.zip) | 8,225,042 |
| 18 | [predictions-part-18.zip](../recovery/stage14/predictions-part-18.zip) | 7,998,778 |
| 19 | [predictions-part-19.zip](../recovery/stage14/predictions-part-19.zip) | 8,189,351 |
| 20 | [predictions-part-20.zip](../recovery/stage14/predictions-part-20.zip) | 8,231,407 |
| 21 | [predictions-part-21.zip](../recovery/stage14/predictions-part-21.zip) | 8,096,195 |
| 22 | [predictions-part-22.zip](../recovery/stage14/predictions-part-22.zip) | 8,095,195 |
| 23 | [predictions-part-23.zip](../recovery/stage14/predictions-part-23.zip) | 8,228,964 |
| 24 | [predictions-part-24.zip](../recovery/stage14/predictions-part-24.zip) | 1,522,167 |
