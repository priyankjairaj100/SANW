# Project recovery index

This branch preserves the original study and the completed matched-assignment follow-up. The new paper is in `manuscript_review_v3/`; original manuscript and scientific outputs remain in their original paths. The checkpoint ZIPs are incremental snapshots, while a fresh clone already combines the files committed in each stage.

```bash
git clone --branch review-followup-controls-20261003 https://github.com/priyankjairaj100/SANW.git
cd SANW
```

## Checkpoints 11–15

| Stage | Preserved scope | Recovery records |
| --- | --- | --- |
| 11 | Frozen follow-up protocol, freeze receipt and gap audit; three project documents | [Protocol](docs/REVIEW_FOLLOWUP_PROTOCOL.json), [freeze receipt](docs/REVIEW_FOLLOWUP_PROTOCOL_FREEZE.json), [gap audit](docs/REVIEW_GAP_AUDIT_20261003.md); these documents are repeated in stage12 |
| 12 | Implementation/tests, six frozen assignments, execution plan/ledger, split and feature provenance, exact dev900/test1000 feature caches | [Guide](docs/STAGE12_RECOVERY_README.md), [manifest](recovery/stage12/RECOVERY_MANIFEST.json), [ZIP receipt](recovery/stage12/receipt.json) |
| 13 | All 72 training histories/completions, all-state manifest, both selectors, training/replay audits, and exact 32 unique selected checkpoints | [Guide](docs/STAGE13_RECOVERY_README.md), [omitted-artifact and selection inventory](results/review_followup/recovery_stage13_inventory.json), [ZIP receipt](recovery/stage13/receipt.json) |
| 14 | All completed indices/pre-score receipts, raw predictions, audits, analysis/bootstrap outputs, figures/data/captions, report, and decision/design documents | [Guide and all 24 raw-part links](docs/STAGE14_RECOVERY_README.md), [raw archive manifest](recovery/stage14/PREDICTION_ARCHIVES.json), [summary ZIP receipt](recovery/stage14/receipt.json) |
| 15 | New v3 PDF, flat Overleaf source ZIP, source/figures/tables/macros/appendices, build/generation code and provenance; this recovery index; seven legacy inputs for the strict paper builder (Git only) | [PDF](output/pdf/acl27_review_followup_v3.pdf), [flat Overleaf ZIP](output/acl27-overleaf-review-followup-v3.zip), [paper source](manuscript_review_v3/main.tex), [checkpoint manifest](recovery/stage15/RECOVERY_MANIFEST.json), [ZIP receipt](recovery/stage15/receipt.json) |

Verified Git snapshots: [stage12](https://github.com/priyankjairaj100/SANW/commit/15026ac00d51df7c519ceeab32b2eb2074d2d581), [stage13](https://github.com/priyankjairaj100/SANW/commit/a237c9380a7fe6ca5f6527ac58913ac42f39b02d), and [stage14](https://github.com/priyankjairaj100/SANW/commit/9032bf0a7c09fee32c6f5ba61470da30b87c26ef). The follow-up branch also carries the later stage15 paper checkpoint.

Stage11's original `checkpoint-11-frozen-followup-protocol.zip` is 11,985 bytes, SHA-256 `44f655e28edb9c9e13bdc86dd72293c0f653d095c511ffac9ce592c20e1a80ac`. Its three scientific documents are already present in Git. Earlier original-study archives are cataloged in [the historical checkpoint index](docs/STAGED_RECOVERY_INDEX.md); original Git transfer scope is recorded in `GIT_SNAPSHOT_MANIFEST.json` and `GITHUB_RECOVERY.md`.

The remaining compact ZIPs can be recreated from their Git-backed files without reacquiring raw images or encoder model files:

```bash
python recovery/stage12/rebuild_checkpoint.py --verify-only
python recovery/stage13/rebuild_checkpoint.py --verify-only
python recovery/stage14/rebuild_summary.py --verify-only
python recovery/stage15/rebuild_checkpoint.py --verify-only
```

Drop `--verify-only` to write a new ZIP under `output/recovery/`, or pass `--output /path/to/new.zip`. Existing output files are never overwritten. The manifest checks payload bytes independently of ZIP compression-library versions.

| Checkpoint ZIP | Bytes | SHA-256 |
| --- | ---: | --- |
| `checkpoint-12-review-followup-inputs-source-features.zip` | 28,104,429 | `508c13fc4535a8acd89c4f882d33ffa77f7a141ccfb1ab0cdc744500dd9e88ba` |
| `checkpoint-13-completed-training-selected-states.zip` | 20,070,902 | `b68b0b97f7d32611da71ffd4796840d39182034889d1f1a2d7e936c3c2d6b036` |
| `checkpoint-14-completed-evaluation-analysis-summary.zip` | 1,416,448 | `f30684148641a03362e31ce63123ef87226e8895feb7939f0b5db2f7c92ce0f9` |
| Stage15 paper checkpoint | See its [receipt](recovery/stage15/receipt.json) | See its [receipt](recovery/stage15/receipt.json) |

## Reanalyze the completed experiments without training

First restore the raw predictions from the 24 Git-backed archive parts:

```bash
python recovery/stage14/restore_predictions.py
python recovery/stage14/restore_predictions.py --verify-restored
```

This restores exactly 1,080 NPZ files and their 1,080 paired metadata JSONs to the paths recorded in the three evaluation indices. The parts total 189,700,260 bytes and each is below 8 MiB. They are the authoritative raw-prediction Git backup; the 2,160 individual files are not duplicated as flat Git blobs. The standard-library restorer validates whole-archive and member SHA-256 hashes, sizes and CRCs, preserves matching files, and rejects differing existing files.

Install the project's declared Python dependencies and recompute into a fresh directory:

```bash
python scripts/analyze_review_followup.py \
  --indices results/review_followup/evaluation/terminal_index.json \
            results/review_followup/evaluation/selected_index.json \
            results/review_followup/evaluation/trajectory_index.json \
  --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json \
  --protocol-sha256 3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34 \
  --output results/review_followup/analysis_recovered
```

This path performs no training or model inference. It needs no adapter/model weights, image files or feature caches. In addition to the restored NPZs and the three indices/pre-score receipts, it reads the frozen protocol, Python helpers and original `data/visual_entailment/manifest.json` to define original400 versus additional600 queries. All are in Git. NumPy and PyTorch must be installed because the code imports those helpers. The archive receipt in `recovery/stage14/restore_verification_receipt.json` records a complete empty-directory restoration and hash check.

The published [complete report](results/review_followup/COMPLETE_RESULTS.md), [figures](results/review_followup/figures/), [statistical analysis](results/review_followup/analysis/), [story assessment](docs/REVIEW_FOLLOWUP_STORY_ASSESSMENT.md) and [independent primary audit](results/review_followup/analysis/independent_primary_audit.json) can be inspected directly without restoring raw predictions. [The allocation design](docs/REVIEW_FOLLOWUP_ALLOCATION_DESIGN_NOT_RUN.md) is a proposal and has not been run.

## Weights, features and rerun limits

The completed grid contains **72 candidates × 11 saved epochs = 792 checkpoint files**. Git/stage13 preserves the **32 distinct selected checkpoint files** referenced by **48 selections** across two selectors. The other **760 epoch weights** and **792 validation-cache NPZs** are omitted from delivery; their paths, sizes and hashes are recorded in the stage13 inventory. They remained untouched in the execution workspace when packaging.

The unchanged evaluator validates all 792 checkpoint hashes before it filters a scoring suite. Restoring only the 32 selected weights therefore does not enable even its selected-suite CLI. Reanalysis of preserved raw predictions is independent of that restriction. To score again with the strict evaluator, provide the complete checkpoint set and required dataset feature caches; to refit omitted checkpoints, use a separate clean reproduction copy of the original training inputs plus stage12 and follow [the reproduction guide](docs/REVIEW_FOLLOWUP_README.md). Do not prepopulate partial candidate folders with completion receipts before fitting: their missing declared artifacts will fail completion checks.

The two new source-caption feature caches (dev900 and official e-ViL test1000) are included in stage12. Original large feature caches, raw images and encoder model files are not supplied by the follow-up checkpoints; full scoring/training/re-encoding needs the corresponding earlier assets or documented reacquisition. Neither the calibration100 nor held-out test was used as source-retrieval selection data. The source-retrieval validation pool is official e-ViL dev minus calibration100; the test pool is official e-ViL test, not the overlapping Karpathy test split.

## Paper recovery

Stage15 keeps the new paper separate from the original manuscript. Its source ZIP is flat and intended for direct Overleaf import. The compiled PDF, build receipt and source hash inventory identify the delivered version. Local TeX runtime files, rendered page previews and build scratch are excluded. Use the supported commands in [the paper README](manuscript_review_v3/OVERLEAF_README.txt) and inspect [the delivery receipt](output/review_followup_v3_delivery_receipt.json) and [build receipt](output/pdf/paper_build_receipt_v3.json); do not run one-time editorial patch helpers. Rebuilding the supplied TeX requires a compatible LaTeX installation, not model weights or training inputs. Regenerating data-driven tables requires the committed completed analysis records and generator source.

The delivered PDF is 582,796 bytes, SHA-256 `330b0be3c46041e295dcbf48577d8a26d0dd508f49ad69d128aca8f5ad806137`; the flat Overleaf ZIP is 980,671 bytes, SHA-256 `d13d02a3eec30e88895b0085a3388e1572059cf6b35700f7fd342332b3d4b116`. The main text occupies four pages; the complete paper, references and appendices occupy 26 pages.

For the Python builder and packager, install the verified paper dependency separately:

```bash
python -m pip install -r requirements-paper-v3.txt
python scripts/build_review_paper.py --check-only
```

This pins PyMuPDF 1.26.6, matching both the observed build runtime and the existing full `requirements-reproduction.txt`; the original requirements files are unchanged. Direct Overleaf/pdfLaTeX compilation needs no Python package installation.

The strict Python builder additionally validates seven original-study inputs that were omitted from the first Git snapshot: `data/coco_images/000000060623.jpg` and `results/evaluation/{clip,multipositive}_seed_{17,29,43}/coco_karpathy.npz`. Stage15 restores all seven exact files as normal Git files (19,625,569 bytes total), **outside the compact paper checkpoint ZIP**. A fresh clone includes them automatically. The supplied flat Overleaf ZIP remains self-contained and does not need these provenance inputs. If restoring only checkpoint ZIPs, use Overleaf/plain LaTeX or obtain those seven files from the branch before running the strict builder. [The Git recovery audit](docs/PAPER_V3_GIT_RECOVERY_AUDIT.json) records every hashed builder input and the seven added-file hashes.
