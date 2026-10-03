# Reproduce the reconstructed study

Run commands from the repository root. The recorded execution used Linux,
CPython 3.12.14 and CPU PyTorch. All reported results belong to the new
2026-10-03 reconstruction. `historical_context/` contains recovered context and
is never an input to training or analysis.

## Environment

Create an isolated environment using Python 3.12. Install the CPU wheels first,
then the observed primary scientific, acquisition, testing and PDF dependencies:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps --no-build-isolation -e .
```

`results/reproduction_environment.json` records the observed package versions.
The broader `results/environment.txt` is an archival listing, not a portable
installation file: it also includes unrelated host packages and local paths.
PDF compilation additionally requires a TeX installation providing `pdflatex`,
`bibtex`, `kpsewhich`, and the packages used by the supplied ACL template.

## Restore an existing execution or start a fresh one

To resume the recorded execution, restore the exact supplied `data/` and
`results/` artifacts, including feature NPZ files, metadata, checkpoints and
the study ledger. Verify archive checksums and use the same fitting commands
below. A candidate is skipped only after its completion record, ledger identity
and artifact hashes match. An interrupted compatible candidate restarts at
epoch zero. Source, protocol or input changes require a new study output.

For a fresh end-to-end execution, use a separate project tree containing the
source, configuration and manuscript files, with no old `results/` directory.
Pinned raw data and verified model files can be copied into that fresh tree to
avoid downloads. Do not regenerate feature NPZ files inside a tree whose
training ledger already binds the old NPZ byte hashes. Regenerated archives
can differ in bytes even when the numerical feature arrays agree.

The three original benchmark feature caches record encoder source hash
`435f274405d7442a9fdfa26ce051c6183ffbd83cf092686cbc23dc1a462cc702`.
That exact encoder is archived at
`scripts/feature_versions/extract_features_initial.py`. The VE cache records
the current encoder hash
`bddd51416f2723b92baeb03e76d410a2340e3afcf4642ee75a3c6d4863d93f29`.
The update added transactional cache reuse and concurrent-write checks. Keep
the supplied NPZ files and their original metadata together for resume. The
fresh-run commands below use the current encoder in a fresh tree.

## Acquire and audit inputs

```bash
python scripts/prepare_visual_entailment.py --workers 16
python scripts/prepare_benchmarks.py --workers 24
python scripts/audit_inputs.py
python scripts/prepare_model.py
python scripts/prepare_model.py --verify
```

The first command deterministically selects the fixed 1,200/100/100/400
image-disjoint VE splits and retains every official hypothesis and five source
captions per image. The second reconstructs the full 7,511 SugarCrepe pairs,
4,757 SugarCrepe++ triplets and 5,000-image/25,000-caption COCO retrieval pool.
Download failures stop acquisition; examples are not silently substituted.
Source revisions, exact selected IDs, file hashes and source audits are under
each dataset directory. `results/input_audit.json` records actual image-byte
overlap, duplicate captions, conflicting annotations, and reference checks.

Model acquisition pins repository
`laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at revision
`1a25a446712ba5ee05982a381eed697ef9b435cf`. The script checks the protocol's
model pins and fixed SHA256/size values for weights, configuration and model
README. Downloads use Hugging Face Hub with atomic local publication. Existing
mismatched files are never overwritten. `--verify` performs no network access.
The receipt is `data/model/acquisition_receipt.json`.

## Encode frozen features

```bash
python scripts/extract_features.py --dataset visual_entailment --dataset sugarcrepe --dataset sugarcrepe_pp --dataset coco_karpathy --threads 4 --image-batch 32 --text-batch 128
```

Each dataset writes `results/features/DATASET/features.npz` and `metadata.json`.
Manifest row order is preserved. The shared SQLite cache keys images by their
exact bytes and text by its exact string. Text encoding checks causal-padding
truncation against the stock encoder before using it. Encoders, temperature
and normalized 512-dimensional float32 feature definitions remain fixed.

## Fit, select and evaluate

The frozen protocol SHA256 is
`066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847`.
Use the recorded digest explicitly; never replace it with the digest of an
unreviewed edited protocol.

```bash
python scripts/run_study.py plan --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 --threads 2
python scripts/run_study.py fit --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 --threads 2
python scripts/run_study.py select --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 --threads 2
python scripts/evaluate_study.py --torch-threads 2 --block-size 128
python scripts/analyze_study.py --replicates 10000 --bootstrap-seed 20261003
```

Fitting executes 12 methods, three learning rates and three seeds: 108
candidates. Selection uses validation only and writes 36 selected adapters.
Held-out evaluation starts after selection and also evaluates the frozen
reference. The analysis uses paired image bootstrap intervals for the six
declared primary contrasts. Other benchmark/control comparisons remain
descriptive. See `docs/TRAINING_EXECUTION.md` and
`docs/TRAINING_COMMANDS_RECORDED.md` for ledger, selection and resume details.

## Separate relation diagnostic

```bash
python scripts/relation_diagnostic.py --threads 2 --output results/relation_diagnostic_iter10000
```

This trains the image-only, text-only and multimodal logistic classifiers,
selects regularization using validation, and calibrates the frozen selected
classifier on the disjoint calibration images. It never provides labels to
the primary adapter study. The fixed design is in
`docs/RELATION_DIAGNOSTIC_PROTOCOL.md`; the recorded solver-convergence
amendment is `docs/RELATION_DIAGNOSTIC_ITERATION_AMENDMENT.json`. The command
uses the recorded amended output directory, `results/relation_diagnostic_iter10000`.
The original 2000-iteration implementation remains archived under
`scripts/diagnostic_versions/`.

## Compile the paper

The flat Overleaf source is `manuscript/`, with `main.tex` as the root file.
First regenerate the numerical macros, all result tables, both figures and
their data from the new analysis, selected runs and complete diagnostic:

```bash
python scripts/generate_paper_results.py --analysis-dir results/analysis --diagnostic results/relation_diagnostic_iter10000/summary.json --selection results/study/selection.json --output manuscript
```

These paths are the generator's defaults. It validates the complete fresh-run
index and input hash bindings, then writes `manuscript/paper_results_receipt.json`.
`scripts/build_paper.py` verifies the unchanged official ACL style and the
fresh-analysis receipt before compiling. Its final mode rejects pending result
markers, changed analysis inputs, undefined references and overfull boxes.

```bash
python scripts/build_paper.py --check-only
python scripts/build_paper.py
```

The final PDF is `output/pdf/acl27_relation_labels_short.pdf`. A source snapshot
whose result tables are still pending can be inspected with
`python scripts/build_paper.py --draft`; this produces an explicitly named draft.
The supplied paper tables and result receipt describe the supplied execution.
After a new execution, run the generator on that execution's audited analysis
before building its final PDF.

The command-line entry points above were inspected and exercised with `--help`;
`results/reproduction_cli_checks.json` records their source hashes and checks.
No research experiment was rerun merely to write this guide.
