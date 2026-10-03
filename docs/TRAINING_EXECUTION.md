# Reconstructed fitting and selection

This code produces a new execution. It does not load historical result tables.
The immutable ledger records the exact protocol digest, source files, feature
cache, manifest, feature metadata, training and validation IDs, and runtime.
The ledger builder compares every scientific `StudyConfig` field with a
structured field in the frozen protocol. A matching protocol digest cannot
authorize a changed learning-rate grid or other altered scientific setting.
Thread count is a runtime choice and remains part of ledger identity.

## Protocol and execution

Before fitting, freeze `docs/RERUN_PROTOCOL.json` and pass its SHA256 explicitly:

```bash
python scripts/run_study.py plan --protocol-sha256 FROZEN_SHA256
python scripts/run_study.py fit --protocol-sha256 FROZEN_SHA256
python scripts/run_study.py select --protocol-sha256 FROZEN_SHA256
```

`plan` reads and validates inputs without writing study artifacts. `fit` contains
no held-out scoring. `select` requires all 108 completed candidates. Held-out
evaluation runs separately through `scripts/evaluate_study.py`.

The reconstruction declares learning rates 0.0001, 0.0003, and 0.001; seeds
17, 29, and 43; ten epochs; CPU fitting with two Torch threads; image batches of
32; AdamW with weight decay 0.01; gradient clipping at norm 1; and the frozen
feature-cache score scale. Epoch zero participates in selection. There is no
learning-rate schedule. Optimizer betas and epsilon are explicitly 0.9, 0.999,
and 1e-8; AMSGrad, foreach, and fused variants are disabled. All source and hypothesis captions assigned to a batch's
images enter its candidate pool. Missing cross-image annotations have relation
code zero. Separate seeded random generators control batch order and stochastic
loss policies, so policy draws do not change image batch order.

The twelve method definitions and reconstructed semantic-kernel parameters are
in `docs/LOSS_RECONSTRUCTION.md`. These defaults enter the immutable ledger.

## Validation definition

The common validation score averages:

1. Image-to-text R@1 over all validation images and all captions assigned to
   validation images. Both source captions and label-supported hypotheses are
   relevant matches. Exact ties choose the first caption in manifest order.
2. Supported-versus-contradicted accuracy, comparing every supported hypothesis
   with every contradicted hypothesis for an image, awarding half credit for an
   exact tie, and averaging eligible images equally.

The first relevance definition resolves an ambiguity in recovered historical
documentation and is explicitly a reconstruction choice. Validation scores use
float64 dot-product accumulation over float32 adapted normalized features.
The positive frozen score scale is omitted from ranking because it cannot
change the ordering. An image missing either hypothesis class is explicitly
excluded from the relation average and recorded with a null relation score.

Each candidate retains the epoch with highest validation score. Exact ties
retain the earlier epoch. A method chooses one learning rate by mean best score
across all three seeds. Exact ties select the lower learning rate. The three
selected checkpoints are the best epochs for that common rate.

## Output and recovery

`results/study/ledger.json` is immutable. Changed code, protocol, input hashes,
or settings require a new output directory. Each candidate directory contains
`best.pt`, `history.json`, and a final `completion.json`. History retains the
raw per-image validation outcomes for all eleven evaluated epochs. Checkpoint
and history writes are atomic. The completion record binds both artifact
hashes. A rerun skips a candidate only when that record, all identity fields,
and both artifact hashes agree. An interrupted partial candidate restarts
deterministically from epoch zero. Incompatible partial files cause an error.

The root epoch-zero checkpoint is `results/study/epoch_zero.pt`, with its raw
validation outcomes in `epoch_zero_validation.json`. Selection copies the 36
chosen states into `results/study/selected/METHOD/seed_SEED.pt` and writes
`selection.json`, including per-rate scores, chosen epochs, checkpoint paths,
and hashes. All 108 candidate best checkpoints remain available.

Parallel workers may use disjoint execution filters. Filters do not change
the study grid or ledger:

```bash
python scripts/run_study.py fit --protocol-sha256 FROZEN_SHA256 --methods clip multipositive
python scripts/run_study.py fit --protocol-sha256 FROZEN_SHA256 --seeds 17
python scripts/run_study.py fit --protocol-sha256 FROZEN_SHA256 --only-learning-rate 0.0003
```

Keep `--threads` identical across workers. File locks protect ledger creation,
epoch-zero creation, and duplicate candidate requests. Do not edit the four
core fitting source files after the first study ledger is created.

## Synthetic preflight

`scripts/preflight_training.py` measures only randomly generated normalized
features. It never opens the natural-image dataset and is not a research
experiment. On the reconstruction workspace, two-thread 512-dimensional
batches of 32 images and 35 captions per image took about 0.020 to 0.030 seconds
per gradient step across all twelve methods. This projected approximately 17
minutes for the 108 candidates' gradient steps, excluding validation, input
loading, artifact writes, and concurrent-worker contention. Actual caption
counts determine the eventual runtime. The full measurement is saved in
`results/training_preflight.json`.

Twelve training integration tests passed before natural-image fitting. They
exercise held-out isolation, supported-positive validation, relation matrices,
input hashes, immutable ledgers, deterministic partial/full recovery, exact
epoch-zero ties, shared-rate selection, artifact-tampering detection, and
configuration changes supplied alongside the correct frozen-protocol hash.
