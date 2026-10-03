# Relation-assignment follow-up

This follow-up asks what label-informed positive assignment adds beyond the
number of promoted captions, their coarse initial difficulty, and the training
schedule. Its choices were fixed after review of the original results and
before follow-up fitting and scoring. It is a new execution, not an amendment
to the original study's primary statistical family.

The controlling specification is `docs/REVIEW_FOLLOWUP_PROTOCOL.json`, SHA256
`3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34`.
`docs/REVIEW_FOLLOWUP_PROTOCOL_FREEZE.json` records the freeze. The original
protocol, data, feature caches, 108 candidate runs, selected results, and v2
paper remain preserved. `docs/REVIEW_GAP_AUDIT_20261003.md` explains the review
questions that motivated this study.

## What changes and what stays matched

All fits use the original 1,200 training images and their unchanged source
captions and annotated hypotheses. The encoders, feature vectors, adapter
capacity, score scale, optimizer settings, batch size, and tuning grid remain
the same. Every hypothesis stays in the candidate pool. The intervention
changes which hypotheses are marked positive during training.

| Policy family | Hypotheses assigned positive | Assignment draws |
|---|---|---:|
| Source | None; only the five source captions are positive | One fixed rule |
| Supported | All hypotheses labeled supported | One fixed rule |
| Count-only control | A uniform subset of each image's own hypotheses, matching its supported count | Three |
| Score-stratified control | A uniform subset within each of two frozen-score bins, matching the supported count in each bin | Three |

Every positive has equal target probability within its image row. All candidate
weights are one. The loss is the original symmetric mean-positive-log-probability
objective. Hypotheses not promoted remain negatives, including supported
hypotheses in the source policy. This source policy is therefore different from
training with hypotheses absent from the candidate pool.

For stratification, cached image and text features are explicitly normalized
in float64. Each image's hypotheses are sorted by frozen cosine score, with
global manifest text index breaking exact ties. `numpy.array_split` makes two
bins; the lower-score bin receives the extra item for an odd count. Labels
determine the supported quotas, not the bin boundaries. Assignment diagnostics
record coarse-score imbalance, naturally unchanged assignments, exchangeable
and forced bins, support overlap, and changes in both caption IDs and exact
caption-string multisets.

Each random assignment is generated once using NumPy PCG64 and assignment seed
101, 211, or 307. It is then fixed across every learning rate, optimizer seed,
and epoch. Assignment randomness is separate from batch-order randomness.
Original relation annotations are never overwritten and remain the evaluation
labels. These controls use label-derived quotas; they are identification
controls, not annotation-free deployment methods.

One-image ownership of text IDs and matched per-image promotion counts preserve
the number of eligible reverse-direction text anchors in every paired batch.
Their identities can change. The implementation also checks unchanged source
positives and candidate columns. It does not claim that the resulting learned
representations or parameter gradients are identical.

## The 72 fits and the two comparisons

There are eight fixed conditions: source, supported, three count-only draws,
and three score-stratified draws. Each uses learning rates `0.0001`, `0.0003`,
and `0.001`, crossed with optimizer seeds 17, 29, and 43. Thus the study has
`8 × 3 × 3 = 72` fits: 18 source/supported fits and 54 randomized-control fits.
Every fit runs for ten epochs and saves states at epochs zero through ten,
giving 792 state records.

The primary comparison uses terminal epoch ten at each common learning rate.
Seeds are paired across policies and assignment draws. All specified rates
are reported; the test set does not choose a preferred rate. Epochs one and
five provide descriptive trajectories on the declared non-COCO benchmarks.

Separately, each fit is scored by two development-set selection strategies:

| Strategy | Development pool and criterion |
|---|---|
| Native | The original 100 validation images and all 1,875 caption/hypothesis candidates. Average known-positive image-to-text R@1 and image-balanced supported-versus-contradicted accuracy. Source and supported captions count as relevant. |
| Source retrieval | The official e-ViL development images except the original 100 calibration images: 900 images and 4,500 source captions. Average full-pool image-to-text and text-to-image R@1, using source-caption ownership as relevance. |

For each policy or assignment draw, select the best epoch separately for each
optimizer seed, including epoch zero. Then select one learning rate by the
mean of those best development scores across the three seeds. Ties favor the
earlier epoch and then the lower learning rate. Select each assignment draw
separately and average the three draws equally; there is no winner-draw
selection.

These are comparisons of complete selection strategies. They differ in
criterion, relevance, candidate pool, retrieval directions, and development
set size. They do not isolate the effect of changing only the selection
formula. A selected state can also coincide with a terminal or trajectory
state; reuse does not create an independent replication.

## Retrieval pools and prior test exposure

The source-retrieval test uses all 1,000 official e-ViL test images and their
5,000 original Flickr30k Entities captions. Every image query searches all
5,000 captions; every caption query searches all 1,000 images. Only source
ownership defines relevance. This is **e-ViL test-pool Flickr source-caption
retrieval**, not the standard Karpathy Flickr1k benchmark, whose split overlaps
the original fitting/development data.

The full test pool includes the original 400 test images, whose outcomes were
already examined, and 600 additional images. Full-pool results are the declared
endpoint. Subgroup summaries retain the original-400 and additional-600 query
IDs and search the same complete candidate pool. They are descriptive; they
must not be presented as two separately constructed retrieval benchmarks or as
two independent confirmations.

Data preparation records exact image IDs, pinned source revisions, caption
strings, image-file hashes, and feature provenance. Image IDs and image-file
SHA256 values are checked for disjointness across training, calibration,
development, and test pools. Identical existing features are reused by exact
IDs/content and encoder provenance. The expanded development pool includes
the original 100 validation images but excludes all original calibration
images. The expanded test pool contains no fitting or development images.

The original VE400 relation test, 7,511 SugarCrepe pairs, 4,757 SugarCrepe++
triplets, and COCO retrieval pool of 5,000 images and 25,000 captions are
unchanged. COCO is evaluated for all terminal states and the unique states
chosen by either selector, plus frozen. It is not required for otherwise
unselected epoch-one/five trajectory states.

## Primary family and uncertainty

The primary family has 18 effects at epoch ten:

- Three contrasts: supported minus source, supported minus the mean count-only
  control, and supported minus the mean score-stratified control.
- Three fixed learning rates.
- Two endpoints: full e-ViL test-pool image-to-text and text-to-image R@1.

For a randomized comparator, first average the three assignment draws within
each optimizer seed. Then average the three optimizer seeds within each query.
The same three supported fits are reused in those paired contrasts; they are
not duplicated into nine independent supported fits.

Intervals use 10,000 paired image-cluster bootstrap replicates with seed
20261004. A sampled text-to-image cluster includes all five caption queries
owned by that image. The two-sided Bonferroni-adjusted percentile intervals
have confidence level `1 − 0.05/18`, or approximately 99.7222%. These intervals
condition on the fitted optimizer seeds and frozen assignment draws. They do
not estimate uncertainty over newly trained models or newly sampled label
assignments. Report every optimizer-seed/assignment-draw cell, per-draw means,
and per-seed means alongside the conditional image intervals.

All COCO, relation, edited-caption, valid-rewording, non-R@1 retrieval,
trajectory, and selected-strategy comparisons are descriptive. An interval
containing zero is not an equivalence result. The new primary family remains
separate from the original study's six planned contrasts.

## Reproduction and saved artifacts

Use the Python 3.12/CPU PyTorch environment and original pinned inputs described
in `docs/REPRODUCE.md`. Run commands from the project root. Keep original feature
NPZ files together with their metadata; do not regenerate them inside an
execution whose ledger already binds their byte hashes.

These commands reproduce the follow-up from the supplied original execution
artifacts. A newly regenerated base study may have different artifact hashes
even when its numerical features agree. Such a study needs a separate protocol
bound to its own inputs; do not edit the frozen protocol to bypass a mismatch.

The confirmed preparation command is:

```bash
python scripts/review_followup_prepare_flickr_pools.py --phase all --workers 12 --threads 4
```

The `prepare` and `encode` phases can also be run separately. New manifests are
under `data/review_followup/`; their feature caches are under
`results/review_followup/features/`. No original dataset namespace is replaced.

Use the activated project environment for the following commands. The recorded
execution's interpreter is
`/workspace/scratch/3eec744c3d5e/.venv/bin/python`; a restored environment can use
its own `python` executable. Set the immutable digest once:

```bash
FOLLOWUP_PROTOCOL_SHA=3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Freeze the six assignments, inspect the execution plan, fit the full grid, and
export both selections:

```bash
python scripts/run_review_followup.py assignments --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py plan --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py fit --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py select --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
```

The default protocol is `docs/REVIEW_FOLLOWUP_PROTOCOL.json`; the default output
is `results/review_followup/`. The development manifest, features, and metadata
default to the `e_vil_dev900` paths described above. A compatible completed fit
is verified and reused. An interrupted compatible fit restarts from epoch
zero. Changed code, protocol, or input hashes require a new execution namespace.
Subset flags are available for scheduling, but final selection requires the
complete 72-fit grid.

Run evaluation only after all 72 fits are complete and both selections have
been frozen in `state_manifest.json`:

```bash
python scripts/evaluate_review_followup.py --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA" --suite terminal
python scripts/evaluate_review_followup.py --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA" --suite selected
python scripts/evaluate_review_followup.py --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA" --suite trajectory

python scripts/analyze_review_followup.py --protocol docs/REVIEW_FOLLOWUP_PROTOCOL.json --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA" --indices results/review_followup/evaluation/terminal_index.json results/review_followup/evaluation/selected_index.json results/review_followup/evaluation/trajectory_index.json
```

The evaluator defaults to `results/review_followup/state_manifest.json`, writes
under `results/review_followup/evaluation/`, and uses two Torch threads with
retrieval blocks of 128 queries. Suite defaults enforce the protocol's dataset
coverage. The trajectory suite omits COCO; terminal and selected suites include
it. The analysis writes under `results/review_followup/analysis/`.

Before fitting, the execution ledger binds the frozen protocol, original
inputs and imported sources, completed development features, new code, and six
assignment artifacts. All 18 source/supported fits must reproduce the original
training losses, native-validation histories, and best-state tensors at the
recorded original best epochs. A mismatch must be explained before using the
matched-schedule results.

`results/review_followup/state_manifest.json` records the 792 state references
and both selectors without unnecessary copies of selected weights. Evaluation
binds checkpoint, data, code, and protocol hashes before scoring. Saved
predictions retain retrieval ranks and best-relevant indices/scores, relation
outcomes, and triplet scores/correctness. Identical reused state references are
scored once. The analysis must reproduce aggregate metrics from these saved
predictions rather than use handwritten result tables.

The analysis outputs are `all_state_metrics.csv`, `fixed_epoch_metrics.json`,
`primary_contrasts.json`, `primary_bootstrap_samples.npz`,
`selected_strategy_metrics.json`, `e_vil_query_subgroups.json`,
`analysis_audit.json`, and `preanalysis_receipt.json`. Their presence alone is
not a completion claim: use the completed evaluation indices, analysis audit,
and saved artifact hashes to verify the execution.

## Interpretation boundary

The matched controls identify the effect of assigning a fixed positive budget
to the annotated supported-caption identities rather than the declared random
alternatives, including the content of reverse-direction anchors. The
score-stratified control matches coarse initial difficulty, not every
linguistic attribute or exact continuous score. The study does not identify
target redistribution as the cause of retrieval loss, independently adjudicate
visual truth, or measure annotation cost. Target-allocation interventions,
negative-weight sensitivity, valid-caption-source controls, and additional
model capacities are not executed by this protocol.
