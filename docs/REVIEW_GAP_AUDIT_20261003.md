# Review gap audit and follow-up design

Date: 2026-10-03. This audit responds to the four weaknesses supplied after the
v2 manuscript freeze. It supersedes the earlier editorial judgment that the
current design fully attributes the value of relation labels. It does not
replace the frozen protocol, results, manuscript, or primary statistical family.

## Decision

The missing promotion control and the selection mismatch are central design
gaps. A wording revision alone cannot establish the proposed attribution story.
The strongest next question is: **does label-informed caption promotion improve
matching beyond the effects of promotion count, initial difficulty, and training
schedule?** A second, distinct question is whether changing positive target
allocation retains useful source-caption retrieval.

Existing comparisons remain valid as comparisons of complete selected
procedures. In particular, contradiction weighting and neutral exclusion are
tested on top of the same supported-positive set. Their results apply to the
specified weights and selection rule. They neither identify the semantic value
of positive expansion nor establish that such interventions cannot help.

## 1. Positive assignment is not controlled

Source positives and expansion use identical candidate strings. Expansion
changes the positive count, the identity of positive captions, their normalized
target weights, and the set of eligible reverse-direction text anchors. Equation
2 is correct, but its source-target redistribution is independent of whether
promoted captions are supported. It is not evidence that redistribution caused
the observed learned retrieval loss.

Add a count-matched within-image permutation of the support labels. Preserve
the five source positives and promote exactly the original supported count from
the same image's hypotheses. Leave all weights at one. Keep each randomized
assignment fixed across epochs. Include multiple independent permutation draws;
three optimizer seeds on one assignment do not replicate label randomization.

A stronger companion control permutes within coarse, prespecified strata of
frozen image-hypothesis similarity. It preserves supported counts within each
stratum as well as the per-image total. This tests positive assignment beyond
count and coarse initial difficulty. Report how many assignments actually
change, their overlap with supported labels, and score imbalance. Small strata
can leave too few labels exchangeable. The control uses label-derived quotas;
it is an identification control, not an annotation-free deployment method.

With the present image-owned text IDs and identical batches, these controls
also preserve the number of eligible reverse anchors. They change their
content. Use identical learning rates, epochs, batch order, and paired seeds for
the intervention comparison. Separately selected runs answer a different
procedure-comparison question.

If claiming an advantage over ordinary extra correct captions, add a distinct
valid-caption comparison: retain four source positives, and promote either the
held-out fifth source caption or one supported hypothesis from the same fixed
candidate universe. Both expanded arms then have five positives and the same
reverse-anchor count. This answers a content-source question rather than the
label-permutation question. Ordinary source-only adaptation and ignoring
supported hypotheses are also useful reference policies: the present source
baseline treats the hypotheses as negatives. These extensions should follow
the specific claim being made rather than becoming an unfocused policy catalog.

If the paper advances a target-allocation mechanism, intervene on allocation as
well. For K source captions and M supported hypotheses, compare the current
source share K/(K+M) with a prespecified group share rho: each source target is
rho/K, and each supported target is (1-rho)/M. For M=0, use source-only targets.
This varies source share with labels and candidates fixed. Normalized targets
cannot preserve the original source mass of one while assigning nonzero mass
to new positives. The reverse-direction anchor weighting must also be specified;
changing image-row targets alone does not remove reverse-anchor dilution.

## 2. Selection, schedule, and transfer are different comparisons

The recorded selections are:

| Policy | Learning rate | Epochs for seeds 17, 29, 43 |
|---|---:|---|
| Source positives | 0.0001 | 1, 1, 3 |
| Supported expansion | 0.0003 | 5, 10, 5 |
| Pairwise ranking | 0.0001 | 0, 0, 0 |

The learning rate differs as well as the epoch. Matching epochs alone while
retaining these different learning rates would not isolate the loss change.
Both methods were trained for ten epochs; the reported states were retained
by validation selection, rather than training being stopped early.

Validation averages image-to-text R@1 over 100 images and 1,875 captions or
hypotheses, counting both source and supported captions as correct, with
image-averaged supported-versus-contradicted accuracy. COCO retrieval instead
uses 5,000 images, 25,000 source captions, and caption ownership. The observed
8.56 percentage-point I2T gap therefore concerns selected procedures on a
transfer benchmark. It is not an identified matched-schedule effect or an
estimate of human-adjudicated deployment relevance.

There are histories for epochs 0 through 10, but only each candidate's best
weights were retained. No matched source/expansion nonzero best states exist
at the same learning rate and seed. Recovering the relevant intermediate or
terminal states requires replay.

The smallest complete schedule check replays source and expansion at all three
original learning rates and three seeds, for 18 runs, and evaluates the common
terminal epoch 10. Report each learning-rate contrast, without choosing the
best test rate. Save every epoch during replay for a separately specified
trajectory analysis. Recorded original training time for these 18 runs was
147 seconds; original full COCO scoring averaged approximately 6.5 seconds per
state. These are historical timings, not a guarantee for a new execution.

For a practical selection claim, add a separately specified source-only
retrieval validation criterion. Reuse a genuinely disjoint development pool,
define both retrieval directions and all tie rules before scoring, and keep
test results out of checkpoint selection. Compare this criterion with the old
relation criterion explicitly. Representative development selection remains
a hypothesis to evaluate, not an intervention already tested by v2.

## 3. In-domain retrieval and incompatible Flickr splits

The current features permit retrieval over all 400 held-out e-ViL test images
and their 2,000 source captions. The follow-up evaluation uses only those five
source captions per image for relevance and candidate text, in both directions.
It is a descriptive addition to the selected-checkpoint study. It does not
repair checkpoint selection or test a matched schedule.

The standard Karpathy Flickr1k test cannot simply be substituted: it overlaps
the current e-ViL-derived training and validation subsets. Use the complete
official e-ViL test image pool with source captions for a larger clean
in-domain evaluation, or rebuild training so it is disjoint from the Karpathy
test. Verify both image IDs and decoded/file content before evaluation. The
current 400-image pool must not be described as the standard Flickr1k benchmark.

The pinned e-ViL test contains 1,000 image IDs, with all 5,000 original captions
available locally. The current feature cache covers 400 of those images and
2,000 captions. Completing this clean pool needs 600 further images and 3,000
further caption encodings. The ID sets are disjoint from the current training,
calibration, and validation subsets; image-content checks remain necessary for
the additional downloads.

Exact pool construction, overlap counts, provenance, metrics, and saved ranks
are recorded in `results/review_followup_flickr400/`. Execution is complete for
the frozen reference and all 36 selected states. Mean R@1 percentages are:

| Selected procedure | Image-to-text | Text-to-image |
|---|---:|---:|
| Frozen reference | 91.00 | 80.15 |
| Source positives | 90.17 | 79.47 |
| Supported expansion | 84.00 | 73.20 |
| Expansion plus both negative-label operations | 83.83 | 73.22 |
| Source positives with constant negative weight 0.25 | 89.83 | 79.73 |

Expansion minus source positives is -6.17 and -6.27 percentage points in the
two directions. Both differences are negative in every seed. The selected
retrieval loss therefore also occurs within the source domain on a common
pool; switching the evaluation dataset to COCO cannot be its sole explanation.
Training schedule, selection, and attribution remain unresolved. These absolute
recalls are not directly comparable with the much larger COCO candidate pool.

The protocol was saved before the added scoring. All input and checkpoint
hashes remained unchanged, and 814 independently sorted query checks passed.
No confidence interval or new confirmatory test was added. Protocol SHA256:
`dac691d52e3cf24414658a075937fed1f2209c059e9076332ed2972c60aa8917`.

## 4. Scope and ranking selection

One adapter is a permissible scope for a focused empirical study, provided the
claims remain at that scope. The tested contradiction weight 2 and neutral
weight 0 cannot establish the value of the corresponding method families.
After the promotion and schedule controls, a compact sensitivity study could
cross contradiction weights {1, 2, 4} with neutral weights {0, 0.5, 1}. It should
be specified as a follow-up grid and evaluated with the same selection rules.
A second adaptation capacity can then test whether the central pattern depends
on the linear residual parameterization. Neither substitutes for identification.

For the specific constant-weight comparison, a more focused extension comes
before a broad sweep: cross source versus expanded positive targets with
negative weights 1 versus 0.25 under the same schedule. The existing headline
constant-versus-expansion comparison changes both target set and negative
weight. Its missing expanded-positive/0.25 condition would test whether
suppression can retain more retrieval while using the supported captions.

The pairwise runs were trained, but all selected outputs are epoch zero. For
seed 17 and learning rate 0.0001, validation relation accuracy rises from
82.90% to 91.76% at epoch 1 and 93.83% at epoch 3. At those epochs, validation
known-positive I2T R@1 falls from 85% to 54% and 30%. The frozen selected row
thus reflects the joint selection criterion rejecting learned states. It does
not show that the ranking objective cannot learn the relation distinction.

## Manuscript decisions

1. Defer another final four-page rewrite until the promotion and schedule
   controls resolve the central question. Keep v2 available as an exact record.
2. Replace broad attribution claims with the exact comparison. A suitable
   current statement is: "For checkpoints chosen by relation validation,
   supported-caption promotion lowers COCO transfer I2T R@1 by 8.56 points."
3. Keep Equation 2 as a target-allocation identity. A causal account of learned
   retrieval requires an allocation intervention and matched training results.
4. Remove empirically untested procedure-choice advice. Give the selection
   criterion and transfer setting where the headline number is introduced.
5. Treat epoch-zero ranking as a selection outcome. Put learned validation
   trajectories in the appendix if this policy remains.
6. A four-page follow-up should center on the promotion control, matched
   schedule, and in-domain retrieval. Dense ancillary theory and the separate
   relation classifier should not displace those experiments.

The new studies are post-review follow-ups informed by the existing results.
Freeze their choices before new scoring, report all specified cells, and do
not relabel them as part of the original planned primary tests. No new training
or coefficient sweep has been executed as part of this audit.

## Evidence inspected

- `docs/RERUN_PROTOCOL.json`
- `results/study/selection.json`
- `results/study/candidates/*/*/*/history.json` and completion records
- `src/gcr/training.py`, `src/gcr/losses.py`, `src/gcr/evaluation.py`
- `scripts/evaluate_study.py`
- `manuscript/main.tex`, `abstract.tex`, `training_summary.tex`,
  `results_main.tex`, and `implications.tex`

The review does not invalidate the existing computed scores or their audited
provenance. It changes which scientific conclusions the design can support.
