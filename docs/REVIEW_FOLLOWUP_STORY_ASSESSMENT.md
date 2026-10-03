# Follow-up evidence and paper story

Assessment of completed phase-one results, 2026-10-03. The original manuscript
is unchanged. All figures below come from the audited follow-up analysis, not
historical aggregates.

## Decision

Proceed with a focused four-page empirical paper. Phase one supports a stronger
and more precise story than v2: **supported-caption identity improves text-to-image
retrieval against count- and coarse-score-matched random promotion, while
image-to-text retrieval loses ground to source-positive training at every
matched learning rate.** Better discrimination and recognition of valid
captions coexist with that image-to-text cost. Changing development selection
rejects the expanded learned states; it does not produce a trained solution
that preserves those gains.

This is a controlled account of positive assignment and complete selection
strategies, not a new winning adaptation method. A mechanism phase would answer
an additional question about target allocation. It is not needed to support
this short-paper contribution. Do not execute it merely to preserve the old
“dilution causes retrieval loss” interpretation. Keep one frozen backbone and
linear adaptation explicit; broader model generality remains untested.

## Complete primary evidence

Every entry is supported minus its named comparator at epoch ten, in percentage
points, with its adjusted 99.7222% paired image-cluster interval. All 18 planned
effects are shown. The random comparators average three assignment draws within
each training seed, then three seeds. Intervals condition on these fitted runs.
Source: `results/review_followup/analysis/primary_contrasts.json`.

| Learning rate | Comparator | Image-to-text R@1 | Text-to-image R@1 |
|---|---|---:|---:|
| 0.0001 | source | -6.00 [-9.04, -3.06] | +0.12 [-1.09, +1.32] |
| 0.0001 | count only | -0.98 [-3.39, +1.38] | +1.81 [+0.63, +2.99] |
| 0.0001 | score stratified | -0.74 [-3.00, +1.44] | +1.54 [+0.50, +2.58] |
| 0.0003 | source | -7.13 [-10.37, -3.80] | -1.32 [-2.95, +0.38] |
| 0.0003 | count only | +1.32 [-1.73, +4.11] | +4.84 [+3.33, +6.38] |
| 0.0003 | score stratified | +0.12 [-2.66, +2.81] | +3.82 [+2.41, +5.20] |
| 0.001 | source | -10.83 [-14.34, -7.33] | -1.90 [-3.96, +0.15] |
| 0.001 | count only | +3.79 [+0.72, +6.92] | +7.04 [+5.16, +8.83] |
| 0.001 | score stratified | +0.22 [-2.59, +3.02] | +4.08 [+2.43, +5.80] |

The nine supported-versus-source image-to-text seed differences are all negative.
The cost is 60, 71.33, and 108.33 additional failed image queries per 1,000 images,
averaged across seeds at the three rates. This persists under matched schedules
and within the source domain. It is not explained solely by selecting different
learning rates/epochs or switching to COCO.

Text-to-image benefits over both randomized controls are positive in every
individual draw-by-seed comparison at every rate. All six adjusted intervals
exclude zero. The score-stratified effects are +1.54, +3.82, and +4.08 points.
The corresponding image-to-text effects are −0.74, +0.12, and +0.22, with bounds
that include zero. This is not evidence of equivalence, and a comparison of
significance labels is not a tested cross-direction interaction. Describe the
observed endpoint-specific evidence and name the comparator each time.

## Secondary outcomes change the old transfer story

The following descriptive differences are also supported minus source at the
same epoch and learning rate. Source: `fixed_epoch_metrics.json`, `cells`.

| Learning rate | Relation accuracy | SugarCrepe | SC++ both valid captions | COCO I→T | COCO T→I |
|---|---:|---:|---:|---:|---:|
| 0.0001 | +9.86 | +2.94 | +3.54 | -4.73 | -0.53 |
| 0.0003 | +11.33 | +3.21 | +3.38 | -6.12 | -0.90 |
| 0.001 | +15.27 | +4.93 | +5.37 | -7.21 | -0.08 |

The old +0.14-point SugarCrepe result belongs to independently selected native
procedures. It cannot headline the matched study. Supported promotion improves
both edited-caption accuracy and the stricter valid-caption endpoint at every
terminal rate. The descriptive trajectories also show early valid-caption gains;
for example, supported training at learning rate 0.0001 and epoch one reaches
70.00% SC++ both and 75.69% alternative-caption accuracy, compared with frozen
68.95% and 75.01%. Do not carry forward a universal decline claim across all new
nonzero checkpoints.

The source-domain image-to-text loss also appears among the additional 600
images: −4.39, −6.22, and −9.50 points at the three rates. These are descriptive
query subsets of the same complete candidate pool, not additional primary
families. Source: `e_vil_query_subgroups.json`.

## What selection actually does

Source: `selected_strategy_metrics.json`, `strategies` and their selected-state
references. All recalls below are percentages on the full e-ViL test pool.

| Strategy and policy | Selected epochs (17/29/43) | I→T | T→I | SugarCrepe |
|---|---|---:|---:|---:|
| Native, source | 1/1/3 | 84.70 | 69.35 | 83.26 |
| Native, supported | 5/10/5 | 77.00 | 63.30 | 83.39 |
| Source retrieval, source | 1/2/2 | 84.83 | 69.70 | 83.29 |
| Source retrieval, supported | 0/0/0 | 84.50 | 69.76 | 81.21 |

The native supported rule selects learning rate 0.0003; the other rows select
0.0001. Source-retrieval validation selects epoch zero for all seven expansion
conditions, including all six randomized draws. Those 21 selected states are
frozen initialization, not successful trained expansion models. Native selection
also rejects every randomized learned state except seed 17 of score-stratified
draw two, which retains epoch seven.

This resolves the practical selection question in the tested grid: the new
retrieval-oriented strategy does not retain the learned expansion gains. Its
outcome cannot be attributed solely to a different scalar criterion because
pool, relevance, development size, and retrieval directions change together.

## Response to the four review weaknesses

1. **Assignment attribution:** Three fixed draws per randomized family preserve
   promotion and reverse-anchor counts. The coarse-score control further
   constrains initial difficulty. A supported-identity benefit is established
   on text-to-image retrieval relative to these controls, not every linguistic
   confound or ordinary additional correct captions.
2. **Schedule and selection:** All three learning rates are compared at the same
   terminal epoch and paired seeds. Source/support replay is exact. The two
   selectors are reported separately as complete procedures, including frozen
   fallbacks.
3. **In-domain relevance and clean splits:** The main endpoint uses the complete
   disjoint official e-ViL test pool and five source captions per image. It is not
   Karpathy Flickr1k. The old 400 and additional 600 images remain identified.
   Source ownership is the retrieval relevance definition.
4. **Scope:** The design supports a focused study of this backbone, adapter,
   candidate pool, and objective. It does not establish all-model behavior or
   the value of every contradiction/neutral coefficient. Original ranking's
   epoch-zero result is a selection outcome, not evidence of failed learning.

A supported description can be correct yet insufficient to identify one image
among many. That distinction provides accessible problem intuition. Neither
this study nor its fixed-score identities establish it as the learned mechanism.

## Proposed title and abstract

**Beyond Caption Counts: Auditing Positive Supervision for Retrieval**

> An image can have several correct descriptions. Does marking more captions as
> correct improve retrieval, or do gains come from which captions receive those
> labels? We compare supported captions with random promotions matched for count
> and coarse initial similarity. Seventy-two fits use the same pretrained model,
> linear adapters, candidate captions, and matched training schedules. Supported
> captions improve edited-description discrimination and recognition of valid
> rewordings. Against similarity-stratified random promotion, text-to-image recall
> improves by 1.54 to 4.08 percentage points across three learning rates. Against
> training with only original captions marked positive, image-to-text recall
> falls by 6.00 to 10.83 points. This corresponds to 60 to 108 additional failed
> image queries per 1,000. A source-retrieval validation strategy selects frozen
> initialization for every expansion condition. The results separate the value
> of annotated caption identity from promotion count, training schedule, and
> checkpoint selection.

The main paper should center on the full primary-effect figure, a compact
matched discrimination/valid-caption table, and the selector outcome. Move the
old negative-weight controls, detailed identities, 31-state scatter, query photo,
and separate relation classifier to a clearly labeled prior-study appendix.

## Evidence provenance

The follow-up analysis audit reports 5,536 aggregate checks with maximum
recomputation discrepancy 2.22e-16, 1,080 prediction archives, 244 unique scored
states, all 18 primary effects, and no test-based selection. Root's independent
primary audit additionally verifies the 180,000 saved bootstrap draws. The
frozen protocol hash is
`3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34`.

Input SHA256 values for this assessment:

- `primary_contrasts.json`: `3b0e9e8ab84a659f9503f0c0d31acb2ce41d86be9a09d3422ef3350ef4c8eac2`
- `fixed_epoch_metrics.json`: `9bb57cc4dbc77814686e8dfc76aee112e205c62e5e4039342919cbcc54328848`
- `selected_strategy_metrics.json`: `1363a4ca8e78042114feaa303028d57aa1fd36354628a7478f10249687f38a4d`
- `e_vil_query_subgroups.json`: `cd4bcfa599f9501de888ae4ce5a635a64f3a73941afe86712ce30fb11ac596aa`
- `analysis_audit.json`: `a9c820d7405b205e177f02acc1488ba63e656cdbd69df820d5053264d3f2574c`
