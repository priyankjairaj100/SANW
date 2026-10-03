# Final research story and content decisions

## Central question

What do relation labels add beyond extra correct captions, when the candidate
pool, trainable capacity, and tuning budget are held fixed?

The paper is a controlled intervention study. Its practical outcome is whether
better statement discrimination preserves valid descriptions and retrieval
from a full candidate pool. It does not present the combined relation rule as
a new winning training method.

## Evidence that determines the story

The fresh execution has 108 trained candidates, 36 selected states, and the
frozen reference on four benchmarks. Positive expansion improves e-SNLI-VE
accuracy by 5.81 percentage points over source-positive adaptation. SugarCrepe
changes by only 0.14 points, while COCO image-to-text R@1 falls by 8.56 points.
That retrieval change is an average of 428 additional failed image queries
out of 5,000. Text-to-image recall also falls, by 4.54 points.

The combined policy minus positive expansion is -0.26 points on e-SNLI-VE
and -0.16 points on SugarCrepe. Both adjusted primary intervals include zero.
The manuscript therefore does not reuse the lost execution's claim that the
combined policy is significantly worse on both endpoints.

Descriptive edit groups separate a 1.92-point gain for added content from a
0.90-point decline for the combined replacement/swap group. These are not new
confirmatory tests. All 31 selected nonzero adapters reduce alternative-caption
accuracy relative to frozen, but only 26 reduce COCO text-to-image recall.
The former universal retrieval-decline claim is not retained.

Neutral exclusion beats count-matched random exclusion in this execution.
Fixed semantic weights trail their shuffled control. Median and constant
weighting have close SugarCrepe means. The old blanket claim that semantic
weights closely track all nonspecific controls is not retained.

## Main body and appendix

The main body introduces the search problem, defines the three relation labels,
defines the controlled comparison, and presents the primary table, edit groups,
and a concrete query failure. The constant-weight control belongs in the main
table: its SugarCrepe mean is 83.67% and image-to-text recall is 55.21%, compared
with 83.39% and 47.29% for positive expansion. Its e-SNLI-VE mean is lower,
82.15% versus 87.00%. This is a descriptive practical tradeoff, not a planned
significance test or a new winning algorithm. Shared validation uses relation
labels for every method; constant weighting is not an annotation-free pipeline.

The objective and positive-target-budget identity appear late in the main
paper, after readers understand the interventions and outcomes. A compact
similarity-control table uses the remaining late-main space for executed
evidence. The matched negative-normalizer identity, detailed weighting analysis,
ranking fallback, and complete retention scatter move to the appendix. The first two to three
pages prioritize the application, controlled design, and empirical evidence.

The novelty is matched attribution of the roles played by relation labels.
The general discrimination/retention tension, extra-positive controls, and
positive/negative/neutral triage all have prior art. Recent successful methods
also preserve retrieval, so no inevitable tradeoff is claimed. See
`NOVELTY_POSITIONING_FINAL_REWRITE.md` for verified primary sources and
`CONTEXT_INVENTORY_FINAL_REWRITE.md` for the complete item-level disposition.

The appendix retains full loss definitions, rectangular-batch conventions,
all proof assumptions, derivatives, counterexamples, the summed-positive
alternative, parameter-dependent weights, source audits, selection histories,
per-seed results, every primary contrast, benchmark categories, diagnostic
calibration and cross-fitting, and reproduction details. The identities are
fixed-score statements, not guarantees about trained representations.

The relation-classifier diagnostic is separate from adapter training. Its
predictions never replace primary labels. Its solver ceiling amendment is
recorded independently of the frozen main protocol.

## Retained but unexecuted expansions

The recovery includes ideas for a second backbone, full-encoder training,
larger data and seed budgets, independent visual adjudication, a shared final
epoch, denominator-matched controls, hardness-matched exclusion, summed-positive
training, calibrated predicted relations, inverse-multiplicity weighting,
retention-constrained selection, distillation, and broader retrieval relevance.
They remain future work. Historical synthetic runs and model smoke tests are
retained as recovered descriptions, not newly reconstructed experiment files.

## Evidence boundary and recovery

This is a new execution under explicit reconstruction choices. The original
workspace, original raw results, and exact unrecovered implementation details
are not claimed to have been restored. Recovered historical text and tables
are isolated in `historical_context/`; all paper numbers come from audited
fresh artifacts under `results/`.
