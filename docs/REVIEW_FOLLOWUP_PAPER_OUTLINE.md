# Conditional outline for a four-page follow-up paper

Editorial plan only. The frozen follow-up protocol determines the comparisons;
no new outcome, title, abstract claim, or winning method is assumed here.
The original v2 manuscript remains unchanged.

## Central question

Does assigning a fixed positive-caption budget to annotated supported captions
improve matching beyond promotion count, coarse initial difficulty, and training
schedule? Keep this matched-schedule question separate from the practical
comparison of two complete checkpoint-selection strategies.

The novelty is controlled assignment attribution. False negatives, relation
triage, and retention failures already have close precedents; successful
retention methods also exist. Use `NOVELTY_POSITIONING_FINAL_REWRITE.md`.

## Four-page allocation

| Space | Purpose and content |
|---|---|
| Page 1 | A plain-language abstract of roughly 140 words. Introduce correct-match retention and explain why changing positive count and identity together leaves attribution unresolved. State the question and closest prior work. Do not repeat the abstract’s results. |
| Page 2 | Define four policy families, fixed assignments, count/score matching, paired schedules, and 72 fits. Explain retrieval relevance, both directions, clean pools, and the two selectors. State the epoch-ten comparison and conditional interval scope. |
| Page 3 | Show all primary effects and adjusted intervals: two retrieval-direction panels, each displaying three contrasts at three learning rates. Interpret the pattern once. Add discrimination or valid-caption evidence showing what a retrieval change buys. |
| Page 4 | Compare selectors in a compact table, including retained epochs when consequential. Include COCO transfer if informative. End with the tested operational finding. Any target-share explanation is brief; algebra stays in the appendix. |

Prioritize one primary figure and one selection table. Do not select favorable
rates or directions for space. Keep dense notation out of the first three
pages. Limitations, references, and appendix follow four full main pages.

## Definitions the main text must supply

- Positives are training targets. The source policy marks five original captions
  positive while retaining every hypothesis as a competing candidate.
- Random controls preserve each image’s promoted count and eligible reverse-anchor
  count while changing their identities. All weights remain one.
- Score strata use frozen image–caption similarity. Quotas use support labels;
  this is a control for attribution, not an annotation-free system.
- Three assignment draws cross three optimizer seeds and stay fixed across
  rates/epochs. They do not create nine independent supported fits.
- R@1 means an annotated match ranks first. The 1,000-image/5,000-caption test
  uses source ownership and the official e-ViL test pool, not Karpathy Flickr1k.
- The primary comparison fixes learning rate and epoch. The two selectors vary
  several development choices together and estimate complete procedures.

## Outcome-dependent novelty decisions

| Follow-up pattern | Defensible story and editorial consequence |
|---|---|
| The supported-minus-source retrieval gap vanishes or reverses at matched schedules | Reframe the old gap as a selected-procedure result. Lead with matched versus selected outcomes if the complete grid supports it. Drop any general harm headline. |
| The retrieval gap remains at matched schedules | Show persistence under schedule matching, then use assignment controls to locate its benefit or cost. Persistence does not identify target redistribution as the cause. |
| Supported promotion beats both random controls | Lead with annotated assignment beyond count and coarse initial difficulty. Specify supporting rates/directions. Do not generalize to all semantic covariates or ordinary additional correct captions. |
| Supported promotion beats count-only but not score-stratified assignment | The advantage is not established beyond coarse difficulty. Compare estimates and bounds; different significance labels alone do not establish a mediating mechanism. |
| Supported promotion has no clear advantage over either random control | Report differences and bounds as a boundary on this policy, not equivalence or universal label uselessness. |
| Source-retrieval selection preserves discrimination gains with better retrieval | Lead the practical comparison with both capabilities and retained epochs. Attribute it to the complete strategy, not the selection formula alone. |
| Source-retrieval selection mainly returns epoch zero | Describe rejection of learned states, not successful adaptation. Terminal comparisons show what training learned. |

Mixed patterns should produce a narrower claim, not a winning-cell narrative.
Selected-strategy and secondary outcomes remain descriptive under the frozen
protocol. Previously examined 400-image and additional 600-image summaries use
the same full candidate pool and belong in the appendix.

## Disposition of v2 material

Move the common loss, positive-target derivative/example, reverse-anchor identity,
matched-negative-mass construction, transpose counterexample, and remaining
proofs to the appendix. At most retain one late sentence on count-matched source
target share, without a learned-loss mechanism claim.

The original twelve-policy results, weighting controls, edit groups, 31-adapter
rewording scatter, validation trajectories, and relation classifier become
prior-execution appendix material. Keep the old candle-query photo there unless
essential to the selection story. Main figures use follow-up states. Preserve
all previous results and unused directions in the reproducibility archive and
context inventory.
