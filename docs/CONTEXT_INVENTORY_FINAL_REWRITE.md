# Context inventory for the final rewrite

Prepared from the accessible workspace on 2026-10-03. This is an editorial
inventory, not a new experiment. Scientific inputs, outputs, and manuscript
files were read but not changed.

Scope: this project, the eight files in sibling `../recovered_context/`, and
existing project/delivery outputs. All eight sibling recovery files are
byte-identical to `historical_context/`; they contain no additional lost material.
No other conversation or Library content was accessed. Items below enumerate
the accessible scientific content and identify the remaining unrecoverable gaps.

**Status:** **F** = fresh executed artifact/result; **F-proof** = a current
self-contained mathematical derivation, not an empirical result; **H** =
historical text/report only, without its original supporting files;
**U** = unexecuted proposal. **Main**, **Appendix**, and **Archive** are editorial
recommendations for the next four-page paper. Archive means retain in the
recovery project, not discard.

## Source map

| Source group | Available content | Status and use |
|---|---|---|
| `historical_context/manuscript_main_recovered.txt` | Transcribed prior four-page story, notation, equations, old citation keys, and pointers to lost tables/figure | H; source of continuity, not current numerical evidence |
| `historical_context/theory_recovered.txt` | Seven numbered families of analytical statements, including original Lemma 2.1/Theorems 2.2–2.6/Proposition 2.7 and corrections | H plus current reconstructed proofs; preserve every valid identity and retired interpretation explicitly |
| `historical_context/reported_main_results.csv`, `reported_weight_controls.csv`, `reported_edit_groups.csv` | Old means for six main policies, four weight controls, and three edit groups | H; archive separately from all fresh tables |
| `historical_context/study_and_recovery_record.txt` | Old experiment scope, reported statistics, diagnostic, pilot work, prior-art links, unexecuted ideas, model/protocol hashes | H; the only accessible record for several early ideas and pilots |
| `docs/RERUN_PROTOCOL.json`, `LOSS_RECONSTRUCTION.md`, `TRAINING_EXECUTION.md`, `TRAINING_COMMANDS_RECORDED.md`, `INTERFACES.md` | Explicit reconstruction choices and the implemented study contract | F; short main setup, complete appendix specification, full archive |
| `data/*/manifest.json`, `provenance.json`, `results/features/*/metadata.json`, `results/input_audit.json` | Actual row/image identities, source revisions, acquisition checks, overlap/duplicate audits, exact cached features | F; appendix provenance, archive all inputs |
| `results/study/`, `results/evaluation/`, `results/analysis/` | Training grid, selected states, item predictions/ranks, aggregates, intervals, descriptive breakdowns | F; authoritative source of every current empirical claim |
| `results/relation_diagnostic_iter10000/` and its audit | Three classifiers, calibration/gates, cross-fitted predictions, model coefficients, partitions | F; separate diagnostic, not adapter supervision |
| `manuscript/` and `scripts/generate_paper_results.py` | Current complete text, proofs, tables, three substantive figure families, precise figure data, fresh-result source receipt | F/F-proof; rewrite from these verified assets |
| `docs/REPRODUCE.md`, `RECOVERY_README.md`, `STAGED_RECOVERY_INDEX.md`; recovery/verification scripts | Fresh-environment and exact-byte recovery procedures, delivered-stage records | F; archive/reproducibility material |

The current completed manuscript snapshot has four main pages and sixteen
pages overall, with a 144-word abstract according to
`manuscript/final_layout_receipt.json`. The historical lost snapshot was
reported as nineteen pages overall. These are different versions.

## Theory disposition matrix

For all fixed-weight statements, finite scores, a nonempty positive set,
nonnegative detached weights, and unit positive weights are the common setup.
Their proofs are in `manuscript/appendix_math.tex`; current gradient tests are in
`tests/test_losses.py`. Original numbering below is recovered provenance, not a
claim that the original PDF was reread.

| ID / item | Exact substantive content | Source | Status | Recommended disposition |
|---|---|---|---|---|
| T1 Common row gradient | Uniform-positive weighted loss has derivative `q_j - y_j`; rectangular directions have their own eligible-anchor means | Current math, losses, tests; recovered theory §1 | F-proof, tested | Appendix derivation; define the loss and its positive target in Main |
| T2 Positive-budget redistribution | Relabeling M existing candidates after K positives, at unit weights/fixed scores, increases each original-positive derivative by `M/[K(K+M)]` | Current main/math; recovered theory §2; expansion test | F-proof, tested | Main candidate: one identity and a five-plus-five example; full assumptions/proof in Appendix |
| T3 Matched negative normalizer | Detached `c* = sum_N w exp(s) / sum_N exp(s)` matches row loss, each positive gradient, and total negative gradient, but not individual negative allocation | Current main/math; recovered theory §3; matched-normalizer test | F-proof, tested | Main only if weight-control result remains central; otherwise Appendix. Explicitly distinguish from executed constant 0.25 |
| T4 Individual positive can be repelled | With two positives and `q=(0.8,0.1,0.1)`, derivatives are `(0.3,-0.4,0.1)` although their sum is nonpositive | Current math; recovered theory §1 | F-proof | Appendix. Useful explanation of why uniform-positive supervision is not independent attraction to every valid caption |
| T5 Total suppression does not bound every group | Suppression reduces total negative probability, yet group probability can rise from `1/102` to `0.3125` when other negatives are suppressed more | Current math; recovered theory §4 | F-proof | Appendix counterexample; no separate main theorem |
| T6 Summed-positive alternative | `log D - log sum_P exp(s)` gives every positive a nonpositive score derivative; it allocates target mass by positive scores rather than uniformly | Current math; recovered theory §5 | F-proof; training U | Appendix as the precise objective alternative; archive as a leading future experiment |
| T7 Parameter-dependent weights | The parameter derivative includes an additional `sum q_j grad log w_j` term unless weights are detached; exact-zero masks need separate treatment | Current math; recovered theory §6 | F-proof | Appendix assumptions; archive as warning against applying detached proofs to trainable semantic weights |
| T8 Original Lemma 2.1 | Single-positive softmax derivative `p_j - 1[j=+]` | Recovered theory §7; current general gradient | H origin, F-proof special case | Absorb into T1 in Appendix; do not spend a second main statement |
| T9 Original Theorem 2.2 | m equal-score/equal-weight negatives have group gradient mass `m w exp(a)/D`; D also changes | Recovered theory §7; current “Other fixed-score identities” | H origin, F-proof | Appendix exact identity; archive original label |
| T10 Original Theorem 2.3 | For suppression in `[0,1]`, `log q+ - log p+ = -log(1-R/S) >= R/S >= 0` | Same sources | H origin, F-proof | Appendix. It is a normalizer/probability statement, not a raw score-margin or retrieval improvement theorem |
| T11 Original Corollary 2.4 | Suppressed equal-weight group contributes at least `m(1-w)exp(a)` to removed mass; resulting ratio saturates with multiplicity | Same sources | H origin, F-proof | Appendix saturation correction; no unbounded linear-in-m claim |
| T12 Original Theorem 2.5 | Under detached suppression, total negative derivative and magnitude of the single-positive derivative do not increase | Same sources | H origin, F-proof | Appendix. No optimizer stability or parameter-gradient-norm guarantee follows |
| T13 Original Theorem 2.6 | At equal scores and a shared denominator, smaller weight means smaller normalized probability | Same sources | H origin, F-proof | Appendix; semantic correctness is not implied |
| T14 Original Proposition 2.7 | All weights tending to one recover the ordinary loss; a shared negative weight below one does not | Same sources | H origin, F-proof | Appendix boundary case |
| T15 Shifted-score interpretation | Weighted normalization equals ordinary normalization on `s + log w`, with `log 0 = -infinity`; smoothing changes targets instead | Recovered theory §7; current math and loss implementation | F-proof, implemented | Appendix, adjacent to exact loss implementation |
| T16 Retired “safe by default/cannot harm” reading | Fixed-score identities do not prove better representation margins, rankings, generalization, or training stability | Recovered theory ending; current math scope | H, explicitly retired | Archive the correction; never revive as a contribution or experimental win |
| T17 Historical 40-row numerical algebra audit | Reported finite-difference error below `4e-10`; original checking script/output missing | Historical study record | H | Archive only. Current gradient/constant/expansion tests provide separate fresh checks |

The two concise main identities are explanatory tools. Neither establishes
the observed learned retrieval loss as a mathematical consequence. The exact
matched constant is row-specific; the executed global constant and batch-wide
shuffle are not exact mass-matched training controls.

## Every executed training policy

All twelve policies share the fixed candidate pool and adapter capacity.
The source baseline includes supported hypotheses as negatives. Definitions
are in `docs/LOSS_RECONSTRUCTION.md`, `src/gcr/losses.py`, and
`manuscript/appendix.tex`. Results are in `results/analysis/aggregate_metrics.*`
and `manuscript/results_appendix.tex`. Each row below has three rates and three
seeds, so no method has been dropped from the archived grid.

| Item / archive name | Intervention and fresh outcome worth retaining | Status | Main / Appendix / Archive |
|---|---|---|---|
| E1 `clip` | Five source captions are positives; all hypothesis candidates stay in the pool. Comparator for expansion and source-target weight controls | F | Main baseline; exact unusual candidate convention must be defined |
| E2 `multipositive` | Promote supported hypotheses. Versus source: +5.81 pp e-SNLI-VE, +0.14 pp SugarCrepe, -8.56 pp COCO I→T, -4.54 pp T→I | F | Main central contrast; Appendix all seeds and metrics |
| E3 `grounded_no_abstention` | Expanded targets plus contradiction weight 2; highest e-SNLI mean among the main relation policies | F | Main main-policy table; Appendix exact comparison |
| E4 `grounded_no_hardening` | Expanded targets plus neutral weight 0; beats count-matched random exclusion by +0.40 pp e-SNLI and +1.77 pp SugarCrepe | F | Main compact semantic-control result if space; Appendix full outcomes |
| E5 `grounded` | Both relation operations. Versus expansion: -0.26 pp e-SNLI, -0.16 pp SugarCrepe; adjusted intervals include zero | F | Main. “No demonstrated additional gain” is supported; “significantly worse than expansion” is not |
| E6 `random_exclusion` | Remove the same number of nonpositive cells per image as neutral exclusion. Particularly poor transfer/retrieval: I→T 43.43%, T→I 29.95% | F | Appendix full row; Main comparison against E4 prevents erasing a real semantic-label benefit |
| E7 `smoothing` | Source targets mixed with 0.1 candidate-uniform targets. Seed 29 selects epoch 10 and has much weaker retention than seeds 17/43 | F | Appendix per-seed results; archive training histories. Aggregate-only presentation hides this variation |
| E8 `pairwise_rank` | Image-balanced supported-versus-contradicted softplus; source captions are not used in this loss. All three selected states are epoch zero | F | Appendix. One selection note at most in Main; do not describe frozen-selected outcomes as gains from trained ranking |
| E9 `sanw_fixed` | Detached source-text similarity weights, threshold 0.5; two seeds select epoch zero. Trails shuffled weights by 3.78 pp e-SNLI and 1.69 pp SugarCrepe | F | Main only in a compact weight-control table; full kernel/selection in Appendix |
| E10 `shuffled` | Permutes the batch-wide multiset from E9, breaking pair assignment while preserving that multiset | F | Paired with E9 in Main or Appendix; row mass is not matched |
| E11 `sanw_median` | Same similarity kernel with batch-negative median threshold; SugarCrepe 83.73%, COCO I→T 53.97% | F | Paired with E12; Appendix all selection details |
| E12 `constant` | Source positives and negative weight 0.25; SugarCrepe 83.67%, COCO I→T 55.21%, T→I 39.39% | F | Strong underused Main control: more transfer and much better retrieval than expanded targets, but lower annotated-task accuracy |
| E13 Frozen reference | Exact model/normalization reference; five selected epoch-zero states are bitwise equal in evaluated predictions | F | Main reference row; Appendix exact normalization and identical-state accounting |

## Other experiments, measurements, and checks

| Item | Content and source | Status | Recommended disposition |
|---|---|---|---|
| X1 Complete optimization/selection grid | 108 `best.pt`, 108 `history.json`, completion receipts, 36 selected copies; 1,188 epoch-zero-through-ten history records. `results/study/`, `training_final_audit.json` | F | Main one setup sentence; Appendix selection rule, seeds/rates, epoch-zero outcomes, all histories in Archive |
| X2 e-SNLI-VE supervised ranking | 400 held-out images; image-balanced Cartesian supported/contradicted comparisons, half-credit ties. `results/evaluation/` and evaluation index | F | Main primary endpoint; Appendix exact denominator and relation labels |
| X3 Full SugarCrepe | All 7,511 strict caption/foil pairs; seven original categories and per-item scores | F | Main primary transfer endpoint; Appendix seven-category table |
| X4 Addition versus replacement/swap | Expansion-source +1.92 pp on 2,754 addition items versus -0.90 pp on 4,757 combined replacement/swap items; both signs hold in all three seeds. `sugarcrepe_category_descriptive.json` | F, descriptive | Main high-value explanation of aggregate cancellation; Appendix individual categories/seeds |
| X5 SugarCrepe++ valid-caption retention | P1, P2, and both-positive success; all 31 nonzero selected adapters reduce P2, while 26 improve P1. `retrieval_and_caption_retention.json` | F, descriptive | Main visual or concise result; Appendix full seed coordinates. This is across policies, not uniquely caused by expansion |
| X6 Full COCO retrieval | 5,000 images × 25,000 captions, both directions, R@1/5/10 and mean/median rank, per-query ranks | F | Main R@1 and concrete query cost; Appendix other ranks; Archive all raw rankings |
| X7 Query-level success transitions | Expansion versus source: I→T net failures 396/539/349, mean 428; lost and newly gained successes separately saved. `retrieval_and_caption_retention.json` | F, descriptive | Main mean practical scale; Appendix all six direction/seed rows. Raw rank changes remain available for further analysis |
| X8 Planned uncertainty family | Six paired image-bootstrap contrasts, 10,000 replicates, 99.1667% intervals, selected seeds averaged before resampling. `primary_contrasts.json`, `primary_bootstrap_samples.npz` | F | Main interval only for central contrast; Appendix entire fixed family |
| X9 Conditional neutral-exclusion result | Combined minus contradiction-only on e-SNLI: -0.535 pp, interval [-0.890,-0.193]; only interval in the six-test family excluding zero | F | Consider one Main sentence if useful to the intervention story; Appendix exact conditional comparison. Do not replace the primary expansion comparison with this selectively |
| X10 Relation classifier inputs | Image/text/multimodal logistic classifiers; held-out macro-F1 34.06/61.38/64.96. `results/relation_diagnostic_iter10000/` | F, separate diagnostic | Appendix, since classifiers never generated adapter-training labels |
| X11 Calibration and selective gates | Temperature scaling; NLL/Brier/ECE; only multimodal contradiction gate accepts test pairs: 683 accepted, 647 correct, 94.73% precision, 11.65% coverage | F, descriptive | Appendix complete selection/calibration protocol and intervals; Archive models/predictions |
| X12 Image-disjoint cross-fitting | Three folds, all 16,192 training hypotheses; fold-specific C selection, calibration and gates, saved OOF probabilities and models | F; downstream use U | Appendix method/data-resource note; Archive reusable outputs. Strong enabling material for a future predicted-label experiment |
| X13 Diagnostic convergence attempt | Original 2,000-iteration run stopped during text-grid fitting; complete amended 10,000-ceiling rerun kept separately. Old script and image-mode outputs remain | F, superseded attempt | Appendix concise amendment; Archive every old/new log and receipt |
| X14 Input/source correction | Deprecated e-SNLI-VE dev/test overlap rejected before feature scoring; later official e-ViL release selected. Incomplete Flickr caption mirror replaced by full official Entities source | F, data audit | Appendix provenance; Archive rejected sources/audits, not a model-performance result |
| X15 Overlaps, repeated annotations, benchmark anomalies | 8,177 distinct image files verified; 48 duplicated e-SNLI image/text groups without label conflict; 2 SCPP P1=foil cases; all official rows retained. `results/input_audit.json`, source audits | F | Appendix complete audit; keep Main free of a repeated caveat list |
| X16 Source-caption/foil differences between benchmarks | SCPP changes 1,203 P1 strings and 878 foil strings; cannot substitute P1 scores for SugarCrepe replacement/swap. `data/benchmark_annotation_alignment.json` | F | Appendix essential audit, Archive exact source strings |
| X17 Encoder implementation/feature verification | Pinned CLIP bytes, frozen scale, image decoding/hashes, normalized 512-D features, causal-padding parity, cache identity, archived first extraction implementation | F | Appendix reproducibility; Archive preprocessing/repair history and exact NPZs |
| X18 Synthetic preflight and unit tests | Twelve-method forward/backward timing, exact loss/gradient/mask tests, evaluation/training/diagnostic/recovery tests. `results/training_preflight.json`, `tests/`, test logs | F, engineering checks | Appendix one verification statement; Archive detailed tests/timings; not natural-image scientific evidence |
| X19 Independent audits | 629 aggregate recomputations with max difference `2.22e-16`; independent bootstrap reconstruction and selected-state validation; sampled full-pool query rescoring | F | Appendix evidence audit; Archive scripts/receipts. They validate reported computations, not extra independent model runs |
| X20 Historical 27 synthetic runs | Only the old execution's description remains | H | Archive only; not the current synthetic preflight |
| X21 Historical 192-image COCO pilot | Acquisition/pipeline pilot, without recoverable original outputs | H | Archive only; superseded by fresh complete COCO evaluation |
| X22 Historical two-pair SmolVLM smoke test | SmolVLM2-500M-Video-Instruct revision/hash retained, no original checkpoint/output bytes in this project | H | Archive only; not evidence for a second backbone or predicted-relation adaptation |
| X23 Original CIKM reported experiments | Earlier author-reported findings were not pooled with the reconstructed adapter study; original tables/raw evidence unavailable | H, unrecoverable detail | Archive the boundary. Individual missing original experimental rows cannot honestly be enumerated from present files |

**Checkpoint granularity matters for future work:** the archive preserves
each candidate's best state and all validation history entries, not a model
checkpoint for every epoch. A common-final-epoch held-out comparison therefore
cannot be recovered just by selecting another saved state; it requires new
training/checkpoint collection.

## Figures, tables, and display assets

| Item | Asset and data source | Status | Recommended disposition |
|---|---|---|---|
| V1 Valid-caption scatter | `caption_retention.pdf/.png`, `figure_data.json`; all 31 nonzero selected adapters; x=P1 change, y=P2 change against frozen | F | Main if it adds understanding beyond the table; retain exact seed coordinates in Appendix/Archive |
| V2 Concrete image-search failure | `retrieval_query.jpg`, `retrieval_example.tex`, `retrieval_example_data.json`; COCO 60623; correct caption rank 1→6 in seed 17 | F, deterministic post hoc illustration | Main candidate for practical entry point; selection rule in Appendix. It is illustrative, not a representative sample |
| V3 Validation trajectories | `validation_trajectories.pdf`, `validation_figure.tex`; three seed lines at each policy's selected rate, retained-epoch dots | F | Appendix; Archive all 108 full histories, including rates omitted from the figure |
| V4 Main intervention table | `main_results_table.tex` | F | Main; exact policy names and main performance tradeoff |
| V5 Edit-group table | `edit_groups_table.tex` | F, descriptive | Main if not merged into a better figure; no need to repeat every number in prose |
| V6 Weight-control table | `weight_controls_table.tex` | F | Main only if nonspecific suppression is one of the central claims; otherwise Appendix |
| V7 Complete numerical tables | `results_appendix.tex`: all-policy discrimination/retention, per-seed rates/epochs, six intervals, seven categories, query transitions, R@1/5/10 | F | Appendix; preserve all tables through rewrites |
| V8 Diagnostic displays | `diagnostic_results.tex`: calibration metrics, accepted/correct counts and intervals, C/temperature/OOF/convergence | F | Appendix only |
| V9 Provenance/training displays | `split_counts_table.tex`, `training_details.tex`, `training_summary.tex` | F | Main only essential split/model/grid summary; Appendix full definitions |
| V10 Current compiled/source snapshots | `manuscript/paper.pdf`, `output/pdf/acl27_relation_labels_short.pdf`, draft PDF, `output/acl27-overleaf-source.zip`, result/layout receipts | F | Archive previous complete versions before replacement; not independent studies |
| V11 Historical prior scatter/layout | Earlier thirty-adapter scatter, old standard-deviation/seed tables, nineteen-page PDF mentioned in recovery record | H; original data/bytes missing | Archive descriptions only; do not redraw old points from aggregate means |
| V12 Original CIKM illustrations/tables | No original uploaded PDF or separate assets present | H; unrecoverable | Explicit missing slot. Number, captions, and exact contents cannot be recovered from this workspace |

`official_template_example.tex` and its sample figures are ACL formatting
examples, not project results. Style/font repair files and build QA images are
engineering provenance rather than additional scientific figures.

## Every retained unexecuted expansion/application

Sources for U1–U15 are `historical_context/study_and_recovery_record.txt`,
`docs/FINAL_STORY_DECISIONS.md`, and the current appendix extension paragraph.
U16 is additionally explicit in current `manuscript/implications.tex`.

| ID / proposal | Question or use | Status | Recommended disposition |
|---|---|---|---|
| U1 Second backbone | Does the intervention pattern persist beyond this CLIP checkpoint? | U | Archive roadmap; Appendix scope. Historical SmolVLM smoke test does not satisfy it |
| U2 Full-encoder adaptation | Does the result survive updates to pretrained representations rather than two residual maps? | U | Archive roadmap; materially larger study |
| U3 More training data | Distinguish small-data behavior from a broader relation-supervision effect | U | Archive roadmap |
| U4 More training seeds | Quantify training-run uncertainty beyond three selected seeds | U | Archive roadmap; image bootstrap is not a substitute |
| U5 Independent visual adjudication | Replace inherited relation labels with image-verified support/contradiction/uncertainty | U | Archive roadmap; addresses a different label source, not a completed robustness test |
| U6 Shared final epoch and hyperparameters | Isolate an optimization intervention from independently selected procedures | U | Archive experiment design; requires new saved states/training |
| U7 Exact denominator/gradient-mass matching | Hold total negative competition fixed while testing semantic allocation | U; T3 gives a row-level construction | Strong theory-connected follow-up. Appendix identity and Archive experiment design; not equivalent to global 0.25 or global shuffle |
| U8 Hardness-matched exclusion | Control not only the number but difficulty of omitted candidates | U | Archive follow-up to the fresh neutral-versus-random result |
| U9 Summed-positive probability training | Change positive target allocation and test retention rather than only explain uniform-target dilution | U; T6 is proved | Strong mechanistic follow-up; Appendix objective, Archive experiment plan |
| U10 Predicted-relation adapter training | Replace supplied training labels with calibrated, cross-fitted predicted relations | U; classifier/OOF assets F | Strong reusable extension. No support gate accepts pairs under the current rule, so a full predicted-positive pipeline is not established |
| U11 Inverse-multiplicity weighting | Remove repeated-annotation influence while preserving unique semantic examples | U; duplicate audit F | Archive targeted robustness experiment, not a completed correction |
| U12 Retention-constrained selection | Choose the best discrimination model under a predeclared acceptable retrieval loss on representative validation queries | U | Most direct application extension; one Main implication, detailed prospective protocol in Archive |
| U13 Feature distillation | Preserve frozen-model retrieval behavior while adding relation supervision | U | Archive prospective method; compare at matched data and tuning budget |
| U14 Corrected many-to-many retrieval relevance | Judge additional valid image/caption matches beyond ownership annotations | U | Appendix limitation/future data design; requires new relevance labels |
| U15 Natural deployment | Evaluate actual image-search users/queries and task costs | U | Archive long-term application; no deployment/user study exists |
| U16 Frozen first-stage retrieval plus relation-adapter reranking | Keep the frozen model's candidate shortlist, use relation scores only to reorder it | U | Strong practical next test with existing feature/checkpoint assets. One restrained Main implication; no performance claim before evaluation |

These proposals remain distinct. In particular, a reranking evaluation, a new
selection rule, a new target objective, and a new predicted-label training
pipeline are four different experiments, not interchangeable descriptions of a
completed “retention solution.”

## Prior-art continuity and missing references

The current bibliography contains CLIP, The Hard Positive Truth, FSC-CLIP,
CLIC, SupCon, e-ViL, SugarCrepe, and SugarCrepe++. Current citation verification
metadata is retained in `manuscript/citation_verification.json`.

The recovered introduction also names four absent citation keys:
`jiang2023regulation`, `gao2024softclip`, `liu2023sat`, and `byun2024mafa`.
Their retained roles are similarity regulation, softened targets, semantic
triage, and grounded filtering for false negatives. A MAFA arXiv link remains
in the historical study record. The complete old BibTeX is unavailable.
Preserve these pointers in the archive and verify their primary sources before
adding specific claims or reconstructed bibliography entries.

The existing related-work records already state that hard-negative improvement
can harm valid rewordings and that generic extra positives need not solve it.
That broad observation cannot carry this paper's novelty by itself. The
available distinctive evidence is the matched relation-policy/control audit
and its measured transfer/retrieval consequences.

## Historical claims that must not migrate unchanged

| Recovered old claim | Fresh evidence / editorial consequence |
|---|---|
| Combined policy significantly worse than positive expansion on both primary tasks | Both fresh adjusted intervals include zero. State lack of an established incremental gain, not significant inferiority |
| Extra positives give +0.94 pp SugarCrepe with -7.53 pp image retrieval | Fresh comparison is +0.14 pp and -8.56 pp; use only fresh values |
| Addition +3.44 pp versus replacement/swap -0.50 pp | Fresh descriptive grouping is +1.92 pp versus -0.90 pp |
| All 30 nonzero adapters lower both alternative-caption accuracy and T→I retrieval | Fresh count is 31 nonzero; all lower alternative-caption accuracy, but only 26 lower T→I retrieval |
| Fixed/shuffled and median/constant controls closely track and select the same rates/epochs | Fresh fixed weighting trails its shuffle and has two frozen selections. Only the close median/constant SugarCrepe mean survives in that form |
| Old diagnostic 66.28% multimodal F1 and 635 accepted contradictions | Fresh diagnostic is 64.96% multimodal F1 and 683 accepted contradictions, of which 647 are correct |
| Old 105-test / 40-row audit / nineteen-page PDF receipts | Retain as historical records. Current test/audit/build receipts identify the fresh execution separately |

## Strongest material to promote, retain, or leave out

1. **Promote the useful control comparison.** Constant suppression scores
   83.67% on SugarCrepe versus 83.39% for positive expansion and retains 55.21%
   versus 47.29% COCO I→T R@1. Its supervised relation score is lower. This
   exposes a concrete objective-dependent tradeoff without requiring a new
   model or claiming one policy wins every task.
2. **Preserve the real conditional semantic benefit.** Neutral exclusion beats
   count-matched random removal, yet provides no stronger overall result than
   positive expansion and has a negative primary contrast when added to
   contradiction weighting. A blanket “relation labels do not help” story
   would erase executed evidence.
3. **Use one practical scale and one illustration.** The 428 additional failed
   image queries and the recorded rank-1-to-rank-6 image example make retrieval
   cost concrete. Full lost/new-success tables belong in the appendix.
4. **Keep the calibrated OOF resource complete but secondary.** It is substantial
   executed work and a direct basis for future predicted-label training. It
   does not answer the current supplied-label intervention question.
5. **Keep theory complete through hierarchy.** One or two main identities can
   explain target redistribution and negative competition. All other valid
   special cases, assumptions, counterexamples, the alternative objective,
   and retired interpretations have explicit appendix/archive homes above.

## Explicit unrecoverable original items

- Original uploaded CIKM manuscript bytes, exact original figures/tables and
  captions, and the full original author bibliography/source repository.
- The lost earlier complete four-page/nineteen-page rewrite's compiled PDF,
  complete appendix/BibTeX, original figure coordinates, and complete source.
- Its raw experimental data snapshots, exact image/split selections, feature
  caches, all checkpoints, per-item predictions, seed logs, frozen ledgers,
  and original runtime environment. Fresh equivalents exist but are a new run.
- Exact unrecovered historical implementation choices, including the old
  similarity kernel, smoothing base target convention, and optimizer/grid
  details now fixed explicitly in the reconstruction protocol.
- The actual files behind the historical 27-run synthetic study, 192-image
  acquisition pilot, two-pair SmolVLM smoke test, and 40-row algebra check.
- Any original idea or experiment never described in the retained recovery
  text. Its existence or detailed content cannot be inferred from absent files.

The old PDF and 3.93 GB ZIP hashes survive in
`historical_context/READ_ME_FIRST.txt`; a hash identifies missing content but
does not reconstruct it. Current full project/Overleaf/reproducibility ZIPs and
staged checkpoints are available in the workspace and preserve the fresh
study. They do not fill these original-content gaps.
