# Theory disposition for the final rewrite

## Decision

Keep **positive-target redistribution** in the main paper. Move the matched-negative-mass constant and its proof to the appendix, together with the other fixed-score identities. The strongest current story is that relation supervision buys discrimination on the annotated task, with little aggregate transfer gain and substantial retrieval cost. The positive-target calculation directly clarifies the intervention at the center of that story. The constant identity primarily motivates a secondary weighting-control family and needs more qualifications to connect it to the executed symmetric objective.

The main-paper statement should concern the **image-to-text target distribution**. If an image has K original captions and M additional supported captions, the original captions jointly receive K/(K+M) of the target probability. This is simpler than introducing an extra derivative equation and then paraphrasing it. Retain the exact gradient difference M/[K(K+M)] in the appendix proof. Neither statement establishes a learned-score, ranking, or retention guarantee.

No training source, result, or manuscript source was edited for this audit. This document records recommendations for the root editor's decision.

**Editorial resolution from root:** retain both the target-budget interpretation and exact derivative identity late in the main paper; move c* and the complete negative-weight theory to the appendix. The recommendation above records the initial space-priority assessment. The inventory and assumption checks below apply to this accepted layout. The common loss and every required symbol must be defined before the late-main derivative statement.

## Material actually recovered and checked

Every file in `../recovered_context/` has an identical counterpart in `historical_context/`, including the recovered theory, manuscript, result tables, and study record. They are two copies of the same retained context, not independent evidence. The original submitted PDF and original author implementation are not present in either directory. Therefore original theorem numbers below refer to the retained inventory, not a newly inspected source PDF.

Checked sources include:

| Source | SHA256 at audit |
|---|---|
| Both copies of `theory_recovered.txt` | `f661acf658c4e764d60e72a78c3641900ea1cc594fb1fd1a18d8c35dc7ac6af9` |
| `historical_context/study_and_recovery_record.txt` | `e9c842f0b8ebc2ac9c2770267929dc85d3b3878ed48b671c08a9107d286656ef` |
| `historical_context/manuscript_main_recovered.txt` | `67c32bc6b124a855e8007eb6b195a101e95e14e45d385f20d3856d0d07d577a6` |
| `manuscript/appendix_math.tex` | `2837d994dbde09d1b98e100698680cb80d182d45f2ae499ed8e38b1590b9ba7a` |
| `src/gcr/losses.py` | `2e3a06a0c22d94b44ae29dbed9b6e44a2c2737ac6fe71afc4c2a23495dc51c35` |
| `tests/test_losses.py` | `268f93e2b79cfe26b3a73a3350b3bb2553006ada8415f0a807473d5d35ce4956` |

I also read the current main text, policy appendix, current generated result tables, and `docs/LOSS_RECONSTRUCTION.md`. The recorded loss-test receipt reports 31 passing tests. No experiments were rerun for this editorial audit.

## Objective and minimum assumptions

For a finite candidate set J, nonempty positive set P, finite scores s_j, and nonnegative weights w_j fixed for differentiation, with w_p=1 on every positive, define

- D = sum over J of w_j exp(s_j), which is positive;
- q_j = w_j exp(s_j)/D;
- y_j = 1[j in P]/|P|;
- ell = log D - sum over J of y_j s_j.

These definitions support every retained uniform-target row identity. Zero weights have zero target and are exact omissions. The current implementation computes the normalizer with logsumexp and log zero equal to negative infinity. It detaches semantic weights, masks, and the pretrained score scale.

The executed objective averages eligible image-row losses and eligible text-column losses separately, then gives each direction weight one half. A text column without a positive image contributes no reverse-anchor loss. Expanding the positive set can therefore change both row targets and the set and normalization of reverse anchors. The pairwise-ranking control is a different objective. Smoothing also uses a different target, so the uniform-positive derivative specialization must not be silently applied to it.

## Complete retained identity and counterexample inventory

| ID / retained origin | Exact content and assumptions | Current disposition and recommendation |
|---|---|---|
| Original Lemma 2.1 | Unit-weight, single-positive softmax derivative p_j minus the positive indicator. | Valid special case of the common gradient. Appendix only; no separate main-paper lemma. |
| Recovered item 1: common gradient | Under the stated common assumptions, derivative of ell with respect to s_j is q_j-y_j. | Valid and foundational. Keep definition and two-line proof in appendix. Main needs only the loss definition and target-budget interpretation. |
| Recovered item 1: individual-positive sign example | With two positives and q=(0.8,0.1,0.1), derivatives are (0.3,-0.4,0.1). A positive candidate can have a positive score derivative. | Valid. Keep one numerical example in appendix; do not describe every positive as being pulled upward. |
| Recovered item 1: total-positive identity | Sum of positive derivatives equals minus the total negative probability, hence is nonpositive. Requires uniform targets summing to one on P. | Valid. Keep beside the preceding example, rather than repeating it elsewhere. |
| Recovered item 2: positive expansion | With unchanged candidates, scores, and unit weights, relabel M existing nonpositives as positives. An old positive derivative increases by M/[K(K+M)]; each new positive derivative changes by -1/(K+M); all other derivatives are unchanged. | Valid; strongest main-relevant theory. Main should state the target-share consequence. Full piecewise derivative and proof stay in appendix. |
| Recovered item 2: conserved target budget | Old positives jointly lose M/(K+M) target mass; new positives gain precisely that mass. Their remaining share is K/(K+M). | Valid direct consequence already implicit in the retained proof. Use this simple form in main. It is not an additional independent contribution. |
| Recovered item 3: matched negative mass | For a nonempty negative set N and fixed scores, c* = sum_N w_j exp(s_j) / sum_N exp(s_j). Replacing negative weights by a detached c* preserves D, row loss, positive probabilities and derivatives, and total negative derivative. | Valid row identity. Move display/proof to appendix. The executed constant is 0.25, not this fitted c*. |
| Recovered item 3: allocation differs | The original negative derivative w_j exp(s_j)/D generally differs from c* exp(s_j)/D even though their sums agree. | Valid. Retain once immediately after the matched-mass proof. |
| Recovered item 4: total-negative suppression | With one positive, w_+=1 and 0<=w_n<=1, D<=S and q_+>=p_+, so total negative derivative does not increase. | Valid. Appendix only. Contradiction weight 2 violates the suppression premise. The aggregate statement also extends to multiple unit-weight positives by replacing the positive exponential by its sum, but this generalization adds little editorial value. |
| Recovered item 4: negative-group counterexample | Exponential masses 1 positive, 1 selected negative group, 100 other negatives; weights 0.5 and 0.001. Group mass grows from 1/102 to 0.3125 although total negative mass falls. | Valid. Keep in appendix as the exact counterexample needed to delimit the suppression claim. |
| Recovered item 5: summed-positive objective | A=sum_P exp(s_p), ell_set=log D-log A, derivative on p is q_p-exp(s_p)/A<=0 because D>=A. | Valid distinct objective under unit positive weights. Keep definition and proof in appendix; mark as analytically described, untrained. It can favor already high-scoring positives. |
| Recovered item 6: parameter-dependent weights | With fixed targets, differentiable scores, and differentiable strictly positive nonmasked weights, parameter gradient adds sum_j q_j grad log w_j to sum_j(q_j-y_j) grad s_j. | Valid. Use a fixed nonzero-support set and state w_p identically one for continuity with the studied objective. Keep in appendix. |
| Recovered item 6: zero-weight boundary | The log-weight derivative formula treats exact zeros as a fixed mask and does not cover a support change through zero. | Valid scope condition. Put once next to the formula, not as repeated main-body prose. |
| Original Theorem 2.2 | A group of m negatives with common score a and weight w contributes m*w*exp(a)/D derivative mass. | Valid identity. Appendix, grouped with other single-positive facts. D also changes with m and w. |
| Original Theorem 2.3 | With suppression weights in [0,1], R=sum_N(1-w_j)exp(s_j), D=S-R, and log q_+-log p_+=-log(1-R/S)>=R/S>=0. | Valid fixed-score probability statement. Appendix. It does not increase the raw score margin or raw ranking. |
| Original Corollary 2.4 | For a suppressed equal-score group, R>=m(1-w)exp(a). The resulting lower bound m(1-w)exp(a)/(A+m exp(a)) is bounded as m grows. | Valid under nonnegative suppression elsewhere. Current appendix explains saturation but omits the explicit R lower-bound step; add it if preserving every old numbered result completely. Do not claim that all possible log-probability gains saturate: with w=0 the actual log gain can grow, while this particular lower bound still saturates. |
| Original Theorem 2.5 | For a single positive under detached suppression, total negative derivative and absolute positive derivative do not increase. | Valid corollary of total-negative suppression. Merge with that proof; retire optimization-stability and parameter-gradient-norm interpretations. |
| Original Theorem 2.6 | Within one fixed normalizer, equal scores and smaller weights imply smaller normalized probabilities. | Valid ordering identity. Appendix. Across different training states or separately changed normalizers, the premise does not apply. |
| Original Proposition 2.7 | As all weights approach one, the loss continuously approaches the ordinary loss. A shared negative weight below one while positive weights remain one is not the same ordinary objective. | Valid. Appendix. Do not label arbitrary constant suppression as exact recovery of CLIP. |
| Recovered shifted-score identity | Weighted normalization equals softmax normalization on s_j+log w_j, treating zero weights as negative infinity. Because positive weights are one, this gives the same uniform-positive cross-entropy. | Valid. Appendix; useful for exact implementation audit. It is not target smoothing. |
| Retired original interpretation | “Safe by default,” “cannot harm,” representation margin, generalization, optimizer stability, and parameter-gradient guarantees. | Unsupported by these identities. Preserve the retirement record in the audit, not as a catalogue of defensive sentences in the main paper. |

No retained mathematical item requires deleting a valid result. The editorial change is to give the main paper one coherent analytical role and consolidate complete proofs in the appendix.

## Two useful exact clarifications for the appendix

### Expansion also changes reverse-anchor weighting

This is a direct clarification of the implemented symmetric objective, not a new empirical mechanism claim.

Use the unsmoothed uniform-positive objective. Hold all scores, candidate image rows, text candidates, and unit weights fixed. Let C0 be the text columns that had at least one positive image before expansion. Let H be the additional columns that acquire a positive image after supported hypotheses are promoted, with the new eligible set exactly the disjoint union of C0 and H. Assume the positive sets of columns in C0 are unchanged, as they are for source-caption columns in this dataset. Write T0=|C0|>0 and Hn=|H|. Each old source-column loss is unchanged. Hence, for Hn>0,

`L_reverse_expanded = [T0 * L_reverse_source + sum over j in H of ell_j_expanded] / (T0+Hn)`.

For nonempty H this is the corresponding weighted mean of the old and new anchor means. The sum form also handles empty H without an undefined mean.

If Hn=0, the old reverse mean is unchanged. The factor one half from the complete symmetric loss can be applied to both sides. Existing source-anchor contributions are diluted in the reverse average by T0/(T0+Hn). This is separate from K/(K+M) redistribution inside each image-row target.

Recommendation: include this identity or its two-sentence explanation immediately after the row expansion proof. It makes the symmetric implementation fully auditable and prevents the fixed-row statement being mistaken for an account of the entire update. It does not need main-paper space.

### Row-matched constants need not match the transposed objective

The scalar c* theorem applies independently to a fixed row. A row-specific replacement generally changes reverse-direction normalizers when the same weight matrix is transposed. It does not construct a symmetric training control that simultaneously matches both directions.

For an explicit example, take two images, three texts, every score zero, positive pairs (1,1) and (2,2), and weight rows `[1,0.2,0.8]` and `[0.6,1,0.4]`. Each row has negative mass one, so its matching constant is 0.5. Both image-row normalizers remain two. In the two eligible text columns, however, normalizers change from 1.6 and 1.2 to 1.5 and 1.5. The full symmetric loss changes by `0.25*log(2.25/1.92) = 0.03965125754415964`.

Recommendation: state the row-only restriction explicitly near the existing “single global constant need not match every row” sentence. The numeric counterexample is optional; retaining it costs little in the appendix and makes the restriction easy to audit. This restriction further favors moving c* out of the main story.

## Definitions and assumption fixes to apply during the rewrite

1. Define K as the number of original source positives and M as the number of already-present supported candidates that become positive. State that the simple main calculation is for one image-to-text target distribution. The denominator is unchanged only with fixed scores, unchanged candidates, and unit weights.
2. Prefer “assigned target probability” or “target share” to “target pressure.” A positive derivative can already be positive, so an unqualified statement that every original positive is attracted less invites a sign misunderstanding.
3. In the matched-mass proposition, put “compute c* at the specified scores and hold it fixed for differentiation” before every derivative conclusion. The present statement eventually says this, but placing it in the assumptions is easier to audit.
4. Explain once that exact c*(s) recomputed without detachment makes its weighted normalizer equal the original weighted normalizer as a function of scores. The negative-gradient redistribution argument then disappears because differentiating c*(s) restores the original individual negative derivatives. The detached analytical reference and the executed fixed 0.25 control are different constructions.
5. Add the explicit original Corollary 2.4 lower-bound step if the appendix promises to retain every earlier result. Qualify saturation as a property of that rational lower bound.
6. In the parameter-dependent-weight formula, define the fixed set of unmasked entries, state their weights are differentiable and strictly positive locally, and retain w_p=1 for positive targets. No log derivative is taken at an exact zero.
7. The table of policies currently uses `Unif(P)` and `Unif(J)` before J is explained in a later paragraph. Define J as the current candidate set before the table, or replace the table notation with “all candidates.” Use distinct labels for source-caption sets and contradicted-hypothesis sets instead of relying only on C versus C-prime.
8. Keep the full symmetric normalization and reverse-anchor eligibility rule in the policy appendix. The current rectangular-loss test correctly verifies that reverse columns without a positive are skipped.
9. All derivations concern the mean-positive-log-probability objective. The pairwise and smoothed controls need their own equations and should remain separate. No theorem should be presented as simultaneously applying to all twelve methods.

## Main-paper wording and space recommendation

Use one short paragraph late in the main paper, after the empirical results, approximately 90 words:

> **Adding positives reallocates the target.** The image-to-text loss divides one unit of target probability equally among positive captions. With K original captions and M additional supported captions, the originals retain K/(K+M) of that probability. Five originals and five additions give the originals half the target mass. At fixed scores and unit weights, this change comes entirely from the targets. Appendix [math] gives the exact gradient change and the corresponding change in text-anchor averaging. This makes retention a relevant measurement even when every additional label is correct.

Use the displayed K/(K+M) expression only if the surrounding layout benefits from it; otherwise the inline form saves space. Do not repeat the gradient display, a paraphrase, and the same numerical example in the main body. If the result section already motivates retention clearly, the final sentence can also be cut.

For similarity controls, a short empirical sentence can point to the appendix: their assignments are compared with a shuffle and fixed suppression. The c* formula does not need to interrupt the practical story. Avoid presenting the 0.25 constant or shuffled weights as an exact mass-matched experimental control; neither preserves each row's weighted exponential mass.

## Every retained unused direction

The following are research directions retained in the recovered record or current manuscript. They remain unexecuted unless a future protocol and fresh results establish otherwise. The current diagnostic and its out-of-fold predictions are completed, but downstream adapter training with those predictions is not.

| Direction | Why it matters / relation to theory | Recommended location |
|---|---|---|
| Second backbone | Tests whether observed intervention outcomes depend on the frozen representation. | Appendix future-work inventory. |
| Full-encoder adaptation | Changes the score Jacobian and optimization scope; row identities still hold locally but give no outcome guarantee. | Appendix. |
| Larger training data | Tests whether relation supervision transfers with more coverage. | Appendix. |
| More independent training seeds | Quantifies training-run variation beyond image-bootstrap uncertainty. | Appendix. |
| Independent image-adjudicated relation labels | Tests whether annotation-source labels explain task-specific gains. | One application-oriented next-step sentence if room; details appendix. |
| Common-final-epoch contrasts | Separates some selection effects from comparisons of independently selected procedures. | Appendix. |
| Denominator-matched natural-image training | Tests whether the weighting effect comes from negative mass or its assignment. Requires specifying both symmetric directions. | Appendix; the row c* identity motivates design but does not supply a ready symmetric control. |
| Exact gradient-mass-matched training | A finer version of the preceding control. Must define which rows, directions, or groups are matched. | Appendix. |
| Hardness-matched exclusion | Tests semantic neutrality beyond removal count and score hardness. | Appendix; stronger than the executed count-only random control. |
| Summed-positive-probability training | Tests an alternative allocation across positives. Its attractive positive derivative does not imply better retrieval. | Appendix beside the analytical alternative. |
| Predicted-relation adapter training | Uses the completed classifier's calibrated/cross-fitted predictions in a new intervention study. | Appendix; current primary adapters use supplied labels. |
| Inverse-multiplicity weighting | Separates semantic evidence from repeated source/hypothesis annotations and target-count effects. | Appendix. |
| Retention-constrained adaptation or selection | Compares discrimination at an explicit acceptable retrieval loss. | Strongest practical follow-up; one main implications sentence, experimental design appendix. |
| Frozen-feature distillation | Could constrain representation movement while relation objectives train. | Appendix; no guarantee without an evaluated objective and constraint. |
| Corrected many-to-many retrieval relevance | Tests whether ownership labels miss additional valid matches. | Appendix limitations/extensions. |
| Natural deployment | Tests actual query distributions and user relevance rather than benchmark-only outcomes. | Appendix or short implications line. |
| Frozen retrieval followed by relation-adapter reranking | Current manuscript's application idea, not present as an executed comparison. Restricts the adapter's role to candidate reordering; preservation of end-to-end retrieval quality is not automatic. | Short main implications proposal only if clearly future work. |

The theory does not support selecting a stronger performance claim than the fresh experiments show. It supports a sharper experimental interpretation: extra positives change a normalized target budget, while negative weighting changes both aggregate competition and its allocation. The main paper should emphasize the first because it directly concerns the principal tested intervention and measured retrieval cost.

## Independent scope review

A second agent independently inspected the current loss and appendix, checked
the reverse-anchor identity and the two-by-three transpose counterexample,
and confirmed the detached-versus-differentiable c* distinction. It found no
false fixed-score identity under the assumptions above. This was an algebraic
and static-code review; no new training or empirical evidence was generated.
