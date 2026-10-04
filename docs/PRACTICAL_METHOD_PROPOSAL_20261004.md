# Remaining practical comparison and a possible next method

Date: 2026-10-04.

Status: design assessment only. This document reports no new fit or evaluation.

## 1. Complete the missing comparison first

The related-work plan specifies four cells: U, A, D, and A+D.
The frozen retention protocol contains U, A, and D.
It does not contain A+D.
This is an unresolved design recommendation, not an incomplete fit within the frozen protocol.
The inspected documents do not explain the omission.

Let `S` denote Source loss and `U` denote Supported loss.
Let `D` denote the existing bidirectional teacher-to-student KL term.
The term uses source-caption columns and temperature `T=2`.

| Cell | Objective |
|---|---|
| U | `U` |
| A | `lambda*S + (1-lambda)*U` |
| D | `U + beta*T^2*D` |
| A+D | `lambda*S + (1-lambda)*U + beta*T^2*D` |

No change to `beta` generally makes D equivalent to A+D.
Allocation changes the supervised gradient.
Distillation adds a separate gradient from the teacher distribution.
Their separate results do not identify their interaction.

Use the original `lambda={0.5,0.8}` and `beta={1,4,16}` values.
Keep both encoders, all three rates, all three seeds, and the existing development selector.
This gives 54 new candidates per encoder.
Each candidate has ten training epochs and an epoch-zero reference.
Keep the original candidate pool, source features, batch schedule, and optimizer settings.
Bind this extension to a new protocol.
Do not change the frozen retention protocol.

Use two comparisons after development selection.

1. Compare each independently selected procedure under the same development rule.
2. Compare U, A, D, and A+D at the A+D-selected rate and epochs.

The second comparison must use the selected `lambda` for A and A+D.
It must use the selected `beta` for D and A+D.
It isolates the interaction from schedule differences.
Earlier checkpoints can supply controls only when their inputs, code, and schedule match exactly.
Otherwise, refit the required controls and identify the new execution.

For an outcome `Y`, the matched interaction is

`[Y(A+D)-Y(D)] - [Y(A)-Y(U)]`.

Report both retrieval directions and caption discrimination.
Use paired image clusters when constructing uncertainty estimates.
Freeze the new comparison family before evaluation.
Keep the previous failure results in the research record.

The separate execution protocol includes randomized promotion under the selected schedule, regardless of the practical result.
It requires all nine seed-by-draw controls for each encoder.
This control tests whether caption identity contributes beyond preservation.
The separate frozen protocol takes precedence over this design assessment.

## 2. What remains if A+D fails

Average minibatch KL does not constrain every query on the complete gallery.
The existing rank theory already proves this limitation.
The next intervention should act on complete-gallery retrieval events.
Another coefficient grid would not address that distinction.

The proposed next model uses one score for every task:

`s_A(x,y) = c * x^T (I+A) y`.

Here `x` and `y` are the fixed, normalized encoder features.
The original logit scale is `c`.
The learned matrix is `A`.
At `A=0`, this score equals the frozen score.
The model does not normalize the transformed features again.
That choice makes each score affine in `A`.

Use this same score for retrieval and caption discrimination.
Do not choose a head from the benchmark identity or candidate count.
Both retrieval directions use the same matrix.
This changes the score architecture from the previous normalized residual adapter.
All method comparisons must therefore include controls with this architecture.

## 3. Exact constraints on the training gallery

Use the 1,200 training images and their 6,000 source captions.
Do not use development or test examples in the constraint bank.
For each query, identify the best relevant item under the frozen score.
Call that item its witness.
Protect only queries with a strictly positive frozen retrieval margin.

For protected query `q`, let `p(q)` be its witness.
Let `N(q)` contain every irrelevant item in the training gallery.
Define the frozen margin

`g_q = s_0(q,p(q)) - max_{j in N(q)} s_0(q,j) > 0`.

Require every constraint

`s_A(q,p(q)) - s_A(q,j) >= rho*g_q`, for `j in N(q)`.

Use the same requirement for source-caption queries against all training images.
An initial exploratory protocol can set `rho=0.5` without a coefficient search.
This fraction reserves half the original minimum margin.
It is a design choice, not a claimed optimum.

Each constraint is affine in `A`.
The feasible set is convex.
It contains `A=0` as a strict interior point.
Every feasible matrix preserves every protected query on that exact training gallery.
Thus each training retrieval direction retains all strictly correct frozen queries.
Queries with frozen ties require a separate, explicit convention.
Exclude them from the theorem and report their count.

This guarantee does not cover unseen queries or a larger gallery.
It also does not guarantee improvement on SugarCrepe++.
Development and test evaluation must establish those outcomes empirically.

The witness restriction can reject harmless changes between relevant captions.
This is a cost of obtaining linear constraints.
Do not describe it as the exact feasible region for retrieval retention.

## 4. A useful optimization guarantee

Write one constraint as `delta_k + <F_k,A> >= rho*g_k`.
For images, `F_k` is the scaled outer product `c*x_q*(y_p-y_j)^T`.
For captions, it is `c*(x_p-x_j)*y_q^T`.
At initialization, every slack exceeds or equals `(1-rho)*g_k`.

For any proposed direction `H`, define its constraint slope as `h_k=<F_k,H>`.
At feasible `A`, let `v_k=delta_k+<F_k,A>-rho*g_k`.
The largest feasible nonnegative step is

`eta_max = min_{k:h_k<0} v_k/(-h_k)`.

The minimum over an empty set is infinite.
This expression uses every gallery item, not only sampled negatives.
It gives an exact feasibility check for a line step.
Numerical code must subtract a stated tolerance from the allowed slack.

A nonzero objective gradient at initialization permits a sufficiently small feasible descent step.
This follows from the strict interior and differentiability.
It does not establish a useful update size.
Near-zero margins can restrict the update severely.
Nor does repeated clipping alone guarantee convergence to the constrained optimum.

## 5. Objective and solver

Keep source contrastive supervision.
Add within-image ranking between valid descriptions and contradicted descriptions.
The old `pairwise_rank` method used relation ranking alone.
It did not include source supervision or complete-gallery constraints.

A suitable group loss for image `i` is

`L_group(i) = log(1 + sum_{p in P_i, n in C_i} exp(s_A(i,n)-s_A(i,p)))`.

Here `P_i` contains source and supported descriptions.
`C_i` contains contradicted descriptions.
Average this loss equally over eligible images.
The loss emphasizes the weakest valid-versus-contradicted comparisons.
It does not equate a supported hypothesis with a paraphrase.
It better matches an all-valid-caption decision than independent average pair accuracy.

Combine source loss and group loss with a declared coefficient.
Keep that coefficient identical across preservation controls.
The resulting objective is convex in `A`.
An additional squared Frobenius penalty makes the objective strongly convex.
Neither convexity nor this penalty is a new principle.

Use projected gradient with an active constraint set.
Compute the projection with dual coordinate updates on the active halfspaces.
Store each constraint through its query and candidate indices.
Construct its outer product when required.
After projection, scan the complete gallery for violations.
Add violated constraints and repeat.
Stop only when the full scan and the projection residual meet declared tolerances.
If the computational limit interrupts this process, retain the last verified feasible state.
Report the interruption; do not claim an exact optimum.

The source score matrix has 7.2 million entries.
One float64 matrix needs about 58 MB.
Both retrieval directions reuse that matrix.
The matrix `A` needs about 2 MB for ViT and 8 MB for RN50.
The active constraints do not require dense storage of every outer product.
This makes a CPU implementation plausible with the existing feature bank.
Actual runtime needs a pilot measurement before a schedule commitment.

## 6. Minimum fair controls

Use the same score architecture and objective for these preservation controls:

| Arm | Preservation mechanism |
|---|---|
| Unconstrained | None |
| KL | Existing source-gallery KL penalty |
| Rank constraint | Complete-gallery affine witness constraints |

Include frozen scores, source-only training, and the completed A+D procedure as absolute references.
Include a relation-label control under the selected rank schedule.
Do not claim that changing the architecture proves the constraint mechanism.
Do not compare a full-gallery constraint only against a weaker minibatch KL implementation.
Add full-gallery KL on the same bank to isolate gallery coverage from constraint type.

A certificate-based alternative can constrain each full-gallery KL by its exact positive-set failure radius.
Those constraints are also convex because scores are affine.
They permit the best relevant item to change.
They can constrain unrelated distribution changes more strongly than direct rank constraints.
Treat this alternative as a separate design, not a hidden implementation substitution.

Verify that the selected matrix changes scores, not merely parameters.
A nonzero matrix can act inside a feature nullspace.
Report score changes and changed margins alongside its parameter norm.
Use only development data for selection and continuation decisions.
The existing test sets remain reused evaluation sets.

## 7. Prior art and novelty limits

The following primary sources were checked on 2026-10-04.

| Source | Consequence for this proposal |
|---|---|
| [FSC-CLIP, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1062/) | Composition gains with retained multimodal ability already motivate an established method. |
| [CLIC, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/526bed91b080ef0cf7f23747215e6353-Abstract-Conference.html) | Joint compositional and retrieval improvements already exist. |
| [Structure-preserving image-text embeddings, CVPR 2016](https://arxiv.org/abs/1511.06078) | Cross-modal margin constraints and structural preservation are established. |
| [FINEST](https://arxiv.org/abs/2402.03481) | Rank-preserving adaptation to a reference model is established in recommendation. |
| [Every Sample Counts](https://arxiv.org/abs/2607.08968) | Per-sample constraints during adaptation are established beyond this retrieval setting. |

These sources rule out broad novelty claims about preservation, ranking constraints, or constrained adaptation.
They do not establish priority for every detail of this proposed experiment.
An exhaustive priority assessment would require the completed method and further comparison.
The present defensible contribution is an identified experimental omission and a precise, testable next intervention.

## 8. Recommended order

1. Complete the documented A+D comparison.
2. Audit its matched interaction and independently selected procedures.
3. Keep every outcome, including practical failure.
4. If needed, conduct a development-only pilot of the constrained score model.
5. Continue only if genuine score changes preserve development retrieval and improve the grouped relation criterion.
6. Freeze the complete follow-up protocol before any further test evaluation.

No result in this document closes the practical gap by itself.
