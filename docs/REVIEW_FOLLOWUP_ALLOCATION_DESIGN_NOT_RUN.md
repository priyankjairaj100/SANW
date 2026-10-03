# Proposed allocation factorial: design only, not run

**Status: unexecuted design. No implementation, fitting, checkpoint selection, or evaluation has been performed for this proposal.** This document is outside the frozen phase-one review follow-up. It does not amend `docs/REVIEW_FOLLOWUP_PROTOCOL.json` (SHA256 `3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34`) or its execution ledger. The allocation values and contrasts below are proposals, not an already frozen second-phase protocol.

**Recommendation: complete phase-one fitting, replay audits, scoring, analysis, and delivery before deciding whether to execute a second phase.** If this experiment remains necessary, freeze a separate protocol before implementing or fitting it. Record that its motivation and design follow the phase-one evidence. Reusing phase-one test sets would not supply a new independent test sample.

## Question and intervention

Does changing the division of training weight between original source captions and supported hypotheses change source retrieval and relation performance when the caption identities, candidate pools, and training schedules remain fixed?

Use the true source and supported groups from the original manifest. Do not use the randomized phase-one promotion assignments. Keep every original source caption and hypothesis in the image-to-text candidate pool, including contradicted and neutral hypotheses. Keep every batch image in each text-to-image candidate pool. All denominator weights remain one. Encoder features, adapter, frozen logit scale, optimizer, batch order, learning rates, training seeds, and epoch counts remain those of phase one.

There are two distinct allocations to intervene on:

1. How image-to-text target probability is divided between source and supported captions within each image row.
2. How the text-to-image average is divided between source-caption anchors and supported-hypothesis anchors.

Changing the first allocation while leaving the second implicit would not isolate these two contributions.

## Exact objective

Consider a paired training batch with image set \(I\), \(B=|I|\), and its unchanged text candidate set \(T\). Let \(z_{ij}\) be the scaled score of image \(i\) and text \(j\). Define the two ordinary, unit-weight softmax distributions

\[
p^I_{ij}=\frac{e^{z_{ij}}}{\sum_{k\in T}e^{z_{ik}}},
\qquad
p^T_{ij}=\frac{e^{z_{ij}}}{\sum_{h\in I}e^{z_{hj}}}.
\]

For image \(i\), let \(S_i\) and \(H_i\) be its disjoint sets of source captions and supported hypotheses, with counts \(K_i=|S_i|>0\) and \(M_i=|H_i|\). The existing training data have five source captions per image. Caption IDs have one owning image, which must be asserted again before fitting. Thus the source and supported text-anchor sets

\[
J_S=\bigcup_i S_i,\qquad J_H=\bigcup_i H_i
\]

are disjoint, and each eligible text anchor \(j\) has one positive image \(o(j)\). Write \(n_S=|J_S|>0\) and \(n_H=|J_H|\). These count eligible **text anchors**, not positive edges in a general many-owner dataset.

For \(M_i>0\), image target probabilities at source-group allocation \(\rho_i\in[0,1]\) are

\[
y_{ij}(\rho_i)=
\begin{cases}
\rho_i/K_i,&j\in S_i,\\
(1-\rho_i)/M_i,&j\in H_i,\\
0,&\text{otherwise}.
\end{cases}
\]

When \(M_i=0\), define \(y_{ij}=\mathbf1\{j\in S_i\}/K_i\), independently of the requested allocation. Keep this image in the average over all \(B\) images. Multiplying its source loss by \(\rho_i\), or omitting the image, would change the intended normalization.

The image-direction loss is

\[
L_I(\rho)=\frac1B\sum_{i\in I}\left[-\sum_{j\in T}y_{ij}(\rho_i)\log p^I_{ij}\right].
\]

For each eligible text anchor, let \(\ell^T_j=-\log p^T_{o(j),j}\). When \(n_H>0\), the reverse-direction loss at source-anchor allocation \(\eta\in[0,1]\) is

\[
L_T(\eta)=
\frac{\eta}{n_S}\sum_{j\in J_S}\ell^T_j
+\frac{1-\eta}{n_H}\sum_{j\in J_H}\ell^T_j.
\]

When \(n_H=0\), use \(L_T=n_S^{-1}\sum_{j\in J_S}\ell^T_j\), without an \(\eta\) multiplier. Contradicted and neutral text columns remain image-direction candidates but do not become reverse anchors.

The full loss is

\[
L(\rho,\eta)=\tfrac12L_I(\rho)+\tfrac12L_T(\eta).
\]

Every image target sums to one. Reverse-anchor coefficients sum to one whenever the direction is defined. Each direction retains weight one half. No additional loss multiplier, denominator change, temperature change, or caption removal is part of this proposal.

## Baseline recovery and batch-specific reverse normalization

The existing uniform multipositive objective is recovered by

\[
\rho_i^U=\frac{K_i}{K_i+M_i},
\qquad
\eta^U=\frac{n_S}{n_S+n_H}.
\]

For each image with support, both types of positive then receive target \(1/(K_i+M_i)\). In the reverse direction, both groups receive the same coefficient per eligible anchor, \(1/(n_S+n_H)\).

**The baseline reverse allocation is batch-specific.** Compute \(n_S\) and \(n_H\) from the current paired batch. A global training-set ratio, a mean of image-level ratios, or a separate average of per-image reverse losses does not reproduce the existing objective. Even in an intervention cell with fixed \(\eta=.8\), the divisors \(n_S\) and \(n_H\) are the current batch's anchor counts.

Setting **both** \(\rho_i=1\) and \(\eta=1\) exactly recovers the existing source-only `clip` objective: image targets include only source captions, and only source-caption reverse anchors contribute. All hypothesis candidates remain in the image denominators. This equivalence concerns the project's full-pool source-only objective, not training on a pool from which hypotheses have been removed. Setting only \(\rho_i=1\) does not recover it because supported reverse anchors still contribute unless their reverse coefficient is also zero.

At the proposed interior allocation .8, all true source and supported positive identities retain nonzero weight. The source-only boundary is an identity check, not an additional proposed training arm.

## Proposed four cells

| Cell | Image allocation | Reverse allocation |
|---|---|---|
| U: existing uniform baseline | \(\rho_i^U=K_i/(K_i+M_i)\) | \(\eta^U=n_S/(n_S+n_H)\), computed per batch |
| I: image allocation only | \(\rho_i=.8\), with the empty-support fallback | Existing batch-specific \(\eta^U\) |
| R: reverse allocation only | Existing \(\rho_i^U\) | \(\eta=.8\), with the empty-support fallback |
| B: both allocations | \(\rho_i=.8\), with the fallback | \(\eta=.8\), with the fallback |

The interpretation is **fixed-budget allocation**, not a source-budget increase for every image. Training-label counts give:

- 1,200 images, 6,000 source captions, and 5,319 supported hypotheses.
- Six images have no supported hypothesis and retain source mass one.
- Eighteen images have exactly one supported hypothesis. Their original source mass is \(5/6\), so .8 slightly decreases it.
- Across all training images, mean image source mass changes from approximately .55467 to .801.
- The pooled training-set source-anchor ratio is approximately .53008. This is descriptive; it must not replace the batch-specific baseline ratio.

Likewise, a fixed reverse allocation .8 increases source-anchor weight only for batches whose original ratio is below .8. These observations use training labels alone, not held-out outcomes.

If the scientific question instead requires a source-budget increase in every eligible row and batch, an alternative parameterization is

\[
\rho_i(\lambda)=\lambda+(1-\lambda)\rho_i^U,
\qquad
\eta(\lambda)=\lambda+(1-\lambda)\eta^U.
\]

It recovers U at \(\lambda=0\), reaches the source-only boundary at \(\lambda=1\), and is monotone in between. A fixed intermediate value such as \(\lambda=.5\) would define a different intervention. Choose between this alternative and fixed .8 before a future execution; do not run alternatives and select one by its outcomes. The current recommendation is the simpler fixed-.8 factorial, explicitly named as such.

## Minimum execution and exact contrasts

If authorized after phase one is complete, reuse its nine uniform-supported fits as U. Add I, R, and B at all three existing learning rates \(10^{-4},3\times10^{-4},10^{-3}\) and all three training seeds 17, 29, 43: **27 new fits**. Retain epochs 0 through 10 and use epoch 10 for the primary matched-schedule comparisons. Report each learning rate; do not choose a winning rate from test performance. Existing source-only fits provide a contextual reference.

A nine-fit version using only the predetermined middle learning rate is a smaller option, but supports a conclusion only at that schedule. The three-rate design better distinguishes a persistent allocation effect from one confined to a particular training rate. No second allocation value is recommended initially.

For each learning rate, seed, and endpoint, let \(Y_U,Y_I,Y_R,Y_B\) denote the four outcomes, oriented so higher is better. Exact candidate contrasts are:

| Contrast | Expression | Interpretation |
|---|---|---|
| Image allocation at baseline reverse weighting | \(C_I=Y_I-Y_U\) | Change only image targets |
| Reverse allocation at baseline image targets | \(C_R=Y_R-Y_U\) | Change only reverse-anchor averaging |
| Joint intervention | \(C_B=Y_B-Y_U\) | Change both allocations |
| Interaction | \(C_X=Y_B-Y_I-Y_R+Y_U\) | Difference between the joint effect and the sum of the two single-direction effects |

These contrasts obey \(C_B=C_I+C_R+C_X\). Report all four cells and all four contrasts. A possible inferential family would use \(C_I,C_R,C_X\) at three learning rates on the two source-retrieval R@1 directions, giving 18 effects; the derived joint contrast would be descriptive unless separately included in the frozen family. This is a proposed analysis choice, not an extension of phase one's existing error-control family. Fix any inferential family, confidence procedure, and endpoint hierarchy in a new protocol before fitting.

Preserve pairing by training seed and evaluation image. Show all seed-level effects. If using image-cluster bootstrap intervals, average the paired seed differences within each query and cluster all five source-caption queries with their image for text-to-image retrieval. Such intervals condition on the fitted seeds. Relation and composition results should accompany source retrieval so that a retrieval gain is not presented without its corresponding change in supported supervision.

## Fixed-logit identities and implementation checks

For \(M_i>0\), changing image source mass from \(\rho_i\) to \(\rho_i+\Delta\rho_i\) changes full-loss score derivatives by

\[
\Delta\frac{\partial L}{\partial z_{ij}}=
\begin{cases}
-\Delta\rho_i/(2BK_i),&j\in S_i,\\
+\Delta\rho_i/(2BM_i),&j\in H_i,\\
0,&\text{otherwise},
\end{cases}
\]

when the reverse allocation and logits are fixed. The softmax denominators are unchanged. No image-target change occurs for \(M_i=0\).

For the reverse direction, define anchor coefficients \(a_j(\eta)=\eta/n_S\) on \(J_S\), \((1-\eta)/n_H\) on \(J_H\), and zero elsewhere, with the stated empty-group fallback. A reverse allocation change gives

\[
\Delta\frac{\partial L}{\partial z_{ij}}
=\tfrac12\Delta a_j\left(p^T_{ij}-\mathbf1\{i=o(j)\}\right).
\]

It reweights whole caption-column gradients. It does not change a caption anchor's positive-image identity or its within-column softmax.

At common logits, the two directional changes are additive. Therefore the four objective values and their full score gradients satisfy

\[
L_B-L_I-L_R+L_U=0,
\qquad
\nabla_z L_B-\nabla_z L_I-\nabla_z L_R+\nabla_z L_U=0.
\]

This **zero interaction at fixed logits** is an exact algebraic identity. A nonzero interaction between trained-model outcomes would arise through their training trajectories and endpoint evaluation, not an explicit interaction term in the loss.

Before any fitting, require meaningful numerical checks of:

1. Baseline recovery against the frozen multipositive loss, in both value and full score gradient, with heterogeneous support counts and rectangular batches.
2. Joint allocation one against the frozen source-only loss, in value and gradient, with every hypothesis retained in the candidate pool.
3. Empty-support rows and batches, normalized target sums, normalized reverse coefficients, and fixed directional weights.
4. Image-gradient and reverse-gradient identities above, checked independently with finite differences.
5. Zero factorial interaction at common logits and in full gradients.
6. Exact candidate IDs, true positive-group identities, unit denominator weights, and one-owner text IDs across all cells.
7. Identical seeded batches, optimizer settings, and schedules; unchanged evaluation labels; complete input/code/checkpoint hash binding.

Baseline U should use the original implementation and existing verified states. Algebraic equivalence tests for a new mixed-target implementation should use suitable floating-point tolerances rather than assume different summation orders are byte-identical.

## What this would establish

The factorial intervenes directly on source-versus-support training allocation while holding their identities and the total objective coefficient fixed. It can distinguish the effects of image-target allocation and reverse-anchor averaging within this model, loss family, data, and schedule. This is a more direct mechanism test than comparing source-only targets with a larger, differently labeled positive set.

It does not hold parameter-gradient norms fixed. Reallocation can change update magnitudes and directions even though the loss coefficients remain normalized. Increasing source target mass necessarily decreases supported target mass; this experiment cannot treat those as two independent interventions. The reverse intervention likewise exchanges weight between the two anchor groups.

Consequently, a favorable result would support the practical value of a specified allocation intervention. It would **not prove that every earlier retrieval loss was caused by target dilution**, establish a general optimization guarantee, independently adjudicate semantic labels, or demonstrate that all other factors have been excluded. The scientific contribution would be the controlled directional decomposition and its observed tradeoffs. The weighted cross-entropy formula alone is not the proposed basis for a novelty claim.

