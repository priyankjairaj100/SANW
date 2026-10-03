# Strengthening the positive-caption study: related work and fair competitors

Assessed 2026-10-03. This document is a design and positioning assessment, not a report of new experiments. It was prepared after reading the v3 main paper, its completed-result assessment, and the unexecuted allocation design. Primary sources below were opened or retrieved from the official proceedings or authors' arXiv records on this date.

## Recommendation

The strongest extension is a controlled test of **which intervention preserves source retrieval while retaining the benefit of supported supervision**. Separate source-target allocation from generic retention regularization. Replicate the useful intervention on a different pretrained representation and adaptation capacity. The contribution should be the empirical decomposition and a demonstrably useful operating point, not a claim that weighted cross entropy or frozen-teacher distillation is new.

A clean sequence is:

1. Compare uniform Supported with the predeclared image/reverse allocation factorial. This directly tests the allocation explanation at fixed captions and fixed total directional weight.
2. Compare the joint allocation intervention with ordinary frozen-teacher distillation and with weight interpolation. If practical retention requires distillation, include uniform Supported plus the same distillation term. This determines whether allocation contributes beyond a standard preservation tool.
3. Transfer a fixed design to a second backbone or pretraining family. A nonlinear residual adapter on the same cached features tests capacity dependence, but does not alone establish backbone generality.

A positive result here would replace a diagnostic-only contribution with a tested design rule: how to exploit supported descriptions without spending the retrieval performance the user needs. The result must come from nonzero learned states selected on development data. Returning the frozen state or reporting a nonsignificant retrieval difference is not, by itself, evidence that adaptation retained retrieval.

## Closest primary work and the novelty boundary

| Prior work | Verified relevant content | Consequence for this project |
|---|---|---|
| **MAFA**, CVPR 2024. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Byun_MAFA_Managing_False_Negatives_for_Vision-Language_Pre-training_CVPR_2024_paper.html); [author full text](https://arxiv.org/html/2312.06112v2). | Sections 1 and 4 convert mined false negatives to positives, compare conversion with removal, and use label smoothing in the contrastive objective. Section 4.2 also discusses teacher-generated soft targets. | Positive conversion, removal comparisons, and soft targets are established. The present opportunity is isolating how the image target budget and reverse anchor average change when known positives are added, then testing that intervention directly. |
| **FSC-CLIP**, EMNLP 2024. [Official proceedings](https://aclanthology.org/2024.emnlp-main.1062/); [author full text](https://arxiv.org/html/2410.05210v1). | Sections 3.2–3.4 combine a patch/token local hard-negative loss with selective calibrated regularization. The latter combines focal weighting and label smoothing. Its objective and evaluations explicitly target composition gains with broader capability retention. | Neither the retention problem nor regularizing difficult caption supervision is new. Full FSC-CLIP requires token and patch features, which are absent from the current pooled-feature cache. A global-only focal/smoothing variant must be named as an ablation or adaptation, not reproduced FSC-CLIP. |
| **CLIC**, NeurIPS 2025. [Author full text](https://arxiv.org/html/2505.24424v2); [official implementation](https://github.com/AmitPeleg/CLIC). | Sections 3.1–3.2 construct concatenated images and captions, use multiple positives, a separate hard-negative objective, and text invariance. Training alternates composed inputs with ordinary image-caption CLIP steps. The paper evaluates several architectures and pretraining variants. | Joint improvements on valid-caption tests and retrieval already exist. Do not present a tradeoff as inherent. Full CLIC needs new composed-image and text encodings; applying its loss to existing annotations is not its full method. Its invariance assumption is inappropriate for a generic supported hypothesis and source caption that need not mean the same thing. |
| **CLIP-Refine**, CVPR 2025. [Author record and venue status](https://arxiv.org/abs/2504.12717); [full text](https://arxiv.org/html/2504.12717v1). | Section 3.4, equations 4–8, gives a bidirectional KL loss with targets mixing the frozen pretrained model's distribution and ground-truth pairs. The full method adds random feature alignment. | Frozen-teacher distribution matching plus positive supervision is established, including its capability-retention motivation. A source-restricted or rectangular multi-positive version is a setting-specific adaptation. Do not rename it as a new distillation principle or claim reproduction of full CLIP-Refine without random feature alignment. |
| **CLIP-KD**, CVPR 2024. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html); [author record](https://arxiv.org/abs/2307.12732). | Studies relation, feature, gradient, and contrastive distillation; simple feature mean-square matching is effective. The original problem is teacher-to-smaller-student transfer. | Feature matching and relational KL are standard baseline ingredients. An equal-size frozen teacher used to regularize adaptation changes their role, not their underlying novelty. |
| **WiSE-FT**, CVPR 2022. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html); [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf). | Ensembles zero-shot and finetuned model weights to balance adaptation with robustness. | Weight interpolation is a necessary cheap retention comparator here. For a zero-initialized linear residual adapter, it is implemented exactly by multiplying its learned correction matrices by the interpolation coefficient. |

The retrieved sources establish close prior art; this is not an exhaustive priority search. None of them alone establishes that this project's two-direction allocation factorial has already been executed. The factorial's value still has to be demonstrated by its results, not asserted from the loss formula.

## Two strongest competitors available from pooled features

### 1. WiSE-FT interpolation of the ordinary Supported adapter

Let the frozen embedding be `x` and the learned image correction matrix be `W_I`; the deployed image feature is `normalize(x + W_I x)`. Interpolating the model parameters with the initial model gives `normalize(x + a W_I x)`, and likewise for text. The same coefficient `a` should be used for both modalities in the primary baseline. This is weight interpolation, not an average of already normalized features or scores.

- Use the existing Supported states. No training and no image re-encoding are required.
- Freeze a small interpolation grid and a common development objective before looking at test outcomes. A practical grid is `a = 0, .1, .25, .5, .75, 1`.
- Permit the same epoch and learning-rate choices for this baseline as for the proposed method. Report the search budget; a denser baseline grid is inexpensive and preferable to an artificially weak comparator.
- `a=0` is the frozen model and must remain visible. If the selected coefficient is zero, the baseline has rejected the learned update.
- Add the analogous source-adapter interpolation if making claims against the best retained source-only adaptation, rather than only against the frozen encoder.

This baseline answers whether a proposed intervention offers more than moving a conventional learned model closer to initialization.

### 2. Uniform Supported plus frozen-teacher distribution matching

Use the exact Supported supervised objective with a bidirectional teacher-to-student KL penalty. The teacher is the original frozen encoder. Cached features make its similarity matrix inexpensive to compute. The same candidate and anchor domains must be explicit for teacher and student.

For a specified image/text submatrix, let `q_I, q_T` be teacher row and column softmax distributions and `p_I, p_T` the corresponding student distributions at the same declared temperature. Then

`D = 0.5 mean_image KL(q_I || p_I) + 0.5 mean_text KL(q_T || p_T)`.

The baseline is `L_supported + beta D`, or its declared normalized version. Preserve the same convention across the allocation and nonallocation arms. Select `beta` on development data only, and state whether the usual temperature-square multiplier is used.

There are two legitimate domains, answering different questions:

- **Source-domain KL:** both directions use only the original source-caption columns. This preserves the teacher's source-retrieval geometry while allowing supported hypotheses to move. It is especially suitable when the proposed method uses the same source preservation term.
- **Full-candidate KL:** the image direction uses all unchanged caption candidates, including hypotheses; the reverse direction uses an explicitly defined anchor set. This also regularizes the teacher's relations to hypotheses and should not be silently mixed with the source-domain design.

For the main comparison, use **the identical source-domain KL term in uniform Supported and source-weighted Supported**. Then any increment from allocation is identifiable relative to the ordinary retention ingredient. Call the comparator “Supported + frozen-teacher KL” and cite CLIP-Refine/CLIP-KD. It is not the complete CLIP-Refine system. A feature-MSE term is a reasonable alternative but should not replace the KL comparator merely after results favor MSE.

## Minimal decomposition that would support a useful new claim

| Cell | Source allocation | Frozen-teacher KL | Question |
|---|---|---|---|
| U | Original uniform multi-positive weights | Off | Existing expansion reference |
| A | Declared source-group allocation in both directions | Off | Does allocation itself help? |
| D | Original uniform weights | On | How much does standard preservation accomplish? |
| A+D | Same allocation as A | Identical term and coefficient as D | Does allocation add value beyond distillation? |

The directional image/reverse factorial can sit inside A's mechanism analysis. WiSE-FT is an additional post-training competitor. Frozen and ordinary Source remain absolute references. Keep any source-only candidate-pool baseline distinct from the existing Source policy, which retains supported hypotheses as image-direction competitors.

The four-cell analysis should report caption discrimination and both retrieval directions for every cell. Retrieval recovery bought by erasing the supported-supervision benefit is not a joint improvement. Conversely, a small retrieval cost can be a useful operating point if it is measured against a predeclared practical tolerance and earns a meaningful gain. A confidence interval spanning zero does not establish that tolerance.

## What to change in the eventual central claim

If allocation improves the composition/retrieval frontier against D and WiSE-FT across independent representations, the strong story is: **adding valid descriptions changes two training budgets; separating those budgets makes the added supervision useful for retrieval.** The main evidence is the controlled intervention, its replication, and the deployable frontier. The algebra explains the intervention.

If only D or WiSE-FT helps, the practical weakness is partly solved but the allocation mechanism is not established. The paper would instead contribute a controlled finding about positive supervision under preservation. Do not describe a standard regularizer's success as a new loss breakthrough.

If no nonzero learned state meets the practical target, these runs cannot be reframed as a successful solution. A materially different data/model regime is then the next scientific step. More runs of the current adapter alone do not resolve generality.

## Mapping to the current strengthening implementation

The implementation inspected during this assessment is `src/gcr/retention_losses.py`. Its `allocated_positive_loss` is exactly `lambda * L_source + (1-lambda) * L_supported`, with unchanged hypothesis candidates. Thus `source_mix=.5/.8` denotes a convex objective mixture, not fixed source-group probability .5/.8 for each image. For an image with K source captions and M supports, its source probability is `lambda + (1-lambda) K/(K+M)`. The reverse source-anchor mass has the analogous current-batch expression. This monotone parameterization avoids decreasing source mass for low-support images.

`source_retrieval_kl` takes only source-caption columns, divides already scaled scores by its temperature, and averages the two teacher-to-student KL directions. `retention_distilled_loss` explicitly adds `beta * temperature^2 * KL`. The proposed coefficients 1, 4, and 16 at temperature 2 are therefore effective KL multipliers 4, 16, and 64. This objective is inspired by standard distillation, but it is not algebraically identical to HyCD target blending on a single shared candidate domain: its supervised image loss contains hypotheses, while its distillation image loss does not.

The planned source/expanded **directional boundary factorial** is also useful. Its four cells are Source/Source, Expanded/Source, Source/Expanded, and Expanded/Expanded for image/reverse directions. This answers which direction's addition of supported supervision changes outcomes. An intermediate-allocation factorial answers a narrower weight-rebalancing question. They should not be described as the same intervention. The boundary factorial plus the two mixture doses provides a concise causal intervention within the fixed training setup.

If a KD arm supplies the useful operating point, repeat score-stratified promotion with the same coefficient and development rule. That distinguishes the retained contribution of caption identity from the benefit of regularization. For independent representation replication, keep two targets distinct: Source/Supported/score-stratified tests assignment generality; replicating the useful preservation arm tests practical generality.

## Role of the rank-retention certificate

The inspected implementation uses the exact two-coordinate KL threshold for reversing a correct and incorrect candidate's order. With multiple valid captions it supplies a conservative sufficient condition. This is a useful diagnostic on the actual gallery, but its proof is standard convex KL projection/calibration reasoning. General quantitative relations between surrogate loss and classification error have a long history, including [Bartlett, Jordan, and McAuliffe, *Convexity, Classification, and Risk Bounds*](https://statistics.berkeley.edu/tech-reports/638) and [Ávila Pires and Szepesvári, *Multiclass Classification Calibration Functions*](https://arxiv.org/abs/1609.06385).

Its value for this paper is empirical: does full-gallery KL certify a useful fraction of retained source matches, and do the interventions improve this coverage while preserving caption-discrimination gains? Compare the certificate with actual retention and report both the certified fraction and the fraction of correct retained queries it certifies. Use one declared diagnostic temperature across methods. A minibatch KL penalty does not provide a deterministic certificate for a larger held-out gallery. Keep this proposition subordinate to the directional intervention and replication unless its diagnostic results are unusually compelling.

### Independent verification of the exact positive-set extension

The mechanism reviewer subsequently proposed replacing the two-coordinate lower bound by the exact distance to failure for multiple relevant captions. I independently verified the derivation. Let the teacher's relevant probabilities be `a_1 >= ... >= a_r` and let `b` be its largest irrelevant probability. Assume `a_1 > b`. Starting with `b`, pool the largest relevant probabilities until their mean `t = (b + sum_{i<=k} a_i)/(k+1)` is at least the next relevant probability. The exact forward-KL distance to the closed failure event is

`C = sum_{i<=k} a_i log(a_i/t) + b log(b/t)`.

For a fixed wrong candidate, the constrained minimizer sets that candidate and all active relevant probabilities to `t`; it leaves the other probabilities unchanged. The simplex Lagrange multiplier is one because the pooled probabilities preserve their total mass. On an interval with fixed active set, the derivative of the cost with respect to `b` is `log(b/t) <= 0`. The cost is continuous when a new relevant coordinate joins the pool. Therefore the teacher's largest wrong probability minimizes the cost over all wrong candidates, proving that it suffices to consider that candidate. Relevant coordinates exactly equal to `t` may join or leave the active set without changing the result. Teacher-incorrect or tied cases have zero distance to failure.

This exact set-level diagnostic is preferable to the pairwise bound for the five-caption source task. It still uses ordinary convex KL projection machinery; the research value would be its measured coverage and usefulness in explaining preservation, not a claim that the underlying projection principle is new.
