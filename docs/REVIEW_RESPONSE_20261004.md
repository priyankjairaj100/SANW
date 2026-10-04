# Response to the SANW review comments

Date: 2026-10-04.

Status: both new replications and the AD study passed independent verification.
The AD study establishes no practical success.
Original-retention and directional outcomes remain pending.

## Central response

The revised study asks which caption assignments improve discrimination while preserving retrieval.
The completed v3 study separates caption identity from promotion count, initial similarity, training schedule, and model selection.
The completed extension tests broader settings and the missing allocation-plus-distillation comparison.
It also restores the complete directional and retention program under its original frozen protocol.
It also gives an exact distance to retrieval failure on a fixed gallery.

The response distinguishes completed comparisons from successful practical adaptation.
A completed experiment does not establish that its model meets the practical criterion.

The preserved v3 artifacts support the earlier numerical findings below.
The current manuscript uses new strengthening results only after their prediction archives pass independent checks.
Reported results from the unavailable 270-fit execution do not supply current tables or claims.

## Four original comments

### 1. Positive expansion lacked a matched promotion control

**Comment.** Source and Supported change the identities and number of positive captions together.
The fixed-score derivative does not identify the value of supported labels.

**Completed response.** The v3 study adds two randomized controls.
One matches the number of promoted hypotheses for each image.
The other also matches supported counts within two groups of frozen similarity scores.
Each control uses three fixed assignment draws.
Every policy retains the same candidate captions, source positives, batch order, learning rate, and terminal epoch.
The controls also preserve the number of eligible reverse anchors.

Supported labels improve text-to-image R@1 over the similarity-matched control at all three tested rates.
The gains are 1.54, 3.82, and 4.08 percentage points.
All three adjusted intervals exclude zero.
The corresponding image-to-text intervals include zero.
The paper therefore identifies a caption-assignment benefit for the stated endpoint and control.

**Current extension.** Both new replication settings retain all three similarity-matched draws.
The AD study also includes nine randomized controls for each encoder.
These controls copy the primary AD coefficients, learning rate, and seed-specific epochs.
They test caption identity under the selected preservation schedule.

**Verified nonlinear extension.** Supported exceeds stratified promotion by 0.94, 0.51, and 2.69 caption-to-image points across increasing rates.
The smallest and largest rates have positive adjusted lower bounds.
The middle adjusted interval includes zero.
This pattern supports caption-assignment value at two tested nonlinear learning rates.
It does not support a uniform advantage across rates.

**Verified RN50 extension.** Supported exceeds stratified promotion by 2.10, 5.39, and 4.63 caption-to-image points.
The adjusted intervals are [0.87, 3.30], [3.84, 6.92], and [2.93, 6.31] points.
All three adjusted lower bounds exceed zero.
Every corresponding image-to-caption interval includes zero.

**Status.** The original attribution comparison and both new replications are complete.
The AD study also verifies a relation-accuracy advantage over its matched randomized control.
The gains are 4.27 points for each encoder, after rounding.
The respective adjusted intervals are [3.44, 5.17] and [3.41, 5.17].
SugarCrepe++ improves against that control for ViT only: 1.10 [0.28, 1.93] points.
The RN50 effect is 0.11 [-0.65, 0.85] points.
These controlled gains do not establish improvement against frozen initialization.

### 2. Selected checkpoints used different schedules and objectives

**Comment.** Different selected rates and epochs confound interpretation of the source-versus-supported comparison.
A native development criterion also differs from full-pool retrieval evaluation.

**Completed response.** The v3 study compares every declared learning rate at epoch ten with paired seeds.
It reports independently selected procedures separately.
It adds a development criterion based on both source-caption retrieval directions.
The corresponding pool contains 900 images and 4,500 source captions.

Supported training loses 6.00, 7.13, and 10.83 image-to-text points against Source at the matched terminal schedules.
All three adjusted intervals exclude zero.
Thus differing selected schedules do not fully explain this observed loss.

The source-retrieval selector chooses epoch zero for every expanded-positive condition in the v3 grid.
The paper identifies these outputs as frozen selections.
It does not present them as successful trained models.

**Current extension.** Every preservation family uses the same constrained development objective.
The primary tolerance permits one percentage point of loss in each retrieval direction.
The zero-loss tolerance provides a separate descriptive sensitivity analysis.
The original 100-image relation pool is a subset of the 900-image retrieval pool.

The AD decomposition also fixes rates and epochs across U, A, D, and AD.
A uses the AD-selected allocation coefficient.
D uses the AD-selected distillation coefficient.
These matched comparisons identify interactions conditional on the AD-selected schedule.
Comparisons between independently selected procedures answer a separate practical question.

**Verified nonlinear extension.** Supported loses 3.03, 3.43, and 8.37 image-to-caption points against Source at matched terminal schedules.
All three adjusted intervals lie below zero.
The retrieval cost therefore recurs after changing adaptation capacity.

**Verified RN50 extension.** Supported loses 5.00, 6.17, and 7.27 image-to-caption points against Source.
The adjusted intervals are [-8.17, -1.69], [-9.43, -2.80], and [-10.67, -3.73] points.
These comparisons retain the same terminal epoch, learning rate, and paired seeds.
All six new Source comparisons therefore retain negative adjusted intervals.

**Status.** The original schedule comparison and both new replications are complete.
The AD-selected decomposition is complete.
Allocation and distillation each improve both retrieval directions on both encoders at those matched schedules.
All eight adjusted main-effect intervals lie above zero.
No adjusted interaction interval has a positive lower bound.
The original directional comparison remains pending.

### 3. Clean full-pool retrieval within the source domain was absent

**Comment.** COCO transfer alone cannot establish the effect on retrieval within the training domain.
The standard Karpathy Flickr1k split also overlaps the existing fitting data.

**Completed response.** The v3 study uses all 1,000 official e-ViL test images and 5,000 original source captions.
Both retrieval directions search the complete pool.
Each image has five relevant source captions.
Each caption has one relevant source image.
The study verifies image identities and image-content overlap against fitting and development data.

This pool supplies the primary in-domain endpoint.
COCO remains a separate transfer evaluation with 5,000 images and 25,000 captions.
The paper names the e-ViL pool explicitly.
It does not call that pool the Karpathy Flickr1k benchmark.

**Current extension.** The new studies retain these definitions and complete candidate galleries.
The evaluator binds feature archives, manifests, selected checkpoints, and scoring code before evaluation.

**Status.** The original evaluation gap is resolved.
Both replication evaluations and the AD evaluation passed their content and aggregate checks.
The original-retention evaluation remains pending.

### 4. One adaptation setting and fixed weights supported broad advice

**Comment.** One backbone and one adapter cannot support broad claims about relation supervision.
Fixed negative weights cannot establish the performance of whole method families.

**Completed response.** The v3 paper narrows its claims to the implemented assignment comparisons.
It removes broad training advice based on one selected coefficient.
It reports the ranking policy's frozen selection as a selection outcome.
The policy's learned development trajectories remain available.

**Current extension.** The program adds a nonlinear ViT adapter and an independently pretrained RN50 encoder.
Each replication contains 45 fits across five assignments, three rates, and three seeds.
The nonlinear setting changes adaptation capacity.
RN50 changes encoder architecture, pretraining, and feature dimension together.

The paper does not treat these changes as isolated causal effects of architecture.
It tests whether the assignment findings recur in materially different settings.
It also retains the limited scope of untested negative-weight families.

**Status.** The original overstatement was corrected in v3.
The verified replications extend the evidence to another adapter and another pretrained encoder.
Neither comparison establishes the effect of changing architecture alone.
The nonlinear advantage over stratified promotion still varies across rates.

## Remaining requirements from the final review

### Broader evidence

The new replications directly test the final review's recommendation for a materially different adaptation setting.
They preserve matched assignment counts, schedules, and scoring rules.
Each uses its declared twelve-contrast family.
Neither selects a favorable test learning rate.

**Verified result.** The nonlinear independent audit passed all 300 prediction archives and 2,400 numeric aggregates.
It also reproduced twelve effects and all 120,000 saved bootstrap samples.
The largest bootstrap difference was below $1.6\times10^{-16}$.
This audit reconstructs statistics from saved ranks and score-derived correctness.
It does not independently extract features or recompute all retrieval dot products.

The nonlinear results reproduce the Source comparison's retrieval cost at all three rates.
The caption-assignment advantage over stratified promotion holds at two rates.
The middle-rate interval includes zero.
The paper reports that variation explicitly.

The RN50 independent audit also passed 310 prediction archives and 2,480 numeric aggregates.
It reproduced twelve effects and all 120,000 saved bootstrap samples.
Its largest bootstrap difference was below $1.9\times10^{-16}$.
Its scope has the same feature-extraction and dot-product exclusions.

RN50 reproduces the Source comparison's image-to-caption retrieval cost at every tested rate.
It also improves caption-to-image retrieval against stratified promotion at every tested rate.
The image-to-caption intervals against that randomized control include zero.
Together, the replications show a directional caption-assignment benefit and a recurring source-retrieval cost.
They do not establish the separate practical criterion.

### Stronger conceptual contribution

The revised theory derives the exact forward-KL distance to losing a relevant top-K result.
The event includes all relevant candidates and pessimistic ties.
The proof supplies an attaining distribution through probability pooling.
It also gives a sharp score-change radius and an aggregate divergence budget bound.

The result goes beyond the fixed-score target identity by connecting distribution change to a retrieval decision.
Its domain remains the specified query, gallery, relevance set, and temperature.
A minibatch penalty alone does not satisfy that domain requirement for a larger gallery.

General surrogate calibration and distribution preservation have established precedents.
The contribution is the explicit positive-set retrieval calculation and its role in this controlled study.
The paper does not claim a new general principle of distillation or calibration.

The AD decomposition supplies a separate empirical contribution.
It tests whether source allocation adds value beyond the same distillation term.
The interaction is `(AD-D)-(A-U)` at the matched schedule.
It does not establish target dilution as the sole cause of learned retrieval loss.

The restored directional factorial separately changes image targets and eligible reverse anchors.
Its four cells retain the same candidate captions and total directional weighting.
It evaluates both retrieval directions at each declared rate and epoch ten.
Its 36 contrasts keep their original correction family.
The reverse intervention changes both caption participation and the averaging weights.
The resulting effects concern that defined intervention.
They do not isolate target dilution as the only possible cause.

**Status.** The derivations, implementation checks, and AD decomposition are complete.
The AD decomposition establishes positive allocation and distillation main effects on retrieval within its matched cells.
Distillation also improves SugarCrepe++ by 2.17 [1.18, 3.17] and 3.43 [2.26, 4.59] points.
Those effects use the separate 24-effect adjustment and the AD-selected schedule.
They do not compare each selected procedure against frozen initialization.
No positive interaction is established.
The original directional effects and certificate coverage remain pending.

### Practical improvement

The current study tests Source, U, A, D, AD, and WiSE under the same development rule.
It uses linear adapters for both encoders.
Each encoder has 117 base fits and nine matched randomized controls.
The full AD stage therefore contains 252 fits.

The earlier retention stage separately contains 180 executions across both encoders.
It includes seven selected families and matched randomized distillation controls.
Those controls copy D's selected schedule, rather than the later AD schedule.
Its 80 selected-procedure contrasts remain separate from the AD inference family.

The full restored program includes 90 replication executions, 180 retention executions, and 252 AD executions.
Some baseline configurations recur across these stages.
Repeated deterministic executions do not provide independent training replicates.
Each analysis uses only its declared seed and assignment sets.

The practical criterion requires all three selected seeds to contain nonzero trained updates.
Both adjusted retrieval bounds must exceed minus one percentage point relative to frozen initialization.
The adjusted SugarCrepe++ both-caption gain must exceed zero.
The same family must satisfy all conditions for both encoders.

The primary inference family contains 80 contrasts across both encoders.
The matched decomposition has a separate family of 24 contrasts.
Both use 100,000 paired image-cluster bootstrap samples.
The intervals condition on the fitted seeds, assignment draws, and fixed galleries.

**Verified result.** None of the six selected AD-study families passes the joint criterion on either encoder.
Every family fails the adjusted SugarCrepe++ improvement requirement.
Supported selects epoch zero for all seeds on both encoders.
RN50 allocation also includes one epoch-zero seed.
AD trains all three seeds on both encoders, but fails both adjusted retrieval requirements and SugarCrepe++ improvement.
Only ViT AD caption-to-image retrieval has an adjusted interval wholly below zero among these retrieval comparisons.
Its loss is 1.47 [-2.91, -0.02] points.
The other failed retrieval requirements reflect intervals that extend across the one-point tolerance.
Those failures do not establish large retrieval losses.

| Encoder | Family | Three trained seeds | I2T retention | T2I retention | SugarCrepe++ gain | Joint criterion |
|---|---|---|---|---|---|---|
| ViT | Source | Yes | Fail | Fail | Fail | Fail |
| ViT | Supported | No | Pass | Pass | Fail | Fail |
| ViT | Allocation | Yes | Fail | Fail | Fail | Fail |
| ViT | Distillation | Yes | Fail | Pass | Fail | Fail |
| ViT | AD | Yes | Fail | Fail | Fail | Fail |
| ViT | WiSE | Yes | Fail | Fail | Fail | Fail |
| RN50 | Source | Yes | Fail | Pass | Fail | Fail |
| RN50 | Supported | No | Pass | Pass | Fail | Fail |
| RN50 | Allocation | No | Fail | Pass | Fail | Fail |
| RN50 | Distillation | Yes | Fail | Pass | Fail | Fail |
| RN50 | AD | Yes | Fail | Fail | Fail | Fail |
| RN50 | WiSE | Yes | Fail | Pass | Fail | Fail |

AD improves relation accuracy against frozen initialization by 6.67 [5.21, 8.17] ViT points.
Its RN50 relation gain is 9.02 [7.15, 10.96] points.
Its SugarCrepe++ changes are -0.13 [-1.46, 1.20] and +0.62 [-0.90, 2.10] points.
Both SugarCrepe++ intervals include zero.
Relation learning therefore cannot supply the missing practical result.

The independent audit checked 99 states, 495 prediction archives, 104 effects, and 10.4 million bootstrap values.
Its largest bootstrap difference was below $1.3\times10^{-16}$.
The intervals remain conditional on the fixed training seeds, assignment draws, and galleries.
Failure to establish noninferiority does not prove that the true retrieval loss exceeds one point.
Practical improvement remains unsolved.
The original retention study still requires completion.

## The omitted A+D comparison

The earlier related-work plan specified U, A, D, and A+D.
The frozen retention protocols included U, A, and D but omitted A+D.
No identified revision withdrew that recommendation.

The new protocol completes this comparison without changing the earlier protocols.
Its objective is

`lambda * L_source + (1-lambda) * L_supported + beta * T^2 * D_source`.

The allocation coefficient mixes complete symmetric objectives.
The KL term uses source-caption columns in both directions, with temperature two.
The four matched cells share the selected schedule and corresponding coefficients.
Separate failures of A and D cannot determine the combined method's result.

The new stage is exploratory because previous outcomes informed its design.
It keeps all declared cells and does not revise the practical tolerance after test inspection.

**Status.** The protocol and implementation passed independent checks.
All 252 fits and the complete evaluation passed independent checks.
The combined method does not establish the declared practical improvement.
The matched decomposition finds no positive interaction.

## Manuscript review completed

The independent review checked the main objective, comparison structure, theorem scope, and appendix proofs.
It identified four concrete corrections before results entered the draft.

1. The numerical projection example used probabilities inconsistent with its reported radii.
2. Two descriptions incorrectly assigned ten epochs to the shortened matched controls.
3. One phrase conflated source captions with hypotheses.
4. One sentence incorrectly excluded retrieval intervals that cross zero but satisfy the negative tolerance.

The manuscript owner corrected all four issues.
The independent reviewer reran all thirteen rank-retention test groups.
All passed, including numerical projection checks and the corrected multiple-positive example.
The main structure now separates assignment evidence, preservation evidence, and mathematical scope.
The final page balance and empirical claims require another review after results are inserted.

## Evidence and completion record

| Item | Evidence | Status |
|---|---|---|
| Original comments and final review | [Review task ledger](REVIEW_TASK_LEDGER_20261004.md) | Recovered and mapped |
| Completed v3 findings | [Follow-up story assessment](REVIEW_FOLLOWUP_STORY_ASSESSMENT.md) | Preserved evidence |
| v3 primary effects | [Primary contrasts](../results/review_followup/analysis/primary_contrasts.json) | Audited historical study |
| Exact rank geometry | [Theory note](STRENGTHEN_RANK_THEORY.md) | Proof and numerical checks available |
| New AD protocol | [Frozen protocol](../results/allocation_distillation/protocol_v1.json) | Frozen before affected fits |
| New AD implementation | [Independent design audit](ALLOCATION_DISTILLATION_DESIGN_AUDIT_20261004.md) | Training and evaluation checks passed |
| Nonlinear replication findings | [Independent audit](../recovery/current_turn_audit/nonlinear_replication_independent_audit.json) | Passed: twelve effects and 120,000 bootstrap samples |
| RN50 replication findings | [Independent audit](../recovery/current_turn_audit/rn50_replication_independent_audit.json) | Passed: twelve effects and 120,000 bootstrap samples |
| Restored directional and retention findings | Original frozen protocols and new raw predictions | Pending |
| New AD findings | [Independent audit](../recovery/current_turn_audit/allocation_distillation_independent_audit.json) | Passed: 104 effects and 10.4 million bootstrap values |
| New practical findings | [Declared success gates](../results/allocation_distillation/analysis/practical_success_gates.json) | No selected family passes on either encoder |
| Four-page manuscript | [Main source](../manuscript_strengthened_v5/main.tex) | Structure complete; results pending |

Protocol SHA256:
`f5ba7e389660bf9930b2713a7173e7a6cd7d55bf755111b259c512bc74c5a4cd`.

## Final update checklist

- Insert only independently verified new results.
- Update each pending status from the corresponding raw evidence.
- Distinguish completed evaluation from a passed practical criterion.
- Preserve all failed methods and frozen selections.
- Check every abstract and conclusion claim against the recorded comparison.
- Render four readable main pages and retain the complete appendices.
- Record the final paper, source archive, and recovery checkpoint.
