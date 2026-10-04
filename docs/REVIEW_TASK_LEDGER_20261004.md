# Review task ledger, 4 October 2026

## Purpose and source boundary

This ledger separates completed review work from unresolved scientific requirements.
It also records the current recovery status.

The [shared SANW conversation](https://chatgpt.com/share/6ac1c8d5-621c-83ee-b9e5-ff7c5108dbc4) contains 769 messages.
Its last message is the ARR assessment, message 768.
It does not contain the later instruction to resolve those remaining weaknesses.

Transcript references below use `/tmp/shared_research_transcript.txt`.
Message numbers use `/tmp/shared_research_messages.json`.
These references identify the extraction examined during this recovery.

The restored repository base is commit `72cba41209e622fb2a5fdf0e3a3570c10501a48d`.
The source branch is `strengthen-resume-20261004`.
[Open the restored commit](https://github.com/priyankjairaj100/SANW/tree/72cba41209e622fb2a5fdf0e3a3570c10501a48d).

## Review chronology

| Stage | Source | Decision or status |
|---|---|---|
| Choose the research direction | Messages 39–43; transcript lines 144–199 | The user selected grounded decisions about supported, contradicted, and uncertain captions. |
| Build the project | Messages 44–48; lines 201–208 | The user approved the direction and required implementation from scratch. |
| Initial empirical program | Lines 255–303 | Public relation labels supported controlled experiments. Results showed discrimination gains and retrieval costs. |
| Recover missing work | Message 288; lines 364–366 | The user required rerunning missing work and sending periodic recovery ZIPs. |
| Rewrite the paper | Message 411; line 502 | The user required a complete inventory, stronger central story, four main pages, and complete appendices. |
| Preserve the project remotely | Messages 463 and 480; lines 567–591 | The user requested GitHub upload and supplied the SANW repository. |
| Address four objections | Message 566; lines 684–689 | The objections concerned promotion controls, schedules, retrieval coverage, and limited scope. |
| Execute the follow-up | Messages 594–598; lines 740–757 | The user approved promotion controls, matched schedules, and clean full-pool retrieval. |
| Complete the follow-up | Messages 657–749; lines 789–860 | All 72 fits completed. The results, analysis, and four-page revision passed independent checks. |
| Assess the revised paper | Message 768; lines 869–888 | Remaining weaknesses concerned breadth, conceptual advance, and practical improvement. |
| Strengthen the science | Restored strengthening documents | The accepted program adds replication, directional interventions, retention comparisons, and rank-retention theory. |

The 72-fit follow-up was complete before the shared snapshot ended.
Its completion does not establish that the final review requirements were satisfied.

## Four original review objections

| Objection | Exact location | Approved response | Current evidence status |
|---|---|---|---|
| Supported captions lacked matched promotion controls. | Message 566; line 686 | Add count-matched and similarity-matched randomized promotion. | Completed in the preserved v3 study. |
| Selected procedures used different training schedules. | Message 566; line 687 | Match learning rate, epoch, and seed. Report selected procedures separately. | Completed in the preserved v3 study. |
| Clean in-domain retrieval was absent. | Message 566; line 688 | Use 1,000 e-ViL test images and 5,000 source captions. Keep COCO as transfer evaluation. | Completed in the preserved v3 study. |
| One setting and fixed weights supported overly broad advice. | Message 566; line 689 | Remove broad procedure advice. Restrict conclusions to the tested setting. | The v3 rewrite narrowed claims. Broader evidence remained a later requirement. |

The schedule audit also found different learning rates.
See message 586, transcript line 707.

The standard Karpathy Flickr1k pool overlapped the training split.
The clean e-ViL pool replaced that unsuitable evaluation choice.
See transcript lines 704 and 738.

The approved follow-up used three randomized assignments, three learning rates, and three training seeds.
It included a 900-image development pool for source-caption retrieval.
See transcript lines 760–775.

The original replay checks reproduced the source and supported runs exactly.
See transcript lines 778–794.

The v3 audit reproduced 18 primary effects and 180,000 bootstrap samples.
See message 681, transcript line 809.

Recovery details remain in [RECOVERY_INDEX.md](../RECOVERY_INDEX.md).
The [v3 reproduction guide](REVIEW_FOLLOWUP_README.md) records the completed follow-up.

## Remaining requirements from the final review

| Requirement | Exact source | Evidence needed | Current status |
|---|---|---|---|
| Broader evidence | Message 768; line 882: “The evidence comes from one adaptation setting.” | Repeat Source, Supported, and score-stratified promotion in materially different settings. | Nonlinear and RN50 results passed independent verification. |
| Stronger conceptual contribution | Message 768; line 883: “The conceptual advance is limited.” | Connect substantive theory or controlled interventions to useful empirical conclusions. | Exact rank-retention derivations and the audited AD decomposition are complete. Original directional effects and certificate coverage remain pending. |
| Practical improvement | Message 768; line 884: “Practical relevance is diagnostic.” | Show a trained model that preserves retrieval and improves caption discrimination. | The complete AD study finds no family that passes the declared gate on either encoder. The requirement remains unsolved. |

The explicit next recommendation was “one independent replication in a materially different adaptation setting.”
See message 768, transcript line 888.

The practical requirement needs more than a completed grid.
Selecting the frozen model does not demonstrate useful adaptation.
See transcript lines 791, 827, and 884.

Target allocation remained an explanation to test.
The fixed-score derivative alone did not establish the cause of retrieval loss.
See transcript lines 697, 724, and 746.

The pairwise ranking policy learned relation discrimination but lost retrieval performance.
Its selected epoch-zero checkpoint did not establish a failure to learn.
See transcript lines 709 and 726.

## Preserved strengthening program

The [execution plan](STRENGTHENING_EXECUTION_PLAN.md) defines four separate questions.

| Question | Required work |
|---|---|
| Generality | Test a nonlinear adapter and an independently pretrained RN50 encoder. |
| Mechanism | Cross source and expanded supervision independently in both retrieval directions. |
| Practical utility | Compare allocation, source-gallery distillation, directional interventions, and WiSE-FT interpolation. |
| Theory | Derive and check rank-retention certificates on the actual evaluation gallery. |

Each question needs its own evidence.
Success on one question cannot substitute for another.
The [design audit](STRENGTHENING_DESIGN_AUDIT.md) states this distinction.

### Frozen protocols

| Protocol | File | SHA-256 |
|---|---|---|
| Completed v3 follow-up | [REVIEW_FOLLOWUP_PROTOCOL.json](REVIEW_FOLLOWUP_PROTOCOL.json) | `3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34` |
| Nonlinear replication | [STRENGTHEN_REPLICATION_PROTOCOL.json](STRENGTHEN_REPLICATION_PROTOCOL.json) | `3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920` |
| Encoder replication | [STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json](STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json) | `53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e` |
| Retention and directional interventions | [protocol_v3.json](../results/strengthen_retention/protocol_v3.json) | `5624342a20dd33b4f235264f57599a8bad0486d70fa587a6d9269371ea00b5d4` |
| Allocation-plus-distillation extension | [protocol_v1.json](../results/allocation_distillation/protocol_v1.json) | `f5ba7e389660bf9930b2713a7173e7a6cd7d55bf755111b259c512bc74c5a4cd` |

The retention predecessors remain design history: `protocol.json` and `protocol_v2.json`.
Do not overwrite frozen protocols when adding experiments.

The practical gate requires nonzero updates for all three selected seeds.
Both retrieval directions must satisfy the declared one-percentage-point tolerance.
SugarCrepe++ accuracy for both valid captions must improve.
The protocol defines adjusted confidence bounds for these conditions.

Existing test datasets remain reused evaluation sets.
They are not newly collected confirmation data.
Development data determines hyperparameters and checkpoint selection.

## Missing allocation-plus-distillation comparison

The [related-work assessment](STRENGTHENING_RELATED_WORK.md) explicitly recommends a four-cell decomposition.
See that file's lines 59–72.

| Cell | Allocation | Source-gallery KL distillation | Purpose |
|---|---|---|---|
| U | Uniform supported targets | Off | Reference expansion procedure |
| A | Declared source allocation | Off | Test allocation alone |
| D | Uniform supported targets | On | Test distillation alone |
| A+D | Same allocation as A | Identical distillation term | Test allocation beyond distillation |

Retention protocols v1–v3 omit the A+D cell.
No identified revision withdraws the four-cell recommendation.
Separate failures of A and D do not establish their interaction.

**Design status: complete and frozen before affected fitting.**
The new exploratory protocol defines the missing comparison and all analysis families.
Its fixed scope is 126 fits per encoder, or 252 fits across both encoders.
The [freeze receipt](../results/allocation_distillation/protocol_freeze.json) binds the protocol and implementation.
Its primary comparison family contains 80 effects across both encoders.
Its separate matched decomposition contains 24 effects.
All 252 fits, evaluation, and the independent AD audit are complete.
No selected family passes the joint practical criterion on either encoder.
The original-retention program remains separate and unfinished.

The new stage must preserve the completed v3 evidence and earlier frozen protocols.
Its analysis must distinguish prior evidence from new exploratory findings.

## Current workspace recovery status

The earlier working directory disappeared: `/workspace/scratch/6d3ae43d7acb/SANW`.
The current directory is `/workspace/scratch/81995298881b/SANW`.

The prior conversation reported completion of 270 strengthening fits and a revised v4 paper.
It also reported independent audits and a final local checkpoint.
Those reports remain historical context.

Those historical raw outputs were not recovered for independent verification.
The current program therefore executes the frozen designs again.
These fresh outputs supply the current tables.
The paper does not reconstruct numerical results from remembered historical claims.

The current recovery restored remote commit `72cba41209e622fb2a5fdf0e3a3570c10501a48d`.
The original v3 evidence remains available in that restored repository.
Do not populate new tables from remembered strengthening results.
Use restored raw outputs or clearly identified fresh executions.

### Verified progress after the user resumed work

Both replication training grids completed 45 fits.
The nonlinear evaluation contains 60 states and 300 prediction archives.
The independent nonlinear audit passed all twelve effects and 120,000 bootstrap samples.
It checked 2,400 numeric aggregates.
The largest bootstrap difference was `1.5265566588595902e-16`.
The audit reconstructs results from saved ranks and score-derived correctness.
It excludes independent feature extraction and complete dot-product reconstruction.

| Nonlinear endpoint | Increasing-rate effects, in percentage points | Adjusted interval result |
|---|---|---|
| Supported minus Source, image-to-caption R@1 | -3.03, -3.43, -8.37 | All three intervals lie below zero. |
| Supported minus stratified promotion, caption-to-image R@1 | +0.94, +0.51, +2.69 | The smallest and largest rates exclude zero. The middle interval includes zero. |

The rates are `0.0001`, `0.0003`, and `0.001`.
The [independent receipt](../recovery/current_turn_audit/nonlinear_replication_independent_audit.json) binds these results.
The evidence extends the retrieval tradeoff to nonlinear adaptation.
It does not establish successful preservation.

The RN50 evaluation contains 62 states and 310 prediction archives.
Its independent audit passed all twelve effects and 120,000 bootstrap samples.
It checked 2,480 numeric aggregates.
The largest bootstrap difference was `1.8041124150158794e-16`.
Its scope has the same feature-extraction and dot-product exclusions.

| RN50 endpoint | Increasing-rate effects, in percentage points | Adjusted interval result |
|---|---|---|
| Supported minus Source, image-to-caption R@1 | -5.00, -6.17, -7.27 | All three intervals lie below zero. |
| Supported minus stratified promotion, caption-to-image R@1 | +2.10, +5.39, +4.63 | All three intervals lie above zero. |

The [independent RN50 receipt](../recovery/current_turn_audit/rn50_replication_independent_audit.json) binds these results.
Every RN50 image-to-caption interval against stratified promotion includes zero.
Both replications retain the defined directional caption-assignment benefit and source-retrieval cost.
Neither establishes the independent practical criterion.

### Completed AD study and its negative practical result

The AD study completed 126 fits per encoder.
Its locked evaluation contains 99 unique states and 495 prediction archives.
The independent audit checked 80 primary effects and 24 matched decomposition effects.
It reproduced all 10.4 million saved bootstrap values.
The maximum bootstrap difference was `1.249000902703301e-16`.
The [independent AD receipt](../recovery/current_turn_audit/allocation_distillation_independent_audit.json) binds the exact inputs.

None of six selected families passes the joint practical criterion on either encoder.
All twelve family-encoder combinations fail the SugarCrepe++ improvement requirement.
AD trains every selected seed, but fails both adjusted retrieval requirements for each encoder.
Supported instead selects frozen initialization for every seed.
Those frozen selections are not trained improvements.

| AD minus frozen | ViT difference and adjusted interval | RN50 difference and adjusted interval |
|---|---|---|
| Image-to-caption R@1 | -1.13 [-4.07, 1.86] | -0.73 [-4.00, 2.40] |
| Caption-to-image R@1 | -1.47 [-2.91, -0.02] | +0.47 [-1.21, 2.13] |
| Relation accuracy | +6.67 [5.21, 8.17] | +9.02 [7.15, 10.96] |
| SugarCrepe++ both-caption accuracy | -0.13 [-1.46, 1.20] | +0.62 [-0.90, 2.10] |

All values use percentage points and the declared 80-effect adjustment.
Relation accuracy cannot substitute for SugarCrepe++ improvement.
Failure to establish noninferiority does not prove that the true loss exceeds one point.
The practical requirement remains unsolved.

The matched decomposition uses the AD-selected schedule and a separate 24-effect adjustment.
Allocation and distillation each improve both retrieval directions on both encoders within those matched cells.
Distillation also improves SugarCrepe++ within those cells.
No adjusted interaction interval has a positive lower bound.
These effects do not establish improvement against frozen initialization for selected procedures.

AD primary contrast SHA256: `bbe15dece9bf6afce64b8bbb1186229a0a8f509948a62fdbf8720a89f03afd3d`.
AD decomposition SHA256: `aeb2ab90de5b93e1fd75d62655a0c549d104699a37e1636c63aa855ad2f08c12`.
AD practical-gate SHA256: `e56e276bf6ef295b6b718b6aa282f9ded0b26422f62a31cfee8fee8802dbabee`.

The primary contrast hashes remain unchanged:

- Nonlinear: `ebf63d3947fef079b914fcadee611def09ad12757cfe484c45b469e4f5a2d538`.
- RN50: `88e2fc0bb1e0f1f7d6058f16cd079cd0f510b82da797c8b814b10d85818bd83b`.

At the restart checkpoint, the new ViT AD grid had completed all 126 fits.
The RN50 AD grid had completed 60 fits.
The original RN50 retention grid had completed 16 fits.
The original ViT retention grid had not started.
These are historical restart counts, not a live execution report.
Repeated baseline executions do not add independent training seeds.

The manuscript includes both independently verified replication families and the complete negative AD result.
The expanded interim draft retains four main pages, with complete numerical tables before the theory appendix.
Final layout review remains pending the original-retention results.
Original-retention and certificate claims stay marked pending.
The final generator requires exact independent audit bindings for every new result family and certificate index.

The restored [resume record](RESUME_20261004.md) describes an earlier recovery stage.
Read it as historical provenance alongside this current ledger.

## Superseded or conditional directions

The initial SANW attenuation study does not define the current paper's central question.
The user selected grounded relation decisions over duplication invariance.
See messages 39–43, transcript lines 144–199.

The learned relation estimator depended on useful oracle headroom.
It was not an unconditional requirement after the later controlled assignment study.
See transcript lines 182–197.

Broad superiority for three-way labeling remained a hypothesis.
See transcript lines 191–195.

The v3 paper reports controlled assignment findings.
It does not propose a generally superior adaptation method.
See transcript lines 811–819.

The favorable early-epoch example belongs in the complete appendix comparison.
It must not replace the full planned comparisons.
See transcript line 845.

## Paper and recovery requirements

These requirements come from user message 411, transcript line 502.

- Inventory all available theories, experiments, ideas, and figures.
- Choose the strongest central story supported by the evidence.
- Explain practical relevance.
- Fill four readable main pages.
- Define notation and concepts before use.
- Make the abstract accessible to nonspecialists.
- Use simple, precise English.
- Avoid em dashes, long sentences, and repeated explanations.
- Keep complete proofs, qualifications, and audits in the appendices.
- Provide the paper PDF and a flat Overleaf archive.
- Preserve code, results, and reproduction instructions.
- Provide usable recovery files during execution.

The user requested GitHub upload and supplied the SANW destination.
See messages 463 and 480, transcript lines 567–591.
Repository authorization and any later execution restriction are separate records.

## Completion checks

- Verify the restored v3 evidence and retain its original hashes.
- Recover strengthening outputs where possible.
- Label fresh executions separately when recovery cannot supply required outputs.
- Freeze the missing A+D stage before affected experiments.
- Complete all declared fits, selection records, evaluations, and audits.
- Assess scientific requirements independently of experiment completion.
- Update the four-page paper only from verified evidence.
- Deliver complete appendices, a flat Overleaf archive, and recovery instructions.
- Record the final repository checkpoint and its exact recovery scope.
