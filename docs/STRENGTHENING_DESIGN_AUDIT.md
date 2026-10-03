# Independent audit of the strengthening design

Audit opened 2026-10-03 at 13:04 UTC. This is a prospective design and implementation review, not a report of new held-out outcomes. The auditor does not choose successful test cells or change other agents' protocols. Original v3 findings remain unchanged.

## Scope and current decision

The extension can address three distinct weaknesses with three distinct pieces of evidence: a materially different pretrained encoder and adaptation family address breadth; directional interventions test the allocation explanation; a nonzero development-selected model meeting a joint retention-and-discrimination criterion addresses practical utility. None substitutes for the others.

Reviewed inputs: `results/strengthen_retention/protocol.json`, `protocol_v2.json`, `docs/STRENGTHEN_REPLICATION_PROTOCOL.json`, the v3 main paper, the original image-cluster bootstrap implementation, and the nonlinear training/selection implementation. RN50 feature extraction is separately pinned to the official OpenAI checkpoint. No new test outcome was inspected for this audit.

**Pending before test scoring:** finalize and hash the expanded retention protocol, including direct baseline and distillation-matched assignment contrasts; freeze the RN50 replication protocol and encoder-specific control assignments; bind selected checkpoints and code before evaluation. Fitting may use the already fixed training and development rules. A definitive implementation/results sign-off follows raw-prediction audit.

## Decisions that strengthen the design

| Question | Required comparison | What it establishes |
|---|---|---|
| Does the assignment effect depend on a linear correction? | Source, Supported, and three score-stratified draws under a ReLU bottleneck adapter; every declared learning rate at epoch ten | Replication in a different function class and capacity, on the original encoder and corpus |
| Does it depend on one pretrained representation? | The same comparison under independently pretrained OpenAI RN50, with RN50-specific frozen-score strata | Breadth across encoder architecture and pretraining; not a fresh test dataset |
| Which training direction causes a change under this intervention? | Source/source, expanded/source, source/expanded, expanded/expanded directional factorial at every declared rate | Main and interaction effects of the two defined supervision interventions |
| Can adaptation improve discrimination while retaining search? | Development-selected allocation, supported-plus-source-KL, and weight-interpolated Supported, each compared with frozen | A joint practically useful operating point if the declared statistical gate passes |
| Does KL offer more than the cheap alternatives? | Direct KL-minus-allocation and KL-minus-WiSE effects on all four primary endpoints | Increment relative to competing complete selection procedures |
| Is useful KL-regularized learning attributable to the supported assignment? | KL-Supported versus count-and-score-matched KL-random at the Supported-selected schedule, averaging all draws | Assignment effect conditional on a supported-development-selected schedule, not a comparison of independently optimized random strategies |

## Selection and test-use requirements

1. These are follow-up experiments designed after the original test results were known. They are prospective with respect to the new fits and predictions, but the test corpus is reused. Do not call the new score a fresh confirmatory holdout. A new pretrained encoder does not make the same images unobserved data.
2. Every epoch, learning rate, KL coefficient, interpolation coefficient, source mixture, and tie rule must be fixed before new test evaluation. The development set may select only within this frozen search.
3. Record per-seed feasible epochs, all development metrics, common method-family hyperparameters, and final state hashes before evaluating test outcomes. No best random draw may be selected.
4. The distillation-matched random controls must use the selected Supported KL coefficient, learning rate, and per-seed epoch. If they instead receive independent selection, they answer a different question. Their assignments remain fixed and all three draws receive equal weight.
5. Test-based method selection, new hyperparameters, endpoint reclassification, or changing the practical margin after inspecting outcomes requires a new explicitly exploratory stage. It cannot overwrite this stage's protocol.

## Practical success and inference

The existing v2 gate is appropriate in substance: a learned model must have adjusted lower confidence bounds above minus one percentage point for **both** in-domain retrieval directions, and an adjusted lower bound above zero for SugarCrepe++ both-valid-caption accuracy, compared with frozen. Relation accuracy remains a separately reported primary endpoint. A nonsignificant retrieval difference, a point decline smaller than one point, or selection of epoch zero is not this success criterion.

The nonzero requirement should be explicit for all three selected seeds. Report every seed's epoch and whether its update norm is nonzero. If the protocol permits a mixture of frozen and trained selected seeds, name that mixed selection procedure and do not describe all its outputs as learned models.

The primary family under discussion comprises 28 family-versus-frozen effects, eight direct KL-versus-allocation/WiSE effects, and four KL-supported-versus-KL-random effects: **40 effects**. This controls the declared endpoints and method comparisons together. Mechanism and architecture replications can retain their separately declared families because their hypotheses and claims are distinct; no inference should pool the smallest intervals from those families to support an unplanned overarching winner.

Use the same paired query-image resamples for matched comparisons. Average random draws within each optimization seed, then average seed predictions. For text retrieval, keep the five captions with their owning image; for SugarCrepe++, keep every triplet from an image in one cluster and recompute the item-weighted mean. The existing `paired_image_bootstrap` implements that weighting correctly. These intervals condition on the fitted seed/draw set and fixed galleries; they do not estimate variability from refitting new models or sampling new galleries.

At family size 40, a two-sided 95% Bonferroni family uses tail probability 0.000625. With 10,000 replicates each tail is represented by only about six order statistics. The audit recommends **100,000 replicates before observing new test effects** for a more stable noninferiority gate. Save all bootstrap effects and the exact quantile convention.

Direct baseline effects do not by themselves require every endpoint to be superior. A claim such as "more discrimination at retained retrieval" needs a positive discrimination increment and the stated retention comparisons. If only the joint frozen-reference gate passes, the valid result is a useful operating point, not superiority to all competitors.

## Fair interpolation and KL baselines

For a zero-initialized linear residual branch, multiplying both learned correction matrices by one shared interpolation coefficient is exactly interpolation between the frozen and adapted parameters. It must occur before output normalization. Interpolating normalized features or scores is a different baseline.

WiSE receives the same training states, learning-rate choices, development feasibility rules, and epoch range as ordinary Supported, including epoch zero, with its coefficient selected from the frozen grid. Both modalities use the same coefficient. State the coefficient and effective update norm. The alpha-zero output must exactly match the frozen encoder. Do not hide a zero-selected baseline or label it a learned remedy.

The KL term uses frozen teacher distributions on original source-caption columns in both directions, the same native scale and declared temperature, and a declared temperature-square multiplier. Hypothesis scores are not directly KL targets. The source-domain restriction is part of the method, not a new distillation principle. Describe this as supported-caption adaptation with a standard retention ingredient and cite the relevant prior methods. The matched random arm should receive precisely the same source KL term.

The method families have different search budgets. Reporting all frozen grids and the full development search is enough for the present procedure comparison; do not claim equal tuning expenditure. A coarse or zero-winning interpolation grid should not be silently refined after test inspection.

## Interpretation of the factorial and theory

The image intervention changes the target distribution between the five source captions and the expanded positive set. The reverse intervention changes which captions act as training anchors and the average over those anchors. All caption columns remain available as image-query competitors, and the total image-versus-text directional weighting remains fixed.

Therefore the reverse intervention changes both anchor participation and the weighting of existing source anchors. It is not an intervention on source weights alone. The 2-by-2 main and interaction effects are causal effects of the implemented directional supervision changes, conditional on the controlled training setup. They do not identify target dilution as the sole source of generalization loss.

At fixed logits, the earlier source-column gradient identity remains an exact property of count-matched expanded assignments. It does not assert that parameter gradients, trained features, or losses are identical. The nonlinear adapter preserves that loss-level identity but has a different parameter Jacobian.

A stability certificate based on a measured held-out KL or logit drift must use the exact distribution domain and temperature it proves. A per-query bound may certify a top-rank decision only when its margin condition holds for that query. Average training-batch KL is not a certificate for full held-out-gallery ranks. Certificates computed after fitting are explanatory measurements and cannot become a hidden checkpoint selector.

## Breadth and reproducibility checks

The nonlinear adapter is a genuine nonlinear function class: independent bias-free dimension-to-128-to-dimension branches with ReLU and zero output initialization. At dimension 512 it has 262,144 trainable parameters. It begins exactly at the frozen representation, then its input projection can learn after the zero output matrix departs from zero. The replication agent reports tests for frozen parity, two-step gradient reachability, and superposition failure. Same rates and epochs support a controlled sensitivity study, not a claim of equal optimization difficulty or nonlinear superiority.

The RN50 checkpoint changes architecture, pretraining, and feature dimension together. A successful replication establishes a second setting; it does not isolate which of those changed factors matters. Linear image and text 1024-by-1024 corrections contain 2,097,152 parameters. Frozen-score strata must be recomputed from RN50 training features if the control is called RN50-similarity-matched. Reusing ViT strata would instead be a fixed-assignment transfer experiment.

Feature caches must bind checkpoint, tokenizer, preprocessing, dimension, numeric precision, manifest, IDs, normalization, and feature hashes. Feature extraction on test items is not fitting, provided no test outcomes direct the design. Causal-padding optimization should pass parity against the unmodified text encoder. Training/evaluation manifests must preserve all existing split/byte-overlap checks.

## Required post-run audit

- Verify every planned fit and assignment draw is present, every selected state was fixed before scoring, and no protocol or input digest changed.
- Recompute selected-family choices independently from development histories, including feasibility and all ties.
- Check epoch-zero and alpha-zero predictions against frozen, source/supported replay parity where promised, and nonzero update norms for any claimed learned success.
- Recompute all primary aggregate endpoints from raw ranks/item correctness, aligning query and cluster IDs exactly.
- Independently reproduce every primary effect and bootstrap interval, using the frozen family size, seed, replicate count, draw/seed averaging order, and cluster weights.
- Evaluate the practical gate mechanically. Report which component passed or failed for every family, without moving the margin.
- Retain all failed and successful cells, complete trajectories, model/input hashes, raw prediction arrays, software versions, and a runnable regeneration command.

This audit will be updated with exact final protocol receipts and results checks after the corresponding artifacts exist.
