# Independent allocation and distillation audit

Date: 2026-10-04.

Status: training design, implementation, and evaluation integration checks passed.
The frozen protocol may govern the new training grid.
Scoring must wait for completed training and the cross-encoder selection lock.

Protocol: `results/allocation_distillation/protocol_v1.json`.

Protocol SHA256: `f5ba7e389660bf9930b2713a7173e7a6cd7d55bf755111b259c512bc74c5a4cd`.

## Scientific scope

The related-work plan requires U, A, D, and A+D.
The previous frozen protocols contain U, A, and D.
They omit A+D without recording an explicit withdrawal.
The new stage completes that unresolved comparison.
It does not replace the completed v3 study.

The new stage is exploratory because earlier test results informed its design.
Freezing its new protocol does not make the reused datasets fresh confirmation samples.

The intended objective is

`lambda * L_source + (1-lambda) * L_supported + beta * T^2 * (KL_image + KL_text)/2`.

Both supervised losses use the same complete candidate matrix.
The source policy changes positive targets, not candidate contents.
The coefficient `lambda` mixes complete objectives.
It does not assign a fixed probability mass to each source group.

The KL term uses source-caption columns at temperature two.
The reverse term uses every batch image for each source-caption anchor.
The teacher remains detached.
Both directional means receive weight one half.

## Required experiment

Each encoder has thirteen training policies, three rates, and three seeds.
This gives 117 base fits per encoder.
Nine additional controls use three assignment draws for each training seed.
The total is 252 fits across two encoders.

The selected families are Source, U, A, D, A+D, and WiSE.
WiSE uses the original six interpolation coefficients.
It scales both residual matrices before normalization.
An alpha-zero output remains a frozen output.

The primary selector permits one percentage point of development retrieval loss.
The sensitivity selector permits no loss.
Each selector checks both retrieval directions for every seed.
Each family shares its learning rate and method parameters across seeds.
Epochs may differ across seeds.
Selection maximizes development relation accuracy among feasible epochs.
The original 100-image relation pool is a subset of the 900-image retrieval pool.
The two development criteria do not use independent image pools.
Ties favor earlier epochs, lower rates, and lower method parameters.
A+D parameter ties use `lambda`, then `beta`.

The matched U/A/D/A+D cells copy the primary A+D schedule.
A uses the A+D allocation coefficient.
D uses the A+D KL coefficient.
All four cells use identical rates and seed-specific epochs.
This comparison conditions on a schedule selected for A+D.
It differs from comparisons between independently selected procedures.

The random controls copy A+D's allocation coefficient, KL coefficient, rate, and seed-specific epoch.
All three fixed assignment draws receive equal weight within each seed.
All three training seeds then receive equal weight.
The controls receive no independent development selection.

## Inference contract

The four primary endpoints are:

- In-domain image-to-text R@1.
- In-domain text-to-image R@1.
- Visual-entailment image-averaged accuracy.
- SugarCrepe++ both-caption accuracy.

Six families versus frozen give 48 contrasts across two encoders.
A+D versus D, A, WiSE, and matched random gives another 32 contrasts.
All 80 primary contrasts share the declared Bonferroni correction.

The matched decomposition has these effects:

`allocation_main = 0.5 * ((A-U) + (AD-D))`

`distillation_main = 0.5 * ((D-U) + (AD-A))`

`interaction = AD-A-D+U`

Four endpoints across two encoders give 24 decomposition contrasts.
These contrasts use a separate correction family.

The bootstrap resamples paired image clusters after averaging fixed training seeds.
Text queries remain grouped with their source image.
Triplet datasets retain item weighting after each cluster resample.
The intervals condition on the fitted seeds, fixed draws, and fixed galleries.
They do not estimate variation from new model fits or new galleries.

Practical success requires three nonzero selected epochs and three nonzero updates.
Both adjusted retrieval bounds must exceed minus one percentage point.
The adjusted SugarCrepe++ both-caption bound must exceed zero.
The same family must pass for both encoders to establish cross-encoder success.
Epoch-zero selections, nonsignificant changes, and small point losses do not satisfy these requirements.

## Independent checks completed

The independent checker uses explicit log-sum-exp formulas instead of the project's loss helpers.
It checks sixteen allocation and KL combinations, including boundary cases.
It also checks gradients, teacher detachment, and the teacher candidate domain.

The final source-bound run passed all sixteen cases.
The largest objective difference was `1.14e-13`.
The largest gradient difference was `1.28e-13`.
Changing teacher hypothesis features left the objective unchanged.

Independent selector checks passed for both tolerances.
They covered both retrieval constraints, shared parameters, epoch ties, and parameter ties.
They also covered a selection containing frozen and trained states.

Checker:
`recovery/review_completion_20261004/audit_allocation_distillation_design.py`.

Receipt:
`recovery/review_completion_20261004/allocation_distillation_independent_design_checks.json`.

These checks perform no scientific training and read no test data.

The independent inference checker reproduced 2,018 synthetic bootstrap samples.
It gathered individual observations from sampled clusters without using the production aggregation formula.
Cluster sizes were one, two, and three items.
The maximum sample difference was `2.78e-17`.
It verified both correction families and the factorial formulas.
It also verified strict success thresholds and rejection of zero epochs or updates.

Checker:
`recovery/review_completion_20261004/audit_allocation_distillation_inference.py`.

Receipt:
`recovery/review_completion_20261004/allocation_distillation_independent_inference_checks.json`.

The auditor also ran all 24 targeted implementation tests.
All tests passed in 2.72 seconds.
Synthetic tests verified exact baseline trajectories against the previous training implementation.
They checked complete family selection, WiSE export, decomposition schedules, and corruption rejection.
The nonzero control test used epochs one, two, and one across seeds.
It observed assignment application and verified unchanged source masks.
It also verified positive updates and the selection digest in each control checkpoint.

The tests used synthetic fixtures only.
They did not use scientific training data or held-out data.

The auditor also ran all eleven evaluation integration tests.
All tests passed in 2.82 seconds.
The tests passed a synthetic executor manifest through the actual scoring and analysis checks.
They rejected changed control rates, epochs, allocation coefficients, and KL coefficients.
They also rejected incomplete controls, incorrect decomposition parameters, and changed item ordering.
Triplet checks retained strict tie handling.

Final receipt:
`recovery/review_completion_20261004/allocation_distillation_final_design_receipt.json`.

## Findings resolved before freezing

1. Training and evaluation now share one complete evaluation specification.
2. Protocol validation rejects any change to that specification.
3. The third primary endpoint is visual-entailment accuracy.
4. RN50 metadata must match the original protocol, checkpoint digest, dimension, and feature digest.
5. The development manifest must match the original review ledger.
6. Assignment records bind the new protocol and exact training features.
7. The training ledger includes the evaluation contract and its imported module.
8. Nonzero control tests now exercise actual assignment application.
9. The decomposition export uses the family names expected by analysis.
10. The protocol explicitly requires the same family to pass for both encoders.

Final checked training source SHA256:
`2b0b8e5e2bbf68d74f2fa81360bbce9908fd06fd2cd45746fc88018bb14fee82`.

Final checked analysis module SHA256:
`19131b5bcbb25c612499e05f03c12ba4f01a424353ec508ef5bb57ea0826771b`.

## Remaining execution checks

The evaluator locks both encoders' selections before scoring.
It binds the selected checkpoints, feature files, source files, and dataset configuration.
The analysis verifies matched parameters, schedules, raw observations, and aggregate metrics.
The final CLI tests verified these integrations using synthetic fixtures.

After training, independently reconstruct selections from every development history.
After evaluation, independently reconstruct every declared contrast and practical gate.
Retain failed gates and all selected frozen states in the final report.

The auditor performs no scientific fits, test scoring, or Git changes.
