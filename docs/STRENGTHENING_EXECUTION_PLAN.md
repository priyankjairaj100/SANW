# Strengthening the empirical short paper

Date: 2026-10-03. This extension follows the completed v3 study and the author's request to resolve its remaining scientific weaknesses through new evidence.

## Questions and deliverables

1. **Generality:** repeat the controlled assignment comparison with a nonlinear residual adapter and with a separately pretrained OpenAI RN50 encoder. Preserve source, supported, and three score-stratified random assignments, three training seeds, matched schedules, and frozen initialization.
2. **Mechanism:** cross source-only versus expanded positive supervision independently in the image and text directions. The four cells distinguish image-target assignment from reverse-anchor expansion. Measure the trained factorial effects rather than infer their cause from a fixed-score derivative.
3. **Practical utility:** compare source/support objective mixing, frozen-teacher retrieval distillation, the two directional interventions, and WiSE-FT interpolation. Select using development data under an explicit retrieval constraint, then evaluate retrieval and caption discrimination together.
4. **Theory:** derive and numerically audit a rank-retention certificate connecting source-gallery distribution fidelity to retained correct retrieval. State its scope precisely. A minibatch training penalty is not itself a full-gallery guarantee.

The method ingredients have existing precedents. The proposed contribution is a tested account of which positive assignments, directions, and preservation strategies work across settings. Mathematical or method novelty will be judged against the closest published work before drafting.

## Execution boundaries

- Original v3 source, evidence, and frozen study files remain intact. New code and results use `strengthen_*` paths.
- Every subexperiment binds its input, protocol, implementation, and output hashes. Design amendments are retained and must precede affected fitting or evaluation.
- Existing held-out datasets have already informed the earlier study. They are reused evaluation sets, not newly collected independent confirmation samples.
- Hyperparameters and checkpoint selection use development data only. Full grids, failed policies, selected states, and frozen baselines are retained in the scientific record.
- The primary practical success criterion requires a nonzero trained state, simultaneous evidence that both retrieval directions lose less than one percentage point relative to frozen initialization, and improved SugarCrepe++ two-valid-caption accuracy. The retention protocol specifies its multiplicity correction and development selector.
- Architecture and encoder replications will be named separately. A nonlinear adapter on the same encoder is not a second pretrained model.

## Frozen subprotocols

- Retention and directional intervention: `results/strengthen_retention/protocol_v3.json` (SHA256 `5624342a20dd33b4f235264f57599a8bad0486d70fa587a6d9269371ea00b5d4`). Its two predecessors are retained as pre-fitting design history. The final design includes both encoders, direct comparisons with interpolation and allocation, a matched distilled random-promotion control, and 100,000 bootstrap resamples. Primary inference adjusts jointly over 80 contrasts across the two encoders; the directional factorial has a separate 36-contrast family. All three selected seeds must represent nonzero training for the practical success gate.
- Nonlinear replication: `docs/STRENGTHEN_REPLICATION_PROTOCOL.json`, SHA256 `3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920`, frozen before fitting.
- Encoder replication: `docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json`, SHA256 `53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e`, frozen before fitting. Score-stratified random assignments are recomputed using this encoder's own frozen scores.
- Second encoder: official OpenAI RN50 checkpoint with source-pinned weight hash, preprocessing, and separate content-addressed feature cache; execution protocol will bind the resulting feature manifests before fitting.

## Delivery

Deliver compact checkpoints during execution, followed by complete raw results, updated figures, a four-page manuscript with complete appendices, a flat Overleaf archive, and a verified Git snapshot. Completed archives from previous stages remain available.
