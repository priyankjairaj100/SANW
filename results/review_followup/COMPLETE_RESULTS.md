# Complete review-follow-up results

All 72 fits, 792 checkpoint records and 48 selection references are complete. The frozen evaluation covers 244 unique state references and 1,080 prediction archives. Independent aggregate recomputation passed 5,536 checks with maximum absolute discrepancy 2.22e-16.

The primary family contains 18 epoch-10 effects. Each learning rate is reported. The intervals below are 99.722% conditional image-bootstrap intervals, with a Bonferroni family of 18.

| Learning rate | Comparator | Direction | Supported minus comparator (pp) | 99.722% interval |
|---:|---|---|---:|---:|
| 0.0001 | Source targets | i2t | -6.000 | [-9.037, -3.063] |
| 0.0001 | Source targets | t2i | +0.120 | [-1.087, +1.322] |
| 0.0001 | Count-matched random | i2t | -0.978 | [-3.389, +1.379] |
| 0.0001 | Count-matched random | t2i | +1.811 | [+0.631, +2.995] |
| 0.0001 | Score-stratified random | i2t | -0.744 | [-3.001, +1.436] |
| 0.0001 | Score-stratified random | t2i | +1.540 | [+0.504, +2.583] |
| 0.0003 | Source targets | i2t | -7.133 | [-10.367, -3.796] |
| 0.0003 | Source targets | t2i | -1.320 | [-2.955, +0.381] |
| 0.0003 | Count-matched random | i2t | +1.322 | [-1.730, +4.114] |
| 0.0003 | Count-matched random | t2i | +4.838 | [+3.332, +6.376] |
| 0.0003 | Score-stratified random | i2t | +0.122 | [-2.659, +2.815] |
| 0.0003 | Score-stratified random | t2i | +3.820 | [+2.412, +5.198] |
| 0.001 | Source targets | i2t | -10.833 | [-14.341, -7.330] |
| 0.001 | Source targets | t2i | -1.900 | [-3.962, +0.148] |
| 0.001 | Count-matched random | i2t | +3.789 | [+0.717, +6.922] |
| 0.001 | Count-matched random | t2i | +7.038 | [+5.163, +8.832] |
| 0.001 | Score-stratified random | i2t | +0.222 | [-2.591, +3.025] |
| 0.001 | Score-stratified random | t2i | +4.084 | [+2.426, +5.804] |

At all three learning rates, supported promotion improves text-to-image Recall@1 over both randomized controls. Every one of those six intervals is above zero. Its image-to-text Recall@1 remains below source-only positive targets at all three learning rates. The corresponding source-target text-to-image intervals include zero. Against score-stratified random promotion, all three image-to-text intervals include zero; against count-matched random promotion, the interval is above zero only at the largest planned learning rate.

These comparisons identify the specified positive-assignment interventions within the fixed adapter setting. Source targets retain every hypothesis as a competing candidate. The randomized controls use label-derived positive quotas. The intervals do not resample training seeds or assignment draws. Full cell values and both sources of observed variation remain in the analysis files.

## Files

- `analysis/all_state_metrics.csv`: every scored state and scalar endpoint.
- `analysis/fixed_epoch_metrics.json`: all learning-rate/epoch/seed/draw cells and equal-draw policy means.
- `analysis/primary_contrasts.json` and `primary_bootstrap_samples.npz`: all 18 effects and 180,000 bootstrap samples.
- `analysis/selected_strategy_metrics.json`: native and source-retrieval strategies, with each random draw selected separately.
- `analysis/e_vil_query_subgroups.json`: original 400 and additional 600 query summaries against the same full candidate pool.
- `evaluation/native_replay_prediction_audit.json`: 259 bitwise-equal retained arrays for frozen and native source/supported replay.
- `figures/`: vector PDF, 300 dpi PNG, source CSVs, captions and complete figure provenance.

All new figure inputs are hash-bound in `figures/figure_provenance.json`. The evaluator, analyzer and protocol were not modified during scoring or figure generation.
