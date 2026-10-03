# Reconstructed relation-estimator diagnostic

This is an appendix-only classifier diagnostic. It does not supply labels,
weights, checkpoint selection, or training examples to any primary adapter.
Its outputs are a separate empirical measurement of how well a small model
predicts the released three-way relation labels. These labels are the reference
for the measurement; the diagnostic does not establish ground-truth visual
correctness or an oracle.

The original diagnostic implementation was unavailable. The following
hyperparameters and splitting rules are newly specified reconstruction choices,
fixed before fitting the new frozen feature cache. Historical numerical results
are not read by the script and are not targets for reconstruction.

## Examples and features

Read the canonical `visual_entailment` manifest and its exact hash-bound feature
cache. Use only annotated supported, contradicted and neutral image/hypothesis
pairs. Exclude source-caption pairs. An image remains entirely in its declared
training, validation, calibration or test split. Missing cross-image relations
are not classifier examples. Each ordered image/hypothesis pair contributes
one example; duplicate manifest pairs and cross-split hypothesis IDs are errors.

Class order is supported (0), contradicted (1), neutral (2). The three modes are:

- **Image:** frozen unit-normalized image features.
- **Text:** frozen unit-normalized hypothesis features.
- **Multimodal:** concatenation `[image, text, image*text, abs(image-text)]`.

No explanation text or source-caption feature enters this diagnostic. Features
are standardized by a `StandardScaler` fitted only on the classifier's training
rows. The full classifier is fitted on all training hypotheses. Repeated image
features across hypotheses remain repeated training examples.

## Model and selection

Fit multinomial logistic regression with an intercept, L2 regularization,
`C in [0.01, 0.1, 1, 10]`, `lbfgs`, `max_iter=2000`, `tol=1e-6`, no class weights,
and seed 17. Choose C by validation macro-F1 averaged equally over the three
classes. Exact ties choose the smaller C. Scaling is refitted using training
rows for each candidate. A convergence warning aborts that fit rather than
being silently ignored. The validation, calibration and test images never
enter classifier or scaler fitting. The selected model remains trained on
the training split; there is no train-plus-validation refit.

## Calibration and gates

Sort unique calibration image IDs and permute them with NumPy `default_rng(0)`.
Assign the first floor(N/2) images to temperature fitting and the rest to gate
selection. The intended 100-image calibration split therefore gives 50 images
per half. All hypotheses of an image remain together. Use this same partition
for all modes and all cross-fitting folds.

On the temperature half, minimize three-class mean negative log likelihood
(NLL) with a positive scalar temperature in `[0.05, 20]`. Optimize log temperature
using bounded scalar optimization with tolerance `1e-8`. Include temperature 1
and both bounds as explicit candidates, so the selected fit NLL cannot exceed
the uncalibrated fit NLL. Apply `softmax(logits / temperature)` afterward.
Temperature scaling preserves argmax class predictions.

For each of supported and contradicted, a gate accepts a pair only when that
class is the argmax and its calibrated probability is at least the gate's
threshold. Search observed confidence thresholds using whole confidence-tie
groups. A threshold is feasible when it accepts at least 20 gate-half examples
and its two-sided 95% Wilson lower endpoint is at least 0.90. Use the feasible
threshold with the greatest accepted count, which maximizes coverage on the
gate-selection half. If none qualifies, the gate accepts nothing.

The Wilson endpoint uses the conventional normal quantile
`z=1.959963984540054`. This selection rule is a heuristic gate criterion, not a
simultaneous guarantee after threshold search or an image-cluster-adjusted
confidence claim. Accepted counts, correctness and empirical precision are
reported on the separate test split. Coverage divides accepted count by all
test hypotheses. Precision divides correct accepted count by accepted count.
An empty gate has precision `null`, not zero or one.

## Metrics and uncertainty

Report both uncalibrated and calibrated accuracy, three-class macro-F1,
10-bin expected calibration error (ECE), multiclass Brier score, and NLL on
validation, temperature-fit, gate-selection and test partitions. Only test
metrics are held-out final diagnostic outcomes. Metrics on the calibration
halves disclose the behavior of their fitted/selected quantities.

ECE uses equal-width confidence bins `[0, .1), ..., [.9, 1]`; each bin contributes
its count fraction times the absolute accuracy-minus-mean-confidence gap.
Brier score is the mean **sum** of squared error across all three classes.
NLL clips a numerical zero probability at `1e-300` before taking its logarithm.
Argmax ties use the first class in the declared class order.

For each nonempty test gate, provide a descriptive 95% percentile precision
interval from 10,000 resamples of whole test images, seed 20261003. A sampled
image brings all its accepted and nonaccepted hypotheses with it. Record the
number of replicates that contain no accepted predictions; percentiles use the
nonempty replicates. The interval is conditional on the fitted classifier,
selected C, temperature and gate. It is separate from the primary paper's six
preplanned adapter contrasts and is not multiplicity-adjusted.

## Three-fold training predictions

Save three-fold out-of-fold predictions for future work. Sort unique training
image IDs, permute with `default_rng(17)`, and split into three blocks with
`numpy.array_split`. An entire training image and every associated hypothesis
belong to exactly one held-out fold.

For each fold, fit each C using only the other two image blocks, select by the
external validation split, then fit its temperature and gates using the same
external calibration halves. Repeating selection and calibration within the
fold prevents the held-out fold's labels from indirectly entering through the
full-training estimator's hyperparameters or calibrator. Predict the held-out
block and save its fold ID, raw logits, calibrated probabilities and gate masks.
The shared validation/calibration splits may be reused across folds; the
held-out **training** labels are excluded from every fitting and selection step
that generates their predictions. These outputs are not used in this study's
adapter training, and they are not an independent test set.

## Reproducibility and files

Run, after frozen feature creation and protocol approval:

```
python scripts/relation_diagnostic.py --threads 2
```

The output directory is `results/relation_diagnostic/`. A protocol ledger binds
input hashes, source hash and reconstruction settings before fitting. It cannot
be reused with changed inputs or settings. Each mode saves its selected model
as numeric scaler/coefficient/intercept arrays (no pickle), temperature, C
selection scores, image partitions, calibration/test predictions, gates,
three fold models and predictions, combined out-of-fold predictions, and
metrics. Each mode receives its own file-hash receipt before completion; a final receipt
covers the complete diagnostic. Completed mode summaries can be reused only
under the same ledger and after their file hashes pass verification. Partially
completed modes are refit.

The tests use constructed data only. They check group disjointness, coverage of
all out-of-fold rows, train-only scaling, temperature behavior, tie-respecting
gate selection, Wilson boundaries, hand-computed calibration metrics, image
cluster bootstrap, feature/manifest alignment, source-caption exclusion, and
invariance of out-of-fold predictions to held-out training labels.

## Numerical convergence amendment

The original 2,000-iteration ceiling stopped the text-mode grid before accepting
a nonconverged fit. The diagnostic-only amendment in
`RELATION_DIAGNOSTIC_ITERATION_AMENDMENT.json` raises the ceiling to 10,000. All
scientific settings remain unchanged. The original script, ledger, failure log
and completed image-mode outputs are preserved. The amended run recomputes all
modes in `results/relation_diagnostic_iter10000/`; it does not reuse estimates
from the first run. The primary adapter protocol is unchanged.
