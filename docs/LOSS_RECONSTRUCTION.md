# Reconstructed objectives, frozen before execution

This implementation was rebuilt on 2026-10-03 from the retained study and
analytical records. The original source code was unavailable. The exact
similarity kernel and the base target set of the smoothing control were not
retained. The choices below are explicit reconstruction decisions, fixed
before inspecting new experiment outcomes. They are not claims of bitwise
historical reproduction. No historical result table is read by these modules.

## Shared target and candidate conventions

A batch contains unique image rows and all selected text candidates. Relation
codes are 0 (unannotated), 1 (source caption), 2 (supported hypothesis),
3 (contradicted hypothesis), and 4 (neutral hypothesis). An absent cross-image
annotation is an eligible negative, not a neutral relation. Source captions
are positive for every contrastive policy. The source-only positive policy
keeps supported hypotheses in its denominator as negatives. Expanded policies
promote those already-present supported hypotheses.

For an eligible row, the loss is

`logsumexp_j(s_j + log(w_j)) - sum_j(y_j * s_j)`.

Weights are nonnegative and one on positives. A zero weight is represented by
negative infinity inside logsumexp, which excludes that candidate exactly.
There is no positive floor on a zero weight. Default targets are uniform over
positive candidates. The loss is half the mean over eligible image rows plus
half the independently normalized mean over eligible text columns. Columns
without any positive image are skipped. The same positive and weight matrices
are transposed, rather than rebuilding a separate reverse policy.

All semantic weights, masks, thresholds and the fixed logit scale are detached
from autograd. Gradients flow through both residual feature adapters. Each
adapter is a bias-free d-by-d linear map initialized to zero, used as
`normalize(x + A*x)`. Images and texts have separate maps.

## Exact policies

| Method | Positive targets | Candidate weights / target change |
|---|---|---|
| clip | Source | All weights one |
| multipositive | Source + supported | All weights one |
| grounded | Source + supported | Contradiction weight 2; neutral weight 0 |
| grounded_no_hardening | Source + supported | Neutral weight 0 |
| grounded_no_abstention | Source + supported | Contradiction weight 2 |
| random_exclusion | Source + supported | Per image, uniformly sample nonpositive cells without replacement; zero exactly as many as its neutral annotations |
| smoothing | Source | Targets `0.9*uniform(positives) + 0.1*uniform(all candidates)`; weights one |
| constant | Source | All nonpositive weights 0.25 |
| sanw_fixed | Source | Detached caption-similarity weights below, threshold 0.5 |
| sanw_median | Source | Same kernel, batch median threshold |
| shuffled | Source | Globally permute sanw_fixed nonpositive weights within the batch |
| pairwise_rank | Supported vs contradicted | Within each eligible image, average `softplus(s_contradicted - s_supported)` over all supported-by-contradicted pairs; then average images |

The ranking loss has no reverse direction and does not train on source,
neutral or unannotated cells. A batch with no eligible ranking image returns a
differentiable zero. In the contrastive policies, source-caption positives
ensure that a legitimate batch has eligible images and text anchors.

## Reconstructed similarity kernel

Let `t_j` be the unit-normalized **frozen** text feature of candidate j. Let
`C_i` be source-caption indices for image i. For every image-text cell,

`a_ij = (1 + mean_{k in C_i} dot(t_k, t_j)) / 2`.

This is the mean mapped cosine. Averaging source features and then normalizing
the average would define a different kernel and is not performed. Source
captions are required for every image under this policy. Candidate weights
before forcing positives to one are

`w_ij = sigmoid((tau - a_ij) / 0.05)`.

The fixed threshold is `tau=0.5`. The median threshold is the statistical median
of `a_ij` over all nonpositive cells in the current image-by-text batch. For an
even number of cells, it is the mean of the two central values. This threshold
is shared by both directions because the resulting weight matrix is transposed.
The shuffled control uses the fixed-threshold kernel and permutes the complete
batch-wide multiset of nonpositive weights. It preserves every positive weight
and the negative-weight multiset, but not each row's mass.

No semantic image feature is used. The optional `semantic_image_features`
argument exists only for a shared caller interface. `semantic_text_features`
must be separately supplied; adapted text features are never silently used.

## Defaults and reproducibility

The public API accepts `semantic_threshold=0.5`, `semantic_softness=0.05`,
`negative_weight=0.25`, `hardening_weight=2.0`, and `smoothing=0.1`. Training
records these values. Random exclusion and shuffling accept a seeded
`torch.Generator`; the caller maintains its state across batches and epochs.
Random exclusion preserves each image's exclusion count. Its reverse-direction
column exclusion counts need not match neutral counts per text.

The tests compare the implementation to ordinary bidirectional cross entropy,
manual rectangular normalization, analytic and finite-difference gradients,
exact zero masks, extreme logits, positive expansion, matched-normalizer
identities, exact policy masks, permutation invariants, candidate-uniform
smoothing, image-balanced ranking, and parameter/semantic gradient boundaries.
