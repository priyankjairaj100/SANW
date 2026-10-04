# Practical-goal experiments, v6

This study follows the completed reviewer revision. Earlier candidates did not
pass the practical gate. These experiments introduce two bounded scoring
families and preserve the original practical requirements.

## Fixed practical requirements

- The same trained family must pass on ViT-B/32 and RN50.
- Each encoder uses three nonzero fitted states: seeds 17, 29, and 43.
- Both full-gallery e-ViL retrieval directions require adjusted lower bounds
  strictly above -1 percentage point, relative to frozen features.
- SugarCrepe++ both-positive correctness requires an adjusted improvement lower
  bound strictly above zero.
- Confidence intervals use 100,000 paired image-cluster bootstrap draws,
  the mean of the three fitted seeds, family size 80, and alpha 0.05.
- COCO retrieval is a secondary endpoint.

These evaluations remain exploratory because earlier test results informed this
research. Selection before the new evaluations does not remove historical test
exposure. The specified interval adjustment does not establish error control
over every adaptive decision in the research history.

## Candidate families

The pooled-feature family adds a learned scalar correction to frozen cosine.
The correction is bounded by epsilon through a tanh output. Its inputs include
centered image-text products, a low-rank interaction, and frozen cosine.

The token family trains the final text-transformer block and final layer norm.
Images, the earlier text blocks, and the text projection remain frozen. Its
score is:

    s(v,t) = s0(v,t) + epsilon * tanh(v dot (t_new - t_reference) / epsilon)

The reference and learned text branches share the same frozen prefix. Their
identical initialization produces an exactly zero correction. Every dataset
uses the same scoring function. Retrieval labels and task identifiers do not
enter the score.

Training uses source/support/contradiction triplets and source-caption retrieval
losses. Neutral hypotheses never become contradictions. Two valid positive
captions need not have identical meanings or embeddings.

## Selection and numerical consistency

Training uses only the 1,200-image training split. Selection uses the
100-image relation-validation split and the 900-image retrieval development set.
Every selected checkpoint must retain both development retrieval counts.
Among feasible checkpoints, selection maximizes image-averaged joint triplet
accuracy, followed by mean triplet margin and then the earliest epoch.

The pooled scorer evaluates each pair with model batch size one. The token
scorer encodes each caption independently, through its own end-of-text token.
These policies prevent neighboring examples from changing floating-point ties.
Development checkpoints are rescored under the same canonical inference rules.
The original training histories remain preserved separately.

Both complete six-state selections must be locked before either new family
accesses held-out outcomes. Locks bind checkpoints, source files, protocols,
development evidence, and feature provenance.

## Evidence locations

- `protocol_v1.json`: initial pooled protocol and unchanged practical gate.
- `inference_amendment_v1.json`: canonical pooled inference rule.
- `token_protocol_v*.json`: immutable token pilot and replication protocols.
- `canonical_development/`: canonical pooled checkpoint trajectories.
- `canonical_text_development_v1/`: canonical token development evidence.
- `pooled_selection/`: pooled manifests and selection lock.
- `token_final_selection/`: token manifests, assembly receipt, and the
  evaluator-compatible `evaluation_lock.json`.
- `cross_family_lock.json`: both complete production locks, verified before
  either family began new held-out evaluation.

Training outputs and prediction archives are scientific artifacts. Public source
files and manifests alone do not contain the fitted weights or feature arrays.
Final benchmark conclusions must come from completed analyses and their
independent audits, rather than these development results.
