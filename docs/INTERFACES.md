# Reconstruction interfaces

This is a new execution reconstructed on 2026-10-03. Historical result tables
are context only. Never copy them into newly computed results.

Repository root: `/workspace/scratch/3eec744c3d5e/grounded-contrastive-rerun`.
Python: `/workspace/scratch/3eec744c3d5e/.venv/bin/python`.
Source package: `src/gcr/`. Agents own separate modules/scripts.

## Canonical dataset manifest

Each dataset has `data/DATASET/manifest.json` with:

```json
{
  "schema_version": 1,
  "dataset": "visual_entailment",
  "images": [{"id": "flickr:123", "path": "data/visual_entailment/images/123.jpg", "split": "train"}],
  "texts": [{"id": "source:123:0", "text": "A caption."}],
  "pairs": [{"image_id": "flickr:123", "text_id": "source:123:0", "relation": "source"}],
  "triplets": []
}
```

Image paths are relative to the repository root. IDs are stable strings.
Relations are `source`, `supported`, `contradicted`, `neutral`. Pair annotations
apply to that ordered image/text pair only; missing cross-image relations are
unannotated negatives. Source captions are known positive targets. Images are
unique within a manifest. Text IDs are unique, even if strings repeat.

e-SNLI splits: `train`, `calibration`, `validation`, `test`, image-disjoint,
sizes 1200/100/100/400. Select image IDs deterministically with seed 42 from
official training and official development/test pools; report exact source
choice. Do not put any official test/development image into training.

SugarCrepe `triplets`: each has `id`, `image_id`, `positive1_id`, `negative_id`,
`category`. SugarCrepe++ adds `positive2_id`. For both, use all official items.
COCO has source pairs, five captions per test image in original source order.
All benchmark image splits are `test`.

Save source URLs, immutable revisions, file hashes and selection decisions
in `data/DATASET/provenance.json`. No candidate images may be silently skipped.

## Frozen feature output

`results/features/DATASET/features.npz`, loaded with `allow_pickle=False`:
`image_features` float32 [N,512], `text_features` float32 [M,512],
`image_ids` string [N], `text_ids` string [M]. Features normalized to unit L2.
`metadata.json`: input manifest SHA256, model revision/weight hash, encoding
parameters, source code hash, logit_scale (exponentiated scalar), row counts.
Features preserve manifest image/text order. Shared image/text content may be
reused only with exact hash/text identity and the same encoder/preprocessing.

## Model and objective interface

`gcr.losses.contrastive_loss(image_features, text_features, relations,
method, logit_scale, generator=None, **kwargs) -> scalar Tensor`.
Relations int matrix: 0 unannotated, 1 source, 2 supported,
3 contradicted, 4 neutral. Inputs are normalized adapted features. Semantic
similarity weights use separately provided frozen `semantic_text_features`
and optional `semantic_image_features`, never trainable features.
`gcr.adapters.ResidualAdapter(dim=512)`: image/text linear residual maps,
zero initialized, no bias, normalize outputs. Methods `encode_image(x)`,
`encode_text(x)`, `forward(images,texts)` returning both normalized features.

Twelve method names exactly: clip, sanw_fixed, sanw_median, constant, shuffled,
multipositive, grounded, grounded_no_hardening, grounded_no_abstention,
pairwise_rank, random_exclusion, smoothing.

## Execution

Training has a separate fit/selection phase before held-out evaluation.
All runs use frozen feature caches. Save epoch-zero and best states, full
validation histories, exact IDs, hyperparameters and code/input hashes.
Metrics and results must be computed anew. Primary image-bootstrap intervals
average selected seeds before sampling images; six planned effects use
Bonferroni family size six. Historical context is excluded from new outputs.
