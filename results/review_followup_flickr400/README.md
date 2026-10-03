# Post-review Flickr400 source-caption retrieval

This is a completed descriptive follow-up to the v2 study, dated 2026-10-03.
It evaluates all 36 originally selected adapters and the frozen reference.
There is no new training, checkpoint selection, or change to the original
primary statistical family.

The candidate pool is all 400 existing e-ViL test images and all 2,000 of their
Flickr30k Entities source captions. Each image has five source captions.
Only source ownership defines relevance. Hypotheses do not enter this candidate
pool. This is not the standard Karpathy Flickr1k retrieval benchmark.

| Procedure | I2T R@1 (%) | T2I R@1 (%) |
|---|---:|---:|
| Frozen | 91.00 | 80.15 |
| Source positives | 90.17 | 79.47 |
| Supported expansion | 84.00 | 73.20 |
| Expansion plus contradiction weighting and neutral exclusion | 83.83 | 73.22 |
| Source positives, constant negative weight 0.25 | 89.83 | 79.73 |

All three expansion seeds fall below their source-positive counterparts in
both directions. These are selected-procedure comparisons with different
learning rates and retained epochs. The new measurement supplies in-domain
retrieval evidence; it does not identify a target-allocation mechanism.

## Files

- `protocol.json`: saved before scoring; exact ordered image/caption IDs,
  caption strings, relevance, input and checkpoint hashes, methods, metrics,
  comparison plan, and numeric/tie conventions.
- `*_ranks.npz`: query-level ranks and best relevant scores/indices. Query IDs
  follow the image and caption orders in `protocol.json`.
- `per_run_results.json`: every run's metrics, provenance and paired differences.
- `summary.json` and `summary.csv`: means, seed values, sample standard deviations,
  and differences for R@1/5/10 in each direction.
- `execution_audit.json`: 814 independent full-sort query checks, plus input,
  checkpoint, protocol and rank-file integrity checks.
- `split_overlap_audit.json`: source hashes, exact split intersections, caption
  provenance, and feasibility of expanding to the full e-ViL test pool.

## Reproduction

Use the pinned environment in the repository's `REPRODUCE.md`. Restore the
original visual-entailment manifest, feature NPZ/metadata and selected adapter
checkpoints first. The lightweight Git snapshot omits the feature NPZ and
adapter weights; its recovery instructions explain how to recreate them.

Run `python scripts/evaluate_flickr_source_pool.py` from a working copy in which
`results/review_followup_flickr400/` has not yet been created. Preserve the
recorded output separately when rerunning. The script refuses to overwrite an
existing protocol or output file. It uses the original evaluator, float32
adaptation, float64 dot products and deterministic candidate-index tie order.

The standard Karpathy Flickr1k test overlaps the current training split by
42 images and validation by 3. It cannot be treated as held out for these
adapters. The full official e-ViL test has 1,000 disjoint image IDs and 5,000
available source captions; completing that pool requires 600 additional image
downloads and feature extraction. See the split audit before changing pools.

See `docs/REVIEW_GAP_AUDIT_20261003.md` for the scientific assessment and proposed
controls. Neither the original paper PDF nor its source has been rewritten by
this follow-up.
