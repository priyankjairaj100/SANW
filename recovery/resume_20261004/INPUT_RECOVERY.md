# Frozen-input recovery, 4 October 2026

The source manifests and original feature metadata were preserved unchanged.
All 3,200 unique Flickr image paths and 6,377 unique COCO image paths were
reacquired from the existing pinned sources and matched their original SHA256s.
`final_input_audit.json` is the final independent byte audit: no failures.
The 605,143,316-byte ViT model matched the original SHA256
`ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`.
All four original Flickr annotation inputs also matched their original hashes.

Exact original SugarCrepe cache archives were located by name in the saved file
collection, but the supported download route repeatedly returned HTTP 502.
The five checkpoint-05 split archives containing original visual-entailment
and COCO feature caches were not found by exact-title/asset-title search.
Consequently a new encoding execution was required; its outputs are separate.

## New feature execution

`run_vit_reencode.py` invokes a byte-identical copy of the original
`scripts/extract_features.py` under `encode_copy/scripts/`, with the same
pinned encoder, manifests, batch sizes and four CPU threads. Only its filesystem
root is redirected through symlinks. Outputs are under `results/resume_features/`;
the original `results/features/*/metadata.json` files are never replaced.

`audit_reencoded_features.py` records each new NPZ/container hash, per-array
C-order byte hashes, shapes/dtypes, exact original-container comparison and
numerical parity against every shared row of the preserved dev900/test1000
feature caches. The output `reencoded_feature_audit.json` distinguishes completed
and still pending datasets. An exact original SHA match may be used to recover
that old file; a different SHA remains a new execution input with provenance.

## Acquisition receipts and transient failures

`flickr_bulk_receipt.json` records every recovered Flickr image and the observed
whole-archive SHA. The pinned Hugging Face ETag is an object identity and differs
from the file SHA; each selected member was checked by original CRC32, byte size
and SHA256. The temporary whole archive was deleted after successful extraction.
`coco_pooled_receipt.json` records successful restoration of every COCO image.

An earlier per-image transfer was interrupted by disk exhaustion caused by
repeated filesystem-sync staging copies of the temporary bulk download. Those
reproducible incomplete staging copies were removed, freeing about 25 GB;
completed model, image and feature files were preserved. Earlier
`*_acquisition.log`/`*_acquisition_receipt.json` files may contain these transient
failures. They are superseded for input completeness by `final_input_audit.json`.

The `helpers/`, `downloads/`, `library-download-*.json`, `encode_copy/`, bulk
archives and `.rsync-tmp` are runtime materialization/execution aids, not paper
source or scientific evidence to stage in Git. Preserve the small recovery
scripts, final receipts and new feature archives as directed by the root task.
