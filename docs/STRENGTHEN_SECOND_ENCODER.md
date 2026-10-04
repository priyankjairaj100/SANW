# Independent frozen encoder: OpenAI CLIP RN50

The second encoder changes both image architecture and pretraining relative to
the original LAION2B ViT-B/32. It uses the publicly released OpenAI CLIP RN50
checkpoint and the original study's unchanged image/text manifests. The same
test examples are reused to study model dependence; they are not a new dataset.

## Identity and preprocessing

- Architecture: OpenAI CLIP modified ResNet-50 image encoder and 12-layer text
  transformer, with QuickGELU and a 1,024-dimensional shared embedding.
- Checkpoint: `RN50.pt`, 255,827,503 bytes.
- SHA-256: `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`.
- Official URL:
  `https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt`.
- Official checkpoint registry: `https://github.com/openai/CLIP/blob/main/clip/clip.py`.
- Runtime: OpenCLIP 2.32.0, PyTorch 2.7.1+cpu, float32 inference, evaluation mode,
  frozen parameters, L2-normalized features, deterministic algorithms enabled.
- Images: RGB, bicubic resize of the shorter side to 224, 224-by-224 center crop,
  mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation
  `(0.26862954, 0.26130258, 0.27577711)`.
- Text: OpenCLIP's RN50 tokenizer. Only positions beyond the last EOT in the
  entire batch are omitted. Causal attention makes those positions irrelevant
  to the EOT feature. A stock-text parity check is recorded for each manifest.
- Fixed checkpoint logit scale: `100.00000762939453`.

## Reproduction

Download the checkpoint from the URL above to `data/model_rn50_openai/RN50.pt`.
The extraction script refuses a different checkpoint hash.

```sh
python scripts/strengthen_extract_second_encoder.py \
  --dataset visual_entailment \
  --dataset review_followup/e_vil_dev900 \
  --dataset review_followup/e_vil_test1000 \
  --dataset sugarcrepe \
  --dataset sugarcrepe_pp \
  --dataset coco_karpathy \
  --threads 4
```

This writes `results/strengthen_second_encoder/features/<dataset>/features.npz`
and `metadata.json`. NPZ keys match the original study: `image_features`,
`text_features`, `image_ids`, and `text_ids`. The feature dimension is 1,024.
Metadata records model identity, normalization, preprocessing, manifest and
feature hashes, runtime versions, logit scale, and extraction-source hash.

The SQLite bank keys images by exact image bytes and text by exact UTF-8 text.
It commits each batch and permits interruption/resume. Completed exports are
reused only after identity and file-hash verification. The initial run removed
process-specific function memory addresses from the preprocessing representation
in its signature; `cache_metadata_normalization.json` records this metadata-only
change and no feature values were changed.

Feature extraction does not calculate downstream retrieval or caption-choice
outcomes. Model fitting, random assignment controls, and statistical families
are specified separately by the replication protocol.

## Reconstruction on 2026-10-04

The saved repository contained this protocol but did not contain its named
extraction script, RN50 weights, feature bank, or completed RN50 feature exports.
`scripts/strengthen_extract_second_encoder.py` was reconstructed from this
protocol and `scripts/extract_features.py`; this reconstruction does not imply
that any earlier RN50 extraction or fit completed. Newly generated feature
metadata identifies the reconstruction and binds the extraction source hash.

The official checkpoint was reacquired and passed the exact size and SHA256
checks above. `data/model_rn50_openai/acquisition_receipt.json` records the
source and verification. All installed distributions in
`requirements-reproduction.txt` matched their preserved version pins at the
start of this run; `recovery/resume_20261004/rn50_runtime_diff.json` records that
comparison. Actual Python, operating-system, device, and package versions are
also saved in `results/strengthen_second_encoder/runtime.json` and each export.

The reconstructed script supports an input inventory without model loading:

```sh
python scripts/strengthen_extract_second_encoder.py \
  --dataset visual_entailment --preflight
```

`--text-only` can fill the same transactional bank while original image bytes
are being restored. It writes a stock-text parity receipt for each manifest
but does not create a completed feature export. The normal command above then
reuses those vectors and exports the full image/text arrays. Image keys use
the SHA256 of actual bytes, checking each manifest's declared image size and
hash where supplied; text keys use exact UTF-8 strings. Export metadata binds
the ordered input-content receipt as well as the original manifest.

Bank reuse requires an identical encoder signature, including actual runtime,
preprocessing, text encoding mode, and extraction-source hash. Completed exports
are checked for manifest identity, feature-file hash, exact row IDs, float32
values, finite entries, and L2 normalization before reuse. Preprocessing
signatures use the explicit configuration rather than process-specific function
addresses. This avoids the metadata normalization issue described above without
modifying any older feature values.

`elapsed_seconds` measures the exporting invocation's work on that manifest;
it is not a full encoding-time benchmark when the bank already contains vectors.
The execution logs and `execution_recovery.json` preserve earlier bank fills,
interruptions, and resumes. In particular, a disk-full interruption during the
text-only phase was followed by a successful SQLite integrity check and reuse
of all 45,066 previously committed vectors.
