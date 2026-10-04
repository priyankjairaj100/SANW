# Lossless epoch checkpoint archives

The archive helper preserves every original byte in verified ZIP parts.
Archive creation never deletes source files.
No scientific training or scoring code is imported.

## Retained states

Supply an explicit retained manifest and its SHA256.
Every checkpoint named in `states` remains local.
An independent retained manifest must bind the unchanged full state manifest.
The helper checks every selected and decomposition state against the retained list.
The original all-epoch manifest therefore allows no checkpoint removal.

The tested protocols are allocation-distillation v1 and retention v3.
The expected candidate grid comes from the ledger-bound protocol.
Matched controls use the primary selected policy, rate, and per-seed epochs.
Both selection hashes are required.
Every completion artifact is verified before archive creation.
This includes validation arrays, history files, and retained weights.
Unknown checkpoint files cause an error.

The older replication evaluation manifest has different frozen-state normalization.
That manifest is not supported by this workflow.

## Archive command

Run from the repository root after all candidates finish.
The verified ViT A+D retained manifest has SHA256:

```
88b59a3231f558bf5b729ba202d1cf2cc1540b268391794188f35baf87f1e608
```

```
python scripts/archive_completed_grid_checkpoints.py archive \
  --grid-root results/allocation_distillation/vit_b32 \
  --retained-manifest results/allocation_distillation/vit_b32/archival_keep_manifest.json \
  --retained-manifest-sha256 88b59a3231f558bf5b729ba202d1cf2cc1540b268391794188f35baf87f1e608 \
  --expected-completions 126 \
  --output-root recovery/epoch_archives_20261004 \
  --prefix SANW_AD_ViT_epoch_weights_20261004 \
  > recovery/archive_AD_ViT_public_index_20261004.json
```

Each ZIP includes `MANIFEST.json` with repository-relative paths, byte counts, and SHA256 digests.
Each part holds at most 128 MiB of uncompressed data, including its manifest.
CRC checks and member digest checks run before success is reported.
The public index contains names, digests, and byte counts.
It contains no remote file IDs.
Existing archive prefixes cannot be overwritten.

The archive command rechecks every protected file and source checkpoint after writing all parts.
Standalone verification also works after local source weights have been removed.

```
python scripts/archive_completed_grid_checkpoints.py verify \
  --manifest PATH_TO_ARCHIVE_MANIFEST \
  --manifest-sha256 ARCHIVE_MANIFEST_SHA256
```

## Separate upload and prune

The root agent owns uploads and deletion authorization.
All parts and the archive manifest must upload successfully before pruning.
The private upload receipt must cover exactly those files.
Each `results` item must contain these fields:

| Field | Required value |
|---|---|
| `local_path` | Exact absolute uploaded path |
| `status` | `succeeded` |
| `purpose` | `create_library_file` |
| `file_id` | Finalized remote file ID |
| `library_file_id` | Finalized remote Library ID |
| `bytes` | Uploaded source byte count |
| `sha256` | Uploaded source SHA256 |

Capture byte counts and digests before uploading.
Merge them with final successful results using exact paths.
The plain upload helper receipt lacks these digest fields.
That receipt alone cannot authorize pruning.

```
python scripts/archive_completed_grid_checkpoints.py prune \
  --manifest PATH_TO_ARCHIVE_MANIFEST \
  --manifest-sha256 ARCHIVE_MANIFEST_SHA256 \
  --upload-receipt PRIVATE_ENRICHED_UPLOAD_RECEIPT \
  --confirm-uploaded-and-authorize-deletion
```

Prune verifies all ZIPs again, including every member digest.
It repeats the complete grid audit and compares the entire archived snapshot.
Any changed manifest, retained checkpoint, completion artifact, or eligible set blocks deletion.
All checks finish before the first source file is removed.
The helper reserves a writable prune receipt before removal starts.
An interruption leaves an explicit `in_progress` receipt and intact archives.
Review that receipt before any recovery attempt.
The helper never deletes ZIP files.

## Verification completed

Eight synthetic safety tests passed on 2026-10-04.
They include a complete lossless archive and a prune using disposable byte fixtures.
They also test both matched-control schemas and stale uploaded digests.
Missing runs, corrupt validation files, unknown weights, missing selected states, and symlinks are rejected.
An all-epoch retained manifest archives no checkpoints.

The completed ViT A+D grid also passed a read-only preflight:

| Item | Count |
|---|---:|
| Completed candidates | 126 |
| Retained checkpoint paths | 166 |
| Eligible epoch checkpoints | 1,204 |
| Eligible original bytes | 2,529,507,652 |
| Protected files | 1,789 |

This preflight created no real archives and removed no research files.
