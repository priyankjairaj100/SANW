# Stage 13: completed training records and exact selected checkpoints

This checkpoint preserves the completed 72-candidate follow-up training record
and the exact checkpoint files referenced by both frozen selectors. It is
incremental: restore the original reconstructed project and stage12 inputs and
source before adding this checkpoint.

The protocol SHA256 is
`3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34`;
the execution-ledger identity is
`2ac7376a3212155099a987e6f73f63f0f9917a36846872afb71ba22d75b34f4c`.
The full state manifest records 72 candidates and all 792 epoch-zero-through-ten
states. Native and source-retrieval selectors contribute 24 selections each.

## What this checkpoint actually contains

- All 72 `history.json` and 72 `completion.json` records.
- The frozen ledger, complete state manifest, both selector files, execution
  plan, replay-parity summary, final training audit, fit-progress record, and
  completed training/control test receipt.
- All **32 distinct selected checkpoint files** referenced by the 48 selections,
  at their original relative paths. These comprise 23 epoch-zero and nine
  nonzero files. Files with equal model tensors but different checkpoint
  metadata are retained separately, preserving each original SHA256.
- The current paper outline, reproduction guide, protocol/freeze, and an
  explicit included/omitted inventory with the complete selector-to-file map.

The training audit reports all 72 completed candidates, all 792 checkpoint
hashes/payloads checked, and exact replay of 18 original source/supported
candidates: 198 native-validation epoch records and 18 original best states.
These are training/development checks, not follow-up held-out findings.

## What is deliberately outside the ZIP

The other **760 epoch checkpoint files** remain untouched in the working
project. They are listed by state ID, path, byte count, and recorded SHA256 in
`results/review_followup/recovery_stage13_inventory.json`; the complete state
manifest also retains their identities. The 792 saved per-epoch validation NPZ
files are omitted and listed separately. Completion records still name these
artifacts, so their presence alone does not assert complete local recovery.

Raw images, model weights, earlier base inputs, stage12 feature caches, and
active follow-up evaluation logs/predictions are not included again. Restore
their earlier checkpoints or follow the pinned acquisition instructions where
needed. No partially written evaluation artifact is part of this checkpoint.

## Extract and verify the packaged scope

The archive uses the common `project/` prefix. Extract into the same recovery
parent as the preceding checkpoints; source and selected-state paths then join
the existing project tree.

```bash
python -m zipfile -e checkpoint-13-completed-training-selected-states.zip recovery
python recovery/verify_recovery.py --archive checkpoint-13-completed-training-selected-states.zip
python recovery/verify_recovery.py --root recovery --allow-unlisted
```

The verifier checks exactly this ZIP's manifest. `--allow-unlisted` permits the
earlier project files in a combined extraction. Keep the immutable ZIP to retain
this snapshot even as the live project gains later evaluation results.

## Recovering states and reproducing omitted artifacts

Every packaged selected state is an actual original PyTorch checkpoint. Its
bytes and state-manifest hash can be verified without retraining. Use the
selector-to-file map to locate it; do not infer a selected state from a rounded
metric or substitute a different epoch-zero payload.

The frozen `evaluate_review_followup.py` deliberately hashes **all 792 states
before choosing a suite**, including for `--suite selected`. Consequently the
current strict evaluation CLI cannot run from this selected-only checkpoint
set. The exact selected tensors are recoverable, but that CLI additionally
requires the remaining states. Its guard is preserved unchanged during the
current execution.

To reconstruct unbundled training states or validation NPZs, use a separate
reproduction copy of the original project plus stage12, with the pinned
environment, exact original features, all six assignments, and the frozen
follow-up code. Do not copy this checkpoint's partial candidate folders into
that fresh reproduction output before fitting: completion-file validation
correctly rejects candidates whose declared artifacts are missing.

```bash
FOLLOWUP_PROTOCOL_SHA=3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/run_review_followup.py plan --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py fit --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
python scripts/run_review_followup.py select --protocol-sha256 "$FOLLOWUP_PROTOCOL_SHA"
```

Run all declared candidates; do not choose a preferred learning rate or control
draw from test outcomes. Compare reconstructed histories, tensors, selector
identities, and file hashes against the preserved records. Exact recovery is
guaranteed for the packaged bytes; a fresh execution should be verified against
the recorded environment and audits before being treated as equivalent.

See `docs/REVIEW_FOLLOWUP_README.md` for the full scientific design, evaluation
suites, and the separate 18-effect inferential family. This checkpoint makes no
claim about evaluation that was still running when it was created.
