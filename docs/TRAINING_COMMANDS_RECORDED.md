# Commands for the completed reconstructed training grid

Run from the project root after creating the pinned environment and reproducing
the frozen feature cache. `python` must resolve to that environment. The
recorded frozen protocol SHA256 is:

```text
066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847
```

The following commands reproduce the complete 108-candidate grid and the
36 selected checkpoints. The original execution used disjoint method workers;
sequential execution has the same scientific configuration and per-candidate
random streams.

```bash
python scripts/run_study.py fit \
  --protocol docs/RERUN_PROTOCOL.json \
  --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 \
  --learning-rates 0.0001 0.0003 0.001 \
  --threads 2 \
  --output results/study

python scripts/run_study.py select \
  --protocol docs/RERUN_PROTOCOL.json \
  --protocol-sha256 066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847 \
  --learning-rates 0.0001 0.0003 0.001 \
  --threads 2 \
  --output results/study

python scripts/audit_training_outputs.py \
  --study results/study \
  --output results/training_final_audit_reproduced.json
```

Default inputs are `data/visual_entailment/manifest.json`,
`results/features/visual_entailment/features.npz`, and its sibling
`metadata.json`. Defaults fix seeds 17, 29, and 43, ten epochs, and image batches
of 32. The saved ledger binds all source, input, protocol, configuration, and
runtime identities. Selection uses only the validation split. Held-out
evaluation is a separate command and follows completed selection.

On the archived project, `fit` verifies and skips existing compatible completed
candidates. It does not retrain them. To deliberately execute all candidates
again, use the same new output directory for both phases, such as
`--output results/study_repeat`, and pass that directory to the audit. Do not
delete or overwrite the archived execution to force a rerun.

An interrupted candidate without its final completion receipt restarts from
epoch zero. A completed candidate is skipped only after its ledger identity and
the SHA256 values of `best.pt` and `history.json` match its completion record.
Changed input files, core fitting sources, protocol, or runtime identity cause
an error when reusing an existing output directory. The protocol guard also
rejects altered scientific settings even when the caller supplies the correct
protocol digest.

The completed audit is `results/training_final_audit.json`. It verifies all
108 candidate receipts, 1,188 epoch histories, the shared learning-rate and
earliest-epoch selection rules, all 36 copied checkpoints, and independent
validation recomputation from the selected states. All 21 audit checks passed;
the maximum independently recomputed per-image validation metric discrepancy
was zero. Five selected checkpoints retained epoch zero. No held-out metric
or historical aggregate was read by this audit.
