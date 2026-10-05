SANW local execution

This wrapper runs unchanged frozen scientific programs.
It does not implement a new experiment or alter an existing contract.
Every execution phase prints commands by default.
Add --execute only after inspecting the printed commands.

Use x86_64 Linux or WSL2 with Docker.
Docker runs the linux/amd64 image explicitly.
ARM emulation can be substantially slower.
Build the CPU fitting and evaluation image:
  docker build --platform linux/amd64 -t sanw-v10-local - < local/Dockerfile

The prepare phase restores exact sources, inputs, and completed fits.
Inspect the plan, then execute it:
  bash local/docker.sh prepare
  bash local/docker.sh prepare --execute
  bash local/docker.sh preflight
  bash local/docker.sh status

Docker mounts this checkout at the historical absolute path.
This preserves old ledger paths without changing their bytes.
No host symlink or privileged Docker option is needed.
CPU fitting uses three BLAS threads.
Audits and evaluation use one BLAS thread.
New fits use local/cache/<encoder> for rebuildable score arrays.
Historical cache metadata remains unchanged.
An exclusive file lock prevents overlapping local wrapper executions.
Direct commands outside this wrapper do not share that protection.

Build the encoding image before fresh-encode:
  docker build --platform linux/amd64 -t sanw-v10-encoder - < local/Dockerfile.encoder

The encoder runtime differs from fitting and evaluation.
Its qualification subprocess uses a separate fitting runtime at the historical path.
Fresh encoding uses four Torch threads as specified by its frozen source.
The Dockerfiles reproduce recorded package versions.
The handoff receipt states whether the images were actually built and tested.

Run these phases separately, in order:
  rn-replicas
  audit-replicas
  lock-replicas
  preserve-development
  evaluate-replicas
  aggregate

Stop if aggregate fails.
If aggregate passes, continue:
  no-retention
  retrieval-only
  labclip
  prepare-benchmark
  core-lock
  audit-labclip
  supplementary-locks
  release-benchmark
  preserve-benchmark
  score-benchmark
  analyze-benchmark
  fresh-lock
  preserve-fresh-inputs
  fresh-prepare
  fresh-encode
  fresh-feature-lock
  fresh-release
  preserve-fresh
  fresh-score

Example:
  bash local/docker.sh rn-replicas
  bash local/docker.sh rn-replicas --execute

Never skip preserve phases.
They save exact states, locks, and sources in a separate archive.
They verify the archive and source identities before scoring.
They do not prove remote persistence.
Copy each preservation archive and receipt to separate durable storage before the next phase.
The preserve-fresh-inputs phase saves the 12-state lock before fresh input preparation.

Completed fit directories are reused; later frozen audits verify them.
Incomplete fits move into local/retired before restarting from epoch zero.
No partial directory is silently deleted.
No checkpoint is selected using held-out outcomes.
Existing output files cause a stop instead of an overwrite.
A partially completed multi-command phase needs inspection before restarting.
Do not delete earlier results to make a phase run.
Use the printed remaining scientific commands after checking their prerequisites.

Every executed phase records its status, runtime, and Git identity.
Each scientific child command also records its output and exit code.
Return a compact review bundle:
  bash local/docker.sh report --execute

Send the resulting local/reports archive and its JSON manifest.
For independent numerical review, create the full evidence bundle:
  bash local/docker.sh report --full-evidence
  bash local/docker.sh report --full-evidence --execute

The full bundle includes V10 states, checkpoints, predictions, paired arrays, and bootstrap arrays.
Both report modes include fresh completion records, metadata, and input manifests.
Full reports exclude fresh feature matrices, raw images, backbone weights, and rebuildable caches.
Send the archive and its JSON manifest.
Keep the complete local campaign until the independent review finishes.
The compact report does not contain every numerical array.

Source authority:
  results/practical_v10/REMAINING_EXECUTION_CHECKLIST_V2.txt
  results/practical_v10/SUPPLEMENT_EXECUTION_CHECKLIST.txt
  results/practical_v10/fresh_confirmation_execution_order_v1.md

The older checklist references private historical preservation receipts.
This wrapper uses new local preservation archives instead.
It does not recreate those private receipts or claim historical remote persistence.
