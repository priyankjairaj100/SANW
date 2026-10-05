# SANW — local empirical program

This handoff targets an ACL 2027 short paper.
All new empirical work runs on your local machine.
Return the evidence packages here for analysis and paper development.

**Empirical superiority remains unproven.** Four joint fits are complete.
RN50 seeds 29 and 43 remain unfinished.
The next decision is the fixed three-seed development gate.

## Start

Use Bash with Docker on Linux, WSL2, or Docker Desktop.
The reference image runs Linux AMD64.

```bash
git clone --branch local-handoff-20261005 https://github.com/priyankjairaj100/SANW.git
cd SANW
docker build --platform linux/amd64 -t sanw-v10-local - < local/Dockerfile
bash local/docker.sh prepare --execute
bash local/docker.sh preflight
bash local/docker.sh status
```

`prepare` restores the exact scientific tree, training inputs, and four completed fits.
The download includes roughly 1.42 GB of checked archive payloads.
Allow additional space for Git objects, Docker images, restored inputs, caches, and results.
A conservative working allowance is 50 GB free disk and 16 GB RAM.
This is a planning allowance, not a certified minimum.

Give your local LLM [AGENTS.md](AGENTS.md).
Follow [the exact operator commands](docs/local_handoff/OPERATOR_COMMANDS.txt).
Read [the scientific experiment program](docs/local_handoff/EMPIRICAL_PROGRAM_FROZEN_V10.txt) before execution.

## First milestone

Run these phases separately and check each exit code:

```bash
bash local/docker.sh rn-replicas --execute
bash local/docker.sh audit-replicas --execute
bash local/docker.sh lock-replicas --execute
bash local/docker.sh preserve-development --execute
```

Copy the preservation archive and receipt outside this checkout before scoring.
Then run:

```bash
bash local/docker.sh evaluate-replicas --execute
bash local/docker.sh aggregate --execute
bash local/docker.sh report --full-evidence --execute
```

Send the printed report archive and manifest back to the research chat.
Stop this V10 recipe if its aggregate gate fails.
Do not replace seeds or relax thresholds.

If the gate passes, the operator guide covers 18 control fits, benchmark evaluation, and fresh confirmation.
Read [the proposed extensions](docs/local_handoff/EMPIRICAL_EXTENSIONS.txt) before making broader superiority claims.
Those extensions require separate implementation and protocols.

## Included context and evidence

- [Current scientific status](docs/local_handoff/CURRENT_STATUS.txt).
- [Theory and novelty context](docs/local_handoff/SCIENTIFIC_CONTEXT.txt).
- [Results to return](docs/local_handoff/RESULTS_TO_RETURN.txt).
- Exact frozen code and 4,805 scientific files through the checked snapshot.
- Seven original input archives and four complete fit histories.
- Source manuscripts, current paper draft, earlier evidence, and audit receipts.
- Pinned public acquisition for fresh images and pretrained weights.

Scientific files are restored at their original repository paths.
Large feature arrays and fit histories use separate exact archives.
Private transfer records and chat references are excluded.
Rebuildable score caches are generated locally.
The recovered RN50 seed 29 partial run was not available in the saved archives.
Its completed replacement must start from epoch zero.

## Validation limits

The [workflow receipt](local/WORKFLOW_VALIDATION.json) records tests, command checks, and source identities.
The [independent review](local/RECOVERED_HANDOFF_REVIEW.json) states its exact scope.
Docker images could not be built here because Docker is unavailable.
Build-time assertions check the pinned runtime before local execution.
No new empirical results were produced while preparing this handoff.
