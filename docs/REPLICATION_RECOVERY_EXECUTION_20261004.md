# Fresh replication execution

Date: 2026-10-04.

The previous workspace no longer contains the reported replication outputs.
This execution recomputes both replications from the original frozen design.
It does not reconstruct missing results from conversation summaries.
The held-out datasets remain reused evaluation sets.

## Scope

| Setting | Fits | Frozen protocol | Output |
|---|---:|---|---|
| Nonlinear ViT | 45 | `docs/STRENGTHEN_REPLICATION_PROTOCOL.json` | `results/strengthen_replication` |
| Linear RN50 | 45 | `docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json` | `results/strengthen_second_encoder` |

Each setting uses five policies, three rates, and three seeds.
Each fit contains epochs zero through ten.
All three score-stratified draws remain in the analysis.
Both original development selectors remain unchanged.
The original 100-image validation subset belongs to the 900-image retrieval pool.

The nonlinear execution uses recovered ViT training features.
Its assignments retain the original bytes after a successful semantic audit.
The RN50 execution recomputes assignments from its recovered, pinned encoder features.
The plans bind all input, assignment, and implementation hashes.

| Setting | Ledger SHA256 |
|---|---|
| Nonlinear | `ed152bbd9038e10c37cb10ede3c9adc5bff173193b24fe83aef83ed5f8dd9879` |
| RN50 | `a88e93ea6b2ac025abdb109a7017a67d93ae2621f0e6bea031eca77840f25c9d` |

Full plans are in `recovery/current_replication/nonlinear_plan.json` and `recovery/current_replication/rn50_plan.json`.
The training logs use corresponding names ending in `_fit.log`.

## Training and selection commands

Run these commands from the repository root.
The environment limits each process to two CPU threads.
Completed candidates are verified before reuse.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# Run assignments, plan, fit, and select in this order.
# The existing assignments and plans are already complete.
for phase in assignments plan fit select; do
  .venv/bin/python scripts/run_strengthen_replication.py "$phase" \
    --setting nonlinear \
    --protocol-sha256 3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920 \
    --train-features results/resume_features/visual_entailment/features.npz \
    --train-metadata results/resume_features/visual_entailment/metadata.json || break
done

for phase in assignments plan fit select; do
  .venv/bin/python scripts/run_strengthen_replication.py "$phase" \
    --setting rn50 \
    --protocol-sha256 53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e || break
done
```

Do not launch a second fit while the corresponding process remains active.
Selection requires all 45 completion receipts.
It exports `evaluation_manifest.json` and the complete 495-record `state_manifest.json`.
The evaluation manifest contains the frozen reference, all terminal states, and all selected nonzero states.

## Evaluation configuration

Evaluation shares the allocation/distillation scoring backend.
It retains float32 feature normalization and float64 similarities.
The original tie rules and raw prediction formats remain unchanged.

Each configuration has this structure:

```json
{
  "encoder": "vit_b32",
  "datasets": {
    "e_vil_test1000": {
      "manifest": "data/review_followup/e_vil_test1000/manifest.json",
      "features": "results/review_followup/features/e_vil_test1000/features.npz",
      "metadata": "results/review_followup/features/e_vil_test1000/metadata.json"
    }
  }
}
```

The complete file must also include `visual_entailment`, `sugarcrepe`, `sugarcrepe_pp`, and `coco_karpathy`.
RN50 uses `encoder: rn50` and its separate feature root.
The input recovery agent prepares the final complete configuration files.
The loader checks every manifest, feature order, benchmark size, and encoder identity.

## Evaluation and analysis commands

Substitute the complete configuration path for `DATASET_CONFIG` below.
Use `--plan-only` for a check without scoring.
Actual scoring requires the separate execution authorization from the root agent.

```bash
.venv/bin/python scripts/evaluate_strengthen_replication.py \
  --setting nonlinear \
  --manifest results/strengthen_replication/evaluation_manifest.json \
  --protocol docs/STRENGTHEN_REPLICATION_PROTOCOL.json \
  --protocol-sha256 3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920 \
  --dataset-config DATASET_CONFIG \
  --output results/strengthen_replication/evaluation_recovered

.venv/bin/python scripts/analyze_strengthen_replication.py \
  --setting nonlinear \
  --index results/strengthen_replication/evaluation_recovered/index.json \
  --protocol docs/STRENGTHEN_REPLICATION_PROTOCOL.json \
  --protocol-sha256 3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920 \
  --output results/strengthen_replication/analysis_recovered

.venv/bin/python scripts/evaluate_strengthen_replication.py \
  --setting rn50 \
  --manifest results/strengthen_second_encoder/evaluation_manifest.json \
  --protocol docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json \
  --protocol-sha256 53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e \
  --dataset-config DATASET_CONFIG \
  --output results/strengthen_second_encoder/evaluation_recovered

.venv/bin/python scripts/analyze_strengthen_replication.py \
  --setting rn50 \
  --index results/strengthen_second_encoder/evaluation_recovered/index.json \
  --protocol docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json \
  --protocol-sha256 53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e \
  --output results/strengthen_second_encoder/analysis_recovered
```

Each setting has twelve primary contrasts.
They compare Supported against Source and all three score-stratified draws.
The analysis uses three rates and both retrieval directions at epoch ten.
It averages draws within optimizer seed before averaging the three seeds.
All five caption queries remain in their owning image cluster.
Intervals use 10,000 paired bootstrap samples and the original Bonferroni correction.
The nonlinear bootstrap seed is `20261013`.
The RN50 bootstrap seed is `20261014`.
Intervals condition on the fitted seeds, fixed draws, and fixed gallery.

## Checks completed before scoring

`tests/test_recovered_replication_analysis.py` checks six behaviors with synthetic inputs.
It verifies all thirty development selections, including frozen aliases and deterministic ties.
It rejects duplicate epoch records and unselected evaluation states.
It checks nonlinear scoring against direct stable ranks.
It independently gathers sampled image clusters for every bootstrap effect.
Both settings match across all 240,000 synthetic bootstrap samples.

```bash
.venv/bin/python -m unittest discover -s tests \
  -p test_recovered_replication_analysis.py -v
```

These checks do not substitute for the independent audit of actual results.
The final audit must inspect prediction archives, selections, and all twelve contrasts for each setting.
