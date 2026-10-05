#!/usr/bin/env python3
"""Summarize already-computed results; never train or evaluate a model."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text())


def identity(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    names = {"vit_b32": "ViT-B/32", "rn50": "RN50"}
    summaries = {}
    evidence = []
    for family in ("pooled", "token"):
        base = f"results/practical_v6/{family}_evaluation/analysis/"
        audit_path = (base + "independent_audit.json" if family == "pooled"
                      else "results/practical_v6/audits/token_heldout_independent_audit.json")
        analysis = read(base + "analysis.json")
        audit = read(audit_path)
        assert audit["passed"] is True
        assert audit["analysis_sha256"] == identity(base + "analysis.json")["sha256"]
        assert not analysis["cross_encoder_passed"]
        summaries[family] = {"analysis": analysis, "audit": audit}
        evidence.extend([identity(base + "analysis.json"), identity(audit_path)])

    diagnostic_path = "results/practical_v7/alpha_resolution_diagnostic_v1/result.json"
    diagnostic = read(diagnostic_path)
    stop_path = "results/practical_v7/pilot_development_decision.json"
    assert diagnostic["official_stop_decision_unchanged"] == identity(stop_path)
    assert diagnostic["candidate_count"] == 728
    assert diagnostic["coarse_grid_candidates_reproduced_exactly"] == 48
    assert diagnostic["held_out_data_read"] is False
    counts = {encoder: sum(x["encoder"] == encoder for x in diagnostic["feasible_candidates"])
              for encoder in names}
    assert counts == {"vit_b32": 16, "rn50": 0}
    evidence.extend([identity(stop_path), identity(diagnostic_path),
                     identity("results/practical_v7/pilot_checkpoint_integrity_audit.json"),
                     identity("results/practical_v7/final_interpretation_audit.json")])
    combined = {
        "schema": "sanw_practical_goal_v6_v7_summary_v1",
        "practical_goal_met": False,
        "evaluation_status": "exploratory_after_historical_test_reuse",
        "new_fits": {"pooled_v6": 16, "token_v6": 6, "source_pair_token_v7": 2, "total": 24},
        "independent_audits_passed": True,
        "evaluated_prediction_archives": sum(v["audit"]["counts"]["prediction_archives"] for v in summaries.values()),
        "evaluated_effects": sum(v["audit"]["counts"]["effects"] for v in summaries.values()),
        "verified_bootstrap_values": sum(v["audit"]["counts"]["bootstrap_values"] for v in summaries.values()),
        "v6_gates": {k: v["analysis"]["gates"] for k, v in summaries.items()},
        "v7_diagnostic_feasible_counts": counts,
        "v7_replication_performed": False,
        "v7_heldout_evaluation_performed": False,
        "evidence": evidence,
    }
    (ROOT / "results/practical_v7/practical_goal_summary.json").write_text(json.dumps(combined, indent=2) + "\n")

    lines = [
        "# Practical-goal results: v6 and v7",
        "",
        "The practical goal remains unmet. No evaluated family passes the required gate on both encoders.",
        "Both v6 independent audits passed. Audit success verifies computation; it does not imply scientific success.",
        "",
        "## Unchanged requirements",
        "",
        "The same family must pass on ViT-B/32 and RN50, using nonzero trained seeds 17, 29, and 43.",
        "Both e-ViL full-gallery retrieval lower bounds must exceed -1 percentage point against the frozen model.",
        "The SugarCrepe++ both-positive correctness improvement lower bound must exceed zero.",
        "COCO retrieval is secondary. Equality at a boundary fails.",
        "",
        "Intervals use 100,000 paired image-cluster bootstrap draws and a fixed mean of three selected seeds.",
        "The adjustment uses family size 80, familywise alpha 0.05, and two-sided tail probability 0.0003125.",
        "These intervals condition on the selected states. They do not estimate variability across new training seeds.",
        "The results remain exploratory after historical test reuse.",
        "The stated adjustment does not cover every adaptive choice across the research history.",
        "",
        "## Primary benchmark results",
        "",
        "Values are changes in percentage points against frozen features, followed by adjusted intervals.",
        "I2T means image-to-text retrieval; T2I means text-to-image retrieval. Retrieval uses Recall@1.",
        "SugarCrepe++ requires both valid captions to outrank the negative caption.",
        "",
        "| Family | Encoder | e-ViL I2T | e-ViL T2I | SugarCrepe++ | Gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    def cell(e):
        return f"{100*e['difference']:+.3f} [{100*e['ci_lower']:+.3f}, {100*e['ci_upper']:+.3f}]"
    for family, data in summaries.items():
        effects = {(e["encoder"], e["dataset"], e["metric"]): e for e in data["analysis"]["effects"]}
        for encoder, name in names.items():
            vals = [cell(effects[(encoder, "e_vil_test1000", "i2t.r1")]),
                    cell(effects[(encoder, "e_vil_test1000", "t2i.r1")]),
                    cell(effects[(encoder, "sugarcrepe_pp", "both_accuracy")])]
            lines.append(f"| {family} v6 | {name} | " + " | ".join(vals) + " | Fail |")
    lines += [
        "",
        "RN50 token adaptation establishes both retrieval gains under the specified exploratory intervals.",
        "Its caption gain is positive, but its lower bound remains below zero.",
        "ViT token adaptation does not establish retrieval retention or a caption gain.",
        "Failure to establish retention does not prove a loss larger than one percentage point.",
        "",
        "## Secondary COCO results",
        "",
        "| Family | Encoder | COCO I2T | COCO T2I | Retention gate |",
        "|---|---|---:|---:|---|",
    ]
    for family, data in summaries.items():
        effects = {(e["encoder"], e["metric"]): e for e in data["analysis"]["effects"] if e["dataset"] == "coco_karpathy"}
        for gate in data["analysis"]["gates"]:
            encoder = gate["encoder"]
            lines.append(f"| {family} v6 | {names[encoder]} | {cell(effects[(encoder, 'i2t.r1')])} | "
                         f"{cell(effects[(encoder, 't2i.r1')])} | {'Pass' if gate['secondary_coco_retention'] else 'Fail'} |")
    lines += [
        "",
        "Both token models retain secondary COCO retrieval under the specified bounds.",
        "This secondary result does not replace either primary requirement.",
        "",
        "## Source-pair development experiment",
        "",
        "V7 trains the final text block using two distinct source captions and a contradiction.",
        "Both positives describe the same image; their meanings need not be equivalent.",
        "Training focuses on triplets within the correction's frozen-margin range.",
        "Retrieval training remains active when a batch contains no eligible composition triplets.",
        "One conflicting normalized negative was excluded. Digits were preserved during normalization.",
        "",
        "Two seed-17 pilots completed. Neither encoder had an eligible candidate on the original six-amplitude grid.",
        "The official development stop remains unchanged. Seeds 29 and 43 were not trained for v7.",
        "No v7 held-out evaluation occurred.",
        "",
        "A separately recorded diagnostic examined alpha=0.10, 0.11, ..., 1.00 across four nonzero epochs.",
        "It reused canonical development deltas, with no new training or encoding.",
        "All 728 candidates were computed; all 48 original coarse-grid results reproduced exactly.",
        "ViT had 16 feasible points. RN50 had zero among its 364 tested points.",
        "The best ViT diagnostic point used epoch 4 and alpha 0.46.",
        "Its composition gain was +0.3424 points; retrieval lower bounds were -0.8889 and -0.2667 points.",
        "This diagnostic does not provide a three-seed result or satisfy the both-encoder requirement.",
        "It does not prove infeasibility at every real-valued alpha or for another training method.",
        "",
        "## Completed work and verification",
        "",
        "This continuation completed 24 fits: 16 pooled v6 fits, six token v6 fits, and two v7 pilots.",
        "Both v6 families were locked before either began new benchmark evaluation.",
        "The evaluations produced 48 prediction archives and 20 effects, with 2,000,000 bootstrap values independently verified.",
        "Independent audits recomputed full-gallery rankings, caption correctness, and all bootstrap results.",
        "Frozen baseline parity was checked across families. Canonical inference avoids batch-dependent floating-point ties.",
        "Training integrity checks confirmed nonzero learned updates and unchanged frozen reference tensors.",
        "",
        "The pooled scorer uses a bounded learned correction to frozen feature cosine.",
        "The token scorer uses a bounded correction from the final text block and layer normalization.",
        "Images, earlier text blocks, and text projection remain frozen.",
        "Neither method uses task-specific scoring or benchmark labels during inference.",
        "",
        "ViT-B/32 uses LAION2B weights, not OpenAI ViT weights. RN50 uses OpenAI weights.",
        "Pinned revisions, checksums, selections, and inference rules are recorded in the protocol files.",
        "",
        "## Remaining research problem",
        "",
        "The tested methods have not jointly established caption improvement and retrieval retention across both encoders.",
        "More amplitude tuning of these pilots is not justified by the completed diagnostic.",
        "A useful next experiment would change the training signal, then validate on development data before replication.",
        "Semantically matched positive pairs and verified minimal negative edits are one untested direction.",
        "This is a research hypothesis, not an established remedy.",
        "A fresh independent test sample would be needed for a new confirmatory claim.",
        "The present failures do not establish that the practical goal is impossible.",
        "",
        "## Replay and preservation",
        "",
        "Recorded repository branch: `practical-goal-20261004` in `priyankjairaj100/SANW`.",
        "The last recorded source snapshot before the final results was commit `cf676a31a09dfc52667bdb3250f9c5740831f9e4`.",
        "Automatic approval review rejected final repository publication. This restored report and compact evidence have not been published in a final commit.",
        "",
        "Retained compact evidence paths:",
        "",
        "- `results/practical_v6/cross_family_lock.json`: pre-evaluation lock for both families.",
        "- `results/practical_v6/pooled_selection/selection_lock.json`: pooled production selection.",
        "- `results/practical_v6/token_final_selection/evaluation_lock.json`: token production selection.",
        "- `results/practical_v6/{pooled,token}_evaluation/analysis/`: analysis summaries and the pooled independent audit receipt; raw predictions and bootstrap arrays are unavailable.",
        "- `results/practical_v6/audits/token_heldout_independent_audit.json`: token independent audit.",
        "- `results/practical_v7/token_source_pair_protocol_v1.json`: v7 fitting and continuation rules.",
        "- `results/practical_v7/pilot_development_decision.json`: unchanged official v7 stop.",
        "- `results/practical_v7/alpha_resolution_diagnostic_v1/`: finer-grid summaries and results; raw prediction arrays are unavailable.",
        "- `results/practical_v7/practical_goal_summary.json`: machine-readable outcome and evidence hashes.",
        "",
        "All attempts to save the research archives failed before a workspace reset.",
        "The reset removed fitted checkpoints, raw prediction arrays, bootstrap samples, and archive bundles; these artifacts are currently unavailable.",
        "Compact source code, analysis summaries, protocol and selection records, and audit receipts were restored from retained session data.",
        "The numerical audit statements above describe checks completed before this loss. Their receipts retain evidence of that work; the audits have not been replayed against the lost artifacts after recovery.",
        "Recorded hashes identify missing artifacts but do not recover them. Numerical replay and use of the selected trained models require recovering the original artifacts or performing a separately recorded rerun.",
        "The loss and its impact are recorded in `results/practical_v7/workspace_loss_impact_20261005.json`.",
        "",
        "Evidence identities:",
        "",
    ]
    for item in evidence:
        lines.append(f"- `{item['path']}` — SHA-256 `{item['sha256']}`.")
    out = ROOT / "docs/PRACTICAL_GOAL_RESULTS_20261004.md"
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(out), "summary": combined}, indent=2))


if __name__ == "__main__":
    main()
