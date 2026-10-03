#!/usr/bin/env python3
"""Create scientific figures and a hash-bound result handoff from frozen outputs.

This script performs no model selection, statistical tests or new evaluation.
All primary intervals come directly from the frozen 18-effect analysis.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate_study import sha256, write_json

BASE = ROOT / "results/review_followup"
FIGURES = BASE / "figures"
LRS = (0.0001, 0.0003, 0.001)
LR_LABELS = (r"$10^{-4}$", r"$3\!\times\!10^{-4}$", r"$10^{-3}$")
SEEDS = (17, 29, 43)
COMPARATORS = ("source", "count_only", "score_stratified")
LABELS = {"source": "Source targets", "supported": "Supported promotion",
          "count_only": "Count-matched random", "score_stratified": "Score-stratified random"}
COLORS = {"source": "#0072B2", "supported": "#8A448B", "count_only": "#D55E00", "score_stratified": "#009E73"}
MARKERS = {"source": "o", "supported": "D", "count_only": "s", "score_stratified": "^"}
DIRECTIONS = (("i2t.r1", "Image to text"), ("t2i.r1", "Text to image"))


def dump_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, stem):
    outputs = []
    for extension in ("pdf", "png"):
        path = FIGURES / f"{stem}.{extension}"
        metadata = {"Creator": "plot_review_followup_results.py", "CreationDate": None, "ModDate": None} if extension == "pdf" else {"Software": "Matplotlib scientific plotting"}
        figure.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        outputs.append(path)
    plt.close(figure)
    return outputs


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)


def primary_figure(contrasts, compact=False):
    rows = []
    lookup = {}
    for row in contrasts:
        key = (row["metric"], row["right"], row["learning_rate"])
        if key in lookup or row["epoch"] != 10 or row["dataset"] != "e_vil_test1000":
            raise ValueError("Unexpected or duplicate primary effect")
        lookup[key] = row
        rows.append({"effect_id": row["effect_id"], "learning_rate": row["learning_rate"], "epoch": row["epoch"],
                     "direction": row["metric"].split(".")[0], "comparator": row["right"],
                     "difference_percentage_points": 100 * row["difference"], "ci_lower_percentage_points": 100 * row["ci_lower"],
                     "ci_upper_percentage_points": 100 * row["ci_upper"], "confidence": row["confidence"],
                     "family_size": row["family_size"], "bootstrap_replicates": row["replicates"], "bootstrap_seed": row["bootstrap_seed"]})
    if set(lookup) != {(metric, comparator, lr) for metric, _ in DIRECTIONS for comparator in COMPARATORS for lr in LRS}:
        raise ValueError("Figure requires all 18 frozen primary effects")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65 if compact else 3.25), sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.30 if compact else .28, top=.88 if compact else .78, wspace=.13)
    for panel, (metric, title) in zip(axes, DIRECTIONS):
        style_axis(panel)
        panel.axhline(0, color="#343434", linewidth=.9, zorder=2)
        for offset, comparator in zip((-.16, 0, .16), COMPARATORS):
            records = [lookup[(metric, comparator, lr)] for lr in LRS]
            values = np.asarray([row["difference"] * 100 for row in records])
            lower = np.asarray([row["ci_lower"] * 100 for row in records])
            upper = np.asarray([row["ci_upper"] * 100 for row in records])
            panel.errorbar(np.arange(3) + offset, values, yerr=[values - lower, upper - values],
                           color=COLORS[comparator], marker=MARKERS[comparator], markersize=4.1,
                           linestyle="none", linewidth=1.1, capsize=2.4, capthick=.9, label=LABELS[comparator], zorder=3)
        panel.set_xticks(np.arange(3), LR_LABELS)
        panel.set_xlim(-.43, 2.43)
        panel.set_ylim(-16, 10)
        panel.set_yticks([-15, -10, -5, 0, 5, 10])
        panel.set_xlabel("Learning rate", labelpad=4)
        panel.set_title(title, fontsize=10, pad=7)
    axes[0].set_ylabel("Difference in Recall@1\n(percentage points)", labelpad=5)
    if not compact:
        fig.suptitle("Supported promotion minus each comparator", fontsize=10.5, y=.965)
        fig.text(.5, .885, "Matched epoch 10 on the full e-ViL test retrieval pool", ha="center", fontsize=8.2, color="#444444")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .008 if compact else .073), ncol=3, frameon=False, handletextpad=.35, columnspacing=1.5, fontsize=8)
    if not compact:
        fig.text(.5, .018, "99.722% Bonferroni intervals. Paired image bootstrap, conditional on the fitted seeds and assignment draws.", ha="center", fontsize=7.2)
    dump_csv(FIGURES / "primary_effects_data.csv", rows)
    return save_figure(fig, "primary_effects_paper" if compact else "primary_effects"), rows


def trajectory_figure(fixed):
    lookup = {}
    for row in fixed["cells"]:
        if row["method"] in ("source", "supported"):
            if not row["complete_seed_cell"]:
                raise ValueError("Trajectory is missing a training seed")
            for metric, _ in DIRECTIONS:
                lookup[(row["method"], row["learning_rate"], row["epoch"], metric)] = row["metrics"][f"e_vil_test1000.{metric}"]["mean"]
    for row in fixed["equal_draw_policy_means"]:
        for metric, _ in DIRECTIONS:
            lookup[(row["condition"], row["learning_rate"], row["epoch"], metric)] = row["metrics"][f"e_vil_test1000.{metric}"]["mean"]
    rows = []
    groups = ("source", "supported", "count_only", "score_stratified")
    for lr in LRS:
        for epoch in (1, 5, 10):
            for group in groups:
                for metric, _ in DIRECTIONS:
                    rows.append({"condition": group, "learning_rate": lr, "epoch": epoch, "direction": metric.split(".")[0],
                                 "recall_at_1_percent": 100 * lookup[(group, lr, epoch, metric)],
                                 "averaging": "equal assignment draws within training seed, then equal training seeds"})
    lower = 5 * math.floor(min(row["recall_at_1_percent"] for row in rows) / 5)
    upper = 5 * math.ceil(max(row["recall_at_1_percent"] for row in rows) / 5)
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 6.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.225, top=.88, wspace=.12, hspace=.32)
    for r, (lr, lr_label) in enumerate(zip(LRS, LR_LABELS)):
        for c, (metric, title) in enumerate(DIRECTIONS):
            ax = axes[r, c]
            style_axis(ax)
            for group in groups:
                values = [100 * lookup[(group, lr, epoch, metric)] for epoch in (1, 5, 10)]
                ax.plot([1, 5, 10], values, color=COLORS[group], marker=MARKERS[group], markersize=3.6,
                        linewidth=1.1, label=LABELS[group])
            ax.set_xticks([1, 5, 10])
            ax.set_xlim(.5, 10.5)
            ax.set_ylim(lower - 1, upper + 1)
            ax.set_title((title + ", " if r == 0 else "") + "learning rate " + lr_label, fontsize=9, pad=5)
            if c == 0:
                ax.set_ylabel("Recall@1 (%)")
            if r == 2:
                ax.set_xlabel("Epoch")
    fig.suptitle("Source-caption retrieval during matched training", y=.965, fontsize=11)
    fig.text(.5, .923, "Descriptive fixed-epoch means on the complete 1,000-image / 5,000-caption e-ViL test pool", ha="center", fontsize=8.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .059), ncol=2, frameon=False, fontsize=8, columnspacing=2)
    fig.text(.5, .024, "Every planned learning rate and epoch is shown. Three training seeds; randomized policies average three fixed assignment draws.", ha="center", fontsize=7.2)
    dump_csv(FIGURES / "fixed_trajectory_data.csv", rows)
    return save_figure(fig, "fixed_trajectories"), rows


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    analysis = BASE / "analysis"
    audit = json.loads((analysis / "analysis_audit.json").read_text())
    if audit["status"] != "passed" or audit["primary_effects"] != 18:
        raise ValueError("Complete audited follow-up analysis is required")
    primary = json.loads((analysis / "primary_contrasts.json").read_text())
    fixed = json.loads((analysis / "fixed_epoch_metrics.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42})
    output_files, primary_rows = primary_figure(primary["contrasts"])
    paper_files, _ = primary_figure(primary["contrasts"], compact=True)
    output_files += paper_files
    trajectory_files, _ = trajectory_figure(fixed)
    output_files += trajectory_files + [FIGURES / "primary_effects_data.csv", FIGURES / "fixed_trajectory_data.csv"]
    input_paths = [ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json", BASE / "state_manifest.json", analysis / "primary_contrasts.json",
                   analysis / "fixed_epoch_metrics.json", analysis / "analysis_audit.json", analysis / "preanalysis_receipt.json"]
    dataset_hashes = json.loads((analysis / "preanalysis_receipt.json").read_text())["dataset_hashes"]
    provenance = {"schema_version": 1, "plotting_source": str(Path(__file__).resolve().relative_to(ROOT)), "plotting_source_sha256": sha256(Path(__file__)),
                  "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                  "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
                  "dataset_hashes": dataset_hashes, "output_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in output_files},
                  "primary_interval_source": "frozen analysis values, no recomputation or new tests",
                  "trajectory_status": "descriptive; all planned learning rates/epochs/conditions retained", "significance_stars": False}
    write_json(FIGURES / "figure_provenance.json", provenance)
    captions = """# Follow-up figure captions

**primary_effects.pdf and primary_effects_paper.pdf.** Supported promotion minus each comparator on full e-ViL test-pool Flickr source-caption retrieval. All 18 planned epoch-10 contrasts are shown at matched learning rates. Points average three training seeds. Randomized comparators first average three fixed assignment draws within each seed. Bars are 99.722% Bonferroni intervals from 10,000 paired image-cluster bootstrap samples. Text-to-image resampling keeps all five captions of an image together. Intervals condition on the fitted seeds and assignment draws. The zero line denotes equal Recall@1.

**fixed_trajectories.pdf.** Descriptive Recall@1 at every planned learning rate and epoch for the four policy families. Both retrieval directions use the full 1,000-image / 5,000-caption pool. Randomized policies average assignment draws within seed before averaging seeds. Lines connect observed checkpoints and do not imply intermediate measurements. No extra confidence intervals or tests are introduced.
"""
    (FIGURES / "captions.md").write_text(captions)
    lines = ["# Complete review-follow-up results", "", "All 72 fits, 792 checkpoint records and 48 selection references are complete. The frozen evaluation covers 244 unique state references and 1,080 prediction archives. Independent aggregate recomputation passed 5,536 checks with maximum absolute discrepancy 2.22e-16.", "", "The primary family contains 18 epoch-10 effects. Each learning rate is reported. The intervals below are 99.722% conditional image-bootstrap intervals, with a Bonferroni family of 18.", "", "| Learning rate | Comparator | Direction | Supported minus comparator (pp) | 99.722% interval |", "|---:|---|---|---:|---:|"]
    for row in primary_rows:
        lines.append(f"| {row['learning_rate']:g} | {LABELS[row['comparator']]} | {row['direction']} | {row['difference_percentage_points']:+.3f} | [{row['ci_lower_percentage_points']:+.3f}, {row['ci_upper_percentage_points']:+.3f}] |")
    lines += ["", "At all three learning rates, supported promotion improves text-to-image Recall@1 over both randomized controls. Every one of those six intervals is above zero. Its image-to-text Recall@1 remains below source-only positive targets at all three learning rates. The corresponding source-target text-to-image intervals include zero. Against score-stratified random promotion, all three image-to-text intervals include zero; against count-matched random promotion, the interval is above zero only at the largest planned learning rate.", "", "These comparisons identify the specified positive-assignment interventions within the fixed adapter setting. Source targets retain every hypothesis as a competing candidate. The randomized controls use label-derived positive quotas. The intervals do not resample training seeds or assignment draws. Full cell values and both sources of observed variation remain in the analysis files.", "", "## Files", "", "- `analysis/all_state_metrics.csv`: every scored state and scalar endpoint.", "- `analysis/fixed_epoch_metrics.json`: all learning-rate/epoch/seed/draw cells and equal-draw policy means.", "- `analysis/primary_contrasts.json` and `primary_bootstrap_samples.npz`: all 18 effects and 180,000 bootstrap samples.", "- `analysis/selected_strategy_metrics.json`: native and source-retrieval strategies, with each random draw selected separately.", "- `analysis/e_vil_query_subgroups.json`: original 400 and additional 600 query summaries against the same full candidate pool.", "- `evaluation/native_replay_prediction_audit.json`: 259 bitwise-equal retained arrays for frozen and native source/supported replay.", "- `figures/`: vector PDF, 300 dpi PNG, source CSVs, captions and complete figure provenance.", "", "All new figure inputs are hash-bound in `figures/figure_provenance.json`. The evaluator, analyzer and protocol were not modified during scoring or figure generation.", ""]
    report = BASE / "COMPLETE_RESULTS.md"
    report.write_text("\n".join(lines))
    index_paths = [BASE / "evaluation" / f"{suite}_index.json" for suite in ("terminal", "selected", "trajectory")]
    predictions = {}
    for path in index_paths:
        index = json.loads(path.read_text())
        if index["status"] != "complete":
            raise ValueError("Incomplete evaluation suite")
        for state in index["runs"]:
            for record in state["datasets"].values():
                key, value = record["predictions"], record["predictions_sha256"]
                if key in predictions and predictions[key] != value:
                    raise ValueError("Inconsistent prediction hashes across suite indices")
                predictions[key] = value
    prediction_digest = hashlib.sha256("".join(f"{key}\t{value}\n" for key, value in sorted(predictions.items())).encode()).hexdigest()
    bound_files = sorted(analysis.glob("*")) + index_paths + [BASE / "evaluation/native_replay_prediction_audit.json", BASE / "training_final_audit.json", report, FIGURES / "figure_provenance.json", FIGURES / "captions.md"] + output_files
    independent_audits = sorted(BASE.glob("*audit*.json")) + sorted(analysis.glob("*audit*.json"))
    bound_files = sorted(set(path for path in bound_files + independent_audits if path.is_file()))
    digest = {"schema_version": 1, "status": "complete", "protocol_sha256": audit["protocol_sha256"],
              "state_manifest_sha256": sha256(BASE / "state_manifest.json"), "analysis_audit": audit,
              "evaluation_sources": {str(path.relative_to(ROOT)): sha256(path) for path in [ROOT / "scripts/evaluate_review_followup.py", ROOT / "scripts/analyze_review_followup.py", ROOT / "tests/test_review_followup_evaluation.py"]},
              "prediction_archive_count": len(predictions), "prediction_manifest_digest_sha256": prediction_digest,
              "prediction_manifest_digest_encoding": "UTF-8 sorted relative-path TAB archive-SHA256 NEWLINE",
              "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in bound_files}, "dataset_hashes": dataset_hashes}
    write_json(BASE / "complete_results_digest.json", digest)
    print(json.dumps({"status": "complete", "figures": [str(path.relative_to(ROOT)) for path in output_files],
                      "digest": str((BASE / "complete_results_digest.json").relative_to(ROOT)), "primary_effects": len(primary_rows)}, indent=2))


if __name__ == "__main__":
    main()
