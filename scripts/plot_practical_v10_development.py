#!/usr/bin/env python3
"""Render audited v10 seed17 development summaries; never open outcome arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "vit_b32": ("results/practical_v10/official_development/joint_vit_b32_seed17/result.json",
                "09dba4d91b669cbab1cd6c41d4070b65ba39e4fc2ffb39a09e86f9aade5d54b3"),
    "rn50": ("results/practical_v10/official_development/joint_rn50_seed17/result.json",
             "696080db41576a7ea538af33660d91c553ce0259baca931c837e1bc158dbe8c5"),
}
AUDIT = "results/practical_v10/official_development/independent_actual_result_audit_seed17_v1.json"
PANELS = (("i2t", "A  Image-to-text\nretrieval (R@1)", True),
          ("original", "B  Source + supported\njoint correctness", False),
          ("t2i", "C  Text-to-image\nretrieval (R@1)", True),
          ("source_pair", "D  Two-source\njoint correctness", False))
ENCODERS = ("vit_b32", "rn50")
NAMES = ("ViT-B/32 (LAION2B)", "RN50 (OpenAI)")
COLORS = ("#2166ac", "#b35806")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summaries():
    audit_path = ROOT / AUDIT
    audit = json.loads(audit_path.read_text())
    if (audit.get("study") != "sanw_practical_v10_official_development_independent_audit"
            or audit.get("audit_passed") is not True or audit.get("replication_gate_passed") is not True):
        raise ValueError("Require the successful independent seed17 development audit")
    audited = {row["encoder"]: row for row in audit["fits"]}
    if set(audited) != set(ENCODERS):
        raise ValueError("Independent audit must cover both encoders")
    results = {}
    for encoder, (name, expected) in INPUTS.items():
        path = ROOT / name
        if digest(path) != expected or audited[encoder]["result"]["sha256"] != expected:
            raise ValueError("Figure input differs from independently audited summary")
        value = json.loads(path.read_text())
        if (value.get("study") != "sanw_practical_v10_official_development" or value.get("encoder") != encoder
                or value.get("seed") != 17 or value.get("passed") is not True or audited[encoder].get("audit_passed") is not True):
            raise ValueError("Require both fixed seed17 development results")
        for metric, _, _ in PANELS:
            effect = value["effects"][metric]
            if (effect["replicates"] != 100000 or effect["family_size"] != 80 or effect["bootstrap_seed"] != 20261007
                    or effect["tail_probability"] != .0003125 or effect["familywise_alpha"] != .05
                    or any(effect[key] != audited[encoder]["effects"][metric][key] for key in ("difference", "ci_lower", "ci_upper"))):
                raise ValueError("Effect or bootstrap contract differs from audited summaries")
        results[encoder] = value
    return results, {"path": AUDIT, "sha256": digest(audit_path)}


def render(output):
    results, audit = load_summaries()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "sanw-v10-seed17-development", "axes.labelcolor": "#343a40",
                         "text.color": "#222222", "axes.titlepad": 13})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.1))
    plotted = {}
    for ax, (metric, title, retrieval) in zip(axes.flat, PANELS):
        plotted[metric] = {}
        ax.axvline(0, color="#92989f", linewidth=.85, zorder=1)
        if retrieval:
            ax.axvline(-1, color="#a51e22", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
        for y, (encoder, color) in enumerate(zip(ENCODERS, COLORS)):
            effect = results[encoder]["effects"][metric]
            mid, lo, hi = [100 * effect[key] for key in ("difference", "ci_lower", "ci_upper")]
            plotted[metric][encoder] = {"difference_pp": mid, "ci_lower_pp": lo, "ci_upper_pp": hi}
            ax.errorbar(mid, y, xerr=[[mid-lo], [hi-mid]], fmt="o", color=color,
                        markersize=6, capsize=4, elinewidth=1.8, zorder=3)
            ax.annotate(f"{mid:+.2f}", (mid, y), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=10, color=color, fontweight="medium")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.7)
        ax.set_yticks([0, 1], NAMES)
        ax.set_ylim(1.5, -.6)
        ax.set_xlim(-1.5, 6.1)
        ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6])
        ax.set_xlabel("Change versus frozen encoder\n(percentage points)", fontsize=10)
        ax.grid(axis="x", color="#d9dce0", linewidth=.65, alpha=.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c3c8cd")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", color="#aeb4bb", labelsize=10)
    fig.suptitle("V10 development effects · fixed seed 17", x=.025, y=.976,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.025, .923, "One preselected state per encoder · development only", fontsize=10.5, color="#545c66")
    fig.text(.025, .135, "Adjusted intervals: 100,000 paired image-cluster bootstrap draws; family 80 (α = .05).", fontsize=10)
    fig.text(.025, .100, "Dashed red: −1 pp retrieval boundary. States and galleries are fixed.", fontsize=10)
    fig.text(.025, .065, "Composition intervals are descriptive and include zero.", fontsize=10)
    fig.text(.025, .030, "Both seed17 development gates pass; benchmark success is not established.", fontsize=10)
    fig.tight_layout(rect=(.008, .182, .997, .902), h_pad=2.0, w_pad=1.6)
    files = {}
    for ext in ("png", "svg"):
        path = output / f"v10_development_effects.{ext}"
        metadata = {"Date": None, "Creator": "SANW reproducible v10 development summary plot"} if ext == "svg" else None
        fig.savefig(path, dpi=240, facecolor="white", metadata=metadata)
        files[ext] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}
    plt.close(fig)
    receipt = {"study": "sanw_v10_seed17_development_figure", "source": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
               "inputs": {encoder: {"path": name, "sha256": sha} for encoder, (name, sha) in INPUTS.items()},
               "independent_audit": audit, "outputs": files, "plotted_effects_percentage_points": plotted,
               "matplotlib_version": matplotlib.__version__, "summary_json_inputs_only": True,
               "raw_arrays_or_features_opened": False, "new_analysis_or_gate": False, "practical_success_claimed": False,
               "scope": "Fixed seed17 exploratory official development; conditional on fixed fitted states and galleries; composition intervals descriptive."}
    path = output / "v10_development_effects_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(path), "sha256": digest(path), "outputs": files}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/practical_v10/figures")
    render(parser.parse_args().output.resolve())
