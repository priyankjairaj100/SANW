#!/usr/bin/env python3
"""Plot the locked v9 development results without changing their analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/practical_v9/figures"
OUT.mkdir(exist_ok=True)
encoders = ("vit_b32", "rn50")
names = ("ViT-B/32 (LAION2B)", "RN50 (OpenAI)")
results = {
    e: json.loads((ROOT / f"results/practical_v9/official_development/{e}_seed17/result.json").read_text())
    for e in encoders
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.7))
panels = [("i2t", "Image-to-text retrieval", -1),
          ("original", "Source + supported joint correctness", 0),
          ("t2i", "Text-to-image retrieval", -1),
          ("source_pair", "Two-source joint correctness", 0)]
for ax, (metric, title, reference) in zip(axes.flat, panels):
    for y, (e, name, color) in enumerate(zip(encoders, names, ("#2166ac", "#b35806"))):
        effect = results[e]["effects"][metric]
        mid, lo, hi = [100 * effect[k] for k in ("difference", "ci_lower", "ci_upper")]
        ax.errorbar(mid, y, xerr=[[mid-lo], [hi-mid]], fmt="o", color=color,
                    markersize=6, capsize=4, elinewidth=1.7)
        ax.annotate(f"{mid:+.2f}", (mid, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=color)
    ax.axvline(0, color="#808080", linewidth=.8)
    if reference == -1:
        ax.axvline(-1, color="#a51e22", linestyle="--", linewidth=1)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_yticks([0, 1], names)
    ax.set_ylim(1.55, -.55)
    ax.set_xlim(-2.3, 5.2)
    ax.set_xlabel("Change versus frozen encoder (percentage points)")
    ax.grid(axis="x", alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Fixed-model development effects · seed 17", x=.02, ha="left", fontsize=15, fontweight="bold")
fig.text(.02, .06, "Intervals: 100,000 paired image-cluster bootstrap draws; family size 80. Red dashed line: −1 pp retrieval boundary.", fontsize=8.5)
fig.text(.02, .027, "Both I2T intervals cross the required boundary. Composition intervals cross zero. Exploratory after historical test reuse.", fontsize=8.5)
fig.tight_layout(rect=(0, .105, 1, .93), h_pad=1.7, w_pad=2)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"v9_development_effects.{ext}", dpi=220, facecolor="white")
plt.close(fig)
print(OUT)
