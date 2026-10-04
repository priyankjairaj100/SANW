#!/usr/bin/env python3
"""Generate v5 empirical tables only from complete, audited analysis files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "manuscript_strengthened_v5"
ENDPOINTS = (
    ("e_vil_test1000", "i2t.r1", r"I$\to$T"),
    ("e_vil_test1000", "t2i.r1", r"T$\to$I"),
    ("visual_entailment", "accuracy", "Relation"),
    ("sugarcrepe_pp", "both_accuracy", "Both"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_audit(audit: dict) -> None:
    """Reject stale receipts and status-only assertions before numerical inclusion."""
    if audit.get("status") != "passed" or not audit.get("input_sha256"):
        raise SystemExit("An independent audit needs a passing status and hashed inputs.")
    if not (audit.get("independence") or audit.get("independent_implementation")):
        raise SystemExit("The receipt does not identify an independent implementation.")
    for field in ("input_sha256", "source_sha256"):
        for name, expected in audit.get(field, {}).items():
            path = root_path(name)
            if not path.is_file() or digest(path) != expected:
                raise SystemExit("Independent audit input changed: " + name)


def require_audit(audits: list[dict], files: list[Path], **scope) -> dict:
    expected = {str(root_path(path).relative_to(ROOT)): digest(root_path(path)) for path in files}
    matches = [audit for audit in audits
               if all(audit.get(key) == value for key, value in scope.items())
               and all(audit["input_sha256"].get(name) == value for name, value in expected.items())]
    if len(matches) != 1:
        raise SystemExit(f"Expected one independent audit for {scope}: {list(expected)}")
    return matches[0]


def selection_values(state: dict) -> str:
    """Use the actual analysis schema; missing numeric fields must never become zero."""
    return (f"{state['seed']} & {state['draw_id'] if state['draw_id'] is not None else '--'} & "
            f"{state['learning_rate']:g} & {state['epoch']} & "
            f"{state['source_mix']:g} & {state['beta']:g} & {state['alpha']:g} & "
            f"{state['update_norm']:.4g}")


def gate_rows(gates: dict, contrasts: dict, scale: float) -> list[str]:
    rows = []
    for gate in gates["gates"]:
        values = []
        for dataset, metric, _ in (ENDPOINTS[0], ENDPOINTS[1], ENDPOINTS[3]):
            matches = [row for row in contrasts["contrasts"]
                       if row["encoder"] == gate["encoder"] and row["left"] == gate["family"]
                       and row["right"] == "frozen" and row["dataset"] == dataset and row["metric"] == metric]
            if len(matches) != 1:
                raise SystemExit("A practical gate lacks its unique adjusted contrast.")
            values.append(f"{scale * matches[0]['ci_lower']:+.3f}")
        rows.append(f"{label(gate['encoder'])} & {label(gate['family'])} & "
                    + ("Yes" if gate["checks"]["all_three_nonzero"] else "No") + " & "
                    + " & ".join(values) + " & " + ("Pass" if gate["passed"] else "Fail") + r"\\")
    return rows


def certificate_report(indices: list[tuple[Path, dict]], audits: list[dict]) -> tuple[str, dict]:
    """Summarize unique checkpoint-query observations without inventing new query counts."""
    if len(indices) != 2 or {data["encoder"] for _, data in indices} != {"vit_b32", "rn50"}:
        raise SystemExit("Certificate reporting requires both encoder indices exactly once.")
    rows, reports = [], []
    for path, data in sorted(indices, key=lambda item: item[1]["encoder"]):
        if data.get("status") != "complete" or data.get("false_exact_kl_certificates") != 0:
            raise SystemExit("Certificate diagnostics are incomplete or contain a false certificate.")
        if (not data["runs"] or data.get("state_count") != len(data["runs"])
                or data.get("archive_count") != 2 * len(data["runs"])):
            raise SystemExit("Certificate diagnostics require a nonempty, complete state inventory.")
        if (data.get("temperature") != 2. or data.get("k") != 1
                or data["protocol_sha256"] != digest(ROOT / "results/strengthen_retention/protocol_v3.json")):
            raise SystemExit("Certificate diagnostics differ from the original gallery diagnostic contract.")
        require_audit(audits, [path], protocol_sha256=data["protocol_sha256"])
        if len({run["state_id"] for run in data["runs"]}) != len(data["runs"]):
            raise SystemExit("Certificate diagnostic states are duplicated.")
        for dataset in ("e_vil_test1000", "coco_karpathy"):
            for direction in ("i2t", "t2i"):
                for trained in (True, False):
                    selected = [run for run in data["runs"]
                                if (run["epoch"] > 0 and run["update_norm"] > 0) == trained]
                    counts = {key: 0 for key in ("queries", "teacher_correct", "actually_retained")}
                    masks = {key: 0 for key in ("certified", "logit_drift_certified", "combined_certified")}
                    for run in selected:
                        record = run["datasets"][dataset]
                        for file_field in ("predictions", "metadata"):
                            artifact = root_path(record[file_field])
                            if digest(artifact) != record[file_field + "_sha256"]:
                                raise SystemExit("Certificate artifact changed: " + str(artifact))
                        summary = record["summary"][direction]
                        for key in counts:
                            counts[key] += summary["counts"][key]
                        for key in masks:
                            coverage = summary["certificate_coverage"][key]
                            if coverage["false_certificates"] != 0:
                                raise SystemExit("A reported certificate contradicts the diagnostic ranks.")
                            masks[key] += coverage["certified_queries"]
                    denominator = counts["teacher_correct"]
                    values = [f"{100 * masks[key] / denominator:.2f}" if denominator else "--" for key in masks]
                    population = "Trained" if trained else "Unchanged"
                    rows.append(f"{label(data['encoder'])} & {'e-ViL' if dataset == 'e_vil_test1000' else 'COCO'} & "
                                + (r"I$\to$T" if direction == "i2t" else r"T$\to$I")
                                + f" & {population} & {len(selected)} & {denominator:,} & "
                                + f"{counts['actually_retained']:,} & " + " & ".join(values) + r"\\")
                    reports.append({"encoder": data["encoder"], "dataset": dataset, "direction": direction,
                                    "population": population, "states": len(selected),
                                    "denominator_unit": "unique selected checkpoint-query observation",
                                    "counts": counts, "certified_counts": masks})
    caption = ("Full-gallery certificate diagnostics for all primary and sensitivity selections in the original retention study. "
               "Shared checkpoints count once. Trained requires positive epoch and update norm; other states appear as Unchanged. "
               "Teacher and Retained count checkpoint-query observations, not distinct queries. "
               "KL, Score, and Union give certified percentages of teacher-correct observations under pessimistic ties.")
    text = "\n".join([
        r"\section{Full-Gallery Certificate Diagnostics}", r"\label{app:certificates}",
        "These diagnostics use the original retention selections without changing their benchmark ranks or practical criteria.",
        "Each unique checkpoint appears once, even when several selection rules choose it.",
        "Different checkpoints reuse the same queries.",
        "The pooled denominator therefore counts checkpoint-query observations, not independent test examples.",
        "The complete artifact retains each checkpoint's selection roles and all query-level certificate masks.",
        "These descriptive counts do not add statistical hypotheses or imply an unseen-gallery guarantee.",
        table(rows, "llllrrrrrr", r"Encoder & Pool & Direction & Group & States & Teacher & Retained & KL & Score & Union",
              caption, "tab:certificates", font="scriptsize"),
    ])
    return text, {"summary": reports, "indices": {data["encoder"]: data for _, data in indices}}


def escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def label(value: str) -> str:
    return {
        "vit_b32": "ViT", "rn50": "RN50", "source": "Source",
        "supported": "U", "allocation": "A", "distilled": "D",
        "allocation_distilled": "AD", "allocation_distillation": "AD",
        "ad": "AD", "wise": "WiSE", "wise_ft": "WiSE",
        "frozen": "Frozen", "matched_ad_random": "AD random",
        "matched_allocation_distillation": "AD random", "score_stratified": "Stratified",
        "matched_distilled": "D random", "image_source_only": r"Image source",
        "reverse_source_only": r"Reverse source", "image_main_effect": "Image main",
        "reverse_main_effect": "Reverse main",
        "allocation_main": "Allocation", "distillation_main": "Distillation",
        "interaction": "Interaction",
    }.get(value, escape(value))


def table(rows: list[str], columns: str, header: str, caption: str, name: str,
          *, wide: bool = True, font: str = "small") -> str:
    env = "table*" if wide else "table"
    return "\n".join([
        rf"\begin{{{env}}}[t]", r"\centering", "\\" + font,
        r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule", header + r"\\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}\label{{{name}}}", rf"\end{{{env}}}",
    ]) + "\n"


def replication_figure(settings: list[tuple[str, list[dict]]], write, outputs: list[Path]) -> None:
    # Plot the two attribution comparisons motivated by the original study.
    # The appendix retains every endpoint and comparator, including negative outcomes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)
    rate_names = {0.0001: "1e-4", 0.0003: "3e-4", 0.001: "1e-3"}
    ylabels = []
    figure_data = []
    for panel, (right, metric, title, color) in enumerate((
        ("source", "i2t.r1", "Supported minus Source: image to caption", "#26547C"),
        ("score_stratified", "t2i.r1", "Supported minus stratified: caption to image", "#087F8C"),
    )):
        ax = axes[panel]
        y = 0
        for setting, contrasts in settings:
            for rate in sorted(rate_names):
                matches = [c for c in contrasts if c["right"] == right and c["metric"] == metric
                           and c["learning_rate"] == rate]
                if len(matches) != 1:
                    raise SystemExit(f"Replication panel needs exactly one effect: {setting}, {right}, {metric}, {rate}")
                c = matches[0]
                point = c["difference_percentage_points"]
                lo, hi = c["ci_percentage_points"]
                ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                            markersize=3.8, capsize=2, elinewidth=1.1)
                figure_data.append({"setting": setting, "right": right, "metric": metric,
                                    "learning_rate": rate, "difference_pp": point, "interval_pp": [lo, hi]})
                if panel == 0:
                    ylabels.append(setting + "  " + rate_names[rate])
                y += 1
        ax.axvline(0, color="#777777", linewidth=.8, linestyle="--")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Difference in R@1 (percentage points)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#E5E5E5", linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout(pad=.7, w_pad=1.5)
    for suffix in ("pdf", "png"):
        path = DEST / f"replication_summary_v5.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    write("replication_figure_data_v5.json", json.dumps(figure_data, indent=2) + "\n")
    write("replication_figure_v5.tex", "\n".join([
        r"\begin{figure*}[t]", r"\centering",
        r"\includegraphics[width=\textwidth]{replication_summary_v5.pdf}",
        r"\caption{Caption identity and source retrieval at matched epoch ten. "
        "Dots average training seeds and fixed assignment draws. "
        "Bars show the adjusted intervals from each setting's separately declared comparison family. "
        "The appendix reports all retrieval directions and comparators.}"
        r"\label{fig:newreplication}", r"\end{figure*}", ""]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-nonlinear", action="store_true",
                        help="Prepare a partial draft while retaining the independent-audit gate.")
    parser.add_argument("--prepare-replications", action="store_true",
                        help="Integrate both audited replications while preservation results remain pending.")
    parser.add_argument("--prepare-ad", action="store_true",
                        help="Integrate audited replications and AD before original retention is complete.")
    parser.add_argument("--nonlinear", type=Path, required=True)
    parser.add_argument("--nonlinear-audit", type=Path,
                        help="Bind the partial nonlinear draft to its independent audit.")
    parser.add_argument("--rn50", type=Path)
    parser.add_argument("--rn50-audit", type=Path)
    parser.add_argument("--preservation", type=Path)
    parser.add_argument("--preservation-audit", type=Path)
    parser.add_argument("--retention", type=Path,
                        help="Original frozen retention analysis with 80 primary and 36 directional effects.")
    parser.add_argument("--audits", nargs="+", type=Path)
    parser.add_argument("--certificates", nargs=2, type=Path,
                        help="Complete original-retention certificate indices for both encoders.")
    parser.add_argument("--certificate-audits", nargs="+", type=Path,
                        help="Independent receipts that bind both certificate indices.")
    parser.add_argument("--preservation-units", choices=("fractions", "percentage_points"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    inputs: list[dict] = []

    def read(path: Path):
        path = path if path.is_absolute() else ROOT / path
        data = json.loads(path.read_text())
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        return data

    if sum((args.prepare_nonlinear, args.prepare_replications, args.prepare_ad)) > 1:
        parser.error("Choose one partial preparation mode.")
    if args.prepare_ad:
        for name in ("rn50", "nonlinear_audit", "rn50_audit", "preservation", "preservation_audit",
                     "protocol", "protocol_sha256"):
            if getattr(args, name) is None:
                parser.error("AD preparation requires --" + name.replace("_", "-"))
        prepare_ad(args, read, inputs)
        return
    if args.prepare_replications:
        for name in ("rn50", "nonlinear_audit", "rn50_audit"):
            if getattr(args, name) is None:
                parser.error("Replication preparation requires --" + name.replace("_", "-"))
        prepare_replications(args, read, inputs)
        return
    if args.prepare_nonlinear:
        prepare_nonlinear(args.nonlinear, read, inputs, args.nonlinear_audit)
        return
    required = ("rn50", "preservation", "retention", "audits", "preservation_units", "protocol", "protocol_sha256",
                "certificates", "certificate_audits")
    for name in required:
        if getattr(args, name) is None:
            parser.error("A complete generation requires --" + name.replace("_", "-"))

    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    if digest(protocol_path) != args.protocol_sha256:
        raise SystemExit("The preservation protocol hash does not match.")
    read(protocol_path)
    audit_data = [read(path) for path in args.audits]
    for audit in audit_data:
        validate_audit(audit)
    certificate_audits = [read(path) for path in args.certificate_audits]
    for audit in certificate_audits:
        validate_audit(audit)
    certificate_text, certificate_data = certificate_report(
        [(path, read(path)) for path in args.certificates], certificate_audits)

    for setting, directory, protocol_name in (
        ("nonlinear", args.nonlinear, "STRENGTHEN_REPLICATION_PROTOCOL.json"),
        ("rn50", args.rn50, "STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json"),
    ):
        require_audit(audit_data, [directory / "primary_contrasts.json"], setting=setting,
                      primary_effects=12, bootstrap_replicates_per_effect=10000,
                      protocol_sha256=digest(ROOT / "docs" / protocol_name))
    for directory, factorial_name, factorial_key, factorial_count, protocol_hash in (
        (args.preservation, "decomposition_contrasts.json", "decomposition_effects", 24, args.protocol_sha256),
        (args.retention, "factorial_contrasts.json", "factorial_effects", 36,
         digest(ROOT / "results/strengthen_retention/protocol_v3.json")),
    ):
        require_audit(audit_data,
            [directory / name for name in ("primary_contrasts.json", factorial_name,
                                          "selected_strategy_metrics.json", "practical_success_gates.json")],
            primary_effects=80, replicates_per_effect=100000, protocol_sha256=protocol_hash,
            **{factorial_key: factorial_count})

    original_replication = read(ROOT / "results/review_followup/analysis/primary_contrasts.json")
    replications = []
    for name, directory in (("ViT nonlinear", args.nonlinear), ("RN50 linear", args.rn50)):
        data = read(directory / "primary_contrasts.json")
        if len(data["contrasts"]) != 12:
            raise SystemExit(f"Expected 12 contrasts for {name}.")
        read(directory / "selected_strategy_metrics.json")
        read(directory / "analysis_audit.json")
        replications.append((name, data["contrasts"]))

    preserved = read(args.preservation / "primary_contrasts.json")
    decomposition = read(args.preservation / "decomposition_contrasts.json")
    strategies = read(args.preservation / "selected_strategy_metrics.json")
    gates = read(args.preservation / "practical_success_gates.json")
    read(args.preservation / "analysis_audit.json")
    if len(preserved["contrasts"]) != 80:
        raise SystemExit("Expected the full 80-contrast preservation family.")
    if len(decomposition["contrasts"]) != 24:
        raise SystemExit("Expected the full 24-contrast decomposition family.")
    old_primary = read(args.retention / "primary_contrasts.json")
    old_factorial = read(args.retention / "factorial_contrasts.json")
    old_strategies = read(args.retention / "selected_strategy_metrics.json")
    old_gates = read(args.retention / "practical_success_gates.json")
    read(args.retention / "analysis_audit.json")
    if len(old_primary["contrasts"]) != 80 or len(old_factorial["contrasts"]) != 36:
        raise SystemExit("The original retention analysis must contain all 80 primary and 36 directional effects.")

    scale = 100.0 if args.preservation_units == "fractions" else 1.0
    outputs: list[Path] = []

    def write(name: str, content: str) -> None:
        path = DEST / name
        path.write_text(content)
        outputs.append(path)

    replication_figure([("ViT linear", original_replication["contrasts"]), *replications], write, outputs)

    family_order = ["frozen", "source", "supported", "allocation", "distilled",
                    "allocation_distillation", "wise_ft", "matched_allocation_distillation"]
    lookup = {(s["encoder"], s["family"]): s for s in strategies["strategies"]
              if s["tolerance_pp"] in (None, 1)}
    main_rows = []
    for family in family_order:
        values = []
        for encoder in ("vit_b32", "rn50"):
            strategy = lookup.get((encoder, family))
            if strategy is None:
                raise SystemExit(f"Missing primary strategy: {encoder}/{family}")
            for dataset, metric, _ in (ENDPOINTS[0], ENDPOINTS[1], ENDPOINTS[3]):
                values.append(f"{strategy['metrics'][f'{dataset}.{metric}']['mean'] * scale:.2f}")
        main_rows.append(label(family) + " & " + " & ".join(values) + r"\\")
    write("preservation_table_v5.tex", table(main_rows, "lrrrrrr",
        r"& \multicolumn{3}{c}{ViT} & \multicolumn{3}{c}{RN50}\\"
        r"Family & I$\to$T & T$\to$I & Both & I$\to$T & T$\to$I & Both",
        "Development-selected procedures under the one-point retrieval tolerance. "
        "Entries are percentages averaged across the fixed training seeds. "
        "Both denotes SugarCrepe++ accuracy for both valid captions. "
        "AD random averages three matched assignment draws within each seed.",
        "tab:newpreservation", wide=False, font="footnotesize"))

    gate_header = r"Encoder & Family & Trained & I$\to$T lower & T$\to$I lower & Both lower & Joint"
    gate_caption = ("Trained requires all three nonzero selected updates. "
                    "Numbers are adjusted lower bounds against frozen initialization, in percentage points. "
                    "Both retrieval bounds must exceed $-1$, and the Both bound must exceed zero.")
    gate_table = table(gate_rows(gates, preserved, scale), "lllrrrl", gate_header,
                      "AD-study practical criteria. " + gate_caption, "tab:newgates")
    old_gate_table = table(gate_rows(old_gates, old_primary, scale), "lllrrrl", gate_header,
        "Original retention practical criteria use their original jointly adjusted family. " + gate_caption,
        "tab:originalgates")

    appendix = [r"\section{Replication and Preservation: Complete Results}",
                r"\label{app:newresults}",
                "All differences below use percentage points.",
                "Intervals use the separately declared comparison families.", gate_table, old_gate_table]
    for setting, contrasts in replications:
        rows = []
        for contrast in contrasts:
            point = contrast["difference_percentage_points"]
            lo, hi = contrast["ci_percentage_points"]
            rows.append(
                f"{contrast['learning_rate']:g} & {label(contrast['right'])} & "
                f"{escape(contrast['metric'])} & {point:+.3f} & $[{lo:+.3f},{hi:+.3f}]$ " + r"\\")
        appendix.append(table(rows, "lllrr", "Rate & Comparator & Endpoint & Difference & Adjusted interval",
            f"All twelve Supported-minus-comparator effects for {setting}. "
            "Every comparison uses epoch ten and a common learning rate.",
            "tab:nonlinearpartial" if setting == "ViT nonlinear" else "tab:rn50partial"))

    endpoint_labels = {f"{d}.{m}": tex for d, m, tex in ENDPOINTS}
    for field, entries, caption, prefix in (
        ("primary", preserved["contrasts"], "Primary preservation contrasts", "adprimary"),
        ("decomposition", decomposition["contrasts"], "AD-selected factorial contrasts", "adfactorial"),
        ("primary", old_primary["contrasts"], "Original retention contrasts", "originalretention"),
        ("directional", old_factorial["contrasts"], "Original directional contrasts at epoch ten", "directional"),
    ):
        for start in range(0, len(entries), 16):
            rows = []
            for c in entries[start:start + 16]:
                name = (label(c["left"]) + " $-$ " + label(c["right"])) if field == "primary" else label(c["effect"])
                if field == "directional":
                    name += f" ({c['learning_rate']:g})"
                endpoint = endpoint_labels.get(f"{c['dataset']}.{c['metric']}", escape(f"{c['dataset']}.{c['metric']}"))
                lo = c["ci_lower"] * scale
                hi = c["ci_upper"] * scale
                point = c["difference"] * scale
                rows.append(f"{label(c['encoder'])} & {name} & {endpoint} & {point:+.3f} & $[{lo:+.3f},{hi:+.3f}]$ " + r"\\")
            appendix.append(table(rows, "lllrr", "Encoder & Comparison & Endpoint & Difference & Adjusted interval",
                caption + f", rows {start + 1} through {min(start + 16, len(entries))}.",
                f"tab:{prefix}{start}"))

    for stage_name, stage_strategies in (("AD", strategies), ("Original", old_strategies)):
        metric_rows = []
        reported_metrics = ["e_vil_test1000.i2t.r1", "e_vil_test1000.t2i.r1",
                            "visual_entailment.accuracy", "sugarcrepe.accuracy",
                            "sugarcrepe_pp.both_accuracy", "coco_karpathy.i2t.r1", "coco_karpathy.t2i.r1"]
        for strategy in stage_strategies["strategies"]:
            tolerance = strategy["tolerance_pp"]
            tolerance_text = "--" if tolerance is None else f"{tolerance:g}"
            values = [f"{strategy['metrics'][metric]['mean'] * scale:.2f}" for metric in reported_metrics]
            metric_rows.append(f"{label(strategy['encoder'])} & {label(strategy['family'])} & {tolerance_text} & "
                               + " & ".join(values) + r"\\")
        for start in range(0, len(metric_rows), 18):
            appendix.append(table(metric_rows[start:start + 18], "llrrrrrrrr",
                r"Encoder & Family & Tol. & e I$\to$T & e T$\to$I & Rel. & SC & Both & C I$\to$T & C T$\to$I",
                f"{stage_name} selected procedures and descriptive zero-tolerance sensitivity. "
                "Entries are percentages. e denotes e-ViL; C denotes COCO. "
                "Rel. is relation accuracy; SC is SugarCrepe; Both is SugarCrepe++ both-caption accuracy.",
                f"tab:{stage_name.lower()}metrics{start}", font="scriptsize"))
        selection_rows = []
        for strategy in stage_strategies["strategies"]:
            if strategy["tolerance_pp"] is None:
                continue
            for state in strategy["states"]:
                selection_rows.append(
                    f"{label(strategy['encoder'])} & {label(strategy['family'])} ({strategy['tolerance_pp']:g}) & "
                    + selection_values(state) + r"\\")
        for start in range(0, len(selection_rows), 20):
            appendix.append(table(selection_rows[start:start + 20], "llrrrrrrrr",
                r"Encoder & Family & Seed & Draw & Rate & Epoch & $\lambda$ & $\beta$ & $\alpha$ & Norm",
                f"{stage_name} study selected states. Family parentheses give the development tolerance in percentage points. "
                "Norm denotes the recorded parameter update norm.",
                f"tab:{stage_name.lower()}selected{start}", font="scriptsize"))
    appendix.append(certificate_text)
    write("results_appendix_v5.tex", "\n\n".join(appendix) + "\n")
    write("certificate_diagnostics_v5.json", json.dumps(certificate_data, indent=2) + "\n")

    # Preserve machine-readable evidence within the flat manuscript package.
    for name, data in (("preservation_primary_v5.json", preserved),
                       ("preservation_factorial_v5.json", decomposition),
                       ("preservation_selected_v5.json", strategies),
                       ("preservation_gates_v5.json", gates),
                       ("original_retention_primary_v5.json", old_primary),
                       ("original_retention_factorial_v5.json", old_factorial),
                       ("original_retention_selected_v5.json", old_strategies),
                       ("original_retention_gates_v5.json", old_gates),
                       ("replication_contrasts_v5.json", dict(replications))):
        write(name, json.dumps(data, indent=2) + "\n")

    receipt = {
        "status": "generated_requires_claim_and_layout_review",
        "execution_kind": "allocation_distillation_exploratory_extension",
        "prior_unavailable_outputs_used": False,
        "preservation_units": args.preservation_units,
        "protocol_sha256": args.protocol_sha256,
        "analysis_inputs": inputs,
        "generated_outputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in outputs],
        "generator_sha256": digest(Path(__file__)),
        "independent_audit_files": [str(p) for p in args.audits],
        "independent_certificate_audit_files": [str(p) for p in args.certificate_audits],
    }
    (DEST / "v5_results_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "generated_files": len(outputs)}, indent=2))


def prepare_replications(args, read, inputs: list[dict]) -> None:
    """Publish both audited replication families without implying practical success."""
    original = read(ROOT / "results/review_followup/analysis/primary_contrasts.json")
    settings = [("ViT linear", original["contrasts"])]
    outputs: list[Path] = []
    documents, lookups = {}, {}

    def write(name: str, content: str) -> None:
        path = DEST / name
        path.write_text(content)
        outputs.append(path)

    appendix = [r"\section{Replication and Preservation: Complete Results}", r"\label{app:newresults}",
                "Both new replication analyses passed independent audits.",
                "The audits reproduced 610 prediction archives, 4,880 numeric aggregates, and 240,000 bootstrap samples.",
                "They reconstruct statistics from saved ranks and score-derived correctness.",
                "They do not independently extract features or reconstruct every retrieval dot product."]
    for setting, name, directory, audit_path, protocol in (
        ("nonlinear", "ViT nonlinear", args.nonlinear, args.nonlinear_audit, "STRENGTHEN_REPLICATION_PROTOCOL.json"),
        ("rn50", "RN50 linear", args.rn50, args.rn50_audit, "STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json"),
    ):
        independent = read(audit_path)
        validate_audit(independent)
        require_audit([independent], [directory / "primary_contrasts.json"], setting=setting,
                      primary_effects=12, bootstrap_replicates_per_effect=10000,
                      protocol_sha256=digest(ROOT / "docs" / protocol))
        data = read(directory / "primary_contrasts.json")
        production = read(directory / "analysis_audit.json")
        if len(data["contrasts"]) != 12 or production["status"] != "passed":
            raise SystemExit("The replication analysis is incomplete.")
        documents[setting] = data
        lookups[setting] = {(row["right"], row["metric"], row["learning_rate"]): row for row in data["contrasts"]}
        settings.append((name, data["contrasts"]))
        rows = []
        for row in data["contrasts"]:
            lo, hi = row["ci_percentage_points"]
            endpoint = r"I$\to$T" if row["metric"] == "i2t.r1" else r"T$\to$I"
            rows.append(f"{row['learning_rate']:g} & {label(row['right'])} & {endpoint} & "
                        f"{row['difference_percentage_points']:+.3f} & $[{lo:+.3f},{hi:+.3f}]$ " + r"\\")
        filename = setting + "_results_table_v5.tex"
        write(filename, table(rows, "lllrr", "Rate & Comparator & Endpoint & Difference & Adjusted interval",
              f"All twelve Supported-minus-comparator effects for {name} at epoch ten. "
              "Differences use percentage points. Intervals adjust over the setting's complete twelve-effect family.",
              "tab:" + setting + "partial"))
        appendix.append(r"\input{" + filename + "}")
    rates = [0.0001, 0.0003, 0.001]
    source_losses, random_gains = {}, {}
    for setting, lookup in lookups.items():
        source_losses[setting] = [lookup[("source", "i2t.r1", rate)]["difference_percentage_points"] for rate in rates]
        random_gains[setting] = [lookup[("score_stratified", "t2i.r1", rate)]["difference_percentage_points"] for rate in rates]
        if not all(lookup[("source", "i2t.r1", rate)]["ci_upper"] < 0 for rate in rates):
            raise SystemExit("The replication source-cost claim differs from the actual intervals.")
    if not all(lookups["rn50"][("score_stratified", "t2i.r1", rate)]["ci_lower"] > 0 for rate in rates):
        raise SystemExit("The RN50 caption-assignment claim differs from the actual intervals.")
    for rate in rates:
        row = lookups["rn50"][("score_stratified", "i2t.r1", rate)]
        if not row["ci_lower"] <= 0 <= row["ci_upper"]:
            raise SystemExit("The RN50 image-query qualification differs from the actual intervals.")
    middle = lookups["nonlinear"][("score_stratified", "t2i.r1", rates[1])]
    if not (middle["ci_lower"] <= 0 <= middle["ci_upper"] and
            all(lookups["nonlinear"][("score_stratified", "t2i.r1", rate)]["ci_lower"] > 0 for rate in (rates[0], rates[2]))):
        raise SystemExit("The nonlinear heterogeneity claim differs from the actual intervals.")
    format_values = lambda values: ", ".join(f"${value:+.2f}$" for value in values)
    write("replication_results_v5.tex", "\n".join([
        r"\input{replication_figure_v5.tex}",
        r"\paragraph{The source-retrieval cost recurs across settings.}",
        "Supported training reduces image-to-caption R@1 against Source at all three rates in both new replications.",
        "The nonlinear effects are " + format_values(source_losses["nonlinear"]) + " percentage points across increasing rates.",
        "The RN50 effects are " + format_values(source_losses["rn50"]) + " points.",
        "Every adjusted interval lies below zero.",
        r"Figure~\ref{fig:newreplication} includes the original linear study and both new replications.",
        r"\paragraph{Caption identity helps the reverse direction.}",
        "Against stratified promotion, RN50 gains " + format_values(random_gains["rn50"]) + " caption-to-image points.",
        "All three adjusted lower bounds exceed zero.",
        "The nonlinear gains are " + format_values(random_gains["nonlinear"]) + " points.",
        "Its smallest and largest rates have positive lower bounds; the middle interval includes zero.",
        "All RN50 image-to-caption intervals against stratified promotion include zero.",
        "Caption assignment therefore has a directional benefit under the matched controls.",
        "That benefit does not establish preservation relative to the frozen model.",
        r"Appendix Tables~\ref{tab:nonlinearpartial} and~\ref{tab:rn50partial} report every new replication effect.", ""]))
    replication_figure(settings, write, outputs)
    appendix.append(r"\pending{Original retention, AD analyses, and certificate diagnostics remain pending.}")
    write("results_appendix_v5.tex", "\n".join(appendix) + "\n")
    write("replication_contrasts_v5.json", json.dumps(documents, indent=2) + "\n")
    receipt = {"status": "replications_independently_verified_preservation_pending", "analysis_inputs": inputs,
               "generated_outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in outputs],
               "generator_sha256": digest(Path(__file__)), "final_build_permitted": False,
               "prior_unavailable_outputs_used": False}
    (DEST / "replication_preparation_receipt_v5.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "new_replication_effects": 24, "generated_files": len(outputs)}, indent=2))


def prepare_ad(args, read, inputs: list[dict]) -> None:
    """Integrate the actual negative AD result without substituting another endpoint."""
    prepare_replications(args, read, inputs)
    protocol_path = root_path(args.protocol)
    if digest(protocol_path) != args.protocol_sha256:
        raise SystemExit("The AD protocol hash changed.")
    read(protocol_path)
    independent = read(args.preservation_audit)
    validate_audit(independent)
    names = ("primary_contrasts.json", "decomposition_contrasts.json",
             "selected_strategy_metrics.json", "practical_success_gates.json")
    require_audit([independent], [args.preservation / name for name in names],
                  protocol_sha256=args.protocol_sha256, primary_effects=80,
                  decomposition_effects=24, replicates_per_effect=100000)
    primary, decomposition, strategies, gates = [read(args.preservation / name) for name in names]
    production = read(args.preservation / "analysis_audit.json")
    if len(primary["contrasts"]) != 80 or len(decomposition["contrasts"]) != 24 or production["status"] != "passed":
        raise SystemExit("The AD analysis is incomplete.")
    if len(gates["gates"]) != 12 or any(row["passed"] for row in gates["gates"]):
        raise SystemExit("The no-success statement differs from the actual practical gates.")
    by_effect = {(row["encoder"], row["left"], row["right"], row["dataset"], row["metric"]): row
                 for row in primary["contrasts"]}
    by_factorial = {(row["encoder"], row["effect"], row["dataset"], row["metric"]): row
                    for row in decomposition["contrasts"]}
    for encoder in ("vit_b32", "rn50"):
        relation = by_effect[(encoder, "allocation_distillation", "frozen", "visual_entailment", "accuracy")]
        caption = by_effect[(encoder, "allocation_distillation", "frozen", "sugarcrepe_pp", "both_accuracy")]
        if relation["ci_lower"] <= 0 or not caption["ci_lower"] <= 0 <= caption["ci_upper"]:
            raise SystemExit("The relation-versus-SugarCrepe++ statement differs from the actual intervals.")
        for direction in ("i2t.r1", "t2i.r1"):
            effect = by_effect[(encoder, "allocation_distillation", "frozen", "e_vil_test1000", direction)]
            if effect["ci_lower"] > -.01:
                raise SystemExit("The AD retrieval-retention failure differs from the actual interval.")
            if encoder == "vit_b32" and direction == "t2i.r1":
                if effect["ci_upper"] >= 0:
                    raise SystemExit("The reported ViT AD retrieval loss is not supported by its interval.")
            elif not effect["ci_lower"] <= 0 <= effect["ci_upper"]:
                raise SystemExit("An AD retrieval-uncertainty statement differs from the actual interval.")
            for intervention in ("allocation_main", "distillation_main"):
                if by_factorial[(encoder, intervention, "e_vil_test1000", direction)]["ci_lower"] <= 0:
                    raise SystemExit("The matched retrieval main-effect statement differs from its interval.")
        if by_factorial[(encoder, "distillation_main", "sugarcrepe_pp", "both_accuracy")]["ci_lower"] <= 0:
            raise SystemExit("The matched distillation effect on SugarCrepe++ differs from its interval.")
    if any(row["effect"] == "interaction" and row["ci_lower"] > 0 for row in decomposition["contrasts"]):
        raise SystemExit("A positive interaction contradicts the prepared statement.")
    outputs: list[Path] = []

    def write(name: str, content: str) -> None:
        path = DEST / name
        path.write_text(content)
        outputs.append(path)

    family_order = ["frozen", "source", "supported", "allocation", "distilled",
                    "allocation_distillation", "wise_ft", "matched_allocation_distillation"]
    lookup = {(row["encoder"], row["family"]): row for row in strategies["strategies"]
              if row["tolerance_pp"] in (None, 1.)}
    rows = []
    for family in family_order:
        values = []
        for encoder in ("vit_b32", "rn50"):
            row = lookup[(encoder, family)]
            for dataset, metric, _ in (ENDPOINTS[0], ENDPOINTS[1], ENDPOINTS[3]):
                values.append(f"{100 * row['metrics'][f'{dataset}.{metric}']['mean']:.2f}")
        rows.append(label(family) + " & " + " & ".join(values) + r"\\")
    write("preservation_table_v5.tex", table(rows, "lrrrrrr",
          r"& \multicolumn{3}{c}{ViT} & \multicolumn{3}{c}{RN50}\\"
          r"Family & I$\to$T & T$\to$I & Both & I$\to$T & T$\to$I & Both",
          "Selected AD-study procedures. Entries are percentages. Both denotes SugarCrepe++ both-caption accuracy. "
          "U is frozen for both encoders. RN50 A includes one frozen seed. "
          "None of six selected families passes the joint criterion.", "tab:newpreservation", wide=False, font="footnotesize"))
    ad_value = lambda encoder, dataset, metric: 100 * by_effect[(encoder, "allocation_distillation", "frozen", dataset, metric)]["difference"]
    relation = [ad_value(encoder, "visual_entailment", "accuracy") for encoder in ("vit_b32", "rn50")]
    caption = [ad_value(encoder, "sugarcrepe_pp", "both_accuracy") for encoder in ("vit_b32", "rn50")]
    write("preservation_results_v5.tex", "\n".join([
        r"\input{preservation_table_v5.tex}", r"\paragraph{No selected procedure meets the practical criterion.}",
        "None of the six families passes the joint criterion on either encoder.",
        "AD produces nonzero updates for all six selected seeds.",
        f"Its relation-accuracy gains over frozen initialization are ${relation[0]:+.2f}$ and ${relation[1]:+.2f}$ points for ViT and RN50.",
        "Both adjusted intervals exclude zero.",
        f"However, its SugarCrepe++ changes are ${caption[0]:+.2f}$ and ${caption[1]:+.2f}$ points; both intervals include zero.",
        f"AD's ViT caption-to-image loss is ${-ad_value('vit_b32', 'e_vil_test1000', 't2i.r1'):.2f}$ points, with an adjusted interval below zero.",
        "The other three AD retrieval intervals include zero but extend below the one-point tolerance.",
        "Those intervals do not establish retention or prove a large retrieval loss.",
        "Thus relation learning does not establish the requested caption improvement with retained retrieval.",
        r"Appendix Table~\ref{tab:newgates} reports every criterion and lower bound.", ""]))
    distill_caption = [100 * by_factorial[(encoder, "distillation_main", "sugarcrepe_pp", "both_accuracy")]["difference"]
                       for encoder in ("vit_b32", "rn50")]
    write("factorial_results_v5.tex", "\n".join([
        r"\paragraph{Matched effects differ from practical success.}",
        "At the AD-selected schedule, allocation and distillation each improve both retrieval directions on both encoders.",
        "All eight adjusted main-effect intervals lie above zero.",
        f"The distillation main effect also improves SugarCrepe++ by ${distill_caption[0]:+.2f}$ and ${distill_caption[1]:+.2f}$ points.",
        "These contrasts compare matched objective cells, rather than each selected procedure with frozen initialization.",
        "All retrieval and SugarCrepe++ interaction intervals include zero.",
        "RN50 has a negative relation-accuracy interaction.",
        "No positive interaction is established.", ""]))
    appendix = [r"\subsection{Allocation and distillation: complete results}",
                "This exploratory analysis reuses previously examined test datasets.",
                "Its independent audit checked 99 states, 495 prediction archives, 104 effects, and 10.4 million bootstrap values.",
                "Every practical family fails at least one declared requirement on each encoder.",
                "Failure to establish noninferiority does not prove the true retrieval loss exceeds the tolerance.",
                "Relation accuracy and SugarCrepe++ accuracy remain separate endpoints.",
                table(gate_rows(gates, primary, 100.), "lllrrrl",
                      r"Encoder & Family & Trained & I$\to$T lower & T$\to$I lower & Both lower & Joint",
                      "AD-study practical criteria. Trained requires three nonzero selected updates. "
                      "Numbers are adjusted lower bounds against frozen initialization, in percentage points. "
                      "Retrieval bounds must exceed $-1$; the SugarCrepe++ Both bound must exceed zero.", "tab:newgates")]
    endpoint_labels = {f"{dataset}.{metric}": tex for dataset, metric, tex in ENDPOINTS}
    for kind, entries in (("primary", primary["contrasts"]), ("factorial", decomposition["contrasts"])):
        for start in range(0, len(entries), 16):
            rows = []
            for row in entries[start:start+16]:
                comparison = (label(row["left"]) + " $-$ " + label(row["right"])) if kind == "primary" else label(row["effect"])
                endpoint = endpoint_labels[f"{row['dataset']}.{row['metric']}"]
                rows.append(f"{label(row['encoder'])} & {comparison} & {endpoint} & "
                            f"{100*row['difference']:+.3f} & $[{100*row['ci_lower']:+.3f},{100*row['ci_upper']:+.3f}]$ " + r"\\")
            appendix.append(table(rows, "lllrr", "Encoder & Comparison & Endpoint & Difference & Adjusted interval",
                f"AD-study {kind} effects, rows {start+1} through {min(start+16,len(entries))}. "
                "Differences use percentage points. Primary and factorial families adjust over 80 and 24 effects respectively.",
                f"tab:ad{kind}{start}"))
    metric_rows, selection_rows = [], []
    reported = ["e_vil_test1000.i2t.r1", "e_vil_test1000.t2i.r1", "visual_entailment.accuracy",
                "sugarcrepe.accuracy", "sugarcrepe_pp.both_accuracy", "coco_karpathy.i2t.r1", "coco_karpathy.t2i.r1"]
    for row in strategies["strategies"]:
        tolerance = "--" if row["tolerance_pp"] is None else f"{row['tolerance_pp']:g}"
        values = [f"{100 * row['metrics'][metric]['mean']:.2f}" for metric in reported]
        metric_rows.append(f"{label(row['encoder'])} & {label(row['family'])} & {tolerance} & " + " & ".join(values) + r"\\")
        if row["tolerance_pp"] is not None:
            for state in row["states"]:
                selection_rows.append(f"{label(row['encoder'])} & {label(row['family'])} ({tolerance}) & "
                                      + selection_values(state) + r"\\")
    for start in range(0,len(metric_rows),18):
        appendix.append(table(metric_rows[start:start+18], "llrrrrrrrr",
            r"Encoder & Family & Tol. & e I$\to$T & e T$\to$I & Rel. & SC & Both & C I$\to$T & C T$\to$I",
            "AD selected procedures and descriptive zero-tolerance sensitivity. Values are percentages. "
            "e denotes e-ViL; C denotes COCO; Rel. denotes relation accuracy; SC denotes SugarCrepe; Both denotes SugarCrepe++.",
            f"tab:admetrics{start}", font="scriptsize"))
    for start in range(0,len(selection_rows),20):
        appendix.append(table(selection_rows[start:start+20], "llrrrrrrrr",
            r"Encoder & Family & Seed & Draw & Rate & Epoch & $\lambda$ & $\beta$ & $\alpha$ & Norm",
            "AD-study selected states. Family parentheses give the development tolerance in percentage points. "
            "Norm denotes the recorded parameter update norm.", f"tab:adselected{start}", font="scriptsize"))
    appendix.append(r"\pending{Original retention, directional effects, and certificate diagnostics remain pending.}")
    write("ad_results_appendix_v5.tex", "\n\n".join(appendix) + "\n")
    existing = (DEST / "results_appendix_v5.tex").read_text()
    existing = existing.replace(r"\pending{Original retention, AD analyses, and certificate diagnostics remain pending.}",
                                r"\input{ad_results_appendix_v5.tex}")
    write("results_appendix_v5.tex", existing)
    for name, document in (("preservation_primary_v5.json",primary), ("preservation_factorial_v5.json",decomposition),
                           ("preservation_selected_v5.json",strategies), ("preservation_gates_v5.json",gates)):
        write(name,json.dumps(document,indent=2)+"\n")
    replication_receipt_path = DEST / "replication_preparation_receipt_v5.json"
    replication_receipt = json.loads(replication_receipt_path.read_text())
    combined = {row["path"]: root_path(row["path"]) for row in replication_receipt["generated_outputs"]}
    combined.update({str(path.relative_to(ROOT)): path for path in outputs})
    receipt = {"status":"replications_and_ad_verified_original_retention_pending", "analysis_inputs":inputs,
               "generated_outputs":[{"path":name,"sha256":digest(path)} for name,path in combined.items()],
               "generator_sha256":digest(Path(__file__)), "final_build_permitted":False,
               "practical_success":False,"prior_unavailable_outputs_used":False}
    (DEST/"ad_preparation_receipt_v5.json").write_text(json.dumps(receipt,indent=2)+"\n")
    replication_receipt["status"] = "superseded_by_ad_preparation_receipt"
    replication_receipt["superseding_receipt"] = "manuscript_strengthened_v5/ad_preparation_receipt_v5.json"
    replication_receipt_path.write_text(json.dumps(replication_receipt,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"AD_effects":104,"passing_practical_families":0,
                      "generated_files":len(outputs)},indent=2))


def prepare_nonlinear(directory: Path, read, inputs: list[dict], independent_path: Path | None = None) -> None:
    """Produce partial nonlinear tables, with optional verified audit binding."""
    data = read(directory / "primary_contrasts.json")
    selections = read(directory / "selected_strategy_metrics.json")
    audit = read(directory / "analysis_audit.json")
    if audit.get("status") != "passed" or len(data["contrasts"]) != 12:
        raise SystemExit("Nonlinear production analysis is incomplete.")
    verified = independent_path is not None
    if verified:
        independent = read(independent_path)
        validate_audit(independent)
        require_audit([independent], [directory / "primary_contrasts.json"], setting="nonlinear",
                      primary_effects=12, bootstrap_replicates_per_effect=10000,
                      protocol_sha256=digest(ROOT / "docs/STRENGTHEN_REPLICATION_PROTOCOL.json"))
    rows = []
    for c in data["contrasts"]:
        lo, hi = c["ci_percentage_points"]
        endpoint = r"I$\to$T" if c["metric"] == "i2t.r1" else r"T$\to$I"
        rows.append(f"{c['learning_rate']:g} & {label(c['right'])} & {endpoint} & "
                    f"{c['difference_percentage_points']:+.3f} & $[{lo:+.3f},{hi:+.3f}]$ " + r"\\")
    text = table(rows, "lllrr", "Rate & Comparator & Endpoint & Difference & Adjusted interval",
                 "All nonlinear Supported-minus-comparator effects at epoch ten. "
                 "Differences use percentage points. Intervals adjust over all twelve contrasts.", "tab:nonlinearpartial")
    path = DEST / "nonlinear_results_table_v5.tex"
    path.write_text(text)
    lookup = {(c["right"], c["metric"], c["learning_rate"]): c for c in data["contrasts"]}
    rates = [0.0001, 0.0003, 0.001]
    losses = [lookup[("source", "i2t.r1", rate)]["difference_percentage_points"] for rate in rates]
    gains = [lookup[("score_stratified", "t2i.r1", rate)]["difference_percentage_points"] for rate in rates]
    if not all(lookup[("source", "i2t.r1", rate)]["ci_upper"] < 0 for rate in rates):
        raise SystemExit("The prepared source-cost claim does not match the actual intervals.")
    if not (lookup[("score_stratified", "t2i.r1", rates[0])]["ci_lower"] > 0
            and lookup[("score_stratified", "t2i.r1", rates[1])]["ci_lower"] <= 0
            <= lookup[("score_stratified", "t2i.r1", rates[1])]["ci_upper"]
            and lookup[("score_stratified", "t2i.r1", rates[2])]["ci_lower"] > 0):
        raise SystemExit("The prepared nonlinear heterogeneity claim does not match its intervals.")
    claim = "\n".join([
        r"\paragraph{Caption identity with a nonlinear adapter.}",
        "The image-to-caption cost persists after changing adaptation capacity.",
        f"Supported-minus-Source effects are ${losses[0]:.2f}$, ${losses[1]:.2f}$, and ${losses[2]:.2f}$ percentage points across increasing rates.",
        "All three adjusted intervals lie below zero.",
        "The caption-to-image advantage over score-stratified promotion varies across rates.",
        f"Its point estimates are ${gains[0]:+.2f}$, ${gains[1]:+.2f}$, and ${gains[2]:+.2f}$ points.",
        "The smallest and largest rates have positive adjusted lower bounds.",
        "The middle interval includes zero.",
        "Supported also reduces image-to-caption recall against the stratified control at the two smaller rates.",
        "The largest-rate interval crosses zero.",
        r"Appendix Table~\ref{tab:nonlinearpartial} gives all twelve effects.",
        (r"\pending{RN50 replication remains pending.}" if verified else
         r"\pending{Independent nonlinear audit and RN50 replication remain pending.}"), ""])
    claim_path = DEST / "replication_results_v5.tex"
    claim_path.write_text(claim)
    appendix_path = DEST / "results_appendix_v5.tex"
    appendix_path.write_text("\n".join([
        r"\section{Replication and Preservation: Complete Results}", r"\label{app:newresults}",
        r"\input{nonlinear_results_table_v5.tex}",
        ("The independent nonlinear audit passed all twelve effects and 120,000 bootstrap samples."
         if verified else r"\pending{Independent nonlinear audit remains pending.}"),
        r"\pending{RN50 replication, original retention, and AD analyses remain pending.}", ""]))
    for name, value in (("nonlinear_primary_prepared_v5.json", data),
                        ("nonlinear_selected_prepared_v5.json", selections)):
        (DEST / name).write_text(json.dumps(value, indent=2) + "\n")
    receipt = {"status": "prepared_independently_verified" if verified else "prepared_pending_independent_audit",
               "analysis_inputs": inputs,
               "generated_outputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                                     for p in (path, claim_path, appendix_path)],
               "generator_sha256": digest(Path(__file__)), "final_build_permitted": False}
    (DEST / "nonlinear_preparation_receipt_v5.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "contrasts": 12}, indent=2))


if __name__ == "__main__":
    main()
