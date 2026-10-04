#!/usr/bin/env python3
"""Compile v5 without allowing pending or unbound results in a final build."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript_strengthened_v5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_sources():
    seen = set()
    queue = [SOURCE / "main.tex"]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text()
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            dependency = SOURCE / name
            if not dependency.suffix:
                dependency = dependency.with_suffix(".tex")
            if not dependency.is_file():
                raise SystemExit(f"Missing source: {dependency}")
            queue.append(dependency)
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    original = json.loads((SOURCE / "v3_preservation.json").read_text())
    for name, expected in original.items():
        if sha(ROOT / "manuscript_review_v3" / name) != expected:
            raise SystemExit(f"Original v3 file changed: {name}")
    style = json.loads((SOURCE / "acl_style_provenance.json").read_text())
    for name in ("acl.sty", "acl_natbib.bst"):
        if sha(SOURCE / name) != style["files"][name]["sha256"]:
            raise SystemExit(f"Official style changed: {name}")
    sources = active_sources()
    pending = [p.name for p in sources if re.search(r"\\pending\{", p.read_text())]
    # The pending command's definition is not an unresolved content marker.
    pending = [name for name in pending if name != "main.tex"]
    for p in sources:
        if "\u2014" in p.read_text() or "---" in p.read_text():
            raise SystemExit(f"Em dash in {p.name}")
    if not args.draft:
        if pending:
            raise SystemExit("Pending empirical text: " + ", ".join(pending))
        receipt_path = SOURCE / "v5_results_receipt.json"
        if not receipt_path.is_file():
            raise SystemExit("Missing empirical result receipt.")
        receipt = json.loads(receipt_path.read_text())
        if receipt["prior_unavailable_outputs_used"]:
            raise SystemExit("Unavailable historical outcomes entered the new paper.")
        if receipt["generator_sha256"] != sha(ROOT / "scripts/generate_strengthened_v5.py"):
            raise SystemExit("Generator changed; regenerate the empirical files.")
        for item in receipt["analysis_inputs"] + receipt["generated_outputs"]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise SystemExit("Empirical provenance mismatch: " + item["path"])
    if args.check_only:
        print(json.dumps({"status": "source_checked", "pending": pending, "draft": args.draft}, indent=2))
        return
    for name in ("pdflatex", "bibtex", "kpsewhich"):
        if not shutil.which(name):
            raise SystemExit("Missing build tool: " + name)
    env = os.environ.copy()
    probe = subprocess.run(["kpsewhich", "pdflatex.fmt"], capture_output=True, text=True)
    if not probe.stdout.strip():
        spec = importlib.util.spec_from_file_location("v5_tex_repair", SOURCE / "repair_tex_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = module.configure()
    build = SOURCE / "build"
    build.mkdir(exist_ok=True)
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-output-directory=build", "main.tex"]
    for index, command in enumerate([cmd, ["bibtex", "build/main"], cmd, cmd], 1):
        run = subprocess.run(command, cwd=SOURCE, env=env, capture_output=True, text=True)
        (build / f"command-{index}.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise SystemExit(run.stdout[-7000:] + run.stderr[-1000:])
    log = (build / "main.log").read_text(errors="replace")
    warnings = [line for line in log.splitlines()
                if re.search(r"Overfull \\[hv]box|undefined|multiply defined", line)]
    if warnings and not args.draft:
        raise SystemExit("Unresolved TeX warnings:\n" + "\n".join(warnings))
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / ("strengthened_v5_draft.pdf" if args.draft else "acl27_strengthened_v5.pdf")
    shutil.copy2(build / "main.pdf", pdf)
    import fitz
    document = fitz.open(pdf)
    limitations = [i + 1 for i, page in enumerate(document)
                   if re.search(r"^Limitations$", page.get_text(), re.MULTILINE)]
    fonts = sorted({f[3] for page in document for f in page.get_fonts() if f[2] == "Type3"})
    if not args.draft and (limitations != [5] or fonts):
        raise SystemExit(f"Layout gate failed: limitations={limitations}, Type3={fonts}")
    receipt = {
        "status": "draft" if args.draft else "compiled_requires_visual_review",
        "pdf": str(pdf.relative_to(ROOT)), "sha256": sha(pdf), "pages": len(document),
        "limitations_pages": limitations, "type3_fonts": fonts, "tex_warnings": warnings,
        "pending": pending, "source_hashes": {p.name: sha(p) for p in sources},
        "supporting_source_hashes": {n: sha(SOURCE / n) for n in ("references.bib", "acl.sty", "acl_natbib.bst")},
        "original_v3_unchanged": True,
    }
    (out / ("v5_draft_build_receipt.json" if args.draft else "v5_build_receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "pages", "limitations_pages", "tex_warnings", "type3_fonts")}, indent=2))


if __name__ == "__main__":
    main()
