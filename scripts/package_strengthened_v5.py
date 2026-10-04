#!/usr/bin/env python3
"""Package v5 only after numerical binding and actual visual review."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manuscript_strengthened_v5"
OUT = ROOT / "output"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads((OUT / "pdf/v5_build_receipt.json").read_text())
    pdf = ROOT / build["pdf"]
    if sha(pdf) != build["sha256"] or build["limitations_pages"] != [5]:
        raise SystemExit("PDF identity or four-page body check failed.")
    if build["tex_warnings"] or build["type3_fonts"] or build["pending"]:
        raise SystemExit("The build contains unresolved issues.")
    visual = json.loads((M / "visual_review_v5.json").read_text())
    if visual["status"] != "passed" or visual["pdf_sha256"] != sha(pdf):
        raise SystemExit("Missing visual review for this exact PDF.")
    for name, expected in (build["source_hashes"] | build["supporting_source_hashes"]).items():
        if sha(M / name) != expected:
            raise SystemExit("Source changed after the checked build: " + name)
    for p in M.glob("*.tex"):
        if p.name.startswith("historical_") or p.name == "official_template_example.tex":
            continue
        for group in re.findall(r"\\input\{([^}]+)\}|\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", p.read_text()):
            name = next(x for x in group if x)
            if "/" in name or not (M / name).is_file():
                raise SystemExit(f"Non-flat dependency: {p.name}: {name}")
    allowed = {".tex", ".bib", ".bst", ".sty", ".pdf", ".png", ".jpg", ".json", ".txt", ".md", ".py"}
    files = sorted(p for p in M.iterdir() if p.is_file() and p.suffix in allowed)
    archive = OUT / "acl27-overleaf-strengthened-v5.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any("/" in name for name in z.namelist()):
            raise SystemExit("Archive integrity or flat-source check failed.")
    receipt = {"status": "packaged", "pdf": {"path": str(pdf.relative_to(ROOT)), "sha256": sha(pdf)},
               "overleaf": {"path": str(archive.relative_to(ROOT)), "sha256": sha(archive), "bytes": archive.stat().st_size},
               "source_hashes": {p.name: sha(p) for p in files}, "original_v3_unchanged": True}
    (OUT / "strengthened_v5_delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "pdf", "overleaf")}, indent=2))


if __name__ == "__main__":
    main()
