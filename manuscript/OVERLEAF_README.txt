Final editorial rewrite v2: ACL short-paper source, fresh execution of 2026-10-03.

Set main.tex as the main document and compile using pdfLaTeX in Overleaf.
Every required LaTeX, bibliography, image, and generated table file is at the
same directory level. The compiled manuscript has four main-content pages;
Limitations starts on page 5; the full document has 17 pages.

The official ACL styles are unchanged and pinned in acl_style_provenance.json.
Standard English hyphenation minima are set in main.tex. Normal text sizes and
margins are used throughout. Citation metadata was checked against primary
paper and publisher pages on 2026-10-03.

Fresh numeric assets are generated, not copied from the recovered old paper.
The full recovery project includes their generator, audited analysis inputs,
raw predictions, input manifests, and complete execution receipts.

From the full recovery project root:
  python scripts/generate_paper_results.py
  python scripts/build_paper.py

The generator defaults are:
  --analysis-dir results/analysis
  --diagnostic results/relation_diagnostic_iter10000/summary.json
  --selection results/study/selection.json
  --output manuscript

paper_results_receipt.json binds the new analysis, selected states, data
provenance, raw prediction inputs for the example, generator, and all generated
outputs. Final project builds reject pending content, altered generated assets,
or input-hash mismatches. The Overleaf source is self-contained for typesetting;
regenerating scientific numbers requires the full recovery project.

The local repair_tex_runtime.py helper repairs an incomplete TeX installation
using installed resources without changing ACL source files. It is unnecessary
on Overleaf or a complete TeX distribution. Compiler caches are omitted.

This v2 rewrites the story around the incremental value of relation labels.
The earlier delivery remains archived separately. The current final PDF is
output/pdf/acl27_relation_labels_final_rewrite_v2.pdf in the full project.
final_layout_receipt_v2.json records the reviewed PDF hash and four main pages.
