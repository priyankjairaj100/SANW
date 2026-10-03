ACL 2027 review-follow-up rewrite (v3)

Open main.tex as the main document and compile with pdfLaTeX + BibTeX.
The ZIP is flat and contains every referenced style, bibliography, TeX fragment,
and figure. It does not require the dataset or model files for typesetting.

The official ACL style and bibliography style are unmodified, pinned to commit
 d5adc823ff0f80f98c80405ca0ab66c68e684409.
English hyphenation limits and float line-number suppression are normal document
settings. The appendix uses ragged-bottom layout to avoid stretched paragraphs.
The four main pages precede Limitations on page five. The complete PDF has
26 pages, including references, new methods/results, and the labeled prior study.

Scientific scope:
- 72 follow-up fits, fixed promotion assignments, matched epoch-ten comparisons.
- All 18 adjusted primary effects; all 72 terminal seed rows; all 48 selected
  seed rows; all 36 declared trajectory means are retained in the appendix.
- Prior-study experiments and proofs are labeled separately. Their original
  project files and earlier delivered PDF/ZIP have not been edited.

Full-project reproduction, from the project root:
  python3 scripts/generate_review_paper_results.py
  python3 scripts/audit_review_paper.py
  python3 scripts/build_review_paper.py

The generator binds audited JSON and figure inputs to generated assets.
The independent audit checks displayed values against those JSON inputs.
The builder checks both studies' numerical receipts, original-manuscript
preservation, official styles, unresolved references, overflow, four-page main
length, and fonts. Its final output is output/pdf/acl27_review_followup_v3.pdf.

repair_tex_runtime.py only repairs this execution host's incomplete local TeX
installation. It is unnecessary on Overleaf. Local runtime and build caches are
excluded from the ZIP. The attached receipts record provenance and visual QA.
