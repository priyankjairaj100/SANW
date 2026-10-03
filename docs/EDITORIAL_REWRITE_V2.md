# Final editorial rewrite, version 2

This revision follows the user's request to reconsider the story after gathering
every accessible earlier result, theory, figure, and unused direction. It changes
presentation and mathematical exposition. The data, protocol, trained models,
selection ledger, predictions, and scientific results are unchanged.

## Editorial decision

The paper asks what each use of relation labels adds to image–text retrieval.
Its contribution is a matched decomposition of label roles and their practical
search outcomes. It does not claim a new state-of-the-art training algorithm or
an inevitable conflict between compositional understanding and retrieval.

The accessible evidence includes twelve training policies, 108 fitted candidates,
36 selected states, a frozen reference, four benchmark evaluations, a separate
relation-classifier diagnostic, and the historical recovery records. The original
CIKM PDF and original raw experiments were not available. The inventories record
this gap and distinguish recovered descriptions from the fresh execution.

## Changes from the previous delivered paper

- Lead with an ordinary search example and define the three relation labels.
- Explain the matched interventions before introducing the objective.
- Promote constant negative weighting to the main result table. It gives a useful
  search comparison to positive expansion, while its lower annotated-task score
  makes the application tradeoff visible.
- Keep the real full-pool query example and the addition versus replacement/swap
  edit breakdown in the main paper.
- Place the compact similarity-control comparison late in the main paper, after
  the positive-target analysis. Keep the complete retention scatter and matched
  negative-normalizer analysis in the appendix.
- Retain the positive-target-budget calculation late in the main text, with its
  fixed-score premises stated before use.
- Complete the appendix's reverse-anchor averaging identity, row-versus-transpose
  counterexample, detached versus differentiable matched-constant distinction,
  and original multiplicity-bound derivation.
- Update the closest primary literature through the research cutoff, including
  methods that preserve retrieval. The generic tradeoff and three-way triage
  are treated as prior work.
- Preserve historical material and the previous flat source/PDF snapshot rather
  than overwriting the evidence trail.

## Review and recovery map

| Record | Purpose |
|---|---|
| `CONTEXT_INVENTORY_FINAL_REWRITE.md` | Item-level map of all accessible scientific material and missing originals |
| `THEORY_DISPOSITION_FINAL_REWRITE.md` | Complete theory disposition, exact assumptions, and unused directions |
| `NOVELTY_POSITIONING_FINAL_REWRITE.md` | Verified primary literature and limits of the novelty claim |
| `STORY_REASSESSMENT_REVIEW.md` | Independent assessment of the central story |
| `FINAL_STORY_DECISIONS.md` | Adopted story and main/appendix allocation |
| `../historical_context/pre_editorial_rewrite_20261003_source.zip` | Previous delivered flat source and compiled PDF |
| `../manuscript/paper_results_receipt.json` | Generated scientific assets and audited input hashes |

Checkpoint 09 preserves the context and editorial audits, scientific records,
code, and previous source. It is an interim checkpoint, not the final rewritten
paper or a complete raw-data/model backup. The final complete project archive
contains the new manuscript and all retained scientific inputs and outputs.

## Scientific invariants

The main protocol SHA256 remains
`066670370c8d0311a5bbeff8cfb37ab5e74f4d6dfcdbc898176bbeabfe54e847`.
The canonical training identity digest recorded as `ledger_sha256` remains
`3eb474e7ba81a348f58462398b1010f7c7c0c3b6b1715743cb6e336e17ddbd51`.
The full serialized ledger file (`results/study/ledger.json`) SHA256 is
`ecd391f48cb6dac4aa1283ec5a4468a117fb8c7109a1d01a3cdb434ad0033e9f`.
The selection record (`results/study/selection.json`) SHA256 remains
`04950c8627a54e0dbd060768d5725c882746e74c71365bf3f1fa81518b4e649e`.
Historical numbers are not inputs to the generated paper assets. Reproducing
the manuscript does not require rerunning the already audited training grid.
