# Independent reassessment of the four-page paper

Reviewed the current manuscript, recovered historical methods and proofs,
`docs/FINAL_STORY_DECISIONS.md`, and the completed fresh analysis. This is an
editorial reassessment, not a new experiment or numerical audit.

## Recommended story and title

**Recommended title: “What Do Relation Labels Add to Image–Text Retrieval?”**

The strongest story is **the practical return of each use of a relation label**.
Under a shared candidate pool and tuning budget, extra supported captions buy
a substantial improvement on the annotated statement task. The improvement
barely transfers to the aggregate edited-caption test and accompanies a large
loss in full-pool retrieval. The controls show which gains come from changing
positive targets and which require choosing particular negative pairs.

The title makes the application visible without claiming a deployed system or
a new retrieval algorithm. The manuscript should retain its explicit frozen
feature and selected-procedure scope in the study description. The first
question for a reader is: **If labels teach a model which descriptions are
correct, does using those labels make it return better matches?**

This is stronger than organizing the paper around “the combined policy does
not beat positive expansion.” That comparison is necessary but is a small
difference with intervals spanning zero. The substantial result is the gap
between what the supervision improves and what a search system must preserve.
The matched intervention study distinguishes this paper from a generic report
of a discrimination–retention tradeoff.

## Evidence worth leading with

1. **The return on extra positives:** +5.81 percentage points on the annotated
   statement task, +0.14 on aggregate SugarCrepe, and −8.56 on image-to-text
   retrieval, all relative to the source-positive procedure. The last change
   means 428 fewer correct first results per 5,000 image queries. Every seed
   loses retrieval; the net losses are 396, 539, and 349 queries. The mean and
   the across-seed consistency are more persuasive than a single striking
   example. Keep the photographed query as an illustration of this measured
   distribution, not its evidence.
2. **The aggregate gain is selective:** +1.92 points on added content and −0.90
   on the combined replacement/swap group, with the directions repeated in all
   three seeds. This identifies the capability being bought. It is more
   informative than calling the overall transfer gain “small.”
3. **A useful existing comparison is underemphasized:** constant negative
   weights reach 83.67 SugarCrepe accuracy and 55.21 image-to-text R@1, versus
   83.39 and 47.29 for positive expansion. Their relation accuracy is lower,
   82.15 versus 87.00. These selected means expose the choice between improving
   the annotation task and preserving the transfer/search outcomes. This is a
   descriptive comparison, not a newly optimized method or a claim of
   statistical dominance. It strengthens the practical interpretation more
   than another paragraph about the similarity kernel.
4. **Relation-specific selection has a real but conditional effect:** neutral
   exclusion beats random removal by 1.77 SugarCrepe points. Yet adding neutral
   exclusion to contradiction weighting reduces the supervised score by 0.535
   points, with the adjusted interval excluding zero. State the actual
   comparisons. They support decomposing the label operations; they do not
   support dismissing all semantic information or asserting a general
   interaction law.

The valid-rewording result remains useful supporting evidence: all 31 nonzero
adapters decline relative to frozen. Its baseline must stay explicit. Positive
expansion actually has slightly higher alternative-caption accuracy than the
source-positive procedure, so that loss cannot be attributed specifically to
adding positives.

## Main body versus appendix

| Content | Placement and reason |
|---|---|
| Search problem; three label meanings; fixed-pool intervention recipe | Early main text. These establish the decision before notation. |
| Source baseline includes supported hypotheses as negatives | Main methods. Essential to understand what promotion changes. |
| Backbone/adapters, common selection rule, source-or-supported validation relevance | Compact main methods. Needed to interpret the controlled comparison. |
| Main intervention table, edit-group table, 428-query result | Main. These carry the central argument. |
| One concrete retrieval example | Main if its image and captions remain legible. It makes the practical consequence accessible. |
| Constant-control comparison above | One concise main paragraph or a compact addition to the main table; do not add a second full control narrative. |
| Two-caption retention | One main result sentence and the SC++ column. Move the full 31-point scatter to appendix before shrinking explanatory text. |
| Common loss and positive-target identity | Late main text, after the reader has seen the problem and evidence. Keep the fixed-score assumption in the statement and the five-to-ten caption example. |
| Matched-negative-normalizer identity; four-row similarity table | Appendix. The identity defines an unexecuted exact reference, while the actual constant is 0.25. Its distinction is useful but starts a second story. |
| Pairwise-ranking epoch-zero outcome and validation trajectories | Appendix. Important for completeness; not central enough for a main paragraph. |
| Full per-seed data, all six intervals, source defects, reconstruction, classifier/calibration diagnostic, historical theorem corrections | Complete appendix and recovery archive. |

Suggested four-page order: page 1 poses the search decision and introduces the
label interventions; page 2 completes the shared study design and presents the
main outcome table; page 3 explains the edit-group and retrieval consequences
with the concrete example; page 4 provides the short objective explanation,
one focused control comparison, and the practical evaluation implication.
This keeps the first two to three pages understandable before derivative
notation appears. Exact float placement may vary.

## Cuts and claim discipline

Do not repeat the numerical result in the introduction, results, and closing
paragraph. State the question and design in the introduction, the measured
effects in results, and the operational implication at the end. The abstract
can retain the three headline deltas but should not also list every control.

The final implication should be concrete: **compare relation-supervised
procedures at an acceptable retrieval loss on representative development
queries**. The present small validation pool already rewards retrieval, so its
failure to preserve COCO performance explains why the development pool matters.
Reranking a frozen candidate set is an untested extension and does not itself
guarantee rank-one retention. It should not be the paper's closing promise.

The scope belongs in the study definition; detailed qualifications belong in
Limitations and the appendix. Keep only assumptions that make a displayed
claim true beside that claim. Avoid “only one,” “cannot conclude,” and repeated
assurances that outcomes were honestly measured in the main body.

## Candid reviewer assessment

The current evidence supports a compact empirical analysis paper. It does not
support a new successful training method, a new general theorem of retention,
or the claim that semantic labels are intrinsically unhelpful. The two
identities are direct objective calculations; treating them as the main
theoretical novelty would weaken the paper. They explain the controls.

As the retained prior-art review already establishes, degradation on valid
rewordings after hard-caption training is not new. The defensible contribution
is the matched decomposition of label roles, tied to item-type transfer and
full-pool retrieval. One backbone and a small adapter study limit the breadth
of that contribution. Stronger prose cannot remove this limit. The best
four-page paper makes this precise evidence easy to understand and useful to
someone choosing how to use additional annotations.
