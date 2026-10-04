# Exact gallery-level rank-retention geometry

Status: implemented and independently numerically audited on 2026-10-04. This note contains mathematical and software results, **not new retrieval results**. It does not change the frozen selection protocol, practical success gate, or primary retrieval endpoints. Exact positive-set R@1 at the already declared temperature is the main diagnostic. R@K and combined certificates below are auxiliary derivations, not new primary hypotheses.

## What improves over the previous argument

The earlier two-coordinate KL bound only protects the teacher's best relevant caption against its best irrelevant caption. Retrieval succeeds when **any** relevant caption appears in the top K. We derive the exact forward-KL distance to losing that event, including multiple relevant captions, zero probabilities, and pessimistic ties. The implementation returns the attaining projection as an independently checkable witness.

For R@1 this is the positive-set pooling extension anticipated in `STRENGTHENING_RELATED_WORK.md`. The new extension handles arbitrary K: select the teacher's K largest irrelevant probabilities, pool the relevant probabilities above a common level together with the selected irrelevant probabilities below that level, and leave all remaining coordinates unchanged. It is unnecessary to enumerate all combinations of K distractors. We also derive an exact radius for shift-invariant logit drift and a sharp aggregate bound given only a total full-gallery KL budget.

These strengthen the diagnostic mathematics. They do not make KL distillation a novel training principle, nor prove that a minibatch retention loss generalizes to a different gallery or previously unseen queries.

## Definitions and assumptions

Fix one query, a finite gallery with n candidates, and a nonempty relevant index set R. Let N be the irrelevant set. Teacher and student probabilities q,p lie in the same probability simplex over exactly this gallery. Natural logarithms give divergences in nats. Define

D(q || p) = sum_i q_i log(q_i / p_i),

with 0 log(0/p)=0 and a positive numerator with zero denominator contributing infinity. Probabilities need not be strictly positive. When probabilities are obtained from softmax scores, both models use the same specified positive temperature T and candidate set.

For an integer K>=1 define the **closed failure event**

F_K = {p: at least K indices j in N satisfy p_j >= max_{i in R} p_i}.

Thus cross-relevance ties count against the relevant item. A point outside F_K has a relevant item in its first K positions under every tie-breaking rule. A point in F_K may still be successful under a favorable fixed tie rule; using the closed event gives a tie-robust sufficient certificate. If |N|<K, failure is impossible.

Write a_1 >= ... >= a_r for the relevant teacher probabilities and b_1 >= ... >= b_s for the irrelevant probabilities. The teacher is strictly correct at K exactly when a_1>b_K, provided K<=s. Define

C_K(q,R) = min_{p in F_K} D(q || p).

## Theorem 1: exact positive-set distance to top-K failure

If |N|<K, C_K=+infinity. If a_1<=b_K, the teacher already belongs to F_K and C_K=0. Otherwise, take S to be the indices of the K largest irrelevant teacher probabilities. Ties within N may be resolved arbitrarily.

There is a unique threshold t between min_{j in S}q_j and max_{i in R}q_i solving

sum_{i in R}(q_i-t)_+ = sum_{j in S}(t-q_j)_+.

Define A={i in R:q_i>t} and B={j in S:q_j<t}. The attaining distribution is

- p_i*=min(q_i,t) for relevant i;
- p_j*=max(q_j,t) for selected irrelevant j in S;
- p_l*=q_l elsewhere.

The total probability is unchanged, because the threshold equation balances the mass removed and added. Moreover,

t = [sum_{i in A}q_i + sum_{j in B}q_j] / (|A|+|B|),

and the exact threshold is

C_K = sum_{i in A}q_i log(q_i/t) + sum_{j in B}q_j log(q_j/t).

A coordinate equal to t may be included in or excluded from a pooled set without changing the distribution or cost. Zero-probability selected irrelevant coordinates can be active: they receive t, contribute zero directly to KL, and affect t through the pool's denominator. Unselected zero coordinates stay zero. A correctly ranked teacher has a_1>0, so its active pooling threshold is positive.

### Proof: which irrelevant candidates are closest to failure?

For a fixed K-element set S, the corresponding failure constraint is p_j>=p_i for all j in S and i in R. The full failure event is the union of these constraints over S.

Suppose u belongs to S, v does not belong to S, both are irrelevant, and q_v>=q_u. Consider any feasible finite-cost p. If p_v>=p_u, the same p is already feasible after replacing u by v. If p_v<p_u, swap p_u and p_v. The swapped distribution is feasible for the replaced set, and its KL cost changes by

(q_v-q_u) log(p_v/p_u) <= 0.

The inequality extends to zero coordinates by the usual KL boundary convention or a limit of positive feasible distributions. Repeating these exchanges gives a set containing the K largest irrelevant teacher probabilities without increasing the attainable cost. This proves that only this one set S needs to be solved. It also avoids relying on differentiability while an active set changes.

### Proof: projection for this fixed set

Introduce an auxiliary t and constraints p_i<=t for i in R and p_j>=t for j in S. The objective, up to a q-only constant, is the convex function -sum_i q_i log p_i. The proposed p* preserves total mass and is feasible.

For a direct optimality certificate, use simplex multiplier lambda=1. For active relevant i use multiplier mu_i=q_i/t-1>=0 on p_i-t<=0. For active selected irrelevant j use nu_j=1-q_j/t>=0 on t-p_j<=0. All inactive order multipliers vanish. At positive coordinates, these choices satisfy stationarity with respect to p. Stationarity with respect to t holds because

sum_{i in A}mu_i = sum_{j in B}nu_j,

which is exactly the pooled-mean identity. For an unpooled zero coordinate, the nonnegativity multiplier can be set to one; it satisfies the boundary stationarity condition. Complementarity and primal feasibility hold. These convex KKT conditions are sufficient for global optimality. Equivalently, the supporting-hyperplane inequality for -q_i log p_i and these multipliers proves that no feasible distribution has lower cost, without requiring strict positivity of q.

The root exists because the continuous function

F(t)=sum_R(q_i-t)_+ - sum_S(t-q_j)_+

is positive at the smallest selected irrelevant value and negative at the largest relevant value in the teacher-correct case. It is strictly decreasing between these bounds: at every interior point there is an active relevant coordinate or an active selected irrelevant coordinate. Its root is unique. Substitution into KL gives the stated cost. QED.

### R@1 specialization and implementation

For K=1 let b be the largest irrelevant teacher probability. Start with pool {b}. Add relevant probabilities a_1,a_2,... in decreasing order as long as the next one exceeds the current pool mean. Stop when the mean t is at least the next relevant probability. If m relevant coordinates joined,

t=(b+sum_{i<=m}a_i)/(m+1),

C_1=sum_{i<=m}a_i log(a_i/t)+b log(b/t).

Adding a relevant coordinate raises the mean but cannot raise it above the smallest relevant coordinate already in the pool, so earlier active coordinates stay active. This proves the simple one-pass pooling algorithm. Gallery cost is O(n+r log r) and working space O(n) when returning the full projection.

For general K, the implementation obtains the top K irrelevant values without a full gallery sort. F is linear between its at most r+K distinct breakpoints, so the root on each candidate interval is the arithmetic mean of that interval's active values. The present implementation scans these small sets directly; its pooling cost is O((r+K)^2), in addition to the O(n) gallery scan and O(K log K) ordering. For the source-caption tasks r is 5 or 1, making this negligible for R@1. A formal interval arithmetic implementation would be needed for machine-verified rather than ordinary floating-point certificates.

## Corollary 1: a strict, optimal KL-radius certificate

If D(q||p)<C_K(q,R), the student is outside F_K and retains a relevant candidate in the top K. This is simply the definition of the distance to failure, with Theorem 1 making that distance computable.

The strict inequality matters. The projection p* is a failure under pessimistic ties and attains equality D=C_K. With a deterministic favorable tie rule, distributions with all selected irrelevant probabilities strictly above all relevant probabilities can approach p* arbitrarily closely. Consequently, no larger open KL radius is valid in general. If the teacher is already tied or incorrect, C_K=0 and this argument certifies nothing. If failure is impossible because there are too few irrelevant candidates, retention holds independently of KL.

Unsafe overflow in score centering, temperature scaling, KL, native margins, or cross-model drift raises `ValueError`; it cannot produce a passing certificate. Independent audit found and corrected the extreme finite-score case teacher=(1e308,-1e308), student=(-1e308,1e308), where unchecked arithmetic would otherwise produce NaN and a false pass. Three regression cases cover this issue, tiny temperatures, and cross-model offset overflow.

The numerical implementation uses D + atol + rtol*|C_K| < C_K, with defaults 1e-12 and 1e-10. These tolerances make near-boundary decisions more conservative. They do not convert floating-point calculations into a formal interval-arithmetic proof.

### Comparison with the earlier pairwise bound

Let a=a_1 and b=b_K in a teacher-correct case. Ignoring the other required ranking constraints gives

C_pair = a log[2a/(a+b)] + b log[2b/(a+b)].

Then

C_K >= C_pair >= (a-b)^2/[2(a+b)] >= (a-b)^2/2.

The middle inequality follows by applying binary Pinsker to the two-coordinate distribution normalized by a+b. Thus the exact positive-set condition weakly dominates the old two-coordinate condition, which in turn dominates the simple unnormalized probability-gap Pinsker bound.

For example, with q=(.35,.30,.25,.10) and the first three captions relevant, the attaining distribution is (.25,.25,.25,.25). The exact threshold is 0.08083268 nats, versus 0.07354844 nats for the two-coordinate bound, an approximately 9.90% larger admissible KL radius. This is an illustrative deterministic calculation, not a retrieval experiment.

The radius always satisfies C_K<=log(K+1) when failure is possible. To see this, mix q with a point mass at each of K selected irrelevant candidates, giving p=(q+sum_{j in S}e_j)/(K+1). This p lies in F_K and p_i>=q_i/(K+1), so D(q||p)<=log(K+1). The bound is attained when q is a point mass on one relevant candidate. This also checks zero-probability behavior of the exact formula.

## Theorem 2: exact shift-invariant logit radius

Let z and z+d be teacher and student scores on the same gallery. Softmax does not change their ranks. Let

g_K = max_{i in R}z_i - z_(K,N),

where z_(K,N) is the K-th largest irrelevant score. For a teacher-correct query, g_K>0. Define the oscillation of the score change by

osc(d)=max_i d_i-min_i d_i = 2 min_c ||d-c*1||_infinity.

If osc(d)<g_K, the student retains a relevant item in the top K. Proof: choose a teacher-best relevant i*. Every irrelevant candidate with teacher score at most z_(K,N) remains below it because

(z_i*+d_i*)-(z_j+d_j) >= g_K-osc(d)>0.

At most K-1 irrelevant candidates had scores above z_(K,N), proving the claim. The radius is sharp. Decrease every relevant score by g_K/2 and increase the K teacher-largest irrelevant scores by g_K/2, leaving the rest unchanged. The oscillation is g_K and the K selected candidates all tie or beat every relevant candidate. Thus the exact distance to F_K under the shift-invariant infinity norm is g_K/2.

This improves the usual uncentered condition 2||d||_infinity<g_K by removing harmless shared score shifts. It is invariant to common positive temperature rescaling when score and margin units are treated consistently.

### KL and logit certificates are complementary

For q=softmax(z/T), p=softmax((z+d)/T), forward KL has the exact identity

D(q||p)=log E_q exp(d/T)-E_q[d/T].

Consequently Hoeffding's elementary bounded-variable inequality gives D<=osc(d)^2/(8T^2). The directly measured KL can be much smaller. Large changes to a tiny-probability distractor can break the worst-coordinate logit condition while preserving the KL condition; harmless changes spread across several distractors can do the reverse.

Two deterministic examples at T=1 are included in the tests:

| Example | Exact KL certificate | Centered logit certificate | Actual result |
|---|---|---|---|
| q=(.6,.3999,.0001), p=(.6,.399,.001), relevant first | Pass: KL=.00067076 < C=.02015784 | Fail: oscillation=2.30484 > gap=.40572 | Retained |
| z=log(.34,.33,.33), d=(0,.029,.029), relevant first | Fail: KL=.00009407 > C=.00007463 | Pass: oscillation=.029 < gap=.02985296 | Retained |

Taking the union of the two sufficient certificates is valid and cannot reduce coverage. Its additional coverage, and that of exact set KL over pairwise KL, should be reported as auxiliary explanatory measurements. They must not select checkpoints or create a new post-hoc practical-success criterion.

## Corollary 2: what a total full-gallery KL budget can establish

For a collection of teacher-correct queries with positive thresholds C_i, any query that loses retrieval costs at least C_i. If all full-gallery divergences sum to at most B, sort the thresholds increasingly, C_(1)<=...<=C_(m). The number of lost queries is at most

max{h: sum_{i=1}^h C_(i)<=B}.

This is sharper than dividing B by the smallest margin, because it uses the distribution of query difficulties. It follows by taking the cheapest h possible failures. The bound is sharp if student distributions may vary independently by query, since Theorem 1 supplies a failure projection attaining every threshold. A shared embedding model may impose extra constraints, so this remains a valid upper bound there but need not be attainable. Infinite thresholds correspond to impossible failures and are excluded from the purchasable failures.

This statement concerns a total KL budget **on the very same evaluated galleries and queries**. It does not transfer an empirical training loss to unseen queries. The helper `failures_from_total_kl` includes a small upward numerical budget allowance, making borderline counts conservative; set its tolerances to zero only when intentionally inspecting raw floating-point comparisons.

## Two deterministic scope counterexamples

1. **Zero minibatch KL, failed full-gallery retrieval.** Teacher scores are (2,1,0), with the first candidate relevant. The training subgallery contains only candidates 1 and 2. Student scores become (2,1,10). Teacher and student distributions on the subgallery are identical: KL=0 and its local rank certificate passes. On the complete gallery the third, irrelevant candidate wins. Hence even zero minibatch KL does not certify full-gallery retrieval.
2. **Tiny average KL, a failed individual query.** Let 100 queries have teacher probabilities (.51,.49), with the first candidate relevant. Leave 99 unchanged and switch one student to (.49,.51). The mean KL is below the individual threshold, but that query fails. Average KL cannot be substituted into each query's certificate. Corollary 2 gives the appropriate aggregate implication when a genuine full-gallery budget is known.

Other limitations follow directly: removing candidates changes the probability normalization and event; supported hypotheses counted relevant in one domain do not become source-owned captions in another; a temperature-square multiplier in a training objective is not part of the unscaled KL in Theorem 1; and native cosine scores versus temperature-scaled logits must be labeled consistently.

## Numerical audit and executable API

Files owned by this work:

- `src/gcr/rank_retention.py`: scalar projection/certificate, stable logit diagnostics, aggregate budget bound.
- `tests/test_rank_retention.py`: 13 test groups, including 45 primal-optimization configurations; every K-subset of wrong candidates is enumerated in the independent small-gallery optimizer, with two initialization choices and explicit simplex/order constraints.
- This document: proof, scope, and usage.

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_rank_retention.py -v
```

The independent SciPy SLSQP objectives contain no pooling formula. Analytic and numerically optimized KL distances agree to absolute tolerance 2e-7 in all 45 audited configurations. Additional checks cover exact single-pair thresholds; multi-positive improvement; teacher errors and ties; absent negatives; zero-support boundaries; top-K monotonicity; projection simplex/KL identities; random sufficient certificates; shift invariance; finite evaluation under extreme student logits; and both counterexamples above. Tests verify both success and failure cases. These finite checks support the implementation; the analytical proof carries the universal claim.

Main evaluation helper:

```python
from gcr.rank_retention import rank_retention_diagnostics

arrays = rank_retention_diagnostics(
    teacher_logits,              # [queries, candidates], native declared scale
    student_logits,              # identical query/candidate order
    relevant_indices_per_query,  # or a same-shape boolean relevance mask
    temperature=2.0,
    k=1,
)
```

The routine processes queries one at a time with O(gallery-size) extra working storage. It returns per-query `threshold`, `divergence`, `certified`, `teacher_correct`, `student_correct`, `probability_margin`, `logit_margin`, `logit_drift_oscillation`, `logit_drift_linf_centered`, `logit_drift_certified`, `pairwise_threshold`, `pairwise_certified`, and `combined_certified`. KL outputs are unmultiplied nats at the declared diagnostic temperature. Logit-margin and drift outputs are in native score units. Probabilities are never clipped to artificial floors; the stable log-probability computation avoids an infinite divergence merely because a finite-logit student's small probability underflows when exponentiated.

Actual empirical ranking may use a stable candidate-index tie break. The diagnostic correctness arrays instead use pessimistic ties, so they must not silently replace the benchmark's existing raw-rank arrays. Coverage denominators should distinguish all queries, strictly teacher-correct queries, and actually retained teacher-correct queries. Compare exact-set coverage to pairwise coverage using aligned per-query indicators; report actual retention alongside both.

## Positioning and claims

This is a setting-specific exact convex projection and robustness derivation. General surrogate-risk calibration and divergence-to-decision analyses are established. Relevant primary references checked on 2026-10-04 include:

- Bartlett, Jordan, and McAuliffe, *Convexity, Classification, and Risk Bounds*, Berkeley technical report 638, 2003: <https://statistics.berkeley.edu/tech-reports/638>. Its stated subject is quantitative relationships between convex surrogate risk and classification risk.
- Avila Pires and Szepesvari, *Multiclass Classification Calibration Functions*, 2016: <https://arxiv.org/abs/1609.06385>. Its stated subject is computing multiclass calibration functions that translate surrogate-risk bounds into classification-risk bounds.

Neither these checks nor this derivation is an exhaustive priority search. Do not claim the projection machinery, frozen-teacher KL, or the general idea of rank stability as a novel principle. An appropriate contribution claim is a precisely scoped, computable diagnostic for this study's many-caption retrieval event, with any practical value supported by its measured coverage. The main experimental contribution remains the controlled directional intervention, fair preservation baselines, and replication.
