#!/usr/bin/env python3
"""Independent bounded-memory audit of completed v10 training states.

The canonical pair scorer is explicitly enumerated rather than reusing the
training score cache, retrieval screener, constraint scanner or loss classes.
Only training rows enter numerical computation; loading the old lineage archive
can materialize other-split bytes. Run after large fits, not alongside them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_practical_constrained_fits_v8 import digest, independent_coordinates
from evaluate_practical_official_development_v10 import require_evaluation_threads, verify_full_pilot
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_training_data_v10 import load_training


def lse_vector(values):
    maximum = values.max()
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def independent_stream(images, texts, owner, model, gamma, scale, tolerance=1e-12, progress=False):
    """Exhaustive canonical scores and inequalities with O(N+M) score storage."""
    n, m = len(images), len(texts)
    assert owner.shape == (m,) and np.all((owner >= 0) & (owner < n))
    x = independent_coordinates(images, model.image_mean, model.image_basis)
    y = independent_coordinates(texts, model.text_mean, model.text_basis)
    xa = np.einsum("nd,dr->nr", x, model.coefficient, optimize=False)
    own_f, own_r = np.empty(m), np.empty(m)
    anchor = np.empty(n, dtype=np.int64)
    rows_by_owner = [np.flatnonzero(owner == i) for i in range(n)]
    for i, rows in enumerate(rows_by_owner):
        assert len(rows)
        own_f[rows] = np.sum(images[i] * texts[rows], axis=1, dtype=np.float64)
        own_r[rows] = np.sum(xa[i] * y[rows], axis=1, dtype=np.float64)
        anchor[i] = rows[np.argmax(own_f[rows])]
    frozen_i = np.empty(n, dtype=np.int64)
    trained_i = np.empty(n, dtype=np.int64)
    frozen_t, trained_t = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    frozen_t_max, trained_t_max = np.full(m, -np.inf), np.full(m, -np.inf)
    text_lse = np.full(m, -np.inf)
    t_minimum, t_violations = np.full(m, np.inf), np.zeros(m, dtype=np.int64)
    i_minimum, i_violations, i_constraint_count = np.inf, 0, 0
    image_losses = np.empty(n)
    for i in range(n):
        frozen = np.sum(images[i] * texts, axis=1, dtype=np.float64)
        residual = np.sum(xa[i] * y, axis=1, dtype=np.float64)
        trained = frozen + residual
        frozen_i[i], trained_i[i] = frozen.argmax(), trained.argmax()
        # Strict updates preserve the lowest manifest index on exact ties.
        better = frozen > frozen_t_max
        frozen_t[better], frozen_t_max[better] = i, frozen[better]
        better = trained > trained_t_max
        trained_t[better], trained_t_max[better] = i, trained[better]
        unowned = owner != i
        if owner[frozen_i[i]] == i:
            slack = (1-gamma)*(own_f[anchor[i]]-frozen) + own_r[anchor[i]]-residual
            i_minimum = min(i_minimum, float(slack[unowned].min()))
            i_violations += int(np.count_nonzero(slack[unowned] < -tolerance))
            i_constraint_count += int(np.count_nonzero(unowned))
        slack = (1-gamma)*(own_f-frozen) + own_r-residual
        t_minimum[unowned] = np.minimum(t_minimum[unowned], slack[unowned])
        t_violations[unowned] += slack[unowned] < -tolerance
        logits = scale * trained
        text_lse = np.logaddexp(text_lse, logits)
        other_owned = ~unowned
        other_owned[anchor[i]] = False
        anchor_logit = logits[anchor[i]]
        logits[other_owned] = -np.inf
        image_losses[i] = lse_vector(logits)-anchor_logit
        if progress and (i+1) % 500 == 0:
            print(json.dumps({"independent_canonical_image_rows": i+1, "total_rows": n}), flush=True)
    protected_i = owner[frozen_i] == np.arange(n)
    protected_t = frozen_t == owner
    current_i = owner[trained_i] == np.arange(n)
    current_t = trained_t == owner
    minimum = min(i_minimum, float(t_minimum[protected_t].min()) if protected_t.any() else np.inf)
    counts = dict(protected_i2t_queries=int(protected_i.sum()), protected_t2i_queries=int(protected_t.sum()),
                  checked_constraints=i_constraint_count+int(protected_t.sum())*(n-1),
                  violated_constraints=i_violations+int(t_violations[protected_t].sum()),
                  lost_frozen_correct_i2t=int((protected_i & ~current_i).sum()),
                  lost_frozen_correct_t2i=int((protected_t & ~current_t).sum()),
                  current_i2t_correct=int(current_i.sum()), current_t2i_correct=int(current_t.sum()),
                  min_constraint_slack=minimum)
    retrieval_loss = float((image_losses.mean() + np.mean(text_lse-scale*(own_f+own_r)))/2)
    return counts, retrieval_loss, dict(frozen_i=frozen_i, trained_i=trained_i, frozen_t=frozen_t,
                                      trained_t=trained_t, best_owned_anchor=anchor)


def pca_check(features, mean, basis):
    covariance = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    for start in range(0, len(features), 1024):
        centered = features[start:start+1024]-mean
        covariance += centered.T @ centered
    values = np.einsum("dr,ds,sr->r", basis, covariance, basis, optimize=True)
    errors = dict(eigenvector_max_error=float(np.max(np.abs(covariance@basis-basis*values))),
                  orthogonality_max_error=float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1])))),
                  leading_eigenvalue_max_error=float(np.max(np.abs(values-np.linalg.eigvalsh(covariance)[::-1][:basis.shape[1]]))))
    assert errors["eigenvector_max_error"] < 1e-7 and errors["orthogonality_max_error"] < 1e-10
    assert errors["leading_eigenvalue_max_error"] < 1e-7
    return dict(rows=len(features), **errors)


def composition_statistics(images, texts, sources, supported, contra, model, config):
    x = independent_coordinates(images, model.image_mean, model.image_basis)
    y = independent_coordinates(texts, model.text_mean, model.text_basis)
    xa = np.einsum("nd,dr->nr", x, model.coefficient, optimize=False)
    losses, canonical_accuracies, training_accuracies, ambiguous_fractions = [], [], [], []
    triplets = disagreements = ambiguous_count = 0
    maximum_margin_band = 0.
    rank = model.coefficient.shape[0]
    coefficient_norm = float(np.linalg.norm(model.coefficient))
    for i, (s,p,n) in enumerate(zip(sources,supported,contra,strict=True)):
        if not len(s) or not len(p) or not len(n):
            continue
        groups = (s,p,n)
        base = [np.sum(images[i]*texts[rows],axis=1) for rows in groups]
        a,b,c = [frozen+np.sum(xa[i]*y[rows],axis=1) for frozen,rows in zip(base,groups,strict=True)]
        first=(c[None,None,:]-a[:,None,None]+config["composition_margin"])/config["temperature"]
        second=(c[None,None,:]-b[None,:,None]+config["composition_margin"])/config["temperature"]
        losses.append(float(np.logaddexp(np.logaddexp(0,first),second).mean()))
        gaps=np.minimum(a[:,None,None],b[None,:,None])-c[None,None,:]
        canonical = gaps > 0
        # This independently reproduces the descriptive training summary's
        # two BLAS reductions. Smooth objective equivalence does not imply
        # thresholded accuracy equality at exact or near-zero margins.
        projected = x[i] @ model.coefficient
        ta,tb,tc = [frozen+y[rows]@projected for frozen,rows in zip(base,groups,strict=True)]
        training = np.minimum(ta[:,None,None],tb[None,:,None])-tc[None,None,:] > 0
        largest_y = max(float(np.linalg.norm(y[rows],axis=1).max()) for rows in groups)
        score_band = (64*(2*rank+4)*np.finfo(np.float64).eps
                      *(1+np.linalg.norm(x[i])*coefficient_norm*largest_y))
        margin_band = 2*score_band
        ambiguous = np.abs(gaps) <= margin_band
        assert np.all(~(canonical != training) | ambiguous)
        disagreements += int(np.count_nonzero(canonical != training))
        ambiguous_count += int(ambiguous.sum())
        maximum_margin_band = max(maximum_margin_band,float(margin_band))
        canonical_accuracies.append(float(canonical.mean()))
        training_accuracies.append(float(training.mean()))
        ambiguous_fractions.append(float(ambiguous.mean()))
        triplets += gaps.size
    canonical_accuracy,training_accuracy = float(np.mean(canonical_accuracies)),float(np.mean(training_accuracies))
    ambiguity_bound = float(np.mean(ambiguous_fractions))
    assert abs(canonical_accuracy-training_accuracy) <= ambiguity_bound+1e-15
    return dict(loss=float(np.mean(losses)),canonical_image_mean_joint_accuracy=canonical_accuracy,
                training_reduction_image_mean_joint_accuracy=training_accuracy,
                eligible_images=len(losses),triplet_count=triplets,
                canonical_vs_training_disagreement_triplets=disagreements,
                roundoff_ambiguous_triplets=ambiguous_count,image_balanced_ambiguity_bound=ambiguity_bound,
                maximum_margin_roundoff_band=maximum_margin_band,
                roundoff_band_rule='2*64*(2*rank+4)*float64eps*(1+norm(x)*frobenius_norm(A)*max(norm(y)))')


def audit_one(run, protocol, protocol_sha):
    started = time.monotonic()
    identity, _, state = verify_full_pilot(run, protocol, protocol_sha)
    config = identity["config"]
    completion = json.loads((run/"completion.json").read_text())
    model = ConstrainedBilinearScorer.load(run/"selected.npz")
    images, texts, source_rows, owner, sources, supported, contra, provenance, scale = load_training(ROOT, identity["encoder"], protocol)
    assert provenance == identity["training_provenance"] and scale == identity["retrieval_logit_scale"]
    assert len(images) == 6000 and len(source_rows) == 30000
    mean_error = max(float(np.max(np.abs(images.mean(axis=0)-model.image_mean))),
                     float(np.max(np.abs(texts.mean(axis=0)-model.text_mean))))
    assert mean_error < 1e-14
    pca_checks = [pca_check(features, mean, basis) for features, mean, basis in
                  ((images, model.image_mean, model.image_basis), (texts, model.text_mean, model.text_basis))]
    checkpoints = completion["checkpoint_history"]
    assert json.loads((run/"history.json").read_text()) == checkpoints
    records = []
    for row, history in zip(checkpoints, completion["history"], strict=True):
        assert {k:v for k,v in row.items() if k != "checkpoint"} == history
        path = run/row["checkpoint"]["path"]
        assert digest(path) == row["checkpoint"]["sha256"]
        saved = ConstrainedBilinearScorer.load(path)
        for key in ("image_mean", "text_mean", "image_basis", "text_basis"):
            assert np.array_equal(getattr(saved, key), getattr(model, key))
        norm = float(np.sqrt(np.sum(saved.coefficient*saved.coefficient)))
        assert np.isfinite(norm) and norm <= config["radius"]*(1+64*np.finfo(np.float64).eps)
        assert row["nonzero"] == bool(np.any(saved.coefficient != 0))
        cert = row["certificate"]
        assert all(cert[k] is True for k in ("ranking_checked_canonically", "ranking_preserved", "feasible_with_tolerance"))
        assert cert["violated_constraints"] == cert["lost_frozen_correct_i2t"] == cert["lost_frozen_correct_t2i"] == 0
        norm_error = abs(norm-cert["coefficient_frobenius_norm"])
        assert norm_error <= 32*np.finfo(np.float64).eps*max(1., norm)
        records.append(dict(epoch=row["epoch"], sha256=digest(path), norm=norm, norm_absolute_error=norm_error))
    assert completion["optimizer_steps"] == config["epochs"]*((len(images)+config["batch_size"]-1)//config["batch_size"])
    chosen = min((row for row in checkpoints if row["nonzero"]), key=lambda row:(row["training_objective"], row["epoch"]))
    assert chosen["epoch"] == completion["selected_epoch"]
    cert, retrieval_loss, _ = independent_stream(images, texts[source_rows], owner, model,
                                                config["retention_fraction"], scale, config["feasibility_tolerance"], True)
    assert cert["violated_constraints"] == cert["lost_frozen_correct_i2t"] == cert["lost_frozen_correct_t2i"] == 0
    for key, value in cert.items():
        if key == "min_constraint_slack":
            assert abs(value-completion["final_certificate"][key]) < 1e-12
        else:
            assert value == completion["final_certificate"][key], key
    composition = composition_statistics(images,texts,sources,supported,contra,model,config)
    comp_loss = composition['loss']
    multiplier = completion["fixed_composition_multiplier"]
    assert multiplier == config["composition_weight"]*completion["initial_retrieval_gradient_norm"]/max(completion["initial_joint_gradient_norm"],1e-12)
    objective = float(retrieval_loss+multiplier*comp_loss+config["ridge"]*np.sum(model.coefficient**2)/2)
    errors=dict(retrieval=abs(retrieval_loss-chosen["retrieval_loss"]), composition=abs(comp_loss-chosen["joint_composition_loss"]),
                objective=abs(objective-completion["selected_training_objective"]),
                training_summary_accuracy=abs(composition['training_reduction_image_mean_joint_accuracy']-chosen["composition"]["image_mean_joint_accuracy"]))
    assert max(errors.values()) < 1e-10
    assert composition['triplet_count'] == chosen["composition"]["triplet_count"]
    assert composition['eligible_images'] == chosen["composition"]["eligible_images"]
    return dict(passed=True, state=state, selected_epoch=completion["selected_epoch"], optimizer_steps=completion["optimizer_steps"],
                checkpoint_records=records, training_mean_max_error=mean_error, pca_checks=pca_checks,
                independent_selected_certificate=cert, full_gallery_canonical_pairs=len(images)*len(source_rows),
                independent_training_objective=objective, scalar_recomputation_errors=errors,
                independent_composition_diagnostics=composition,
                elapsed_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--runs',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an audit')
    require_evaluation_threads()
    protocol=json.loads(args.protocol.read_text()); protocol_sha=digest(args.protocol)
    fits=[]
    for run in args.runs:
        fit=audit_one(run.resolve(),protocol,protocol_sha); fits.append(fit)
        print(json.dumps({k:fit[k] for k in ('passed','selected_epoch','independent_selected_certificate','scalar_recomputation_errors','elapsed_seconds')}),flush=True)
    result=dict(study='sanw_practical_v10_independent_fullfit_audit',passed=True,
                meaning_of_passed='training_artifact_integrity_and_selected_finite_training_numerical_certificate_only',
                created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(__file__),
                direct_audit_dependency_sha256={name:digest(ROOT/name) for name in (
                    'scripts/audit_practical_constrained_fits_v8.py',
                    'scripts/evaluate_practical_official_development_v10.py')},
                protocol=dict(path=str(args.protocol),sha256=protocol_sha), fits=fits,
                total_checkpoints=sum(len(f['checkpoint_records']) for f in fits),
                total_canonical_pairs=sum(f['full_gallery_canonical_pairs'] for f in fits),
                total_constraints=sum(f['independent_selected_certificate']['checked_constraints'] for f in fits),
                no_fit_performed=True,no_development_or_benchmark_outcomes_computed=True,
                scope='All epoch artifact hashes/headers and recorded minimum-objective selection checked. Selected full-gallery canonical winners, every inequality and scalar objective independently recomputed without training cache/scanner/loss classes. Descriptive training-composition accuracy reduction is reproduced separately; canonical discrepancies permitted only for explicitly bounded roundoff-ambiguous margins. Epoch rankings/objectives not individually rerun. Fixed multiplier checked against recorded initial gradient norms; independent finite-difference tests cover loss gradients. Old lineage archive may materialize other-split bytes; numerical computation uses training rows only.',
                memory_scope='O(N+M) score storage; total memory also includes O(MD) feature/product temporaries and O(D^2) PCA covariance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(audit=str(args.output),sha256=digest(args.output),passed=True)),flush=True)


if __name__=='__main__':
    main()
