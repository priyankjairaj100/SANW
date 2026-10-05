#!/usr/bin/env python3
"""Independent selected-state numerical training certificate for full v9 fits.

Reuses the standard artifact verifier and selected training row loader, then
independently enumerates every canonical pair and every declared inequality.
The shared feature archive materializes multiple splits; computations here
use only original training rows. No development or benchmark outcomes enter.
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
from audit_practical_constrained_fits_v8 import digest, independent_coordinates, independent_full_gallery
from evaluate_practical_official_development_v9 import verify_full_pilot
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from run_practical_joint_v9 import training_view


def lse(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def audit_one(run, protocol, protocol_sha, selection):
    started = time.monotonic()
    identity, _, _, state_record = verify_full_pilot(run, protocol, protocol_sha, selection)
    assert identity["family"] == "joint"
    config = identity["config"]
    completion = json.loads((run / "completion.json").read_text())
    model = ConstrainedBilinearScorer.load(run / "selected.npz")
    images, texts, source_rows, owner, sources, supported, contra, provenance = training_view(ROOT, identity["encoder"])
    assert provenance == identity["training_provenance"]
    assert len(images) == 1200 and len(source_rows) == 6000
    mean_error = max(float(np.max(np.abs(images.mean(axis=0) - model.image_mean))),
                     float(np.max(np.abs(texts.mean(axis=0) - model.text_mean))))
    assert mean_error < 1e-14
    pca = []
    for features, mean, basis in ((images, model.image_mean, model.image_basis), (texts, model.text_mean, model.text_basis)):
        covariance = (features - mean).T @ (features - mean)
        values = np.einsum("dr,ds,sr->r", basis, covariance, basis, optimize=True)
        eigen_error = float(np.max(np.abs(covariance @ basis - basis * values)))
        orthogonal_error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
        leading_error = float(np.max(np.abs(values - np.linalg.eigvalsh(covariance)[::-1][:basis.shape[1]])))
        assert eigen_error < 1e-8 and orthogonal_error < 1e-10 and leading_error < 1e-8
        pca.append(dict(rows=len(features), eigenvector_error=eigen_error, orthogonality_error=orthogonal_error, leading_eigenvalue_error=leading_error))
    checkpoints = completion["checkpoint_history"]
    assert json.loads((run / "history.json").read_text()) == checkpoints
    checkpoint_records = []
    for row, history in zip(checkpoints, completion["history"], strict=True):
        assert {k:v for k,v in row.items() if k != "checkpoint"} == history
        path = run / row["checkpoint"]["path"]
        assert digest(path) == row["checkpoint"]["sha256"]
        saved = ConstrainedBilinearScorer.load(path)
        for attr in ("image_mean", "text_mean", "image_basis", "text_basis"):
            assert np.array_equal(getattr(saved, attr), getattr(model, attr))
        norm = float(np.linalg.norm(saved.coefficient))
        assert row["nonzero"] == bool(np.any(saved.coefficient != 0))
        assert norm <= config["radius"] * (1 + 64 * np.finfo(np.float64).eps)
        cert = row["certificate"]
        norm_error = abs(norm-cert["coefficient_frobenius_norm"])
        assert norm_error <= 16*np.finfo(np.float64).eps*max(1.,norm)
        assert all(cert[k] is True for k in ("feasible_with_tolerance", "ranking_preserved", "ranking_checked_canonically"))
        assert cert["violated_constraints"] == cert["lost_frozen_correct_i2t"] == cert["lost_frozen_correct_t2i"] == 0
        checkpoint_records.append(dict(epoch=row["epoch"], sha256=digest(path), norm=norm, norm_reduction_absolute_error=norm_error))
    assert completion["optimizer_steps"] == config["epochs"] * int(np.ceil(len(images) / config["batch_size"]))
    chosen = min((row for row in checkpoints if row["nonzero"]), key=lambda row:(row["training_objective"],row["epoch"]))
    assert chosen["epoch"] == completion["selected_epoch"]
    frozen, residual = independent_full_gallery(images, texts[source_rows], model)
    trained = frozen + residual
    owned = np.arange(len(images))[:, None] == owner[None, :]
    frozen_i, frozen_t = frozen.argmax(axis=1), frozen.argmax(axis=0)
    protected_i = owner[frozen_i] == np.arange(len(images))
    protected_t = frozen_t == owner
    best_positive = np.where(owned, frozen, -np.inf).argmax(axis=1)
    gamma = config["retention_fraction"]
    i_slack = ((1-gamma)*(frozen[np.arange(len(images)), best_positive, None]-frozen)
               +residual[np.arange(len(images)), best_positive, None]-residual)
    t_slack = ((1-gamma)*(frozen[owner,np.arange(len(owner))][None,:]-frozen)
               +residual[owner,np.arange(len(owner))][None,:]-residual)
    i_valid, t_valid = ~owned & protected_i[:,None], ~owned & protected_t[None,:]
    counts = dict(protected_i2t_queries=int(protected_i.sum()), protected_t2i_queries=int(protected_t.sum()),
        checked_constraints=int(i_valid.sum()+t_valid.sum()),
        violated_constraints=int((i_slack[i_valid]<-1e-12).sum()+(t_slack[t_valid]<-1e-12).sum()),
        lost_frozen_correct_i2t=int((protected_i & (owner[trained.argmax(axis=1)] != np.arange(len(images)))).sum()),
        lost_frozen_correct_t2i=int((protected_t & (trained.argmax(axis=0) != owner)).sum()),
        current_i2t_correct=int((owner[trained.argmax(axis=1)]==np.arange(len(images))).sum()),
        current_t2i_correct=int((trained.argmax(axis=0)==owner).sum()))
    assert counts["violated_constraints"] == counts["lost_frozen_correct_i2t"] == counts["lost_frozen_correct_t2i"] == 0
    assert all(completion["final_certificate"][k]==v for k,v in counts.items())
    minimum = min(float(i_slack[i_valid].min()), float(t_slack[t_valid].min()))
    assert abs(minimum-completion["final_certificate"]["min_constraint_slack"]) < 1e-12
    # Independently reconstruct the retrieval and original joint scalar losses.
    logits = trained * identity["retrieval_logit_scale"]
    text_loss = np.mean(lse(logits, 0)-logits[owner,np.arange(len(owner))])
    other_owned = owned.copy(); other_owned[np.arange(len(images)),best_positive] = False
    logits[other_owned] = -np.inf
    image_loss = np.mean(lse(logits, 1)-logits[np.arange(len(images)),best_positive])
    retrieval_loss = float((image_loss+text_loss)/2)
    x = independent_coordinates(images, model.image_mean, model.image_basis)
    y = independent_coordinates(texts, model.text_mean, model.text_basis)
    xa = np.einsum("nd,dr->nr", x, model.coefficient, optimize=False)
    losses, accuracies, triplets = [], [], 0
    for i, (s,p,n) in enumerate(zip(sources,supported,contra,strict=True)):
        if not len(s) or not len(p) or not len(n):
            continue
        a,b,c = [np.sum(images[i]*texts[rows],axis=1)+np.sum(xa[i]*y[rows],axis=1) for rows in (s,p,n)]
        first=(c[None,None,:]-a[:,None,None]+config["composition_margin"])/config["temperature"]
        second=(c[None,None,:]-b[None,:,None]+config["composition_margin"])/config["temperature"]
        losses.append(float(np.logaddexp(np.logaddexp(0,first),second).mean()))
        gaps=np.minimum(a[:,None,None],b[None,:,None])-c[None,None,:]
        accuracies.append(float((gaps>0).mean())); triplets+=gaps.size
    comp_loss=float(np.mean(losses))
    multiplier=completion["fixed_composition_multiplier"]
    assert multiplier == config["composition_weight"] * completion["initial_retrieval_gradient_norm"] / max(completion["initial_joint_gradient_norm"],1e-12)
    objective=float(retrieval_loss+multiplier*comp_loss+config["ridge"]*np.sum(model.coefficient**2)/2)
    errors=dict(retrieval=abs(retrieval_loss-chosen["retrieval_loss"]), composition=abs(comp_loss-chosen["joint_composition_loss"]), objective=abs(objective-completion["selected_training_objective"]), accuracy=abs(float(np.mean(accuracies))-chosen["composition"]["image_mean_joint_accuracy"]))
    assert max(errors.values())<1e-11
    assert triplets==chosen["composition"]["triplet_count"] and len(losses)==chosen["composition"]["eligible_images"]
    return dict(passed=True,state=state_record,selected_epoch=completion["selected_epoch"],optimizer_steps=completion["optimizer_steps"],
        checkpoint_records=checkpoint_records,training_mean_max_error=mean_error,pca_checks=pca,
        independent_selected_certificate=dict(counts,min_constraint_slack=minimum),
        full_gallery_canonical_pairs=int(frozen.size),independent_training_objective=objective,
        scalar_recomputation_errors=errors,elapsed_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path); parser.add_argument('--selection',required=True,type=Path)
    parser.add_argument('--runs',nargs='+',required=True,type=Path); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    protocol=json.loads(args.protocol.read_text()); protocol_sha=digest(args.protocol)
    records=[]
    for run in args.runs:
        record=audit_one(run.resolve(),protocol,protocol_sha,args.selection.resolve()); records.append(record)
        print(json.dumps({k:record[k] for k in ('passed','selected_epoch','independent_selected_certificate','scalar_recomputation_errors','elapsed_seconds')}),flush=True)
    result=dict(study='sanw_practical_v9_independent_fullfit_audit',passed=True,
        meaning_of_passed='training_artifact_integrity_and_finite_training_numerical_certificate_only',
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__),
        protocol=dict(path=str(args.protocol),sha256=protocol_sha),selection=dict(path=str(args.selection),sha256=digest(args.selection)),
        fits=records,total_checkpoints=sum(len(r['checkpoint_records']) for r in records),
        total_canonical_pairs=sum(r['full_gallery_canonical_pairs'] for r in records),
        total_constraints=sum(r['independent_selected_certificate']['checked_constraints'] for r in records),
        no_fit_performed=True,no_development_or_benchmark_outcomes_computed=True,
        scope='All epoch artifact hashes/headers and declared minimum-objective selection checked. Selected full-gallery canonical winners and all inequalities independently recomputed; selected scalar loss independently reconstructed. Epoch rankings not individually rerun. Shared-cache loading may materialize other-split bytes, but numerical computations use original training rows only.')
    with args.output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(json.dumps(dict(audit=str(args.output),sha256=digest(args.output),passed=True)),flush=True)


if __name__=='__main__':
    main()
