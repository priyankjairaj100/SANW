"""Check the independent bounded-memory auditor against dense enumeration."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_practical_streaming_fullfits_v10 import independent_stream, composition_statistics
from audit_practical_constrained_fits_v8 import independent_full_gallery
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer


@pytest.mark.parametrize('tied,seed', [(False, 5), (False, 19), (True, 5)])
def test_streamed_independent_audit_matches_dense(tied, seed):
    rng=np.random.default_rng(seed)
    images=rng.normal(size=(5,6)); texts=rng.normal(size=(11,6))
    images/=np.linalg.norm(images,axis=1,keepdims=True)
    texts/=np.linalg.norm(texts,axis=1,keepdims=True)
    owner=np.array([0,0,0,1,2,2,3,3,3,3,4])
    if tied:
        images[1]=images[0];texts[3]=texts[0]
    model=ConstrainedBilinearScorer.from_training(images,texts,3)
    model.coefficient[:]=rng.normal(size=(3,3))*.2
    gamma,scale,tol=.5,17.,1e-12
    cert,loss,pred=independent_stream(images,texts,owner,model,gamma,scale,tol)
    frozen,residual=independent_full_gallery(images,texts,model)
    trained=frozen+residual
    owned=np.arange(len(images))[:,None]==owner[None,:]
    pi=owner[frozen.argmax(axis=1)]==np.arange(len(images))
    pt=frozen.argmax(axis=0)==owner
    anchor=np.where(owned,frozen,-np.inf).argmax(axis=1)
    a=np.arange(len(images));b=np.arange(len(texts))
    islack=(1-gamma)*(frozen[a,anchor,None]-frozen)+residual[a,anchor,None]-residual
    tslack=(1-gamma)*(frozen[owner,b][None,:]-frozen)+residual[owner,b][None,:]-residual
    imask=pi[:,None]&~owned;tmask=pt[None,:]&~owned
    mins=[x.min() for x in (islack[imask],tslack[tmask]) if x.size]
    assert cert==dict(protected_i2t_queries=int(pi.sum()),protected_t2i_queries=int(pt.sum()),
        checked_constraints=int(imask.sum()+tmask.sum()),
        violated_constraints=int((islack[imask]<-tol).sum()+(tslack[tmask]<-tol).sum()),
        lost_frozen_correct_i2t=int((pi&(owner[trained.argmax(axis=1)]!=a)).sum()),
        lost_frozen_correct_t2i=int((pt&(trained.argmax(axis=0)!=owner)).sum()),
        current_i2t_correct=int((owner[trained.argmax(axis=1)]==a).sum()),
        current_t2i_correct=int((trained.argmax(axis=0)==owner).sum()),min_constraint_slack=min(mins) if mins else np.inf)
    for key,expected in [('frozen_i',frozen.argmax(axis=1)),('frozen_t',frozen.argmax(axis=0)),
                         ('trained_i',trained.argmax(axis=1)),('trained_t',trained.argmax(axis=0)),('best_owned_anchor',anchor)]:
        np.testing.assert_array_equal(pred[key],expected)
    logits=scale*trained
    t_loss=np.mean(np.logaddexp.reduce(logits,axis=0)-logits[owner,b])
    mask=owned.copy();mask[a,anchor]=False
    logits[mask]=-np.inf
    i_loss=np.mean(np.logaddexp.reduce(logits,axis=1)-logits[a,anchor])
    assert abs(loss-(t_loss+i_loss)/2)<1e-13


def test_descriptive_training_accuracy_roundoff_does_not_fail_canonical_audit():
    from gcr.practical_joint_v9 import JointCompositionExamples, JointFitConfig
    rng=np.random.default_rng(2)
    images=rng.normal(size=(5,12));texts=rng.normal(size=(24,12))
    images/=np.linalg.norm(images,axis=1,keepdims=True)
    texts/=np.linalg.norm(texts,axis=1,keepdims=True)
    texts[23]=texts[0]
    model=ConstrainedBilinearScorer.from_training(images,texts,4)
    model.coefficient=rng.normal(size=(4,4))*.3
    sources=[np.arange(5) for _ in images]
    supported=[np.array([22]) for _ in images]
    contra=[np.array([23]) for _ in images]
    config=JointFitConfig(rank=4)
    actual=JointCompositionExamples(images,texts,sources,supported,contra,model)
    checked=composition_statistics(images,texts,sources,supported,contra,model,config.__dict__)
    assert checked['training_reduction_image_mean_joint_accuracy']==actual.summary(model.coefficient)['image_mean_joint_accuracy']
    assert abs(checked['loss']-actual.loss_gradient(model.coefficient,actual.eligible,config)[0])<1e-12
    discrepancy=abs(checked['canonical_image_mean_joint_accuracy']-checked['training_reduction_image_mean_joint_accuracy'])
    assert discrepancy <= checked['image_balanced_ambiguity_bound']
    assert checked['roundoff_ambiguous_triplets']>0
