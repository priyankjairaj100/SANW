"""Independent algebra and exhaustive canonical oracles for blocked storage."""
import numpy as np
import pytest

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_streaming_v10 import FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss, cached_exact_retrieval


def fixture(tmp_path, block=2):
    rng=np.random.default_rng(807)
    images=rng.normal(size=(5,7));images/=np.linalg.norm(images,axis=1,keepdims=True)
    owner=np.repeat(np.arange(5),[1,3,2,4,1])
    texts=images[owner]+.18*rng.normal(size=(len(owner),7));texts/=np.linalg.norm(texts,axis=1,keepdims=True)
    model=ConstrainedBilinearScorer.from_training(images,texts,3)
    cache=FrozenScoreCache.create(tmp_path/'cache',images,texts,owner,block_size=block)
    constraints=StreamingFullGalleryConstraints(images,texts,owner,model,cache,block_size=block)
    return images,texts,owner,model,cache,constraints


def scores(images,texts,model,A):
    x=np.einsum('nd,dr->nr',images-model.image_mean,model.image_basis,optimize=False)
    y=np.einsum('nd,dr->nr',texts-model.text_mean,model.text_basis,optimize=False)
    xa=np.einsum('nd,dr->nr',x,A,optimize=False)
    base=np.array([np.sum(row*texts,axis=1) for row in images])
    residual=np.array([np.sum(row*y,axis=1) for row in xa])
    return base,residual


def source_oracle(images,texts,owner,model,A,queries,scale):
    baseline,residual=scores(images,texts,model,A);current=scale*(baseline+residual)
    image_loss=[];text_loss=[]
    for i in queries:
        owned=np.flatnonzero(owner==i);anchor=owned[np.argmax(baseline[i,owned])]
        columns=np.r_[anchor,np.flatnonzero(owner!=i)]
        image_loss.append(np.logaddexp.reduce(current[i,columns])-current[i,anchor])
        for t in owned:text_loss.append(np.logaddexp.reduce(current[:,t])-current[i,t])
    return (np.mean(image_loss)+sum(text_loss)/(len(queries)*len(texts)/len(images)))/2


@pytest.mark.parametrize('block',[1,2,6])
def test_streamed_full_denominators_and_every_gradient_coordinate(tmp_path,block):
    images,texts,owner,model,cache,c=fixture(tmp_path,block)
    A=np.random.default_rng(79).normal(size=(3,3))*.12
    loss=StreamingFullGallerySourceLoss(c,11.,query_block=block)
    for queries in (np.arange(5),np.array([4,0,2])):
        value,gradient=loss.loss_gradient(A,queries)
        assert value==pytest.approx(source_oracle(images,texts,owner,model,A,queries,11.),abs=3e-13)
        numerical=np.zeros_like(A)
        for coordinate in np.ndindex(A.shape):
            step=np.zeros_like(A);step[coordinate]=1e-6
            numerical[coordinate]=(source_oracle(images,texts,owner,model,A+step,queries,11.)-source_oracle(images,texts,owner,model,A-step,queries,11.))/2e-6
        np.testing.assert_allclose(gradient,numerical,atol=2e-9,rtol=2e-7)
    singleton=[loss.loss_gradient(A,[i]) for i in range(5)]
    value,gradient=loss.loss_gradient(A)
    assert value==pytest.approx(np.mean([v for v,g in singleton]),abs=3e-13)
    np.testing.assert_allclose(gradient,np.mean([g for v,g in singleton],axis=0),atol=3e-13,rtol=1e-13)


def test_all_constraints_and_radial_factor_match_explicit_pair_oracle(tmp_path):
    images,texts,owner,model,cache,c=fixture(tmp_path)
    A=np.random.default_rng(867).normal(size=(3,3))*2
    base,residual=scores(images,texts,model,A);own=owner[None,:]==np.arange(5)[:,None]
    anchor=np.where(own,base,-np.inf).argmax(axis=1)
    pi=owner[base.argmax(axis=1)]==np.arange(5);pt=base.argmax(axis=0)==owner
    mi=base[np.arange(5),anchor,None]-base;mt=base[owner,np.arange(len(owner))][None,:]-base
    di=residual[np.arange(5),anchor,None]-residual;dt=residual[owner,np.arange(len(owner))][None,:]-residual
    si=.5*mi+di;st=.5*mt+dt;vi=(~own)&pi[:,None];vt=(~own)&pt[None,:]
    actual=c.scan(A,canonical=True)
    assert actual['checked_constraints']==int(vi.sum()+vt.sum())
    assert actual['violated_constraints']==int((si[vi]<-1e-12).sum()+(st[vt]<-1e-12).sum())
    assert actual['min_constraint_slack']==pytest.approx(min(si[vi].min(),st[vt].min()),abs=3e-15)
    ratios=[.5*float(m)/-float(d) for margins,deltas,mask in [(mi,di,vi),(mt,dt,vt)] for m,d in zip(margins[mask],deltas[mask]) if d<0]
    assert c.radial_feasibility_factor(A)==pytest.approx(max(0.,min([1.]+ratios)),abs=3e-15)
    np.testing.assert_array_equal(c.protect_i2t,pi);np.testing.assert_array_equal(c.protect_t2i,pt)
    for edge in c.active:
        lhs,rhs,bound=c.edge_factors([edge]);value=float(lhs[0]@A@rhs[0]-bound[0])
        wanted=si[edge.query,edge.negative] if edge.direction==0 else st[edge.negative,edge.query]
        assert value==pytest.approx(wanted,abs=3e-15)


def test_canonical_winners_exact_with_duplicates_ties_and_query_blocks(tmp_path):
    images,texts,owner,model,cache,c=fixture(tmp_path)
    images[1]=images[0];texts[3]=texts[0]
    # New feature set gets a distinct bound cache.
    cache=FrozenScoreCache.create(tmp_path/'tied_cache',images,texts,owner,block_size=2)
    for A in (np.zeros((3,3)),np.eye(3)*1e-12,np.random.default_rng(92).normal(size=(3,3))*.08):
        model.coefficient=A;base,residual=scores(images,texts,model,A);current=base+residual
        for block in (1,3,99):
            raw=cached_exact_retrieval(images,texts,owner,cache,model,query_block=block)
            for direction,axis in [('i2t',1),('t2i',0)]:
                winners=current.argmax(axis=axis);np.testing.assert_array_equal(raw[direction+'_top_indices'],winners)
                expected=current[np.arange(len(images)),winners] if axis==1 else current[winners,np.arange(len(texts))]
                np.testing.assert_array_equal(raw[direction+'_top_scores'],expected)


def test_cache_cannot_silently_attach_to_different_same_shape_features(tmp_path):
    images,texts,owner,model,cache,c=fixture(tmp_path)
    changed=images.copy();changed[0,0]+=1e-4
    with pytest.raises(ValueError,match='identity|feature|input|match'):
        StreamingFullGalleryConstraints(changed,texts,owner,model,cache)


def test_all_owned_cache_entries_use_canonical_dot_even_at_large_magnitudes(tmp_path):
    rng=np.random.default_rng(51);images=rng.normal(size=(4,513))*1e5;texts=rng.normal(size=(8,513))*1e5;owner=np.repeat(np.arange(4),2)
    cache=FrozenScoreCache.create(tmp_path/'large',images,texts,owner,block_size=3)
    canonical=np.sum(images[owner]*texts,axis=1)
    np.testing.assert_array_equal(cache.i2t[owner,np.arange(len(texts))],canonical)

