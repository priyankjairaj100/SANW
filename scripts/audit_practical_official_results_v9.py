#!/usr/bin/env python3
"""Independent canonical prediction, bootstrap and gate audit for v9 pilots."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from audit_practical_constrained_fits_v8 import digest,independent_coordinates,independent_full_gallery
from evaluate_practical_official_development_v9 import load_lock,verify_full_pilot,reconstruct_gate
from evaluate_practical_constrained_development_v8 import load_development,verify_record
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer

METRICS=('i2t','t2i','original','source_pair')

def read(path):return json.loads(Path(path).read_text())
def archive(entry):
    with np.load(verify_record(entry),allow_pickle=False) as data:return {k:data[k] for k in data.files}

def composition(raw,data,model):
    expected=[(i,t,data.pairs[i][t]) for i in data.split_indices['validation'] for t in sorted(data.pairs[i])]
    assert len(expected)==1875
    np.testing.assert_array_equal(np.asarray(expected),np.stack([raw[k] for k in ('raw_image_index','raw_text_index','raw_relation')],axis=1))
    ii,tt=raw['raw_image_index'],raw['raw_text_index']
    pair_scores=np.sum(data.images[ii]*data.texts[tt],axis=1)
    if model is not None:
        x=independent_coordinates(data.images[ii],model.image_mean,model.image_basis)
        y=independent_coordinates(data.texts[tt],model.text_mean,model.text_basis)
        pair_scores+=np.sum(np.einsum('nd,dr->nr',x,model.coefficient,optimize=False)*y,axis=1)
    np.testing.assert_array_equal(pair_scores,raw['raw_score'])
    actual={k:{} for k in ('original','source_pair')}
    for i in data.split_indices['validation']:
        mask=ii==i;ids=tt[mask];labels=raw['raw_relation'][mask];scores=pair_scores[mask]
        src,sup,neg=[scores[labels==j] for j in (1,2,3)]
        if len(src) and len(sup) and len(neg):
            gaps=[min(a,b)-c for a in src for b in sup for c in neg]
            actual['original'][data.image_ids[i]]=(sum(g>0 for g in gaps),len(gaps),float(np.mean(gaps)))
        key={int(t):tuple(re.findall(r'\w+',unicodedata.normalize('NFKC',data.manifest['texts'][int(t)]['text']).casefold())) for t in ids}
        positive={key[int(t)] for t,label in zip(ids,labels) if label in (1,2)}
        negatives=[int(t) for t,label in zip(ids,labels) if label==3 and key[int(t)] not in positive]
        pairs=[(int(a),int(b)) for a,b in combinations(ids[labels==1],2) if key[int(a)]!=key[int(b)]]
        lookup=dict(zip(ids.tolist(),scores.tolist()))
        if pairs and negatives:
            gaps=[min(lookup[a],lookup[b])-lookup[c] for a,b in pairs for c in negatives]
            actual['source_pair'][data.image_ids[i]]=(sum(g>0 for g in gaps),len(gaps),float(np.mean(gaps)))
    total=0
    for metric,values in actual.items():
        ids=raw[f'{metric}_image_ids'];assert len(ids)==len(values)==100
        good,n,margins=map(np.asarray,zip(*[values[str(i)] for i in ids]))
        np.testing.assert_array_equal(n,raw[f'{metric}_triplet_counts'])
        np.testing.assert_array_equal(good/n,raw[f'{metric}_correct'])
        np.testing.assert_array_equal(good/n,raw[f'{metric}_joint_accuracy'])
        np.testing.assert_array_equal(margins,raw[f'{metric}_joint_margin'])
        np.testing.assert_array_equal(ids,raw[f'{metric}_cluster_ids']);total+=int(n.sum())
    return total

def audit_result(path,lock):
    result=read(path);enc=result['encoder'];assert result['state']==lock['states'][enc]
    protocol=read(verify_record(lock['protocol']))
    identity,_,_,state=verify_full_pilot(result['state']['run'],protocol,lock['protocol']['sha256'],verify_record(lock['selection']))
    assert state==result['state']
    data,pool=load_development(identity,protocol)
    model=ConstrainedBilinearScorer.load(verify_record(state['checkpoint']))
    frozen,residual=independent_full_gallery(pool.images,pool.texts,model)
    raws={k:archive(v) for k,v in result['artifacts'].items()}
    owner=pool.owner.numpy();pairs=triples=0
    for name,scores,m in [('frozen',frozen,None),('trained',frozen+residual,model)]:
        raw=raws[name];np.testing.assert_array_equal(raw['owner'],owner)
        np.testing.assert_array_equal(raw['gallery_image_ids'],np.asarray(pool.image_ids))
        np.testing.assert_array_equal(raw['gallery_text_ids'],np.asarray(pool.text_ids))
        for d,axis in [('i2t',1),('t2i',0)]:
            winners=scores.argmax(axis=axis)
            top=scores[np.arange(len(pool.images)),winners] if d=='i2t' else scores[winners,np.arange(len(pool.texts))]
            correct=owner[winners]==np.arange(len(pool.images)) if d=='i2t' else winners==owner
            np.testing.assert_array_equal(winners,raw[f'{d}_top_indices'])
            np.testing.assert_array_equal(top,raw[f'{d}_top_scores'])
            np.testing.assert_array_equal(correct,raw[f'{d}_correct'])
            ids=np.asarray(pool.image_ids) if d=='i2t' else np.asarray(pool.image_ids)[owner]
            np.testing.assert_array_equal(ids,raw[f'{d}_cluster_ids'])
        triples+=composition(raw,data,m);pairs+=scores.size
    exact={};effects={};transition={}
    for metric in METRICS:
        f,t=raws['frozen'],raws['trained']
        np.testing.assert_array_equal(f[f'{metric}_cluster_ids'],t[f'{metric}_cluster_ids'])
        denom=np.ones(len(f[f'{metric}_correct']),dtype=np.int64) if metric in ('i2t','t2i') else f[f'{metric}_triplet_counts']
        if metric not in ('i2t','t2i'):np.testing.assert_array_equal(denom,t[f'{metric}_triplet_counts'])
        numer=np.rint(t[f'{metric}_correct']*denom).astype(np.int64)-np.rint(f[f'{metric}_correct']*denom).astype(np.int64)
        value=sum((Fraction(int(n),int(d)) for n,d in zip(numer,denom)),Fraction())/len(denom)
        exact[metric]=value;assert result['exact_paired_changes'][metric]==dict(numerator=value.numerator,denominator=value.denominator)
        assert result['paired_changes'][metric]==float(value)
        for key,array in [('difference_numerators',numer),('difference_denominators',denom),('difference',numer.astype(float)/denom),('cluster_ids',f[f'{metric}_cluster_ids'])]:
            np.testing.assert_array_equal(array,raws['paired'][f'{metric}_{key}'])
        # Independently regroup image-cluster outcomes and reproduce the fixed RNG recipe.
        delta=numer.astype(float)/denom;ids=f[f'{metric}_cluster_ids']
        unique=sorted(set(ids.tolist()));groups=[delta[ids==i] for i in unique]
        sizes=np.asarray([len(g) for g in groups]);assert np.all(sizes==sizes[0])
        sums=np.asarray([g.sum() for g in groups]);values,counts=np.unique(sums,return_counts=True)
        rng=np.random.default_rng(20261007);blocks=[]
        for first in range(0,100000,4096):
            draws=rng.multinomial(len(unique),counts/len(unique),size=min(4096,100000-first))
            blocks.append(draws@values/(len(unique)*sizes[0]))
        samples=np.concatenate(blocks);np.testing.assert_array_equal(samples,raws['bootstrap'][metric])
        lo,hi=np.quantile(samples,[.05/160,1-.05/160],method='linear')
        effect=result['effects'][metric]
        assert effect['ci_lower']==lo and effect['ci_upper']==hi and effect['difference']==float(value)
        assert (effect['replicates'],effect['bootstrap_seed'],effect['family_size'],effect['items'],effect['image_clusters'])==(100000,20261007,80,len(delta),len(unique))
        effects[metric]=dict(difference=float(value),ci_lower=float(lo),ci_upper=float(hi))
        transition[metric]=dict(improved_items=int((numer>0).sum()),worsened_items=int((numer<0).sum()),unchanged_items=int((numer==0).sum()))
    checks=dict(nonzero_trained_update=state['update_norm']>0,original_joint_improvement=exact['original']>0,source_pair_joint_non_decrease=exact['source_pair']>=0)
    for d in ('i2t','t2i'):
        checks[f'{d}_mean_non_decrease']=exact[d]>=0;checks[f'{d}_strict_adjusted_retention']=effects[d]['ci_lower']>-.01
    expected=dict(passed=all(checks.values()),checks=checks);assert expected==result['gate']==reconstruct_gate(result)
    return dict(encoder=enc,audit_passed=True,result=dict(path=str(path),sha256=digest(path)),gate=expected,effects=effects,
        transitions=transition,exhaustive_canonical_pairs=pairs,composition_triples=triples,bootstrap_draws_checked=400000,
        original_joint_ci_strictly_positive=effects['original']['ci_lower']>0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--results',nargs=2,required=True,type=Path)
    parser.add_argument('--gate',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=load_lock(args.lock,digest(args.lock));assert lock['family']=='joint' and lock['seed']==17
    fits=[]
    for p in args.results:
        row=audit_result(p,lock);fits.append(row);print(json.dumps(row),flush=True)
    assert {r['encoder'] for r in fits}=={'vit_b32','rn50'}
    passed=[r['encoder'] for r in fits if r['gate']['passed']]
    if args.gate:
        gate=read(args.gate);assert gate['passed']==(len(passed)==2)
        assert gate['encoders_passed']==[enc for enc in ('vit_b32','rn50') if enc in passed]
        for row in fits:
            assert gate['encoders'][row['encoder']]['gate']==row['gate']
            assert gate['encoders'][row['encoder']]['result']['sha256']==row['result']['sha256']
    result=dict(study='sanw_practical_v9_official_development_independent_audit',audit_passed=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__),
        lock=dict(path=str(args.lock),sha256=digest(args.lock)),fits=fits,
        replication_gate_passed=len(passed)==2,
        authorized_next_stage='replication' if len(passed)==2 else 'stop_no_replications_or_new_benchmark_evaluation',
        total_canonical_pairs=sum(r['exhaustive_canonical_pairs'] for r in fits),
        total_composition_triples=sum(r['composition_triples'] for r in fits),total_bootstrap_draws=800000,
        data_scope='Official development 900-image/4500-caption retrieval and100-image composition. Shared cached archive may materialize other-split bytes; no benchmark predictions or outcomes computed.',
        no_fit_performed=True)
    if args.gate:result['assembled_gate']=dict(path=str(args.gate),sha256=digest(args.gate))
    with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(audit=str(args.output),sha256=digest(args.output),audit_passed=True,replication_gate_passed=result['replication_gate_passed'])),flush=True)

if __name__=='__main__':main()
