#!/usr/bin/env python3
"""Independent canonical prediction, exact-boundary bootstrap and gate audit for v10 pilots."""
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
from evaluate_practical_official_development_v10 import load_lock,verify_full_pilot,reconstruct_gate,require_evaluation_threads
from evaluate_practical_constrained_development_v8 import load_development,verify_record
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer

METRICS=('i2t','t2i','original','source_pair')

def read(path):return json.loads(Path(path).read_text())
def archive(entry):
    with np.load(verify_record(entry),allow_pickle=False) as data:return {k:data[k] for k in data.files}

def independent_bootstrap(numerators,denominators,ids,retrieval):
    """Fixed equal-size image clusters, independently grouped and resampled."""
    numerators,denominators,ids=map(np.asarray,(numerators,denominators,ids))
    assert numerators.ndim==denominators.ndim==ids.ndim==1
    assert numerators.shape==denominators.shape==ids.shape and np.all(denominators>0)
    unique=sorted(set(ids.tolist()));sizes=np.asarray([np.count_nonzero(ids==i) for i in unique])
    assert np.all(sizes==sizes[0])
    if retrieval:
        assert np.all(denominators==denominators[0])
        sums=np.asarray([int(numerators[ids==i].sum()) for i in unique],dtype=np.int64)
    else:
        delta=numerators.astype(np.float64)/denominators
        sums=np.asarray([delta[ids==i].sum() for i in unique])
    values,counts=np.unique(sums,return_counts=True);rng=np.random.default_rng(20261007);blocks=[]
    for first in range(0,100000,4096):
        draws=rng.multinomial(len(unique),counts/len(unique),size=min(4096,100000-first))
        blocks.append(draws@values)
    totals=np.concatenate(blocks)
    if retrieval:
        denominator=int(denominators[0])*len(numerators)
        samples=totals.astype(np.float64)/denominator
        ordered=np.sort(totals)
        bounds=[]
        for probability in (Fraction(1,3200),1-Fraction(1,3200)):
            position=probability*(len(ordered)-1);index=position.numerator//position.denominator;weight=position-index
            bounds.append(((1-weight)*int(ordered[index])+weight*int(ordered[min(index+1,len(ordered)-1)]))/denominator)
        low,high=bounds
        return samples,dict(ci_lower=float(low),ci_upper=float(high),
            exact_ci_lower=dict(numerator=low.numerator,denominator=low.denominator),
            exact_ci_upper=dict(numerator=high.numerator,denominator=high.denominator))
    samples=totals/(len(unique)*sizes[0]);low,high=np.quantile(samples,[.05/160,1-.05/160],method='linear')
    return samples,dict(ci_lower=float(low),ci_upper=float(high))

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
    assert read(verify_record(result['lock']))==lock
    assert read(path.parent/'start.json')==dict(lock=result['lock'],state=result['state'],no_new_benchmark_access=True)
    protocol=read(verify_record(lock['protocol']))
    identity,_,state=verify_full_pilot(result['state']['run'],protocol,lock['protocol']['sha256'])
    assert state==result['state']
    assert result['study']=='sanw_practical_v10_official_development' and result['seed']==17
    assert result['protocol_sha256']==lock['protocol']['sha256']
    assert result['checkpoint_sha256']==state['checkpoint']['sha256']
    reference={'encoder':enc,'training_provenance':{'inputs':protocol['original_training_inputs'][enc]}}
    data,pool=load_development(reference,protocol)
    assert len(pool.images)==900 and len(pool.texts)==4500
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
        summary=result['summaries'][name]
        for direction in ('i2t','t2i'):
            assert summary['retrieval'][direction+'_r1']==float(raw[direction+'_correct'].mean())
        for metric in ('original','source_pair'):
            comp=summary['composition'][metric]
            assert comp['joint_accuracy']==float(raw[metric+'_correct'].mean())
            assert comp['mean_joint_margin']==float(raw[metric+'_joint_margin'].mean())
            assert comp['triplet_count']==int(raw[metric+'_triplet_counts'].sum())
            assert comp['image_count']==len(raw[metric+'_cluster_ids'])
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
        # Independent integer regrouping and rational percentile endpoints for retrieval;
        # inherited descriptive floating composition intervals remain unchanged.
        samples,interval=independent_bootstrap(numer,denom,f[f'{metric}_cluster_ids'],metric in ('i2t','t2i'))
        np.testing.assert_array_equal(samples,raws['bootstrap'][metric])
        effect=result['effects'][metric]
        lo,hi=interval['ci_lower'],interval['ci_upper']
        assert effect['ci_lower']==lo and effect['ci_upper']==hi and effect['difference']==float(value)
        unique=set(f[f'{metric}_cluster_ids'].tolist())
        assert (effect['replicates'],effect['bootstrap_seed'],effect['family_size'],effect['items'],effect['image_clusters'])==(100000,20261007,80,len(numer),len(unique))
        if metric in ('i2t','t2i'):
            for key in ('exact_ci_lower','exact_ci_upper'):assert effect[key]==interval[key]
            assert effect['exact_difference']==dict(numerator=value.numerator,denominator=value.denominator)
        effects[metric]=dict(difference=float(value),**interval)
        transition[metric]=dict(improved_items=int((numer>0).sum()),worsened_items=int((numer<0).sum()),unchanged_items=int((numer==0).sum()))
    checks=dict(nonzero_trained_update=state['nonzero'] is True,original_joint_improvement=exact['original']>0,source_pair_joint_non_decrease=exact['source_pair']>=0)
    for d in ('i2t','t2i'):
        checks[f'{d}_mean_non_decrease']=exact[d]>=0;checks[f'{d}_strict_adjusted_retention']=Fraction(**effects[d]['exact_ci_lower'])>Fraction(-1,100)
    expected=dict(passed=all(checks.values()),checks=checks);assert expected==result['gate']==reconstruct_gate(result)
    return dict(encoder=enc,audit_passed=True,result=dict(path=str(path),sha256=digest(path)),gate=expected,effects=effects,
        transitions=transition,exhaustive_canonical_pairs=pairs,composition_triples=triples,bootstrap_draws_checked=400000,
        original_joint_ci_strictly_positive=effects['original']['ci_lower']>0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--results',nargs=2,required=True,type=Path)
    parser.add_argument('--gate',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    require_evaluation_threads()
    if args.output.exists():raise FileExistsError('Refusing to overwrite independent result audit')
    lock=load_lock(args.lock,digest(args.lock));assert lock['family']=='joint' and lock['seed']==17
    fits=[]
    for p in args.results:
        row=audit_result(p,lock);fits.append(row);print(json.dumps(row),flush=True)
    assert {r['encoder'] for r in fits}=={'vit_b32','rn50'}
    passed=[r['encoder'] for r in fits if r['gate']['passed']]
    if args.gate:
        gate=read(args.gate);assert gate['passed']==(len(passed)==2)
        assert gate['study']=='sanw_practical_v10_replication_gate' and gate['family']=='joint' and gate['seed']==17
        assert gate['protocol_sha256']==lock['protocol']['sha256'] and gate['lock']['sha256']==digest(args.lock)
        assert gate['encoders_passed']==[enc for enc in ('vit_b32','rn50') if enc in passed]
        for row in fits:
            assert gate['encoders'][row['encoder']]['gate']==row['gate']
            assert gate['encoders'][row['encoder']]['result']['sha256']==row['result']['sha256']
            assert digest(verify_record(gate['encoders'][row['encoder']]['result']))==row['result']['sha256']
            assert gate['encoders'][row['encoder']]['state']==lock['states'][row['encoder']]
        assert len(gate['pilot_runs'])==2 and {x['encoder'] for x in gate['pilot_runs']}=={'vit_b32','rn50'}
        for entry in gate['pilot_runs']:
            enc=entry['encoder'];current=gate['encoders'][enc]
            for field,record in (('result',current['result']),('completion',current['state']['completion']),('checkpoint',current['state']['checkpoint'])):
                assert entry[field]=={key:record[key] for key in ('path','sha256')}
                verify_record(entry[field])
    result=dict(study='sanw_practical_v10_official_development_independent_audit',audit_passed=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__),
        direct_dependency_sha256={p:digest(ROOT/p) for p in ('scripts/audit_practical_constrained_fits_v8.py','scripts/evaluate_practical_official_development_v10.py','scripts/evaluate_practical_constrained_development_v8.py')},
        numerical_scope='All canonical retrieval pairs and raw composition pairs/triples independently replayed; every saved bootstrap draw and exact retrieval percentile boundary independently reconstructed. Composition intervals remain descriptive floating percentiles. Strict gate uses exact retrieval lower bound and exact paired means.',
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
