from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from gcr.practical_fresh_evaluation_v10 import (confirmation_view, score_confirmation, paired_counts,
    fresh_support_gate, ENCODERS, METRICS, CONTRASTS)


def fixture():
    images=np.eye(2);texts=[];rows=[];pairs=[]
    specs=[[("a",.9,"source"),("A!",.9,"source"),("b",.7,"source"),("p",.8,"supported"),
            ("a",.9,"contradicted"),("z",.6,"contradicted"),("q",1.,"neutral")],
           [("c",.9,"source"),("d",.8,"source"),("e",.7,"source"),("p2",.9,"supported"),
            ("p3",.2,"supported"),("z2",.5,"contradicted")]]
    for i,group in enumerate(specs):
        for text,score,relation in group:
            j=len(rows);rows.append({'id':f't{j}','text':text});pairs.append({'image_id':f'i{i}','text_id':f't{j}','relation':relation})
            texts.append([score,0] if i==0 else [0,score])
    manifest={'images':[{'id':f'i{i}','split':'test'} for i in range(2)],'texts':rows,'pairs':pairs}
    return confirmation_view(images,np.array(texts),manifest,['i0','i1'],image_count=2,sources_per_image=3)


def test_raw_joint_conflict_and_distinct_source_pairs():
    summary,raw=score_confirmation(fixture())
    assert raw['original_correct_counts'].tolist()==[3,3]
    assert raw['original_triplet_counts'].tolist()==[6,6]
    assert raw['source_pair_correct_counts'].tolist()==[2,3]
    assert raw['source_pair_triplet_counts'].tolist()==[2,3]
    assert summary['composition']['original']['excluded_conflicting_negatives']==0
    assert summary['composition']['source_pair']['excluded_conflicting_negatives']==1
    assert (raw['raw_relation']==4).sum()==1
    assert raw['gallery_text_ids'].tolist()==['t0','t1','t2','t7','t8','t9']
    assert raw['i2t_correct'].all() and raw['t2i_correct'].all()


def test_counts_are_image_balanced_with_different_denominators():
    view=fixture()
    view['texts'][8]=[0,.4]
    summary,raw=score_confirmation(view)
    assert raw['source_pair_correct_counts'].tolist()==[2,1]
    assert summary['composition']['source_pair']['exact_accuracy']=={'numerator':2,'denominator':3}
    assert summary['composition']['source_pair']['image_mean_accuracy']!=3/5


def test_integer_pairing_and_alignment():
    _,raw=score_confirmation(fixture());selected={s:deepcopy(raw) for s in (17,29,43)}
    reference={s:deepcopy(raw) for s in selected}
    selected[17]['original_correct_counts'][0]+=1
    selected[29]['original_correct_counts'][1]-=2
    out=paired_counts('original',selected,reference)
    assert out['difference_numerators'].tolist()==[1,-2]
    assert out['difference_denominators'].tolist()==[18,18]
    selected[43]['original_triplet_counts'][1]+=1
    with pytest.raises(ValueError):paired_counts('original',selected,reference)
    selected[43]=deepcopy(raw);selected[43]['gallery_text_ids'][0]='bad'
    with pytest.raises(ValueError):paired_counts('original',selected,reference)


def effects(lower=Fraction(1,100),source=Fraction(0)):
    out=[]
    for encoder in ENCODERS:
        for metric in METRICS:
            for contrast,_,_ in CONTRASTS:
                mean=source if metric=='source_pair' else lower
                out.append(dict(encoder=encoder,metric=metric,contrast=contrast,replicates=100000,bootstrap_seed=20261007,family_size=80,
                    difference=float(mean),ci_lower=float(lower),exact_ci_lower={'numerator':lower.numerator,'denominator':lower.denominator},
                    exact_difference={'numerator':mean.numerator,'denominator':mean.denominator}))
    return out


def test_strict_exact_boundaries_and_complete_scope():
    assert fresh_support_gate(effects())['passed']
    assert not fresh_support_gate(effects(Fraction(0)))['passed']
    rows=effects()
    row=next(x for x in rows if x['metric']=='i2t' and x['contrast']=='joint_minus_original_frozen')
    row['exact_ci_lower']={'numerator':-1,'denominator':100};row['ci_lower']=-.01
    assert not fresh_support_gate(rows)['passed']
    row['ci_lower']=np.nextafter(-.01,1)
    with pytest.raises(ValueError):fresh_support_gate(rows)
    with pytest.raises(ValueError):fresh_support_gate(effects()[:-1])
    assert not fresh_support_gate(effects(source=Fraction(-1,10**20)))['passed']


def test_manifest_requires_distinct_occurrence_ids():
    view=fixture();manifest=deepcopy(view['manifest']);manifest['pairs'].append(manifest['pairs'][0])
    with pytest.raises(ValueError):confirmation_view(view['images'],view['texts'],manifest,['i0','i1'],image_count=2,sources_per_image=3)


def test_ineligible_owners_omitted_not_zero_filled():
    view=fixture()
    view['pairs'][1]={j:(4 if code==2 else code) for j,code in view['pairs'][1].items()}
    summary,raw=score_confirmation(view)
    assert raw['original_cluster_ids'].tolist()==['i0']
    assert raw['source_pair_cluster_ids'].tolist()==['i0','i1']
    assert summary['composition']['original']['eligible_images']==1
    assert summary['composition']['original']['image_mean_accuracy']==.5


def test_exact_source_mean_display_consistency():
    rows=effects()
    row=next(x for x in rows if x['metric']=='source_pair' and x['contrast']=='joint_minus_original_frozen')
    row['difference']=1e-20
    with pytest.raises(ValueError):fresh_support_gate(rows)
