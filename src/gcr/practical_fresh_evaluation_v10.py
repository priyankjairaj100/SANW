"""Fresh same-source outcomes with raw annotation rules and integer counts.

No data loading, fitting or model selection occurs here. Callers supply a
previously locked confirmation manifest and one-pass normalized frozen features.
Original-joint outcomes retain every raw contradicted annotation. Only the
source-pair endpoint excludes exact normalized positive/negative conflicts.
"""
from itertools import combinations
from fractions import Fraction
import numpy as np

from .practical_constrained_evaluation_v8 import caption_key, exact_retrieval
from .practical_exact_bootstrap_v10 import integer_cluster_bootstrap

SEEDS = (17,29,43)
ENCODERS = ("vit_b32","rn50")
FAMILIES = ("joint","no_retention")
METRICS = ("i2t","t2i","original","source_pair")
CONTRASTS = (("joint_minus_original_frozen","joint","original_frozen"),
             ("no_retention_minus_original_frozen","no_retention","original_frozen"),
             ("joint_minus_no_retention","joint","no_retention"))
RELATIONS = {"source":1,"supported":2,"contradicted":3,"neutral":4}


def confirmation_view(images,texts,manifest,expected_image_ids,*,image_count=1500,sources_per_image=5):
    images,texts = (np.ascontiguousarray(x,dtype=np.float64) for x in (images,texts))
    image_ids=[str(row['id']) for row in manifest['images']]
    text_ids=[str(row['id']) for row in manifest['texts']]
    if (len(image_ids)!=image_count or image_ids!=list(expected_image_ids) or len(set(image_ids))!=len(image_ids)
            or len(set(text_ids))!=len(text_ids) or images.ndim!=2 or texts.ndim!=2
            or images.shape!=(image_count,texts.shape[1]) or texts.shape[0]!=len(text_ids)
            or not np.isfinite(images).all() or not np.isfinite(texts).all()
            or any(row['split']!='test' for row in manifest['images'])):
        raise ValueError('Confirmation gallery identity, feature shape or split differs')
    ilook,tlook = {x:i for i,x in enumerate(image_ids)},{x:i for i,x in enumerate(text_ids)}
    pairs=[{} for _ in image_ids];text_owner=np.full(len(text_ids),-1,dtype=np.int64)
    for pair in manifest['pairs']:
        if pair['image_id'] not in ilook or pair['text_id'] not in tlook or pair['relation'] not in RELATIONS:
            raise ValueError('Invalid raw confirmation relation')
        i,j=ilook[pair['image_id']],tlook[pair['text_id']]
        if j in pairs[i] or text_owner[j]>=0:
            raise ValueError('Repeated relation or shared text ID; retain occurrences with distinct IDs')
        pairs[i][j]=RELATIONS[pair['relation']];text_owner[j]=i
    if np.any(text_owner<0) or any(sum(code==1 for code in group.values())!=sources_per_image for group in pairs):
        raise ValueError('Every text needs an owner and each image needs the declared source count')
    source_rows=np.asarray(sorted(j for group in pairs for j,code in group.items() if code==1),dtype=np.int64)
    return dict(images=images,texts=texts,manifest=manifest,image_ids=np.asarray(image_ids),text_ids=np.asarray(text_ids),
                pairs=pairs,text_owner=text_owner,source_rows=source_rows,owner=text_owner[source_rows])


def score_confirmation(view,scorer=None):
    images,texts,source_rows=view['images'],view['texts'],view['source_rows']
    retrieval,ranked=exact_retrieval(images,texts[source_rows],view['owner'],scorer,query_block=64)
    raw={"gallery_image_ids":view['image_ids'],"gallery_text_ids":view['text_ids'][source_rows],
         "gallery_owner":view['owner'],"gallery_source_text_indices":source_rows,
         "i2t_cluster_ids":view['image_ids'],"t2i_cluster_ids":view['image_ids'][view['owner']]}
    for direction in ('i2t','t2i'):
        for suffix in ('correct','top_indices','top_scores','candidate_counts','roundoff_bounds',
                       'rescored_query_indices','rescored_gallery_indices','rescored_scores'):
            raw[direction+'_'+suffix]=ranked[direction+'_'+suffix]
    ii,jj,codes=[],[],[]
    for i,pairs in enumerate(view['pairs']):
        for j in sorted(pairs):ii.append(i);jj.append(j);codes.append(pairs[j])
    ii,jj,codes=(np.asarray(x,dtype=np.int64) for x in (ii,jj,codes))
    scores=np.empty(len(ii),dtype=np.float64)
    for lo in range(0,len(ii),4096):
        stop=min(lo+4096,len(ii));v,t=images[ii[lo:stop]],texts[jj[lo:stop]]
        scores[lo:stop]=np.sum(v*t,axis=1,dtype=np.float64) if scorer is None else scorer.pair_scores(v,t)
    raw.update(raw_image_index=ii,raw_text_index=jj,raw_relation=codes,raw_score=scores)
    counters={kind:{'ids':[],'correct':[],'counts':[]} for kind in ('original','source_pair')}
    conflicts=0
    for i,pairs in enumerate(view['pairs']):
        mask=ii==i;indices=jj[mask];labels=codes[mask];values=scores[mask]
        groups=[values[labels==label] for label in (1,2,3)]
        source,supported,negative=groups
        if min(map(len,groups)):
            gap=np.minimum(source[:,None,None],supported[None,:,None])-negative[None,None,:]
            bucket=counters['original'];bucket['ids'].append(view['image_ids'][i])
            bucket['correct'].append(int(np.count_nonzero(gap>0)));bucket['counts'].append(gap.size)
        keys={j:caption_key(view['manifest']['texts'][j]['text']) for j in indices}
        positive_keys={keys[j] for j,code in zip(indices,labels,strict=True) if code in (1,2)}
        negatives=[j for j,code in zip(indices,labels,strict=True) if code==3 and keys[j] not in positive_keys]
        conflicts+=sum(code==3 and keys[j] in positive_keys for j,code in zip(indices,labels,strict=True))
        source_pairs=[(a,b) for a,b in combinations(indices[labels==1],2) if keys[a]!=keys[b]]
        if source_pairs and negatives:
            lookup=dict(zip(indices,values,strict=True))
            gap=np.asarray([min(lookup[a],lookup[b])-lookup[n] for a,b in source_pairs for n in negatives])
            bucket=counters['source_pair'];bucket['ids'].append(view['image_ids'][i])
            bucket['correct'].append(int(np.count_nonzero(gap>0)));bucket['counts'].append(gap.size)
    composition={}
    for name,bucket in counters.items():
        if not bucket['ids']:raise ValueError(f'No eligible raw-annotation {name} confirmation owners')
        counts=np.asarray(bucket['counts'],dtype=np.int64);wins=np.asarray(bucket['correct'],dtype=np.int64)
        raw[name+'_cluster_ids']=np.asarray(bucket['ids']);raw[name+'_correct_counts']=wins;raw[name+'_triplet_counts']=counts
        raw[name+'_correct']=wins/counts
        exact=sum((Fraction(int(a),int(b)) for a,b in zip(wins,counts,strict=True)),Fraction())/len(counts)
        composition[name]={'eligible_images':len(counts),'triplets':int(counts.sum()),'image_mean_accuracy':float(exact),
                           'exact_accuracy':{'numerator':exact.numerator,'denominator':exact.denominator},
                           'averaging':'equal_image_then_all_declared_triplets','strict_score_comparison':True}
    composition['original']['excluded_conflicting_negatives']=0
    composition['source_pair']['excluded_conflicting_negatives']=int(conflicts)
    return {'retrieval':retrieval,'composition':composition,'neutral_as_negative':False,
            'same_canonical_score_all_endpoints':True,'original_training_eligibility_filter':False},raw


def paired_counts(metric,selected,reference):
    if metric not in METRICS or set(selected)!=set(SEEDS) or set(reference)!=set(SEEDS):
        raise ValueError('Require a declared endpoint and all three fixed paired seeds')
    anchor=reference[17]
    identities=('gallery_image_ids','gallery_text_ids','gallery_owner','gallery_source_text_indices',
                'raw_image_index','raw_text_index','raw_relation',metric+'_cluster_ids')
    deltas=[]
    for seed in SEEDS:
        current,baseline=selected[seed],reference[seed]
        for arrays in (current,baseline):
            if any(not np.array_equal(arrays[key],anchor[key]) for key in identities):
                raise ValueError('Confirmation paired owners, galleries or raw annotations changed')
        if metric in ('i2t','t2i'):
            a,b=current[metric+'_correct'],baseline[metric+'_correct']
            if (a.dtype!=np.bool_ or b.dtype!=np.bool_ or a.shape!=b.shape or a.ndim!=1
                    or a.shape!=anchor[metric+'_cluster_ids'].shape):
                raise ValueError('Retrieval outcomes must be aligned Boolean vectors')
            counts=np.ones(len(a),dtype=np.int64);delta=a.astype(np.int64)-b.astype(np.int64)
        else:
            counts=np.asarray(anchor[metric+'_triplet_counts'])
            if (counts.dtype.kind not in 'iu' or counts.ndim!=1 or np.any(counts<=0)
                    or any(int(value)>np.iinfo(np.int64).max//3 for value in counts)):
                raise ValueError('Invalid or overflowing raw triplet denominators')
            for arrays in (current,baseline):
                wins=arrays[metric+'_correct_counts']
                if (arrays[metric+'_triplet_counts'].dtype.kind not in 'iu'
                        or not np.array_equal(arrays[metric+'_triplet_counts'],counts) or wins.dtype.kind not in 'iu'
                        or wins.shape!=counts.shape or wins.shape!=anchor[metric+'_cluster_ids'].shape
                        or any(int(w)<0 or int(w)>int(c) for w,c in zip(wins,counts,strict=True))):
                    raise ValueError('Integer correct/triplet counts changed or are invalid')
            delta=current[metric+'_correct_counts'].astype(np.int64)-baseline[metric+'_correct_counts'].astype(np.int64)
        deltas.append(delta)
    deltas=np.stack(deltas);denominators=3*counts.astype(np.int64)
    if np.any(denominators<=0):raise ValueError('Denominator overflow')
    return {'difference_by_seed':deltas,'difference_numerators':deltas.sum(axis=0,dtype=np.int64),
            'difference_denominators':denominators,'cluster_ids':anchor[metric+'_cluster_ids']}


def effect_from_counts(metric,paired):
    n,d,ids=(paired[key] for key in ('difference_numerators','difference_denominators','cluster_ids'))
    if metric in ('i2t','t2i'):
        if not np.all(d==3):raise ValueError('Retrieval fixed seed mean must have denominator three')
        effect,samples=integer_cluster_bootstrap(n,3,ids,grouped=False)
    else:
        from .practical_fresh_statistics_v10 import image_balanced_bootstrap
        effect,samples=image_balanced_bootstrap(n,d,ids)
    effect['conditioning']='fixed_selected_three_seed_mean; paired_confirmation_image_clusters; fixed_gallery; no_seed_resampling_or_gallery_reranking'
    return effect,samples


def fresh_support_gate(effects):
    from .practical_fresh_statistics_v10 import exact_mean_nonnegative,exact_lower_exceeds
    required={(encoder,metric,contrast) for encoder in ENCODERS for metric in METRICS for contrast,_,_ in CONTRASTS}
    indexed={}
    for row in effects:
        key=row['encoder'],row['metric'],row['contrast']
        if key in indexed or key not in required:raise ValueError('Duplicate or undeclared fresh effect')
        if (row['replicates'],row['bootstrap_seed'],row['family_size'])!=(100000,20261007,80):
            raise ValueError('Fresh uncertainty contract changed')
        indexed[key]=row
    if set(indexed)!=required:raise ValueError('All24 prespecified fresh effects must be reported')
    checks={}
    for encoder in ENCODERS:
        def item(metric):return indexed[encoder,metric,'joint_minus_original_frozen']
        checks[encoder]={'i2t_strict_retention':exact_lower_exceeds(item('i2t'),Fraction(-1,100)),
                         't2i_strict_retention':exact_lower_exceeds(item('t2i'),Fraction(-1,100)),
                         'original_joint_strict_positive_lower':exact_lower_exceeds(item('original'),Fraction(0)),
                         'source_pair_nonnegative_exact_mean':exact_mean_nonnegative(item('source_pair'))}
    return {'passed':all(all(row.values()) for row in checks.values()),'checks':checks,
            'comparison':'joint_minus_original_frozen','fresh_same_source_support_only':True,
            'replaces_practical_gate':False,'unseen_data_guarantee_claimed':False,'candidate_selection_performed':False}
