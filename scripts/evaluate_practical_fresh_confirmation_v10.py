#!/usr/bin/env python3
"""Separately frozen same-source fresh1500 confirmation; no fitting or selection.

Order: state-lock -> external gated prepare/encode -> feature-lock -> release ->
score (both encoders, seven states each) -> analyze (all24 effects). The first
lock permits encoding only. Scoring additionally waits for every planned core
and explanatory state to be frozen. No benchmark success is a prerequisite.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_practical_benchmark_v10 as core
import evaluate_practical_official_development_v10 as development
from evaluate_practical_constrained_development_v8 import digest,read,record,root_path,verify_record,write_json,write_npz
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from gcr.practical_training_data_v10 import validate_encoder_metadata,_read_features
from gcr.practical_fresh_evaluation_v10 import (ENCODERS,SEEDS,FAMILIES,METRICS,CONTRASTS,
    confirmation_view,score_confirmation,paired_counts,effect_from_counts,fresh_support_gate)

SUPPORT_PATH='recovery/training_expansion_private/fresh_confirmation_support.py'
spec=importlib.util.spec_from_file_location('fresh_confirmation_support_contract',ROOT/SUPPORT_PATH)
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
CONTRACT=support.CONTRACT;OWNER_LOCK=support.OWNER_LOCK
PROTOCOL_SHA='ce04032395f185519f3f5012a82d403ff41a7e99b8662e998c99028cf719a826'
NEW_SOURCES=(__file__.removeprefix(str(ROOT)+'/'), 'src/gcr/practical_fresh_evaluation_v10.py',
    'src/gcr/practical_fresh_statistics_v10.py','tests/test_practical_fresh_evaluation_v10.py',
    'tests/test_practical_fresh_statistics_v10.py','tests/test_practical_fresh_confirmation_cli_v10.py',
    'tests/test_fresh_confirmation_inputs_v10.py',SUPPORT_PATH,support.EXPORT_SOURCE)


def checked(entry):
    path=verify_record(entry)
    if 'bytes' in entry and path.stat().st_size!=entry['bytes']:raise ValueError('Artifact byte length changed')
    return path


def source_map(protocol):
    from run_practical_retrieval_only_v10 import NEW_SOURCES as control_sources
    from evaluate_practical_retrieval_only_benchmark_v10 import SOURCES as release_sources
    from run_practical_labclip_v10 import NEW_SOURCES as lab_fit_sources
    from evaluate_practical_labclip_benchmark_v10 import SOURCES as lab_evaluation_sources
    value=dict(protocol['source_sha256'])
    for name in (*core.EXTRA_SOURCES,*control_sources,*release_sources,*lab_fit_sources,*lab_evaluation_sources,
                 *NEW_SOURCES,*support.PINNED_ENCODING_SOURCES):
        actual=digest(root_path(name))
        if name in value and value[name]!=actual:raise ValueError('Frozen source changed')
        value[name]=actual
    for name,sha in support.PINNED_ENCODING_SOURCES.items():
        if value[name]!=sha:raise ValueError('Encoding dependency differs from frozen training exporter')
    return value


def verify_sources(value):
    for name,sha in value.items():
        if digest(root_path(name))!=sha:raise ValueError('Locked source changed: '+name)


def verify_joint_budget(state,protocol):
    run=root_path(state['run']);completion=read(checked(state['completion']));ledger=read(checked(state['ledger']))
    rows=completion['checkpoint_history'];steps=(6000+protocol['fit_config']['batch_size']-1)//protocol['fit_config']['batch_size']
    if (read(run/'history.json')!=rows or completion['optimizer_steps']!=steps*32
            or any(row['optimizer_steps']!=row['epoch']*steps for row in rows)):
        raise ValueError('Joint fit does not have the full frozen query-update budget')
    model=ConstrainedBilinearScorer.load(checked(state['checkpoint']))
    for row in rows:
        entry=row['checkpoint'];path=run/entry['path']
        if (not path.resolve().is_relative_to(run) or digest(path)!=entry['sha256']
                or entry['ledger_sha256']!=ledger['ledger_sha256']):raise ValueError('Joint epoch identity changed')
        epoch=ConstrainedBilinearScorer.load(path)
        if (any(not np.array_equal(getattr(model,key),getattr(epoch,key)) for key in ('image_mean','text_mean','image_basis','text_basis'))
                or bool(np.any(epoch.coefficient!=0))!=row['nonzero'] or not np.isfinite(epoch.coefficient).all()
                or float(np.sum(epoch.coefficient**2))>protocol['fit_config']['radius']**2*(1+128*np.finfo(np.float64).eps)):
            raise ValueError('Joint epoch geometry/nonzero/radius differs')


def build_state_lock(protocol_path,protocol_sha,development_path,source_audit_path,runs):
    development.require_evaluation_threads()
    if protocol_sha!=PROTOCOL_SHA or digest(root_path(protocol_path))!=protocol_sha:raise ValueError('Inherited protocol identity differs')
    protocol=read(protocol_path);checked(CONTRACT);checked(OWNER_LOCK)
    if protocol['study']!='sanw_practical_v10' or protocol['confirmation_owner_lock']!=OWNER_LOCK:raise ValueError('Wrong inherited study or owner lock')
    sources=source_map(protocol);verify_sources(sources)
    audit=read(source_audit_path)
    if (audit.get('study')!='sanw_practical_v10_fresh_confirmation_source_audit' or audit.get('passed') is not True
            or audit.get('blocking_findings')!=[] or audit.get('protocol',{}).get('sha256')!=protocol_sha
            or audit.get('contract')!=CONTRACT or audit.get('source_sha256')!=sources):
        raise ValueError('Independent source audit must bind the complete fresh dependency closure')
    qualified=core.verify_development_gate(development_path,protocol_sha)
    states={}
    for run in runs:
        family=read(root_path(run)/'ledger.json')['identity']['family']
        if family=='joint':
            _,_,state=development.verify_full_pilot(run,protocol,protocol_sha);verify_joint_budget(state,protocol)
            if state!=qualified[state['encoder'],state['seed']]:raise ValueError('Fresh candidate differs from development-qualified state')
        elif family=='no_retention':state=core.verify_control(run,protocol,protocol_sha)
        else:raise ValueError('Only joint and no_retention enter the fresh contract')
        key=f"{state['encoder']}__{family}__{state['seed']}"
        if key in states:raise ValueError('Duplicate fresh state')
        states[key]=state
    if set(states)!={f'{e}__{f}__{s}' for e in ENCODERS for f in FAMILIES for s in SEEDS}:
        raise ValueError('Exactly12 final joint/no_retention states required before fresh encoding')
    for e in ENCODERS:
        models=[ConstrainedBilinearScorer.load(checked(states[f'{e}__{f}__{s}']['checkpoint'])) for f in FAMILIES for s in SEEDS]
        if any(not np.array_equal(getattr(models[0],key),getattr(m,key)) for m in models[1:]
               for key in ('image_mean','text_mean','image_basis','text_basis')):
            raise ValueError('Fresh states do not share the fixed training geometry')
    return {'study':support.STUDY,'protocol':record(protocol_path),'contract':CONTRACT,'owner_lock':OWNER_LOCK,
        'development_gate':record(development_path),'states':states,'source_sha256':sources,'source_audit':record(source_audit_path),
        'raw_inputs':support.INPUT_RECORDS,'encoding_permitted':True,'outcome_scoring_permitted':False,
        'required_scoring_release_families':['retrieval_only','expanded_labclip'],
        'historical_benchmark_pass_required':False,'fresh_owner_rows_or_features_parsed_during_qualification':False}


def _load_input_lock_direct(path,expected_sha256):
    if digest(root_path(path))!=expected_sha256:raise ValueError('Fresh state/source lock hash differs')
    value=read(path)
    if (value.get('study')!=support.STUDY or value.get('contract')!=CONTRACT or value.get('owner_lock')!=OWNER_LOCK
            or value.get('raw_inputs')!=support.INPUT_RECORDS):raise ValueError('Fresh lock scope or input identities differ')
    rebuilt=build_state_lock(value['protocol']['path'],value['protocol']['sha256'],value['development_gate']['path'],
        value['source_audit']['path'],[state['run'] for state in value['states'].values()])
    if rebuilt!=value:raise ValueError('Fresh locked source, state, data or qualification changed')
    return value


def load_input_lock(path,expected_sha256):
    """Validate initial lock without reading fresh owner rows/features.

    Isolate the inherited strict qualifier in the exact existing fitting runtime;
    the caller may be the four-thread, separately pinned feature-export runtime.
    """
    filename=root_path(path)
    if digest(filename)!=expected_sha256:raise ValueError('Fresh state/source lock hash differs')
    value=read(filename)
    if value.get('source_sha256',{}).get(NEW_SOURCES[0])!=digest(Path(__file__)):
        raise ValueError('Fresh evaluator source is not locked')
    verify_sources(value['source_sha256'])
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONPATH=str(ROOT.parent/'practical_python')+os.pathsep+str(ROOT/'src'))
    completed=subprocess.run([sys.executable,str(Path(__file__)),'verify-state-lock','--lock',str(filename),
        '--lock-sha256',expected_sha256],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    verified=json.loads(completed.stdout)
    if verified!=value:raise ValueError('Fresh qualification subprocess returned a different lock')
    return value


def verify_completion(path,lock_path,lock):
    value=read(path);encoder=value.get('encoder');lock_entry=record(lock_path)
    if (value.get('study')!='sanw_practical_v10_fresh_confirmation_encoding' or value.get('complete') is not True
            or encoder not in ENCODERS or value.get('state_source_lock')!=lock_entry or value.get('owner_lock')!=OWNER_LOCK
            or value.get('image_count')!=1500 or value.get('source_caption_count')!=7500 or value.get('outcomes_scored') is not False):
        raise ValueError('Fresh encoding completion scope or decision lock differs')
    for key in ('preparation_receipt','manifest','features','metadata'):checked(value[key])
    prep=read(value['preparation_receipt']['path']);manifest=read(value['manifest']['path']);metadata=read(value['metadata']['path'])
    if (prep.get('study')!='sanw_practical_v10_fresh_confirmation_preparation' or prep.get('complete') is not True
            or prep.get('state_source_lock')!=lock_entry or prep.get('contract')!=CONTRACT or prep.get('owner_lock')!=OWNER_LOCK
            or prep.get('raw_inputs')!=support.INPUT_RECORDS or prep.get('manifest')!=value['manifest']
            or prep.get('source_sha256')!={p:lock['source_sha256'][p] for p in (support.EXPORT_SOURCE,SUPPORT_PATH,*support.PINNED_ENCODING_SOURCES)}
            or prep.get('all_raw_labels_preserved') is not True or prep.get('training_eligibility_filter_applied') is not False
            or prep.get('caption_rows_deduplicated') is not False or prep.get('fit_or_selection_allowed') is not False):
        raise ValueError('Fresh preparation no longer matches locked raw annotation rules')
    owners=read(checked(OWNER_LOCK))['image_ids'];support.validate_manifest(manifest,owners)
    counts=dict(sorted(Counter(row['relation'] for row in manifest['pairs']).items()))
    if (len(owners)!=1500 or counts.get('source')!=7500 or prep['relation_counts']!=counts or metadata['relation_counts']!=counts
            or any(item.get('text_count')!=len(manifest['texts']) for item in (value,prep,metadata))):
        raise ValueError('Fresh encoding/manifest relation counts differ')
    protocol=read(lock['protocol']['path']);old=read(checked(protocol['training_inputs'][encoder]['metadata']))
    validate_encoder_metadata(metadata,old)
    if (metadata.get('encoder_id')!=encoder or metadata.get('manifest_sha256')!=value['manifest']['sha256']
            or metadata.get('features_sha256')!=value['features']['sha256'] or metadata.get('state_source_lock')!=lock_entry
            or metadata.get('owner_lock')!=OWNER_LOCK or metadata.get('image_count')!=1500 or metadata.get('source_caption_count')!=7500
            or metadata.get('all_rows_confirmation_only') is not True or metadata.get('fit_or_selection_allowed') is not False
            or metadata.get('training_eligibility_filter_applied') is not False or metadata.get('caption_rows_deduplicated') is not False
            or metadata.get('outcomes_scored') is not False):raise ValueError('Fresh feature provenance changed')
    identity=metadata['encoding_identity']
    weight=support.INPUT_RECORDS['vit_weights' if encoder=='vit_b32' else 'rn50_weights']
    if (identity.get('encoder')!=encoder or identity.get('state_source_lock')!=lock_entry or identity.get('contract')!=CONTRACT
            or identity.get('owner_lock')!=OWNER_LOCK or identity.get('manifest')!=value['manifest'] or identity.get('weights')!=weight
            or identity.get('preparation_receipt')!=value['preparation_receipt'] or identity.get('runtime')!=support.EXPECTED_RUNTIME
            or identity.get('source_sha256')!=prep['source_sha256'] or identity.get('preprocess_config')!=support.PREPROCESS
            or identity.get('dtype')!='float32' or identity.get('normalization')!='L2'
            or identity.get('text_encoding')!='batch_causally_trimmed_after_last_eot'
            or (identity.get('threads'),identity.get('image_batch'),identity.get('text_batch'))!=(4,32,128)):
        raise ValueError('Fresh encoding runtime, identity, or source changed')
    return {'completion':record(path),**{key:value[key] for key in ('preparation_receipt','manifest','features','metadata')}},manifest,owners


def build_feature_lock(state_lock_path,state_sha,completions):
    lock=load_input_lock(state_lock_path,state_sha);inputs={}
    for path in completions:
        encoder=read(path)['encoder']
        if encoder in inputs:raise ValueError('Repeated fresh encoder completion')
        inputs[encoder],_,_=verify_completion(path,state_lock_path,lock)
    if set(inputs)!=set(ENCODERS) or inputs[ENCODERS[0]]['manifest']!=inputs[ENCODERS[1]]['manifest']:
        raise ValueError('Both fresh encoders must share the complete locked manifest')
    return {'study':'sanw_practical_v10_fresh_confirmation_feature_lock','state_source_lock':record(state_lock_path),
        'inputs':inputs,'source_sha256':lock['source_sha256'],'contract':CONTRACT,'owner_lock':OWNER_LOCK,
        'normalization':'one Torch float32 L2 pass then float64 canonical reductions','fitting_or_selection_allowed':False,
        'outcome_scoring_permitted':False}


def load_feature_lock(path,sha):
    if digest(root_path(path))!=sha:raise ValueError('Fresh feature lock hash differs')
    value=read(path);state=value['state_source_lock']
    rebuilt=build_feature_lock(state['path'],state['sha256'],[entry['completion']['path'] for entry in value['inputs'].values()])
    if value!=rebuilt:raise ValueError('Fresh feature lock changed')
    return value


def load_view(feature_lock,encoder):
    entries=feature_lock['inputs'][encoder];manifest=read(checked(entries['manifest']))
    owners=read(checked(OWNER_LOCK))['image_ids']
    images,texts=_read_features(checked(entries['features']),[row['id'] for row in manifest['images']],
        [row['id'] for row in manifest['texts']],512 if encoder=='vit_b32' else 1024)
    import torch
    import torch.nn.functional as F
    images,texts=(F.normalize(torch.from_numpy(value),dim=1).numpy().astype(np.float64) for value in (images,texts))
    return confirmation_view(images,texts,manifest,owners)


def verify_planned_release(path,sha,states):
    import evaluate_practical_retrieval_only_benchmark_v10 as supplement
    declared,supplemental=supplement.load_release(path,sha)
    baseline=read(checked(declared['core_lock']))
    additional=declared['additional_locks']
    if (baseline['states']!=states or declared['selected_states_frozen_before_scoring']!=24
            or len(additional)!=1 or read(checked(additional[0]))['family']!='labclip'
            or len(supplemental['states'])!=6):
        raise ValueError('Both retrieval_only and expanded LABCLIP controls must freeze before fresh outcomes')
    return declared


def verify_release(path,sha):
    if digest(root_path(path))!=sha:raise ValueError('Fresh release hash differs')
    value=read(path)
    if (value.get('study')!='sanw_practical_v10_fresh_confirmation_scoring_release'
            or value.get('all_planned_selected_states_frozen')!=24 or value.get('historical_benchmark_pass_required') is not False
            or value.get('contract')!=CONTRACT or value.get('no_candidate_selection') is not True):
        raise ValueError('Fresh scoring requires the complete planned model-state release')
    features=load_feature_lock(value['feature_lock']['path'],value['feature_lock']['sha256'])
    lock=read(checked(features['state_source_lock']))
    # Strict benchmark release verification has no practical benchmark pass condition.
    verify_planned_release(value['planned_states_release']['path'],value['planned_states_release']['sha256'],lock['states'])
    if value.get('state_source_lock')!=features['state_source_lock'] or value.get('fresh_inference_states')!=12 or value.get('fresh_effect_count')!=24:
        raise ValueError('Fresh release added an inference family or changed scope')
    return value,features,lock


def release(args):
    features=load_feature_lock(args.lock,args.lock_sha256)
    state=read(checked(features['state_source_lock']))
    verify_planned_release(args.planned_states_release,args.planned_states_release_sha256,state['states'])
    output=root_path(args.output_root)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh scoring release must precede all fresh predictions')
    value={'study':'sanw_practical_v10_fresh_confirmation_scoring_release','feature_lock':record(args.lock),
        'state_source_lock':features['state_source_lock'],'planned_states_release':record(args.planned_states_release),
        'all_planned_selected_states_frozen':24,'fresh_inference_states':12,'fresh_effect_count':24,
        'historical_benchmark_pass_required':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'output_root':str(output.relative_to(ROOT)),'no_candidate_selection':True,'contract':CONTRACT}
    print(json.dumps(write_json(args.output,value)),flush=True)


def score(args,release_value,features,lock):
    development.require_evaluation_threads();output=root_path(args.output)
    if not output.is_relative_to(root_path(release_value['output_root'])):raise ValueError('Fresh score output escapes released root')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh predictions are immutable')
    output.mkdir(parents=True,exist_ok=True)
    start=write_json(output/'prescore_receipt.json',{'release':record(args.release),'encoder':args.encoder,
        'feature_lock':release_value['feature_lock'],'no_fitting_or_selection':True})
    view=load_view(features,args.encoder);artifacts={}
    states={'original_frozen':None}|{f'{family}_{seed}':lock['states'][f'{args.encoder}__{family}__{seed}'] for family in FAMILIES for seed in SEEDS}
    for name,state in states.items():
        scorer=None
        if state:
            model=ConstrainedBilinearScorer.load(checked(state['checkpoint']))
            scorer=CanonicalScorer(model.image_mean,model.text_mean,model.image_basis,model.text_basis,model.coefficient)
        summary,raw=score_confirmation(view,scorer)
        artifacts[name]={'state':state,'summary':summary,'predictions':write_npz(output/f'{name}.npz',raw)}
    for entry in features['inputs'][args.encoder].values():checked(entry)
    for state in states.values():
        if state:checked(state['checkpoint'])
    value={'study':'sanw_practical_v10_fresh_confirmation_predictions','complete':True,'encoder':args.encoder,
        'release':record(args.release),'feature_lock':release_value['feature_lock'],'start_receipt':start,'artifacts':artifacts}
    print(json.dumps(write_json(output/'index.json',value)),flush=True)


def analyze(args,release_value,features,lock):
    development.require_evaluation_threads();loaded={};indices=[]
    for path in args.indices:
        index=read(path);encoder=index.get('encoder')
        if (index.get('study')!='sanw_practical_v10_fresh_confirmation_predictions' or index.get('complete') is not True
                or encoder not in ENCODERS or encoder in loaded or index.get('release')!=record(args.release)
                or index.get('feature_lock')!=release_value['feature_lock']):raise ValueError('Fresh prediction identity differs')
        start=read(checked(index['start_receipt']))
        if start!={'release':record(args.release),'encoder':encoder,'feature_lock':release_value['feature_lock'],'no_fitting_or_selection':True}:
            raise ValueError('Fresh scoring did not follow locked release')
        states={'original_frozen':None}|{f'{f}_{s}':lock['states'][f'{encoder}__{f}__{s}'] for f in FAMILIES for s in SEEDS}
        if set(index['artifacts'])!=set(states):raise ValueError('All seven fixed state predictions required')
        loaded[encoder]={}
        for name,state in states.items():
            item=index['artifacts'][name]
            if item['state']!=state:raise ValueError('Fresh outcome state differs from frozen inference list')
            loaded[encoder][name]=core.archive(item['predictions'])
        indices.append(record(path))
    if set(loaded)!=set(ENCODERS):raise ValueError('Both encoder prediction indices required')
    output=root_path(args.output)
    if not output.is_relative_to(root_path(release_value['output_root'])):raise ValueError('Fresh analysis output escapes released root')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh analysis is immutable')
    (output/'paired').mkdir(parents=True,exist_ok=True)
    effects=[];draws={};pairs={}
    for encoder,raw in loaded.items():
        for contrast,first,second in CONTRASTS:
            by_seed=lambda family:{seed:raw['original_frozen' if family=='original_frozen' else f'{family}_{seed}'] for seed in SEEDS}
            for metric in METRICS:
                paired=paired_counts(metric,by_seed(first),by_seed(second));effect,samples=effect_from_counts(metric,paired)
                key=f'{encoder}__{metric}__{contrast}';effect.update(effect_id=key,encoder=encoder,metric=metric,contrast=contrast)
                effects.append(effect);draws[key]=samples;pairs[key]=write_npz(output/'paired'/f'{key}.npz',paired)
    result={'study':'sanw_practical_v10_fresh_confirmation_analysis','release':record(args.release),'contract':CONTRACT,
        'indices':indices,'effects':effects,'paired_artifacts':pairs,'bootstrap':write_npz(output/'bootstrap.npz',draws),
        'fresh_same_source_support_gate':fresh_support_gate(effects),'fresh_effect_count':24,'no_test_selection':True,
        'historical_benchmark_pass_required':False,'practical_benchmark_gate_replaced':False,'unseen_data_guarantee_claimed':False}
    print(json.dumps({'result':write_json(output/'analysis.json',result),'fresh_support_gate':result['fresh_same_source_support_gate']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    a=sub.add_parser('state-lock');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True)
    a.add_argument('--development-gate',required=True);a.add_argument('--source-audit',required=True);a.add_argument('--runs',nargs=12,required=True);a.add_argument('--output',required=True)
    for command in ('verify-state-lock','feature-lock','release'):
        a=sub.add_parser(command);a.add_argument('--lock',required=True);a.add_argument('--lock-sha256',required=True)
        if command=='feature-lock':a.add_argument('--completions',nargs=2,required=True);a.add_argument('--output',required=True)
        if command=='release':
            a.add_argument('--planned-states-release',required=True);a.add_argument('--planned-states-release-sha256',required=True)
            a.add_argument('--output-root',required=True);a.add_argument('--output',required=True)
    for command in ('score','analyze'):
        a=sub.add_parser(command);a.add_argument('--release',required=True);a.add_argument('--release-sha256',required=True);a.add_argument('--output',required=True)
        if command=='score':a.add_argument('--encoder',choices=ENCODERS,required=True)
        else:a.add_argument('--indices',nargs=2,required=True)
    args=p.parse_args()
    if args.operation=='state-lock':print(json.dumps(write_json(args.output,build_state_lock(args.protocol,args.protocol_sha256,args.development_gate,args.source_audit,args.runs))),flush=True)
    elif args.operation=='verify-state-lock':print(json.dumps(_load_input_lock_direct(args.lock,args.lock_sha256),sort_keys=True),flush=True)
    elif args.operation=='feature-lock':print(json.dumps(write_json(args.output,build_feature_lock(args.lock,args.lock_sha256,args.completions))),flush=True)
    elif args.operation=='release':release(args)
    else:
        value,features,lock=verify_release(args.release,args.release_sha256)
        (score if args.operation=='score' else analyze)(args,value,features,lock)


if __name__=='__main__':main()
