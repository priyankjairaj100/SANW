"""Synthetic tests only. Never launch a child or inspect research outcomes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_queue_test',HERE/'run_remaining_joint_replicas.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_runner_matching_semantics_and_duplicate_rejection():
    wanted=m.expected('vit_b32',29)
    command=['python','scripts/run_practical_streaming_v10.py']
    for key,value in wanted.items():command.extend(['--'+key,value])
    assert m.parse_runner(command,m.REPO)==wanted
    with pytest.raises(ValueError):m.parse_runner(command+['--seed','43'],m.REPO)
    process={'pid':42,'start_ticks':7,'options':wanted,'threads':{key:'3' for key in m.THREADS}}
    assert m.match_target([process],'vit_b32',29)==process
    with pytest.raises(RuntimeError):m.match_target([process,process],'vit_b32',29)
    process['threads']['OMP_NUM_THREADS']='1'
    with pytest.raises(RuntimeError):m.match_target([process],'vit_b32',29)


def sample(current=1<<30):
    return {'valid':True,'limit':8*m.GIB,'current':current,'host_available':8*m.GIB,'swap_current':0,
            'cpu_quota_cores':8,'anon':1*m.GIB,'file':0,'kernel':0,'events':{}}


def test_conservative_memory_decision_and_cache_accounting():
    ok,report=m.overlap_decision([sample() for _ in range(3)])
    assert ok and report['startup_reserve_is_planning_bound_not_measured_guarantee']
    rows=[sample(5*m.GIB) for _ in range(3)]
    for row in rows:row['file']=4*m.GIB
    ok,report=m.overlap_decision(rows)
    assert not ok and report['not_proof_of_inadequate_RAM']
    assert report['samples'][0]['raw_headroom']==3*m.GIB
    assert m.overlap_decision([sample()])[0] is False
    assert m.overlap_decision([{'valid':False}]*3)[0] is False
    rows=[sample() for _ in range(3)];rows[1]['swap_current']=1
    assert m.overlap_decision(rows)[0] is False


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def completed_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'REPO',tmp_path);monkeypatch.setattr(m,'RUN_BASE',tmp_path/'runs')
    monkeypatch.setattr(m,'CACHE',tmp_path/'cache'/'rn50')
    protocol={'fit_config':{'epochs':32,'batch_size':64},'streaming':{'fit_threads':3},'source_sha256':{},
              'training_inputs':{'vit_b32':{'synthetic':True}}}
    protocol_path=tmp_path/'protocol.json';gate_path=tmp_path/'gate.json'
    write_json(protocol_path,protocol);write_json(gate_path,{'synthetic':True})
    monkeypatch.setattr(m,'PROTOCOL',protocol_path);monkeypatch.setattr(m,'GATE',gate_path)
    monkeypatch.setattr(m,'PROTOCOL_SHA',m.sha(protocol_path));monkeypatch.setattr(m,'GATE_SHA',m.sha(gate_path))
    cache=m.CACHE.parent/'vit_b32';write_json(cache/'metadata.json',{'synthetic':True})
    path=m.run_path('vit_b32',29);path.mkdir(parents=True)
    identity={'study':'sanw_practical_v10','family':'joint','mode':'full','encoder':'vit_b32',
              'config':{**protocol['fit_config'],'seed':29},'protocol_sha256':m.PROTOCOL_SHA,'source_sha256':{},
              'protocol':m.record(protocol_path),'replication_gate':m.record(gate_path),'streaming':protocol['streaming'],
              'environment':{'blas_threads':{key:'3' for key in m.THREADS}},'fit_gallery_image_count':6000,
              'fit_gallery_text_count':30000,'official_development_or_benchmarks_used':False,
              'training_provenance':{'inputs':protocol['training_inputs']['vit_b32'],'heldout_used':False},
              'frozen_score_cache':{'path':str(cache),'metadata_sha256':m.sha(cache/'metadata.json')}}
    digest=m.hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    ledger={'identity':identity,'ledger_sha256':digest};write_json(path/'ledger.json',ledger)
    certificate={key:True for key in ('ranking_checked_canonically','ranking_preserved','feasible_with_tolerance')}
    rows=[]
    for epoch in range(1,33):
        checkpoint=path/'checkpoints'/f'epoch_{epoch:03d}.npz';checkpoint.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(checkpoint,'w') as z:z.writestr('coefficient.npy',str(epoch))
        rows.append({'epoch':epoch,'optimizer_steps':94*epoch,'training_objective':float(33-epoch),'nonzero':True,
                     'certificate':certificate,'checkpoint':{'path':str(checkpoint.relative_to(path)),
                     'sha256':m.sha(checkpoint),'ledger_sha256':digest}})
    selected=path/'selected.npz';selected.write_bytes((path/rows[-1]['checkpoint']['path']).read_bytes())
    value={'study':'sanw_practical_v10','family':'joint','mode':'full','encoder':'vit_b32',
           'config':identity['config'],'protocol_sha256':m.PROTOCOL_SHA,'ledger_sha256':digest,
           'checkpoint_history':rows,'history':[{k:v for k,v in row.items() if k!='checkpoint'} for row in rows],
           'optimizer_steps':3008,'selection':'minimum_feasible_nonzero_training_objective_then_earliest_epoch',
           'selected_epoch':32,'selected_training_objective':1.,'final_certificate':certificate,
           'selected_checkpoint':{'path':'selected.npz','sha256':m.sha(selected)}}
    write_json(path/'history.json',rows);write_json(path/'completion.json',value)
    return path,protocol,value


def test_complete_manifest_and_selected_epoch_bytes(monkeypatch,tmp_path):
    path,protocol,value=completed_fixture(monkeypatch,tmp_path)
    assert m.completion('vit_b32',29,protocol)['epoch']==32
    with zipfile.ZipFile(path/'selected.npz','w') as z:z.writestr('coefficient.npy','wrong')
    value['selected_checkpoint']['sha256']=m.sha(path/'selected.npz');write_json(path/'completion.json',value)
    with pytest.raises(ValueError,match='Selected bytes'):m.completion('vit_b32',29,protocol)


def test_complete_budget_and_ledger_tampering(monkeypatch,tmp_path):
    path,protocol,value=completed_fixture(monkeypatch,tmp_path)
    value['optimizer_steps']=3007;write_json(path/'completion.json',value)
    with pytest.raises(ValueError):m.completion('vit_b32',29,protocol)
    value['optimizer_steps']=3008;write_json(path/'completion.json',value)
    ledger=m.read(path/'ledger.json');ledger['identity']['config']['seed']=43;write_json(path/'ledger.json',ledger)
    with pytest.raises(ValueError,match='Ledger'):m.completion('vit_b32',29,protocol)


def test_refuse_existing_output_before_process_launch(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'RUN_BASE',tmp_path);m.run_path('rn50',29).mkdir()
    monkeypatch.setattr(m,'fixed_inputs',lambda:None)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('Child must never launch'))
    with pytest.raises(FileExistsError):m.launch(29,True,SimpleNamespace())


def test_incomplete_cache_refused_without_repair(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'CACHE',tmp_path/'cache')
    assert m.cache_status()['state']=='absent'
    m.CACHE.mkdir()
    with pytest.raises(RuntimeError,match='incomplete'):m.cache_status()


def test_narrow_attested_process_identity_and_ledger(monkeypatch,tmp_path):
    path,protocol,value=completed_fixture(monkeypatch,tmp_path)
    argv=['python','scripts/run_practical_streaming_v10.py']
    for key,val in m.expected('vit_b32',29).items():argv.extend(['--'+key,val])
    raw=b'\0'.join(x.encode() for x in argv)+b'\0'
    item={'encoder':'vit_b32','seed':29,'pid':71,'start_ticks':99,'argv':argv,
          'cmdline_sha256':m.hashlib.sha256(raw).hexdigest(),'ledger':m.record(path/'ledger.json'),
          'recorded_threads':{key:'3' for key in m.THREADS}}
    result=m.attested_process(71,99,argv,raw,{71:item})
    assert result['cwd_thread_provenance']=='recorded_parent_launch_and_validated_ledger'
    assert m.match_target([result],'vit_b32',29)==result
    for pid,ticks,words,bytes_ in [(72,99,argv,raw),(71,100,argv,raw),(71,99,argv+['x'],raw),(71,99,argv,raw+b'\0')]:
        with pytest.raises(PermissionError):m.attested_process(pid,ticks,words,bytes_,{71:item})
    with pytest.raises(PermissionError):m.attested_process(71,99,argv,raw,None)
    (path/'ledger.json').write_text('{}')
    with pytest.raises(PermissionError):m.attested_process(71,99,argv,raw,{71:item})


def test_attestation_refuses_other_family_or_missing_pair(monkeypatch,tmp_path):
    path,protocol,value=completed_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(m,'BASE',tmp_path)
    value={'study':'sanw_existing_vit_process_attestation_v1','protocol_sha256':m.PROTOCOL_SHA,
           'gate_sha256':m.GATE_SHA,'processes':[]}
    attestation=tmp_path/'attestation.json';write_json(attestation,value)
    with pytest.raises(ValueError,match='scope'):m.load_existing_attestation(attestation,m.sha(attestation),protocol)
    item={'encoder':'rn50','seed':29,'pid':71,'start_ticks':99,'recorded_cwd':str(tmp_path),
          'recorded_threads':{key:'3' for key in m.THREADS},
          'recorded_provenance':'parent_launch_and_frozen_ledger_not_proc_observed','argv':[]}
    value['processes']=[item,{**item,'seed':43,'pid':72}];write_json(attestation,value)
    with pytest.raises(ValueError,match='fixed ViT'):m.load_existing_attestation(attestation,m.sha(attestation),protocol)
