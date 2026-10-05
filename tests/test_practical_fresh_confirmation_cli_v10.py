"""Synthetic lock/precondition tests; no actual private data or model states."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fresh_cli_test',ROOT/'scripts/evaluate_practical_fresh_confirmation_v10.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_rejects_wrong_lock_hash_before_subprocess(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'root_path',lambda p:Path(p))
    monkeypatch.setattr(m,'digest',lambda p:'actual')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('Unqualified subprocess launched'))
    with pytest.raises(ValueError,match='hash'):m.load_input_lock(tmp_path/'lock.json','bad')


def test_qualification_isolated_to_pinned_runtime(monkeypatch,tmp_path):
    path=tmp_path/'lock.json';value={'source_sha256':{m.NEW_SOURCES[0]:'sha'}};path.write_text(json.dumps(value))
    monkeypatch.setattr(m,'root_path',lambda p:Path(p));monkeypatch.setattr(m,'digest',lambda p:'sha')
    monkeypatch.setattr(m,'read',lambda p:json.loads(Path(p).read_text()));monkeypatch.setattr(m,'verify_sources',lambda x:None)
    monkeypatch.setenv('PYTHONPATH','/different/encoder/runtime');monkeypatch.setenv('OMP_NUM_THREADS','4')
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(stdout=json.dumps(value))
    monkeypatch.setattr(m.subprocess,'run',run)
    assert m.load_input_lock(path,'sha')==value
    command,kwargs=calls[0]
    assert command[2]=='verify-state-lock'
    assert kwargs['env']['OMP_NUM_THREADS']==kwargs['env']['OPENBLAS_NUM_THREADS']==kwargs['env']['MKL_NUM_THREADS']=='1'
    assert kwargs['env']['PYTHONPATH']==str(ROOT.parent/'practical_python')+':'+str(ROOT/'src')
    assert m.os.environ['OMP_NUM_THREADS']=='4'
    assert kwargs['check'] is True


def test_source_mismatch_blocks_qualification(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'root_path',lambda p:Path(p));monkeypatch.setattr(m,'digest',lambda p:'sha')
    monkeypatch.setattr(m,'read',lambda p:{'source_sha256':{m.NEW_SOURCES[0]:'other'}})
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('Unbound source launched'))
    with pytest.raises(ValueError,match='source'):m.load_input_lock(tmp_path/'lock','sha')


def planned(monkeypatch):
    import evaluate_practical_retrieval_only_benchmark_v10 as supplement
    states={'state':{'checkpoint':'toy'}}
    receipt={'core_lock':{'path':'core'},'additional_locks':[{'path':'lab'}],'selected_states_frozen_before_scoring':24}
    values={'core':{'states':states},'lab':{'family':'labclip'}}
    monkeypatch.setattr(m,'checked',lambda x:x['path']);monkeypatch.setattr(m,'read',lambda p:values[p])
    monkeypatch.setattr(supplement,'load_release',lambda *x:(receipt,{'states':dict.fromkeys(range(6))}))
    return states,receipt,values


def test_all_planned_states_without_benchmark_success(monkeypatch):
    states,receipt,_=planned(monkeypatch)
    assert m.verify_planned_release('unused','sha',states)==receipt
    receipt['selected_states_frozen_before_scoring']=18
    with pytest.raises(ValueError,match='controls'):m.verify_planned_release('unused','sha',states)


def test_planned_release_rejects_family_or_state_change(monkeypatch):
    states,receipt,values=planned(monkeypatch)
    values['lab']['family']='unplanned'
    with pytest.raises(ValueError):m.verify_planned_release('unused','sha',states)
    values['lab']['family']='labclip';receipt['additional_locks']*=2
    with pytest.raises(ValueError):m.verify_planned_release('unused','sha',states)
    receipt['additional_locks']=receipt['additional_locks'][:1]
    with pytest.raises(ValueError):m.verify_planned_release('unused','sha',{'different':1})


def test_encoding_completion_refuses_wrong_lock_before_rows(monkeypatch):
    monkeypatch.setattr(m,'read',lambda p:{'study':'sanw_practical_v10_fresh_confirmation_encoding','complete':True,
        'encoder':'vit_b32','state_source_lock':{'path':'wrong'},'owner_lock':m.OWNER_LOCK,
        'image_count':1500,'source_caption_count':7500,'outcomes_scored':False})
    monkeypatch.setattr(m,'record',lambda p:{'path':'right'});monkeypatch.setattr(m,'checked',lambda x:pytest.fail('Fresh rows read before scope check'))
    with pytest.raises(ValueError,match='scope'):m.verify_completion('unused','lock',{})


def test_initial_lock_direct_requires_exact_inherited_scope(monkeypatch):
    monkeypatch.setattr(m,'root_path',lambda p:Path(p));monkeypatch.setattr(m,'digest',lambda p:'sha')
    monkeypatch.setattr(m,'read',lambda p:{'study':m.support.STUDY,'contract':{**m.CONTRACT,'sha256':'other'},
        'owner_lock':m.OWNER_LOCK,'raw_inputs':m.support.INPUT_RECORDS})
    monkeypatch.setattr(m,'build_state_lock',lambda *x:pytest.fail('Changed contract rebuilt'))
    with pytest.raises(ValueError,match='scope'):m._load_input_lock_direct('unused','sha')
