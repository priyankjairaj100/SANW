"""Fixed zero semantic coefficient, retained constraints, and execution gates."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_practical_retrieval_only_v10 import (validate_contract_payload, verify_development_gate, verify_matched_controls,
                                             CONTRACT_SCHEMA, GATE_STUDY, CONTRASTS, ENDPOINTS, INFERENCE)
from gcr.practical_constrained_v8 import _ball, repair_feasibility, ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import JointFitConfig
from gcr.practical_retrieval_only_v10 import fit_retrieval_only
from test_practical_joint_v9 import fixture


def test_matches_independent_source_only_update_trajectory_and_selection():
    model,constraints,composition,retrieval=fixture()
    config=JointFitConfig(rank=3,radius=.15,epochs=3,batch_size=2,composition_weight=.25)
    rng=np.random.default_rng(config.seed); coefficient=model.coefficient.copy(); states=[]; objectives=[]
    for epoch in range(1,config.epochs+1):
        order=rng.permutation(len(constraints.images))
        for start in range(0,len(order),config.batch_size):
            _,gr=retrieval.loss_gradient(coefficient,order[start:start+config.batch_size])
            _,gp=constraints.penalty(coefficient)
            gradient=gr+config.ridge*coefficient+config.constraint_weight*gp
            norm=np.linalg.norm(gradient)
            if norm>config.gradient_clip: gradient*=config.gradient_clip/norm
            coefficient-=(config.learning_rate/math.sqrt(epoch))*gradient
            coefficient=_ball(coefficient,config.radius)
        coefficient,_=repair_feasibility(coefficient,constraints,config)
        value,_=retrieval.loss_gradient(coefficient)
        objectives.append(value+config.ridge*np.sum(coefficient**2)/2);states.append(coefficient.copy())
    second,c2,j2,r2=fixture(); observed=[]
    result=fit_retrieval_only(second,j2,r2,c2,config,lambda row,point:observed.append(point))
    for actual,expected in zip(observed,states,strict=True):np.testing.assert_array_equal(actual,expected)
    best=int(np.argmin(objectives))
    np.testing.assert_array_equal(second.coefficient,states[best])
    assert result['selected_epoch']==best+1 and result['optimizer_steps']==9
    assert result['fixed_composition_multiplier']==0 and result['candidate_selection_allowed'] is False
    assert result['final_certificate']['ranking_preserved'] is True
    assert result['final_certificate']['ranking_checked_canonically'] is True


def test_composition_gradients_and_diagnostics_cannot_change_fitting():
    config=JointFitConfig(rank=3,radius=.15,epochs=2,batch_size=2,composition_weight=.25)
    first,c1,j1,r1=fixture();normal=fit_retrieval_only(first,j1,r1,c1,config)
    second,c2,_,r2=fixture()
    fake=SimpleNamespace(eligible=np.arange(5),summary=lambda point:{'deliberately_irrelevant_metric':1e9},
                         loss_gradient=lambda point,indices,conf:(float(1e8+np.sum(point)*1e6),np.full_like(point,1e6)))
    result=fit_retrieval_only(second,fake,r2,c2,config)
    np.testing.assert_array_equal(first.coefficient,second.coefficient)
    assert result['selected_training_objective']==normal['selected_training_objective']
    assert result['selected_epoch']==normal['selected_epoch']
    third,c3,j3,r3=fixture();fit_retrieval_only(third,j3,r3,c3,replace(config,seed=29))
    assert not np.array_equal(first.coefficient,third.coefficient)


def test_harmful_gradient_is_still_repaired_and_all_frozen_correct_queries_retained():
    images=np.eye(2);model=ConstrainedBilinearScorer.from_training(images,images,2)
    constraints=FullGalleryConstraints(images,images,np.arange(2),model)
    retrieval=SimpleNamespace(loss_gradient=lambda point,image_indices=None:(float(4*np.trace(point)),4*np.eye(2)))
    composition=SimpleNamespace(eligible=np.arange(2),summary=lambda point:{},
                                loss_gradient=lambda point,indices,config:(0.,np.zeros_like(point)))
    config=JointFitConfig(rank=2,radius=10,epochs=1,batch_size=2,learning_rate=1)
    with patch.object(constraints,'penalty',wraps=constraints.penalty) as penalty:
        result=fit_retrieval_only(model,composition,retrieval,constraints,config)
        assert penalty.call_count==1
    assert not np.array_equal(model.coefficient,-4*np.eye(2))
    assert constraints.active
    cert=result['final_certificate']
    assert cert['ranking_preserved'] and cert['feasible_with_tolerance']
    assert cert['lost_frozen_correct_i2t']==cert['lost_frozen_correct_t2i']==0


def contract_fixture():
    config=asdict(JointFitConfig(radius=1,composition_weight=.25));config.pop('seed')
    protocol={'study':'sanw_practical_v10','fit_config':config,'fresh_confirmation_contract':{'path':'fresh.json','sha256':'fresh-sha'}}
    contract={'study':CONTRACT_SCHEMA,'family':'retrieval_only','inherited_protocol':{'sha256':'protocol-sha'},
              'inherited_fit_config':config,'seeds':[17,29,43],'fixed_composition_multiplier':0.,
              'candidate_selection_allowed':False,'explanatory_control_only':True,'fresh_confirmation_inclusion_allowed':False,
              'fresh_confirmation_contract':protocol['fresh_confirmation_contract'],'fresh_confirmation_contract_unchanged':True,
              'required_gate_study':GATE_STUDY,'required_gate_family':'joint','require_completed_matched_no_retention_seeds':[17,29,43],
              'contrasts':CONTRASTS,'endpoints':ENDPOINTS,'inference':INFERENCE,'encoders':['vit_b32','rn50'],
              'selection':'minimum_feasible_nonzero_training_objective_then_earliest_epoch'}
    return contract,protocol


@pytest.mark.parametrize('key,value',[('candidate_selection_allowed',True),('fixed_composition_multiplier',.25),
                                    ('fresh_confirmation_inclusion_allowed',True),('seeds',[17]),
                                    ('required_gate_family','retrieval_only')])
def test_contract_rejects_candidate_promotion_new_confirmation_or_objective_tuning(key,value):
    contract,protocol=contract_fixture();validate_contract_payload(contract,protocol,'protocol-sha')
    changed=deepcopy(contract);changed[key]=value
    with pytest.raises(ValueError):validate_contract_payload(changed,protocol,'protocol-sha')


@pytest.mark.parametrize('study,passed',[(GATE_STUDY,False),('sanw_practical_v10_replication_gate',True)])
def test_seed17_or_failed_gate_cannot_launch_explanatory_fits(study,passed):
    gate={'study':study,'family':'joint','passed':passed,'protocol_sha256':'sha','encoders':{'vit_b32':{},'rn50':{}}}
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'gate.json';path.write_text(json.dumps(gate))
        with pytest.raises(ValueError):verify_development_gate(Path(directory),path,{},'sha')


def test_changed_aggregate_evidence_must_pass_strict_raw_reconstruction():
    gate={'study':GATE_STUDY,'family':'joint','passed':True,'protocol_sha256':'sha','encoders':{'vit_b32':{},'rn50':{}}}
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'gate.json';path.write_text(json.dumps(gate))
        with patch('run_practical_retrieval_only_v10.strict_development_gate',side_effect=ValueError('bootstrap evidence changed')) as verifier:
            with pytest.raises(ValueError,match='bootstrap evidence changed'):
                verify_development_gate(Path(directory),path,{},'sha')
            verifier.assert_called_once_with(path,'sha')


@pytest.mark.parametrize('tamper',['missing_epoch','epoch_hash'])
def test_matched_control_partial_budget_and_changed_epoch_rejected(tamper):
    _,protocol=contract_fixture();config={**protocol['fit_config'],'seed':17}
    value={'study':'sanw_practical_v10','family':'no_retention','encoder':'vit_b32','config':config,
           'protocol_sha256':'sha','optimizer_steps':3008,'retention_enforced':False,
           'final_training_retention_diagnostic':{'ranking_checked_canonically':True},
           'history':[{'epoch':i,'optimizer_steps':i*94} for i in range(1,33)]}
    if tamper=='missing_epoch':value['history'].pop()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'epoch.npz').write_bytes(b'changed-state')
        model=ConstrainedBilinearScorer(np.zeros(128),np.zeros(128),np.eye(128),np.eye(128),np.eye(128)*.001)
        model.save(root/'selected.npz');value['selected_checkpoint']={'path':'selected.npz'}
        value['checkpoint_history']=[{'checkpoint':{'path':'epoch.npz','sha256':'incorrect-hash','ledger_sha256':'ledger'}}]
        (root/'ledger.json').write_text(json.dumps({'ledger_sha256':'ledger'}))
        path=root/'completion.json';path.write_text(json.dumps(value))
        with patch('run_practical_retrieval_only_v10.strict_control',return_value={'seed':17}):
            with pytest.raises(ValueError):verify_matched_controls(root,[path],'sha',protocol)
