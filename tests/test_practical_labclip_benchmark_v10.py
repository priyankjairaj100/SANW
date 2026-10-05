"""Toy-only canonical scores, fixed coverage, audit receipts and effect reporting."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_practical_labclip_benchmark_v10 as lab
from gcr.practical_labclip_v9 import canonical_transform


def rec(path): return {'path': str(path), 'sha256': 'hash', 'bytes': 1}


def test_model_rejects_scale_dtype_and_learned_scale(tmp_path):
    path = tmp_path / 'model.npz'
    def save(**changes):
        values = dict(schema=np.asarray('sanw_labclip_inner_v9'), weight=np.eye(2, dtype=np.float32), log_scale=np.asarray(math.log(100.)),
                      learned_scale=np.asarray(False), score=np.asarray('image_dot_l2_normalized_full_rank_transformed_text'))
        np.savez(path, **(values | changes))
    save(); assert np.array_equal(lab.load_model(path, 2, 100), np.eye(2))
    for change in ({'learned_scale': np.asarray(True)}, {'weight': np.eye(2)}, {'log_scale': np.asarray(1.)}):
        save(**change)
        with pytest.raises(ValueError): lab.load_model(path, 2, 100)


def test_canonical_scorer_matches_exact_full_gallery():
    rng = np.random.default_rng(91); images = rng.normal(size=(3, 4)); images /= np.linalg.norm(images, axis=1, keepdims=True)
    texts = rng.normal(size=(6, 4)); texts /= np.linalg.norm(texts, axis=1, keepdims=True); weight = rng.normal(size=(4, 4)).astype(np.float32)
    transformed = canonical_transform(texts, weight); scorer = lab.LABCLIPScorer(weight)
    assert np.array_equal(scorer.transform(texts), transformed)
    assert np.array_equal(scorer.pair_scores(images[[0, 1]], texts[[2, 3]]), np.sum(images[[0, 1]] * transformed[[2, 3]], axis=1))
    dataset = {'arrays': {'image_features': images, 'text_features': texts, 'image_ids': np.asarray(list('abc')), 'text_ids': np.asarray(list('uvwxyz'))},
               'manifest': {'pairs': [{'image_id': 'abc'[j // 2], 'text_id': 'uvwxyz'[j], 'relation': 'source'} for j in range(6)]}}
    _, raw = lab.score_retrieval(dataset, scorer); exact = np.array([[np.sum(v * t) for t in transformed] for v in images])
    assert np.array_equal(raw['i2t_top_indices'], np.argmax(exact, axis=1)); assert np.array_equal(raw['t2i_top_indices'], np.argmax(exact, axis=0))


def test_identity_diagnostic_includes_changed_winners_and_scores():
    dataset = {'arrays': {'image_features': np.eye(2), 'text_features': np.eye(2) * 1.001}, 'manifest': {}}
    frozen = {'i2t_top_indices': np.array([0, 1]), 't2i_top_indices': np.array([0, 1]), 'i2t_correct': np.array([True, True]), 't2i_correct': np.array([True, True])}
    changed = {**frozen, 'i2t_top_indices': np.array([1, 1]), 'i2t_correct': np.array([False, True])}; out = lab.identity_parity(dataset, frozen, changed)
    assert out['i2t']['winner_changes'] == out['i2t']['correctness_changes'] == 1
    assert out['i2t']['max_absolute_pair_score_error_on_union_of_winners'] > 0 and out['descriptive_only']


def test_lock_requires_six_states_six_audits_and_empty_root(tmp_path):
    core = {'protocol': {'sha256': 'p'}, 'inputs': {}, 'evaluation_status': 'exploratory'}
    contract = {'inherited_protocol': {'sha256': 'p'}, 'contrasts': [], 'endpoints': {}, 'inference': {}}
    audit = {'study': lab.AUDIT_STUDY, 'passed': True, 'blocking_findings': [], 'contract': {'sha256': 'hash'}, 'sources': {k: 'hash' for k in lab.SOURCES}}
    runs = [(e, s) for e in lab.ENCODERS for s in lab.SEEDS]
    with patch.object(lab.core, 'load_lock', return_value=core), patch.object(lab.fit_api, 'verify_contract', return_value=(contract, {}, None)), \
         patch.object(lab, 'ROOT', tmp_path), patch.object(lab, 'root_path', side_effect=Path), patch.object(lab, 'read', return_value=audit), \
         patch.object(lab, 'digest', return_value='hash'), patch.object(lab, 'record', side_effect=rec), \
         patch.object(lab, 'verify_state', side_effect=lambda r, *args: {'encoder': r[0], 'seed': r[1], 'family': 'labclip'}), \
         patch.object(lab, 'verify_training_audit', side_effect=lambda p, *args: rec(p)):
        lock = lab.build_lock('core', 'hash', 'contract', 'audit', runs, tmp_path / 'out', training_audits=['a'] * 6)
        assert lock['effect_count'] == 10 and len(lock['states']) == len(lock['training_audits']) == 6
        for bad in (runs[:-1], runs[:-1] + runs[:1]):
            with pytest.raises(ValueError): lab.build_lock('core', 'hash', 'contract', 'audit', bad, tmp_path / 'out', training_audits=['a'] * len(bad))
        with pytest.raises(ValueError): lab.build_lock('core', 'hash', 'contract', 'audit', runs, tmp_path / 'out', training_audits=['a'] * 5)
        (tmp_path / 'out').mkdir(); (tmp_path / 'out' / 'old').write_text('toy')
        with pytest.raises(ValueError, match='precede'): lab.build_lock('core', 'hash', 'contract', 'audit', runs, tmp_path / 'out', training_audits=['a'] * 6)


def test_lab_scoring_requires24_state_release():
    with patch.object(lab.release_api, 'load_release', return_value=({'additional_locks': [], 'selected_states_frozen_before_scoring': 18}, {})):
        with pytest.raises(ValueError, match='All24'): lab.load_released(SimpleNamespace(release='r', release_sha256='hash'))


def toy_raw(dataset, correct):
    if dataset == 'sugarcrepe_pp':
        return {'item_ids': np.array(['x', 'y']), 'image_ids': np.array(['a', 'a']), 'categories': np.array(['toy', 'toy']),
                'positive1_ids': np.array(['p', 'p']), 'positive2_ids': np.array(['q', 'q']), 'negative_ids': np.array(['n', 'n']), 'correct': np.array(correct, dtype=bool)}
    return {'image_ids': np.array(['a', 'b']), 'text_ids': np.array(['x', 'y']), 'text_source_image_ids': np.array(['a', 'b']), 'owner': np.array([0, 1]),
            'i2t_correct': np.array(correct, dtype=bool), 't2i_correct': np.array(correct, dtype=bool)}


def test_all_ten_effects_report_even_when_joint_loses(tmp_path):
    lock = {'endpoints': {'e_vil_test1000': ['i2t.r1', 't2i.r1'], 'coco_karpathy': ['i2t.r1', 't2i.r1'], 'sugarcrepe_pp': ['both_accuracy']},
            'core_lock': rec('core'), 'evaluation_status': 'exploratory'}
    def prediction(args, release, supp, lock, entry, encoder, dataset):
        return ({f'joint_{s}': toy_raw(dataset, [False, False]) for s in lab.SEEDS}, {f'labclip_{s}': toy_raw(dataset, [True, True]) for s in lab.SEEDS},
                {'identity_vs_original_frozen': {}, 'artifacts': {}}, rec('index'), rec('core_index'), rec('receipt'))
    writes = []
    with patch.object(lab, 'load_prediction_index', side_effect=prediction), patch.object(lab, 'root_path', side_effect=Path), patch.object(lab, 'record', side_effect=rec), \
         patch.object(lab, 'write_npz', side_effect=lambda p, v: rec(p)), patch.object(lab, 'write_json', side_effect=lambda p, v: writes.append(v) or rec(p)):
        lab.analyze(SimpleNamespace(output=str(tmp_path / 'analysis'), release='release'), {}, {}, lock, rec('lablock'))
    result = writes[-1]; assert result['effect_count'] == len(result['effects']) == 10
    assert all(r['difference'] == -1 for r in result['effects']); assert result['new_gate'] is None and not result['candidate_selection_allowed']
    assert not result['core_practical_result_used_to_filter_effects']


@pytest.mark.parametrize('mutation', [None, 'objective', 'eligibility', 'bank'])
def test_training_only_audit_replays_selected_objective_and_movement(tmp_path, mutation):
    import torch
    from gcr.practical_labclip_v9 import NormalizedFullRankAlignment
    from gcr.practical_labclip_v10 import fixed_audit_batches, audit_bank_record, fixed_training_objective, functional_witness
    rng = np.random.default_rng(20); images = rng.normal(size=(2, 3)); texts = rng.normal(size=(12, 3))
    images /= np.linalg.norm(images, axis=1, keepdims=True); texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    source_rows = np.arange(10); owner = np.repeat(np.arange(2), 5); sources = [list(range(5)), list(range(5, 10))]; negatives = [[10], [11]]
    eligible = [0, 1]; provenance = {'toy': True}; scale = 10.; weight = np.eye(3, dtype=np.float32); weight[0, 1] = .2
    batches = fixed_audit_batches(np.asarray(eligible), sources, negatives); bank = audit_bank_record(batches)
    torch.set_num_threads(3); model = NormalizedFullRankAlignment(3, native_logit_scale=scale, learned_scale=False)
    with torch.no_grad(): model.linear.weight.copy_(torch.from_numpy(weight))
    objective = fixed_training_objective(model, torch.tensor(images, dtype=torch.float32), torch.tensor(texts, dtype=torch.float32), batches)
    witnesses = [functional_witness(images, texts, source_rows, owner, np.asarray(eligible), w) for w in (np.eye(3), weight)]
    state = {'encoder': 'vit_b32', 'epoch': 1, 'ledger': rec('ledger'), 'completion': rec('completion'), 'identity_checkpoint': rec('identity'), 'checkpoint': rec('selected')}
    identity = {'eligible_owner_indices': eligible, 'omitted_no_valid_contradiction_owner_indices': [], 'training_provenance': provenance, 'native_logit_scale': scale}
    completion = {'audit_bank': bank, 'checkpoint_history': [{'epoch': i, 'functional_witness': w, 'training_objective': objective} for i, w in enumerate(witnesses)]}
    if mutation == 'objective': completion['checkpoint_history'][1]['training_objective'] += 1
    elif mutation == 'eligibility': identity['eligible_owner_indices'] = [0]
    elif mutation == 'bank': completion['audit_bank'] = {**bank, 'sha256': 'wrong'}
    writes = []; inputs = (images, texts, source_rows, owner, sources, [[], []], negatives, provenance, scale)
    with patch.object(lab, 'root_path', side_effect=Path), patch.object(lab.fit_api, 'verify_contract', return_value=({}, {}, Path('protocol'))), \
         patch.object(lab.core, 'load_lock', return_value={}), patch.object(lab, 'verify_state', return_value=state), \
         patch('gcr.practical_training_data_v10.load_training', return_value=inputs), \
         patch.object(lab, 'verify_record', side_effect=lambda e: Path(e['path'])), patch.object(lab, 'read', side_effect=lambda p: {'identity': identity} if str(p) == 'ledger' else completion), \
         patch.object(lab, 'load_model', side_effect=lambda p, d, s: np.eye(3, dtype=np.float32) if str(p) == 'identity' else weight), \
         patch.object(lab, 'record', side_effect=rec), patch.object(lab, 'digest', return_value='hash'), \
         patch.object(lab, 'write_json', side_effect=lambda p, v: writes.append(v) or rec(p)):
        args = SimpleNamespace(output=str(tmp_path / 'audit.json'), contract='contract', core_lock='core', core_lock_sha256='hash', run='run')
        if mutation:
            with pytest.raises(ValueError): lab.audit_training(args)
        else:
            lab.audit_training(args); result = writes[-1]
            assert result['passed'] and result['selected_training_objective_recorded'] == result['selected_training_objective_replayed'] == objective
            assert not result['all_checkpoint_objectives_numerically_replayed'] and result['movement']['selected']['nonzero_functional_update']
