"""Original retention-v3 schema, terminal factorial, and recovery inference tests."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from gcr.strengthen_retention import POLICIES, PROTOCOL_SHA256, fit_candidate, select_and_export, fit_matched_controls
from gcr.review_controls import make_promotion_assignment
from gcr.training import StudyConfig
from test_training import fixture_dataset
from test_strengthen_retention import tiny_pool
from evaluate_strengthen_retention import (FAMILIES, RATES, SEEDS, FACTORIAL_CELLS,
    planned_states, factorial_selections, validate_manifest, verify_protocol)
from analyze_strengthen_retention import (factorial_runs, factorial_arrays,
    validate_matched_controls, selected_runs, strategy, PRIMARY_BOOTSTRAP_SEED, FACTORIAL_BOOTSTRAP_SEED)
from gcr.allocation_distillation_analysis import effect


@pytest.fixture(scope="module")
def retention_export(tmp_path_factory):
    repository = tmp_path_factory.mktemp("retention_actual_schema")
    data = fixture_dataset(repository/"data")
    config = StudyConfig(feature_dim=8,epochs=10,image_batch_size=2,threads=1)
    ledger = {"ledger_sha256":"synthetic-retention","identity":{"encoder":"vit_b32","inputs":{"logit_scale":data.logit_scale}}}
    output = repository/"retention"
    for policy in POLICIES:
        for rate in RATES:
            for seed in SEEDS:
                fit_candidate(repository,output,config,data,tiny_pool(),ledger,policy,rate,seed)
    select_and_export(repository,output,config,ledger)
    assignments = {draw:make_promotion_assignment(data.image_ids,data.text_ids,data.images,data.texts,
        data.pairs,data.split_indices["train"],"score_stratified",seed,
        {key:"a"*64 for key in ("manifest_sha256","features_sha256","protocol_sha256")},
        text_strings=[row["text"] for row in data.manifest["texts"]]) for draw,seed in enumerate((101,211,307))}
    manifest = fit_matched_controls(repository,output,config,data,tiny_pool(),ledger,assignments)
    index = {**manifest,"runs":planned_states(manifest),"factorial_selections":factorial_selections(manifest)}
    return repository,output,manifest,index


def test_original_protocol_remains_exact_and_uses_original_inference_seeds():
    protocol = verify_protocol(ROOT/"results/strengthen_retention/protocol_v3.json",PROTOCOL_SHA256)
    assert protocol["evaluation"]["bootstrap_seed"] == PRIMARY_BOOTSTRAP_SEED == 20261005
    assert protocol["evaluation"]["mechanism_factorial"]["bootstrap_seed"] == FACTORIAL_BOOTSTRAP_SEED == 20261006
    assert len(FAMILIES)*4*2 + 3*4*2 == 80
    assert 3*len(RATES)*2*2 == 36
    with pytest.raises(ValueError,match="unchanged"):
        verify_protocol(ROOT/"results/strengthen_retention/protocol_v3.json","bad")


def test_actual_original_executor_export_passes_new_recovery_schema(retention_export):
    repository,output,manifest,index = retention_export
    assert validate_manifest(repository,output/"state_manifest.json") == manifest
    assert len(index["factorial_selections"]) == 36
    for rate in RATES:
        cells = factorial_runs(index,rate)
        assert set(cells) == set(FACTORIAL_CELLS)
        assert all(len(runs)==3 for runs in cells.values())
    assert len(selected_runs(index,"matched_distilled")) == 9
    validate_matched_controls(index)
    assert len(index["runs"]) == len({row["state_id"] for row in index["runs"]})
    assert index["runs"][0]["method"] == "frozen"


@pytest.mark.parametrize("key,value",[("epoch",9),("learning_rate",.002),("beta",999.)])
def test_original_matched_controls_cannot_drift(retention_export,key,value):
    _,_,manifest,_ = retention_export
    bad = copy.deepcopy(manifest)
    next(row for row in bad["states"] if row["family"]=="matched_distilled")[key]=value
    with pytest.raises(ValueError,match="primary selected schedule"):
        planned_states(bad)


def test_recovery_requires_all_terminal_cells(retention_export):
    _,_,manifest,_ = retention_export
    bad = copy.deepcopy(manifest)
    bad["states"]=[row for row in bad["states"] if not
        (row["method"]=="reverse_source_only" and row["seed"]==17 and row["learning_rate"]==1e-4 and row["epoch"]==10)]
    with pytest.raises(ValueError,match="891-state"):
        planned_states(bad)


def test_factorial_restoration_matches_original_signed_contrasts():
    ids=np.array(["a","b"])
    make=lambda value:(np.full((3,2),value),ids,ids)
    values=factorial_arrays(make(.8),make(.1),make(.2),make(.4))
    assert values["image_main_effect"][0][0,0]==pytest.approx(.25)
    assert values["reverse_main_effect"][0][0,0]==pytest.approx(.45)
    assert values["interaction"][0][0,0]==pytest.approx(.3)
    bad=(np.full((3,2),.4),ids[::-1],ids)
    with pytest.raises(ValueError,match="not paired"):
        factorial_arrays(make(.8),make(.1),make(.2),bad)


@pytest.mark.parametrize("seed,family",[(20261005,80),(20261006,36)])
def test_original_bootstrap_seed_and_family_are_used_exactly(seed,family):
    ids=np.array(["a","b","c","d"])
    groups=np.array(["x","x","x","y"])
    left=(np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1.]]),ids,groups)
    right=(np.zeros((3,4)),ids,groups)
    result,samples=effect(left,right,family_size=family,replicates=419,seed=seed)
    draws=np.random.default_rng(seed).integers(0,2,size=(419,2))
    item=left[0].mean(axis=0)
    explicit=np.array([np.concatenate([item[groups==["x","y"][j]] for j in row]).mean() for row in draws])
    np.testing.assert_allclose(samples,explicit,rtol=0,atol=1e-16)
    bounds=np.quantile(explicit,[.05/(2*family),1-.05/(2*family)],method="linear")
    np.testing.assert_allclose([result["ci_lower"],result["ci_upper"]],bounds,rtol=0,atol=1e-16)
    assert result["bootstrap_seed"]==seed and result["family_size"]==family


def test_factorial_ignores_distinct_wise_alphas_at_the_same_schedule(retention_export):
    _,_,manifest,_ = retention_export
    changed=copy.deepcopy(manifest)
    wise=copy.deepcopy(next(row for row in changed["states"] if row["method"]=="wise_ft"))
    wise["state_id"] += "__another_alpha"
    wise["alpha"] = .25 if wise["alpha"] != .25 else .5
    changed["states"].append(wise)
    assert factorial_selections(changed)==factorial_selections(manifest)
