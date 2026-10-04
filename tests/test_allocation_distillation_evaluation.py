"""Integration checks for the frozen executor schema and paired AD inference."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gcr.allocation_distillation import POLICIES, fit_candidate, select_and_export, fit_matched_controls
from gcr.allocation_distillation_analysis import (SEEDS, evaluation_specification, endpoint_values,
    equal_seed_draw_values, decomposition_arrays, effect, practical_gate)
from gcr.review_controls import make_promotion_assignment
from gcr.training import StudyConfig
from test_training import fixture_dataset
from test_strengthen_retention import tiny_pool
from test_allocation_distillation import tiny_ledger
from evaluate_allocation_distillation import planned_states, validate_manifest, score_state
from analyze_allocation_distillation import decomposition_runs, validate_matched_controls, selected_runs


@pytest.fixture(scope="module")
def exported_grid(tmp_path_factory):
    repository = tmp_path_factory.mktemp("actual_extension_schema")
    data = fixture_dataset(repository / "data")
    config = StudyConfig(feature_dim=8, epochs=1, image_batch_size=2, threads=1, learning_rates=(1e-4,))
    ledger = tiny_ledger(data)
    output = repository / "new"
    for policy in POLICIES:
        for seed in SEEDS:
            fit_candidate(repository, output, config, data, tiny_pool(), ledger, policy, 1e-4, seed)
    select_and_export(repository, output, config, ledger)
    assignments = {draw: make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
        data.pairs, data.split_indices["train"], "score_stratified", seed,
        {key: "a"*64 for key in ("manifest_sha256", "features_sha256", "protocol_sha256")},
        text_strings=[row["text"] for row in data.manifest["texts"]]) for draw,seed in enumerate((101,211,307))}
    manifest = fit_matched_controls(repository, output, config, data, tiny_pool(), ledger, assignments)
    index = {**manifest, "runs": planned_states(manifest)}
    return repository, output, manifest, index


def test_real_executor_manifest_passes_prescore_and_analysis_schema(exported_grid):
    repository, output, manifest, index = exported_grid
    assert validate_manifest(repository, output / "state_manifest.json", "synthetic-protocol") == manifest
    assert {row["cell"] for row in manifest["decomposition_selections"]} == {"supported", "allocation", "distilled", "allocation_distillation"}
    assert set(decomposition_runs(index)) == {"supported", "allocation", "distilled", "allocation_distillation"}
    validate_matched_controls(index)
    assert len(selected_runs(index, "matched_allocation_distillation")) == 9
    assert len(index["runs"]) == len({row["state_id"] for row in index["runs"]})


@pytest.mark.parametrize("key,value", [("epoch",9), ("learning_rate",.001), ("source_mix",.9), ("beta",99.)])
def test_prescore_rejects_mismatched_control_schedule(exported_grid,key,value):
    _,_,manifest,_ = exported_grid
    bad = copy.deepcopy(manifest)
    row = next(row for row in bad["states"] if row["family"] == "matched_allocation_distillation")
    row[key] = value
    with pytest.raises(ValueError, match="Random control differs"):
        planned_states(bad)


def test_prescore_rejects_mismatched_decomposition_schedule(exported_grid):
    _,_,manifest,_ = exported_grid
    bad = copy.deepcopy(manifest)
    entry = next(row for row in bad["decomposition_selections"] if row["cell"] == "distilled")
    next(row for row in bad["states"] if row["state_id"] == entry["state_id"])["beta"] = 16.
    with pytest.raises(ValueError,match="Decomposition state differs"):
        planned_states(bad)


def test_prescore_rejects_missing_random_draw(exported_grid):
    _,_,manifest,_ = exported_grid
    bad = copy.deepcopy(manifest)
    bad["selections"] = [row for row in bad["selections"] if not
        (row["family"] == "matched_allocation_distillation" and row["seed"] == 17 and row["draw_id"] == 2)]
    with pytest.raises(ValueError,match="all nine"):
        planned_states(bad)


def test_item_and_cluster_weighting_and_draw_averaging():
    predictions, runs = {}, []
    for seed_i,seed in enumerate(SEEDS):
        for draw in range(3):
            state_id = f"s{seed}d{draw}"
            runs.append({"state_id":state_id,"seed":seed,"draw_id":draw})
            predictions[(state_id,"sugarcrepe_pp")] = {"item_ids":np.array(["a","b","c"]),
                "image_ids":np.array(["x","x","y"]), "correct":np.array([seed_i/2,draw/2,1.])}
    values,ids,clusters = equal_seed_draw_values(runs,predictions,"sugarcrepe_pp","both_accuracy",expected_draws=(0,1,2))
    np.testing.assert_array_equal(values,[[0,.5,1],[.5,.5,1],[1,.5,1]])
    zero = (np.zeros_like(values),ids,clusters)
    result,samples = effect((values,ids,clusters),zero,family_size=80,replicates=301,seed=92)
    draws = np.random.default_rng(92).integers(0,2,size=(301,2))
    averaged = values.mean(axis=0)
    explicit = np.array([np.concatenate([averaged[clusters==["x","y"][j]] for j in row]).mean() for row in draws])
    np.testing.assert_allclose(samples,explicit,rtol=0,atol=1e-16)
    assert result["difference"] == pytest.approx(2/3)
    assert result["family_size"] == 80


def test_alignment_rejects_changed_item_order():
    predictions, runs = {}, []
    for seed in SEEDS:
        sid = str(seed)
        runs.append({"state_id":sid,"seed":seed,"draw_id":None})
        predictions[(sid,"visual_entailment")] = {"image_ids":np.array(["a","b"]),"image_accuracy":np.array([.3,.8])}
    predictions[("29","visual_entailment")]["image_ids"] = np.array(["b","a"])
    with pytest.raises(ValueError,match="identities differ"):
        equal_seed_draw_values(runs,predictions,"visual_entailment","accuracy")


def test_synthetic_scoring_uses_owner_clusters_and_strict_triplet_ties():
    images = np.array([[1.,0.],[0.,1.]],np.float32)
    texts = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],np.float32)
    dataset = {"name":"sugarcrepe_pp", "features":{"image_ids":np.array(["i0","i1"]),
        "text_ids":np.array(["t0","t1","t2","t3"]),"image_features":images,"text_features":texts},
        "manifest":{"triplets":[{"id":"a","image_id":"i0","positive1_id":"t0","positive2_id":"t1","negative_id":"t2","category":"x"},
                                  {"id":"b","image_id":"i1","positive1_id":"t2","positive2_id":"t3","negative_id":"t2","category":"x"}]}}
    metrics,predictions = score_state(None,dataset)
    assert metrics["both_accuracy"] == .5
    np.testing.assert_array_equal(predictions["correct"],[True,False])
    value,ids,clusters = endpoint_values(predictions,"sugarcrepe_pp","both_accuracy")
    np.testing.assert_array_equal(clusters,["i0","i1"])


def test_factorial_and_strict_practical_gate():
    ids = np.array(["x","y"])
    make = lambda x:(np.full((3,2),x),ids,ids)
    result = decomposition_arrays(make(.1),make(.2),make(.4),make(.8))
    assert result["allocation_main"][0][0,0] == pytest.approx(.25)
    assert result["distillation_main"][0][0,0] == pytest.approx(.45)
    assert result["interaction"][0][0,0] == pytest.approx(.3)
    runs = [{"seed":seed,"epoch":1,"update_norm":.1} for seed in SEEDS]
    contrasts = [{"dataset":"e_vil_test1000","metric":direction,"family_size":80,"ci_lower":-.009}
                 for direction in ("i2t.r1","t2i.r1")]
    contrasts += [{"dataset":"sugarcrepe_pp","metric":"both_accuracy","family_size":80,"ci_lower":.001}]
    assert practical_gate(runs,contrasts)["passed"]
    contrasts[0]["ci_lower"] = -.01
    assert not practical_gate(runs,contrasts)["passed"]
    assert "same selected family" in evaluation_specification()["practical_success_gate"]["cross_encoder_rule"]
