"""Identity, gallery scope, tie semantics, and certificate denominator tests."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from gcr.rank_retention import rank_retention_diagnostics
from diagnose_strengthen_retention import (diagnose_direction,coverage_summary,
    stable_ranks,assert_benchmark_alignment,MASK_KEYS)


def test_identity_covers_exact_positive_sets_and_keeps_wrong_teachers_uncertified():
    images=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    texts=np.array([[1.,0.],[.8,.6],[0.,1.],[-1.,0.]])
    relevance=[[0,1],[2],[0]]
    values=diagnose_direction(images,texts,images,texts,relevance,100.,block_size=2)
    np.testing.assert_array_equal(values["teacher_correct"],[True,True,False])
    np.testing.assert_array_equal(values["student_correct"],[True,True,False])
    np.testing.assert_allclose(values["divergence"],0,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(values["logit_drift_oscillation"],0)
    for key in MASK_KEYS:
        np.testing.assert_array_equal(values[key],[True,True,False])
    assert values["threshold"][0]>0 and values["threshold"][2]==0


def test_stable_benchmark_tie_and_pessimistic_certificate_remain_separate():
    query=np.array([[1.,0.]])
    gallery=np.array([[1.,0.],[1.,0.]])
    values=diagnose_direction(query,gallery,query,gallery,[[0]],7.)
    assert values["benchmark_teacher_ranks"][0]==1
    assert values["benchmark_student_ranks"][0]==1
    assert not values["teacher_correct"][0] and not values["student_correct"][0]
    assert not values["certified"][0]
    summary=coverage_summary(values)
    assert summary["counts"]["benchmark_teacher_correct"]==1
    assert summary["counts"]["teacher_correct"]==0
    assert summary["tie_disagreements"]["teacher_stable_vs_pessimistic"]==1
    assert summary["certificate_coverage"]["certified"]["fraction_of_teacher_correct"] is None


def test_minigallery_identity_cannot_certify_a_larger_gallery():
    query=np.array([[1.,0.]])
    teacher=np.array([[.8,.6],[0.,1.],[.6,.8]])
    student=np.array([[.8,.6],[0.,1.],[1.,0.]])
    mini=diagnose_direction(query,teacher[:2],query,student[:2],[[0]],30.)
    full=diagnose_direction(query,teacher,query,student,[[0]],30.)
    assert mini["certified"][0]
    assert full["teacher_correct"][0] and not full["student_correct"][0]
    assert not full["certified"][0]
    assert full["divergence"][0]>=full["threshold"][0]
    with pytest.raises(ValueError,match="identical query and gallery"):
        diagnose_direction(query,teacher,query,student[:2],[[0]],30.)


def test_streaming_matches_intact_diagnostic_at_native_scale_without_t_squared():
    rng=np.random.default_rng(91)
    teacher_queries=rng.normal(size=(7,4))
    teacher_candidates=rng.normal(size=(9,4))
    student_queries=teacher_queries+.003*rng.normal(size=(7,4))
    student_candidates=teacher_candidates+.003*rng.normal(size=(9,4))
    relevance=[[i%9,(i+1)%9] for i in range(7)]
    scale=3.7
    direct=rank_retention_diagnostics((teacher_queries@teacher_candidates.T)*scale,
        (student_queries@student_candidates.T)*scale,relevance,temperature=2.,k=1,atol=1e-12,rtol=1e-10)
    blocked=diagnose_direction(teacher_queries,teacher_candidates,student_queries,student_candidates,relevance,scale,block_size=3)
    for key,expected in direct.items():
        if expected.dtype==np.bool_:
            np.testing.assert_array_equal(blocked[key],expected)
        else:
            np.testing.assert_allclose(blocked[key],expected,rtol=1e-12,atol=1e-13)
    np.testing.assert_array_equal(blocked["benchmark_teacher_ranks"],stable_ranks(teacher_queries@teacher_candidates.T,relevance))


def test_coverage_uses_distinct_teacher_and_retained_denominators():
    teacher_logits=np.array([[3.,1.],[3.,1.],[1.,3.]])
    student_logits=np.array([[3.,1.],[1.,3.],[1.,3.]])
    values=rank_retention_diagnostics(teacher_logits,student_logits,[[0],[0],[0]],temperature=2.)
    values["benchmark_teacher_ranks"]=stable_ranks(teacher_logits,[[0],[0],[0]])
    values["benchmark_student_ranks"]=stable_ranks(student_logits,[[0],[0],[0]])
    summary=coverage_summary(values)
    assert summary["counts"]["teacher_correct"]==2
    assert summary["counts"]["actually_retained"]==1
    coverage=summary["certificate_coverage"]["certified"]
    assert coverage["certified_queries"]==1
    assert coverage["fraction_of_all_queries"]==pytest.approx(1/3)
    assert coverage["fraction_of_teacher_correct"]==.5
    assert coverage["fraction_of_actually_retained"]==1.
    assert coverage["false_certificates"]==0


def test_benchmark_alignment_rejects_changed_ranks_and_gallery_identity():
    raw={"image_ids":np.array(["a"]),"text_ids":np.array(["t"]),"text_source_image_ids":np.array(["a"]),
         "i2t_benchmark_teacher_ranks":np.array([1]),"i2t_benchmark_student_ranks":np.array([1]),
         "t2i_benchmark_teacher_ranks":np.array([1]),"t2i_benchmark_student_ranks":np.array([1])}
    prediction={"image_ids":raw["image_ids"],"text_ids":raw["text_ids"],"text_source_image_ids":raw["text_source_image_ids"],
                "i2t_ranks":np.array([1]),"t2i_ranks":np.array([1])}
    assert_benchmark_alignment(raw,prediction,prediction)
    wrong=dict(prediction,i2t_ranks=np.array([2]))
    with pytest.raises(ValueError,match="student ranks differ"):
        assert_benchmark_alignment(raw,wrong,prediction)
    wrong=dict(prediction,text_ids=np.array(["another"]))
    with pytest.raises(ValueError,match="identities differ"):
        assert_benchmark_alignment(raw,wrong,prediction)


def test_reused_raw_masks_cannot_silently_report_a_false_certificate():
    query=np.array([[1.,0.]])
    gallery=np.array([[1.,0.],[1.,0.]])
    values=diagnose_direction(query,gallery,query,gallery,[[0]],7.)
    values["logit_drift_certified"][0]=True
    with pytest.raises(ValueError,match="false full-gallery certificate"):
        coverage_summary(values)
