"""Toy-only checks for the independent actual-result bootstrap auditor."""
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_practical_official_results_v10 import independent_bootstrap
from gcr.practical_exact_bootstrap_v10 import integer_cluster_bootstrap
from gcr.practical_constrained_evaluation_v8 import paired_cluster_bootstrap


def test_exact_minus_one_percent_boundary_is_not_rounded_into_pass():
    n=np.zeros(900,dtype=np.int64);n[:39]=1;n[39:75]=-1
    ids=np.asarray([str(i) for i in range(900)])
    samples,effect=independent_bootstrap(n,np.full(900,3),ids,True)
    assert Fraction(**effect['exact_ci_lower'])==Fraction(-1,100)
    reference,draws=integer_cluster_bootstrap(n,3,ids,grouped=True)
    assert np.array_equal(samples,draws)
    assert effect['exact_ci_lower']==reference['exact_ci_lower']
    assert effect['exact_ci_upper']==reference['exact_ci_upper']


def test_caption_cluster_integer_grouping_and_exact_zero():
    n=np.array([1,-1,0,1,-1]*3,dtype=np.int64);ids=np.repeat(['a','b','c'],5)
    samples,effect=independent_bootstrap(n,np.ones(15,dtype=np.int64),ids,True)
    assert np.array_equal(samples,np.zeros(100000))
    assert Fraction(**effect['exact_ci_lower'])==0
    n[0]=0
    reference,draws=integer_cluster_bootstrap(n,1,ids,grouped=True)
    samples,effect=independent_bootstrap(n,np.ones(15,dtype=np.int64),ids,True)
    assert np.array_equal(samples,draws)
    assert effect['exact_ci_lower']==reference['exact_ci_lower']


def test_variable_denominator_composition_keeps_original_descriptive_law():
    n=np.array([-1,1,2,-1,0],dtype=np.int64);d=np.array([3,5,7,11,13],dtype=np.int64)
    ids=np.asarray(list('abcde'))
    samples,effect=independent_bootstrap(n,d,ids,False)
    reference,draws=paired_cluster_bootstrap(n.astype(float)/d,ids)
    assert np.array_equal(samples,draws)
    assert effect['ci_lower']==reference['ci_lower'] and effect['ci_upper']==reference['ci_upper']
