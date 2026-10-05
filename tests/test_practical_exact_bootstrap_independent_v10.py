"""Independent regression/oracle checks for strict rational bootstrap gates."""
from fractions import Fraction

import numpy as np

from gcr.practical_exact_bootstrap_v10 import exact_percentile, integer_cluster_bootstrap, lower_exceeds


def test_zero_cluster_effect_cannot_become_positive_by_rounded_thirds():
    numerator=np.array([1,1,1,-1,-1,-1],dtype=np.int64)
    clusters=np.zeros(6,dtype=np.int64)
    legacy=np.bincount(clusters,weights=numerator/3)[0]/6
    assert legacy>0  # Demonstrates the actual inherited false-positive case.
    effect,samples=integer_cluster_bootstrap(numerator,3,clusters)
    assert effect['exact_ci_lower']=={'numerator':0,'denominator':1}
    assert not lower_exceeds(effect,Fraction(0))
    np.testing.assert_array_equal(samples,np.zeros(100000))


def test_900_query_three_seed_retention_equality_fails_exactly():
    numerator=np.r_[np.full(36,-1),np.zeros(825,dtype=np.int64),np.ones(39,dtype=np.int64)]
    effect,samples=integer_cluster_bootstrap(numerator,3,np.arange(900),grouped=True)
    assert effect['exact_ci_lower']=={'numerator':-1,'denominator':100}
    assert effect['exact_difference']=={'numerator':1,'denominator':900}
    assert not lower_exceeds(effect,Fraction(-1,100))
    draws=np.random.default_rng(20261007).multinomial(900,np.array([36,825,39])/900,size=100000)
    legacy=draws@np.array([-1/3,0,1/3])/900
    assert np.quantile(legacy,.05/160,method='linear')>-.01
    np.testing.assert_array_equal(samples,(draws[:,2]-draws[:,0]).astype(float)/2700)


def test_unequal_clusters_preserve_ratio_estimator_and_exact_percentile():
    numerator=np.array([1,0,-1,2,-2,3,1,-1,-2],dtype=np.int64)
    clusters=np.array([0,1,1,2,2,2,3,3,3],dtype=np.int64)
    effect,samples=integer_cluster_bootstrap(numerator,3,clusters)
    counts=np.array([1,2,3,3],dtype=np.int64)
    totals=np.array([1,-1,3,-2],dtype=np.int64)
    draws=np.random.default_rng(20261007).integers(0,4,size=(100000,4))
    n=totals[draws].sum(axis=1);d=3*counts[draws].sum(axis=1)
    np.testing.assert_array_equal(samples,n.astype(float)/d)
    values=sorted(Fraction(int(a),int(b)) for a,b in zip(n,d,strict=True))
    h=Fraction(1,3200)*99999;k=h.numerator//h.denominator;w=h-k
    exact=(1-w)*values[k]+w*values[k+1]
    assert effect['exact_ci_lower']=={'numerator':exact.numerator,'denominator':exact.denominator}


def test_exact_sort_reorders_distinct_ratios_that_round_to_identical_float():
    n=np.array([1073741823,1073741822],dtype=np.int64)
    d=np.array([1073741824,1073741823],dtype=np.int64)
    assert n[0]/d[0]==n[1]/d[1]
    assert Fraction(int(n[0]),int(d[0]))>Fraction(int(n[1]),int(d[1]))
    assert exact_percentile(n,d,Fraction(0))==Fraction(int(n[1]),int(d[1]))
    assert exact_percentile(n,d,Fraction(1,2))==(Fraction(int(n[0]),int(d[0]))+Fraction(int(n[1]),int(d[1])))/2
