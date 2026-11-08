from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
from scipy.optimize import linprog,minimize

COMPONENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMPONENT))
from analysis_math import fit_mse,fit_mae,choose_single
from analysis_data import common_bootstrap_multiplicities,sha
from bca_reference_selection.fit import fit_tv_simplex
from pier_llm_v23.core import fit_mse_simplex,fit_mae_simplex

@pytest.mark.parametrize('loss',['mse','mae'])
def test_exact_clones_mixtures_duplicates_and_removal(loss):
    rng=np.random.default_rng(837);x=rng.uniform(size=(60,4));mass=rng.uniform(size=60)
    fitter=fit_mse if loss=='mse' else fit_mae
    clone=fitter(x,x[:,2],mass)
    np.testing.assert_allclose(x@clone.weights,x[:,2],atol=1e-8)
    y=x@np.array([.2,.3,.1,.4]);fit=fitter(x,y,mass)
    np.testing.assert_allclose(x@fit.weights,y,atol=1e-8)
    duplicated=np.column_stack([x,x[:,2]])
    a=fitter(duplicated,x[:,2],mass);b=fitter(np.delete(duplicated,2,axis=1),x[:,2],mass)
    assert a.train_objective<1e-8 and b.train_objective<1e-8
    c=fitter(x[:,[0,1,3]],x[:,2],mass)
    assert c.train_objective>1e-3
    assert fit.raw_min_weight>=-1e-7 and fit.raw_sum_error<=1e-7
    assert fit.repair_prediction_max_abs<=1e-6
    if loss=='mse':assert fit.kkt_gap<=1e-7


def test_sparse_mae_matches_dense_tiny_reference():
    rng=np.random.default_rng(120);x=rng.random((25,5));y=rng.random(25);mass=rng.random(25)
    fit=fit_mae(x,y,mass)
    ref=fit_mae_simplex(x,y,mass)
    assert abs(np.average(abs(x@ref-y),weights=mass)-fit.train_objective)<1e-7


def test_active_set_mse_matches_original_and_slsqp_random_collinear():
    rng=np.random.default_rng(275)
    for case in range(40):
        x=rng.random((35,6));y=rng.random(35);mass=rng.random(35)
        if case%2==0:x[:,-1]=x[:,0]
        fit=fit_mse(x,y,mass)
        result=minimize(lambda w:np.average((x@w-y)**2,weights=mass),np.ones(6)/6,
            jac=lambda w:2*x.T@(mass*(x@w-y))/mass.sum(),method='SLSQP',bounds=[(0,1)]*6,
            constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(6)},
            options={'ftol':1e-13,'maxiter':2000})
        assert result.success and abs(result.fun-fit.train_objective)<1e-7 and fit.kkt_gap<1e-7
        if case==0:
            old=fit_mse_simplex(x,y,mass)
            np.testing.assert_allclose(x@old,x@fit.weights,atol=2e-6)


def test_tv_sparse_matches_explicit_tiny_dense_lp():
    rng=np.random.default_rng(55);x=rng.dirichlet(np.ones(3),size=(8,4)).transpose(0,2,1)
    y=rng.dirichlet(np.ones(3),size=8);mass=rng.random(8);mask=np.ones(y.shape,dtype=bool)
    actual=fit_tv_simplex(y,x,mask,mass)
    flat=x.reshape(-1,4);fy=y.reshape(-1);n=len(fy)
    ref=linprog(np.r_[np.zeros(4),np.repeat(.5*mass/mass.sum(),3)],
        A_ub=np.block([[flat,-np.eye(n)],[-flat,-np.eye(n)]]),b_ub=np.r_[fy,-fy],
        A_eq=np.r_[np.ones(4),np.zeros(n)][None,:],b_eq=[1],bounds=(0,None),method='highs')
    assert ref.success and abs(ref.fun-actual.train_tv)<1e-7


def test_training_only_single_tie_policy_and_mass_invariance():
    assert choose_single([.3,.2,.2])==1
    assert choose_single([.2+5e-13,.2])==0
    rng=np.random.default_rng(6);x=rng.random((20,5));y=rng.random(20);mass=rng.random(20)
    a=fit_mse(x,y,mass);b=fit_mse(x,y,77*mass)
    np.testing.assert_allclose(x@a.weights,x@b.weights,atol=1e-12)
    for invalid in [np.zeros(20),-np.ones(20),np.full(20,np.nan)]:
        with pytest.raises(ValueError):fit_mse(x,y,invalid)
