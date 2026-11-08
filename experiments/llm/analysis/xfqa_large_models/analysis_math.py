"""Fixed numerical backends for XfqA; no historical source is modified.

MSE uses a deterministic Gram-matrix active-set simplex QP. SLSQP is a fixed,
recorded numerical recovery branch, never a dynamically discovered backend.
MAE uses sparse HiGHS residual constraints with exactly one solver thread.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from dataclasses import dataclass, asdict
from time import perf_counter
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, linprog, OptimizeWarning

BACKEND = 'gram_active_set_v1_with_fixed_slsqp_recovery'

@dataclass
class Fit:
    weights: np.ndarray
    raw_weights: np.ndarray
    train_objective: float
    solver_objective: float
    solver_backend: str
    success: bool = True
    kkt_gap: float | None = None
    raw_min_weight: float = 0.
    raw_sum_error: float = 0.
    repair_prediction_max_abs: float = 0.
    repair_l1: float = 0.
    iterations: int = 0
    solve_seconds: float = 0.

    def record(self):
        d=asdict(self)
        d['weights']=self.weights.tolist()
        d['raw_weights']=self.raw_weights.tolist()
        return d


def validate_scalar(x,y,mass):
    x,y,mass=(np.asarray(a,dtype=np.float64) for a in (x,y,mass))
    if x.ndim!=2 or x.shape[1]<1 or y.shape!=(len(x),) or mass.shape!=(len(x),):
        raise ValueError('Expected x[n,p], y[n], mass[n]')
    if not all(np.isfinite(a).all() for a in (x,y,mass)) or np.any(mass<0) or mass.sum()<=0:
        raise ValueError('Nonfinite data or invalid mass')
    positive=mass>0
    return x[positive],y[positive],mass[positive]/mass.sum()


def _repair(raw,x,objective,solver_objective,backend,iterations,started,kkt_gap=None):
    raw=np.asarray(raw,dtype=np.float64)
    if not np.isfinite(raw).all() or raw.min() < -1e-7 or abs(raw.sum()-1)>1e-7:
        raise RuntimeError('Raw simplex constraint violation')
    w=np.maximum(raw,0);w/=w.sum()
    change=float(np.max(np.abs(x@(w-raw))))
    if w.min() < -1e-14 or abs(w.sum()-1)>1e-12 or change>1e-6:
        raise RuntimeError('Simplex repair violation')
    direct=float(objective(w))
    if abs(direct-float(solver_objective))>1e-7:
        raise RuntimeError('Training objective mismatch')
    return Fit(w,raw.copy(),direct,float(solver_objective),backend,True,kkt_gap,
        float(raw.min()),float(abs(raw.sum()-1)),change,float(abs(w-raw).sum()),
        int(iterations),perf_counter()-started)


def gram_active_set(gram,cross,*,tolerance=2e-11,maxiter=300):
    """Simplex QP using exact active-face solves and blocking-step removals."""
    g=np.asarray(gram,dtype=np.float64);c=np.asarray(cross,dtype=np.float64)
    p=len(c)
    if g.shape!=(p,p) or not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError('Invalid Gram system')
    w=np.zeros(p);w[int(np.argmin(np.diag(g)-2*c))]=1.
    active=set(np.flatnonzero(w))
    for iteration in range(1,maxiter+1):
        ids=np.array(sorted(active),dtype=int)
        k=len(ids)
        system=np.empty((k+1,k+1));system[:k,:k]=g[np.ix_(ids,ids)]
        system[:k,k]=1.;system[k,:k]=1.;system[k,k]=0.
        rhs=np.r_[c[ids],1.]
        try:solution=np.linalg.solve(system,rhs)
        except np.linalg.LinAlgError:solution=np.linalg.lstsq(system,rhs,rcond=1e-13)[0]
        z=np.zeros(p);z[ids]=solution[:k]
        if np.min(z[ids]) < -1e-12:
            direction=z-w
            blocking=ids[direction[ids] < -1e-14]
            if not len(blocking):break
            step=min(1.,float(np.min(-w[blocking]/direction[blocking])))
            w=w+step*direction
            w[np.abs(w)<1e-13]=0.
            active={int(i) for i in ids if w[i]>1e-13}
            if not active:break
            w=np.maximum(w,0);w/=w.sum()
            continue
        w=np.maximum(z,0);w/=w.sum()
        gradient=2*(g@w-c)
        gap=max(0.,float(gradient@w-gradient.min()))
        if gap<=tolerance:return w,iteration,gap,'active_set'
        entering=int(np.argmin(gradient))
        if entering in active:break
        active.add(entering)
    # Fixed, outcome-independent recovery for singular/ill-conditioned faces.
    result=minimize(lambda z:float(z@g@z-2*c@z),w,
        jac=lambda z:2*(g@z-c),method='SLSQP',bounds=[(0.,1.)]*p,
        constraints={'type':'eq','fun':lambda z:float(z.sum()-1.),'jac':lambda z:np.ones_like(z)},
        options={'ftol':1e-14,'maxiter':2000})
    if not result.success:raise RuntimeError('MSE QP recovery failed: '+str(result.message))
    w=np.asarray(result.x)
    gradient=2*(g@w-c);gap=max(0.,float(gradient@w-gradient.min()))
    if gap>1e-7:raise RuntimeError(f'MSE KKT gap exceeds fixed tolerance: {gap}')
    return w,iteration+int(result.nit),gap,'slsqp_recovery'


def fit_mse(x,y,mass,*,gram=None,cross=None):
    started=perf_counter();x,y,mass=validate_scalar(x,y,mass)
    g=x.T@(mass[:,None]*x) if gram is None else np.asarray(gram)
    c=x.T@(mass*y) if cross is None else np.asarray(cross)
    raw,nit,gap,branch=gram_active_set(g,c)
    objective=lambda w:float(mass@((x@w-y)**2))
    result=_repair(raw,x,objective,objective(raw),BACKEND+'/'+branch,nit,started,gap)
    grad=2*x.T@(mass*(x@result.weights-y))
    result.kkt_gap=max(0.,float(grad@result.weights-grad.min()))
    if result.kkt_gap>1e-7:raise RuntimeError('Direct MSE KKT check failed')
    return result


def fit_mae(x,y,mass):
    started=perf_counter();x,y,mass=validate_scalar(x,y,mass)
    n,p=x.shape;eye=sparse.eye(n,format='csr');sx=sparse.csr_matrix(x)
    inequalities=sparse.vstack([sparse.hstack([sx,-eye]),sparse.hstack([-sx,-eye])],format='csr')
    equality=sparse.csr_matrix((np.ones(p),(np.zeros(p,dtype=int),np.arange(p))),shape=(1,p+n))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected:.*')
        result=linprog(np.r_[np.zeros(p),mass],A_ub=inequalities,b_ub=np.r_[y,-y],
            A_eq=equality,b_eq=[1.],bounds=(0.,None),method='highs',
            options={'threads':1,'parallel':False,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError('Sparse MAE LP failed: '+str(result.message))
    objective=lambda w:float(mass@np.abs(x@w-y))
    return _repair(result.x[:p],x,objective,result.fun,'scipy_highs_sparse_mae_threads1',result.nit,started)


def choose_single(losses,tolerance=1e-12):
    v=np.asarray(losses,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('Invalid single-peer objectives')
    return int(np.flatnonzero(v<=v.min()+tolerance)[0])
