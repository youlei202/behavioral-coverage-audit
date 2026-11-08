"""Executable XfqA point analyses and prescribed bootstrap entry points.

All optimization uses fitting questions only. Frozen complete score inputs are
read through the unchanged XWVG loader with a separate eleven-model config.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import fcntl,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from .analysis_data import *
    from .analysis_math import fit_mse,fit_mae,choose_single
except ImportError:
    from analysis_data import *
    from analysis_math import fit_mse,fit_mae,choose_single
from bca_reference_selection.fit import fit_tv_simplex,verify_highs_runtime

_DATA=None

def _base(dataset,job):
    return {'run_id':dataset.root.name,**{k:job[k] for k in ('job_id','panel','target','family','split_seed','response','fit_loss')},
        'peer_ids':json_text(job['peers']),'evaluation_metric':'tv' if job['fit_loss']=='tv' else 'mae',
        'input_fingerprint':dataset.fingerprint,'completion_state':'complete'}


def job_paths(root,job_id):
    p=Path(root)/'analysis/checkpoints/point'/job_id[:2]/job_id
    return p.with_suffix('.json'),p.with_suffix('.npz')


def load_point(root,job,fingerprint):
    record_path,arrays_path=job_paths(root,job['job_id'])
    if not record_path.exists():return None
    record=json.loads(record_path.read_text())
    if record['input_fingerprint']!=fingerprint or record['job']!=job or record.get('completion_state')!='complete':
        raise RuntimeError('Point checkpoint identity mismatch: '+str(record_path))
    if not arrays_path.exists() or sha(arrays_path)!=record['arrays_sha256']:
        raise RuntimeError('Point checkpoint array checksum mismatch: '+str(arrays_path))
    return record


def fit_job(dataset,job,question_multiplicity=None):
    """Fit/evaluate exactly one declared job; never accepts held-out selection."""
    family=job['family'];seed=job['split_seed'];loss=job['fit_loss']
    train=dataset.fit_indices[seed];evaluation=dataset.eval_indices[seed]
    ti=dataset.model_ids.index(job['target']);pis=[dataset.model_ids.index(x) for x in job['peers']]
    cm=dataset.values.masses[family]
    draw=np.ones(len(dataset.values.question_ids)) if question_multiplicity is None else np.asarray(question_multiplicity,dtype=float)
    if draw.shape!=(len(dataset.values.question_ids),) or np.any(draw<0):raise ValueError('Invalid question multiplicities')
    fit_mass=(draw[train,None]*cm).reshape(-1)
    eval_mass=draw[evaluation,None]*cm
    if fit_mass.sum()<=0 or eval_mass.sum()<=0:raise RuntimeError('Empty weighted split')
    eval_mass/=eval_mass.sum()
    if loss in ('mse','mae'):
        y=dataset.scalar[family][ti]
        x=np.moveaxis(dataset.scalar[family][pis],0,-1)
        trainx=x[train].reshape(-1,len(pis));trainy=y[train].reshape(-1)
        fit=(fit_mse if loss=='mse' else fit_mae)(trainx,trainy,fit_mass)
        objective_values=np.average((trainx-trainy[:,None])**2 if loss=='mse' else abs(trainx-trainy[:,None]),axis=0,weights=fit_mass)
        selected=choose_single(objective_values)
        pred=x[evaluation]@fit.weights;truth=y[evaluation];single=x[evaluation,:,selected]
        arrays={'absolute_error':abs(pred-truth),'squared_error':(pred-truth)**2,
            'single_absolute_error':abs(single-truth),'single_squared_error':(single-truth)**2}
        metrics={'e_group':float(np.sum(eval_mass*arrays['absolute_error'])),
            'e_single':float(np.sum(eval_mass*arrays['single_absolute_error'])),
            'mse_group':float(np.sum(eval_mass*arrays['squared_error'])),
            'mse_single':float(np.sum(eval_mass*arrays['single_squared_error']))}
        fitrecord=fit.record()
    else:
        y=dataset.values.probabilities[family][ti]
        x=np.moveaxis(dataset.values.probabilities[family][pis],0,-1)
        mask=dataset.values.mask
        fit=fit_tv_simplex(y[train].reshape(-1,y.shape[-1]),x[train].reshape(-1,y.shape[-1],len(pis)),
            mask[train].reshape(-1,y.shape[-1]),fit_mass)
        if fit.repair_prediction_max_abs>1e-6:raise RuntimeError('TV repair exceeds tolerance')
        objective_values=np.average(.5*np.abs(x[train]-y[train,...,None]).sum(axis=2).reshape(-1,len(pis)),axis=0,weights=fit_mass)
        selected=choose_single(objective_values)
        pred=np.einsum('qcap,p->qca',x[evaluation],fit.weights);truth=y[evaluation];single=x[evaluation,...,selected]
        targetarg=np.argmax(np.where(mask[evaluation],truth,-np.inf),axis=-1)
        arrays={'total_variation':.5*abs(pred-truth).sum(axis=-1),
            'top_option_agreement':(np.argmax(np.where(mask[evaluation],pred,-np.inf),axis=-1)==targetarg).astype(float),
            'single_total_variation':.5*abs(single-truth).sum(axis=-1),
            'single_top_option_agreement':(np.argmax(np.where(mask[evaluation],single,-np.inf),axis=-1)==targetarg).astype(float)}
        metrics={'e_group':float(np.sum(eval_mass*arrays['total_variation'])),
            'e_single':float(np.sum(eval_mass*arrays['single_total_variation'])),
            'group_top_option_agreement':float(np.sum(eval_mass*arrays['top_option_agreement'])),
            'single_top_option_agreement':float(np.sum(eval_mass*arrays['single_top_option_agreement']))}
        fitrecord={'weights':fit.weights.tolist(),'raw_weights':fit.raw_weights,
            'train_objective':fit.train_tv,'solver_objective':fit.solver_objective,
            'solver_backend':'scipy_highs_sparse_tv_threads1','success':fit.status==0,'kkt_gap':None,
            'raw_min_weight':fit.raw_min_weight,'raw_sum_error':fit.raw_sum_error,
            'repair_prediction_max_abs':fit.repair_prediction_max_abs,'repair_l1':fit.repair_l1,
            'iterations':fit.iterations,'solve_seconds':fit.solve_seconds}
    raw_delta=np.asarray(fitrecord['weights'])-np.asarray(fitrecord['raw_weights'])
    evaluation_repair=float(np.max(np.abs(np.einsum('...p,p->...',x[evaluation],raw_delta))))
    fitrecord['repair_prediction_max_abs']=max(fitrecord['repair_prediction_max_abs'],evaluation_repair)
    if fitrecord['repair_prediction_max_abs']>1e-6:raise RuntimeError('Evaluation prediction repair exceeds fixed tolerance')
    if fitrecord['train_objective']>float(objective_values.min())+1e-7:
        raise RuntimeError('Group training objective exceeds available singleton')
    metrics.update(chosen_single=job['peers'][selected],train_single=float(objective_values[selected]),
        train_group=fitrecord['train_objective'],gamma=metrics['e_single']-metrics['e_group'])
    metrics['relative_gamma_percent']=100*metrics['gamma']/metrics['e_single'] if metrics['e_single']>1e-14 else None
    if loss=='tv':metrics['agreement_gain']=metrics['group_top_option_agreement']-metrics['single_top_option_agreement']
    arrays['evaluation_indices']=evaluation
    return fitrecord,metrics,arrays


def _point_worker(job):
    dataset=_DATA
    old=load_point(dataset.root,job,dataset.fingerprint)
    if old:return old
    fit,metrics,arrays=fit_job(dataset,job)
    recordpath,arraypath=job_paths(dataset.root,job['job_id'])
    atomic_npz(arraypath,**arrays)
    record={**_base(dataset,job),'job':job,'role':job['role'],'removed_peer':job['removed_peer'],
        **fit,**metrics,'arrays_path':str(arraypath),'arrays_sha256':sha(arraypath),'completed_at':utc()}
    atomic_json(recordpath,record)
    return record


def direct_partner_losses(dataset,target,partner,family,loss):
    ti,pi=(dataset.model_ids.index(m) for m in (target,partner))
    if loss=='tv':return .5*abs(dataset.values.probabilities[family][ti]-dataset.values.probabilities[family][pi]).sum(axis=-1)
    return abs(dataset.scalar[family][ti]-dataset.scalar[family][pi])


def _mean_test(dataset,values,family,seed):
    return float(np.mean(values[dataset.eval_indices[seed]]@dataset.values.masses[family]))


def make_tables(dataset,records):
    bykey={(r['panel'],r['target'],r['family'],r['split_seed'],r['fit_loss'],r['role'],r['removed_peer']):r for r in records}
    scalar=[];vector=[];removals=[];pairs=[];vpairs=[];expansion=[];vexpansion=[]
    scalar_metrics=['chosen_single','train_single','train_group','e_single','e_group','gamma','relative_gamma_percent','mse_single','mse_group']
    vector_metrics=['chosen_single','train_single','train_group','e_single','e_group','gamma','relative_gamma_percent','single_top_option_agreement','group_top_option_agreement','agreement_gain']
    for r in records:
        job=r['job'];base=_base(dataset,job)
        if r['role']=='group':
            dest,metrics=(vector,vector_metrics) if r['fit_loss']=='tv' else (scalar,scalar_metrics)
            dest.append({**base,**{m:r[m] for m in metrics}})
        elif r['fit_loss']!='tv':
            full=bykey[(r['panel'],r['target'],r['family'],r['split_seed'],r['fit_loss'],'group',None)]
            if r['train_objective']+1e-7<full['train_objective']:raise RuntimeError('Removal training nesting violated')
            removals.append({**base,'removed_peer':r['removed_peer'],'full_job_id':full['job_id'],'removed_job_id':r['job_id'],
                'e_full':full['e_group'],'e_without':r['e_group'],'removal_effect':r['e_group']-full['e_group']})
    for declared in dataset.config['declared_pairs']:
        for target,partner in [(declared['a'],declared['b']),(declared['b'],declared['a'])]:
            for family in sorted(dataset.config['families']):
                for seed in dataset.config['split_seeds']:
                    for loss in ('mse','mae','tv'):
                        group=bykey[('expanded11',target,family,seed,loss,'group',None)]
                        reverse=bykey[('expanded11',partner,family,seed,loss,'group',None)]
                        removed=bykey[('expanded11',target,family,seed,loss,'removal',partner)]
                        if removed['train_objective']+1e-7<group['train_objective']:raise RuntimeError('Directed removal nesting violated')
                        error=_mean_test(dataset,direct_partner_losses(dataset,target,partner,family,loss),family,seed)
                        row={**_base(dataset,group['job']),'partner':partner,'pair_kind':declared['kind'],'pair_priority':declared['priority'],
                            'full_job_id':group['job_id'],'removed_job_id':removed['job_id'],'partner_error':error,'e_group':group['e_group'],
                            'extra_beyond_partner':error-group['e_group'],'e_without':removed['e_group'],
                            'removal_effect':removed['e_group']-group['e_group'],
                            'directional_extra_difference':reverse['e_group']-group['e_group']}
                        (vpairs if loss=='tv' else pairs).append(row)
    for target in dataset.model_ids:
        comparator='baseline8' if target in dataset.config['original_roster'] else 'new_on_old8'
        for family in sorted(dataset.config['families']):
            for seed in dataset.config['split_seeds']:
                for loss in ('mse','mae','tv'):
                    full=bykey[('expanded11',target,family,seed,loss,'group',None)]
                    old=bykey[(comparator,target,family,seed,loss,'group',None)]
                    if full['train_objective']>old['train_objective']+1e-7:raise RuntimeError('Roster-expansion training nesting violated')
                    row={**_base(dataset,full['job']),'comparator_panel':comparator,'comparator_job_id':old['job_id'],
                        'expanded_job_id':full['job_id'],'comparator_error':old['e_group'],'expanded_error':full['e_group'],
                        'large_peer_gain':old['e_group']-full['e_group']}
                    (vexpansion if loss=='tv' else expansion).append(row)
    tables={'scalar_main_split_results':pd.DataFrame(scalar),'scalar_removal_split_results':pd.DataFrame(removals),
        'scalar_directional_pairs':pd.DataFrame(pairs),'scalar_roster_expansion':pd.DataFrame(expansion),
        'vector_main_split_results':pd.DataFrame(vector),'vector_directional_removals':pd.DataFrame(vpairs),
        'vector_roster_expansion':pd.DataFrame(vexpansion)}
    expected={'scalar_main_split_results':880,'scalar_removal_split_results':4400,'scalar_directional_pairs':240,
        'scalar_roster_expansion':440,'vector_main_split_results':440,'vector_directional_removals':120,'vector_roster_expansion':220}
    for name,frame in tables.items():
        if len(frame)!=expected[name]:raise RuntimeError(f'{name} count {len(frame)} != {expected[name]}')
    return tables


LOSS_COLUMNS=['absolute_error','squared_error','total_variation','top_option_agreement',
    'single_absolute_error','single_squared_error','single_total_variation','single_top_option_agreement']


def export_losses(dataset,records,path):
    """Stream all 6,540,800 condition-level observations without a giant frame."""
    tmp=Path(path).with_suffix('.parquet.tmp');writer=None;rows=0
    metadata_columns=['run_id','panel','target','family','response','fit_loss','evaluation_metric','peer_ids','input_fingerprint','completion_state']
    schema=pa.schema([('job_id',pa.string()),('base_question_id',pa.string()),('semantic_condition',pa.string()),
        ('condition_mass',pa.float64()),('split_seed',pa.int64())]+[(c,pa.string()) for c in metadata_columns]+[(c,pa.float64()) for c in LOSS_COLUMNS])
    try:
        for r in records:
            _,arraypath=job_paths(dataset.root,r['job_id'])
            with np.load(arraypath,allow_pickle=False) as a:
                ids=a['evaluation_indices'];n=len(ids)
                conditions=['C0']+[('I' if r['family']=='irrelevant_context' else 'D')+str(i) for i in range(3)]
                values={'job_id':[r['job_id']]*(n*4),
                    'base_question_id':np.repeat(np.asarray(dataset.values.question_ids)[ids],4),
                    'semantic_condition':conditions*n,'condition_mass':np.tile(dataset.values.masses[r['family']],n),
                    'split_seed':[r['split_seed']]*(n*4)}
                for col in metadata_columns:values[col]=[r[col]]*(n*4)
                for col in LOSS_COLUMNS:
                    values[col]=a[col].reshape(-1) if col in a and (not col.startswith('single_') or r['role']=='group') else np.full(n*4,np.nan)
                table=pa.Table.from_pydict(values,schema=schema)
                if writer is None:writer=pq.ParquetWriter(tmp,schema,compression='zstd')
                writer.write_table(table);rows+=n*4
    finally:
        if writer is not None:writer.close()
    if rows!=len(records)*280*4:raise RuntimeError('Per-question inventory mismatch')
    os.replace(tmp,path)
    return rows


def all_records(dataset):
    records=[]
    for job in dataset.jobs:
        r=load_point(dataset.root,job,dataset.fingerprint)
        if r is None:raise RuntimeError('Missing point job: '+job['job_id'])
        records.append(r)
    return records


def analyze(repo_root,plan_root,output_root,cache_path=None,workers=4,resume=False,**kwargs):
    global _DATA
    started=time.perf_counter();dataset=load_dataset(repo_root,plan_root,output_root,cache_path)
    root=register_identity(dataset,resume)
    lock=(root/'.point.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another point-analysis writer owns this run')
    try:
        complete=root/'point_complete.json'
        if complete.exists():
            status=json.loads(complete.read_text())
            if status['input_fingerprint']!=dataset.fingerprint:raise RuntimeError('Point completion identity mismatch')
            for name,digest in status['artifact_hashes'].items():
                if sha(root/name)!=digest:raise RuntimeError('Completed point artifact changed: '+name)
            return status
        _DATA=dataset
        runtime=verify_highs_runtime()
        atomic_json(root/'solver_runtime.json',runtime)
        # Fixed first job of each loss supplies an outcome-blind CPU timing pilot.
        pilot_jobs=[next(j for j in dataset.jobs if j['fit_loss']==loss and j['role']=='group') for loss in ('mse','mae','tv')]
        pilot_started=time.perf_counter();pilot_records=[_point_worker(j) for j in pilot_jobs]
        atomic_json(root/'point_timing_pilot.json',{'jobs':[j['job_id'] for j in pilot_jobs],
            'wall_seconds':time.perf_counter()-pilot_started,'solve_seconds':{r['fit_loss']:r['solve_seconds'] for r in pilot_records},
            'outcome_blind':'first declared expanded11 group job of each fitting loss; pilot fits reused'})
        pending=[j for j in dataset.jobs if load_point(dataset.root,j,dataset.fingerprint) is None]
        workers=max(1,min(int(workers),4))
        done=len(dataset.jobs)-len(pending)
        if workers==1:
            for job in pending:
                _point_worker(job);done+=1
                if done%25==0:atomic_json(root/'point_progress.json',{'complete_jobs':done,'total_jobs':len(dataset.jobs),'updated_at':utc()})
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
                futures={pool.submit(_point_worker,job):job for job in pending}
                for future in as_completed(futures):
                    future.result();done+=1
                    if done%25==0:atomic_json(root/'point_progress.json',{'complete_jobs':done,'total_jobs':len(dataset.jobs),'updated_at':utc()})
        records=all_records(dataset);tables=make_tables(dataset,records)
        artifacts=[]
        for name,frame in tables.items():atomic_csv(root/(name+'.csv'),frame);artifacts.append(name+'.csv')
        drop=['job','arrays_path','arrays_sha256','completed_at']
        coefficients=pd.DataFrame([{k:v for k,v in r.items() if k not in drop} for r in records])
        atomic_parquet(root/'point_fit_coefficients.parquet',coefficients);artifacts.append('point_fit_coefficients.parquet')
        rows=export_losses(dataset,records,root/'per_question_losses.parquet');artifacts.append('per_question_losses.parquet')
        status={'status':'complete','input_fingerprint':dataset.fingerprint,'completed_at':utc(),'point_fit_count':len(records),
            'table_rows':{k:len(v) for k,v in tables.items()},'per_question_rows':rows,
            'wall_seconds':time.perf_counter()-started,'sum_fit_seconds':sum(r['solve_seconds'] for r in records),
            'workers':workers,'artifact_hashes':{name:sha(root/name) for name in artifacts}}
        atomic_json(root/'point_progress.json',{'complete_jobs':len(records),'total_jobs':len(dataset.jobs),'status':'complete','updated_at':utc()})
        atomic_json(complete,status);return status
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def bootstrap(repo_root,plan_root,output_root,cache_path=None,workers=4,resume=False,**kwargs):
    try:from .analysis_bootstrap import run_bootstrap
    except ImportError:from analysis_bootstrap import run_bootstrap
    return run_bootstrap(repo_root,plan_root,output_root,cache_path,workers,resume)
