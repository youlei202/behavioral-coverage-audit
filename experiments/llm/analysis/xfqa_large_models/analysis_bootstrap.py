"""Prespecified true-refit and separately labelled frozen-fit uncertainty.

A replicate is one shared within-subject multiplicity vector across every
model, family and overlapping split. Every split normalizes its own sides.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import fcntl,json,multiprocessing as mp,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
try:
    from .analysis_data import *
    from .analysis import fit_job,all_records,make_tables,direct_partner_losses,job_paths,_base
except ImportError:
    from analysis_data import *
    from analysis import fit_job,all_records,make_tables,direct_partner_losses,job_paths,_base

_BDATA=None
_PRIMARY_JOBS=None
PRIMARY_SCOPE='common-question within-subject refit bootstrap; refits group and declared removal weights and reselects the single peer; ten overlapping splits averaged within replicate'
SECONDARY_SCOPE='descriptive frozen-fit common-question interval; excludes fitting/selection uncertainty; ten overlapping splits averaged within replicate'
TABLE_METRICS={
 'scalar_main_split_results':['e_single','e_group','gamma','relative_gamma_percent','mse_single','mse_group'],
 'scalar_removal_split_results':['e_full','e_without','removal_effect'],
 'scalar_directional_pairs':['partner_error','e_group','extra_beyond_partner','e_without','removal_effect','directional_extra_difference'],
 'scalar_roster_expansion':['comparator_error','expanded_error','large_peer_gain'],
 'vector_main_split_results':['e_single','e_group','gamma','relative_gamma_percent','single_top_option_agreement','group_top_option_agreement','agreement_gain'],
 'vector_directional_removals':['partner_error','e_group','extra_beyond_partner','e_without','removal_effect','directional_extra_difference'],
 'vector_roster_expansion':['comparator_error','expanded_error','large_peer_gain']}
GROUP_KEYS=['panel','target','family','fit_loss','partner','removed_peer','comparator_panel']


def contrast_groups(tables,primary=False,omissions=None):
    result=[]
    omissions=[] if omissions is None else omissions
    for table,frame in tables.items():
        if primary and table not in ('scalar_main_split_results','scalar_directional_pairs'):continue
        metrics=TABLE_METRICS[table]
        if primary and table=='scalar_main_split_results':metrics=metrics[:4]
        keys=[k for k in GROUP_KEYS if k in frame]
        for key,group in frame.groupby(keys,sort=True,dropna=False):
            if len(group)!=10 or group['split_seed'].nunique()!=10:raise RuntimeError('Interval group must contain the original ten splits')
            first=group.iloc[0]
            for metric in metrics:
                descriptor={'table':table,**{k:str(first[k]) if k in group else '' for k in GROUP_KEYS},'metric':metric}
                cid=json_hash(descriptor)
                if metric=='relative_gamma_percent' and not np.isfinite(group[metric].to_numpy(dtype=float)).all():
                    omissions.append({**descriptor,'contrast_id':cid,'stage':'point','reason':'relative percentage omitted because a selected-single denominator is numerically zero','undefined_split_seeds':group.loc[~np.isfinite(group[metric].to_numpy(dtype=float)),'split_seed'].astype(int).tolist()})
                    continue  # Absolute errors and gains remain included.
                point=float(group[metric].mean())
                if not np.isfinite(point):raise RuntimeError('Undefined contrast point; do not silently omit denominator failures')
                evaluation_metric=('mse' if metric in ('mse_single','mse_group') else
                    'top_option_agreement' if metric in ('single_top_option_agreement','group_top_option_agreement','agreement_gain') else first['evaluation_metric'])
                result.append({'contrast_id':cid,**descriptor,'point':point,
                    'run_id':first['run_id'],'response':first['response'],'evaluation_metric':evaluation_metric,
                    'peer_ids':first['peer_ids'],'input_fingerprint':first['input_fingerprint'],'completion_state':'complete',
                    '_rows':group.sort_values('split_seed').to_dict('records')})
    return result


def filter_defined_relative(descriptors,values,omissions):
    keep=[]
    for index,descriptor in enumerate(descriptors):
        missing=~np.isfinite(values[:,index])
        if missing.any():
            if descriptor['metric']!='relative_gamma_percent':raise RuntimeError('Nonfinite absolute bootstrap estimate')
            omissions.append({k:v for k,v in {**descriptor,'stage':'bootstrap',
                'reason':'relative percentage omitted because a selected-single denominator is numerically zero',
                'undefined_replicates':(np.flatnonzero(missing)+1).tolist()}.items() if k!='_rows'})
        else:keep.append(index)
    return [descriptors[i] for i in keep],values[:,keep]


def interval_frame(descriptors,values,method):
    if values.shape[1]!=len(descriptors) or not np.isfinite(values).all():raise RuntimeError('Invalid interval replicate matrix')
    out=[]
    for i,d in enumerate(descriptors):
        lo,hi=np.percentile(values[:,i],[2.5,97.5])
        out.append({k:v for k,v in {**d,'low':float(lo),'high':float(hi),'replicates':len(values),
            'method':method,'uncertainty_scope':PRIMARY_SCOPE if method=='primary_refit' else SECONDARY_SCOPE,
            'includes_refitting':method=='primary_refit','includes_single_reselection':method=='primary_refit'}.items() if k!='_rows'})
    return pd.DataFrame(out)


def _primary_tables(dataset,tables):
    targets=dataset.config['uncertainty']['primary_refit']['group_targets']
    main=tables['scalar_main_split_results']
    main=main[(main.panel=='expanded11')&(main.fit_loss=='mse')&main.target.isin(targets)].copy()
    pair=tables['scalar_directional_pairs']
    directions={tuple(x) for x in dataset.config['uncertainty']['primary_refit']['removal_directions']}
    pair=pair[(pair.fit_loss=='mse')&pd.Series([(a,b) in directions for a,b in zip(pair.target,pair.partner)],index=pair.index)].copy()
    return {'scalar_main_split_results':main,'scalar_directional_pairs':pair}


def _chunk_valid(path,dataset,start,end):
    meta=path/'meta.json'
    if not meta.exists():return False
    record=json.loads(meta.read_text())
    if record['input_fingerprint']!=dataset.fingerprint or record['replicate_start']!=start or record['replicate_end']!=end:
        raise RuntimeError('Primary chunk identity mismatch')
    for name,digest in record['artifact_hashes'].items():
        if not (path/name).exists() or sha(path/name)!=digest:raise RuntimeError('Primary bootstrap checkpoint corrupted: '+str(path/name))
    return True


def _primary_chunk(bounds):
    dataset=_BDATA;start,end=bounds
    path=dataset.root/'analysis/checkpoints/primary_refit'/f'chunk_{start:04d}_{end:04d}'
    if _chunk_valid(path,dataset,start,end):return str(path)
    began=time.perf_counter();fitrecords=[];splitrecords=[];draws=[]
    seed=dataset.config['uncertainty']['primary_refit']['seed']
    directions=dataset.config['uncertainty']['primary_refit']['removal_directions']
    for replicate in range(start,end+1):
        draw=multiplicities(dataset,replicate,seed);draws.append(draw)
        lookup={}
        for job in _PRIMARY_JOBS:
            fit,metrics,_arrays=fit_job(dataset,job,draw)
            record={**_base(dataset,job),'replicate':replicate,'role':job['role'],'removed_peer':job['removed_peer'],**fit,**metrics}
            fitrecords.append(record)
            lookup[(job['target'],job['family'],job['split_seed'],job['role'],job['removed_peer'])]=record
            if job['role']=='group':splitrecords.append({**record,'table':'scalar_main_split_results'})
        for target,partner in directions:
            for family in sorted(dataset.config['families']):
                partner_loss=direct_partner_losses(dataset,target,partner,family,'mse')@dataset.values.masses[family]
                for split_seed in dataset.config['split_seeds']:
                    full=lookup[(target,family,split_seed,'group',None)]
                    reverse=lookup[(partner,family,split_seed,'group',None)]
                    removed=lookup[(target,family,split_seed,'removal',partner)]
                    ids=dataset.eval_indices[split_seed];mass=draw[ids]
                    partner_error=float(np.average(partner_loss[ids],weights=mass))
                    splitrecords.append({**{k:full[k] for k in ('run_id','panel','target','family','split_seed','response','fit_loss',
                        'evaluation_metric','peer_ids','input_fingerprint','completion_state','replicate')},
                        'table':'scalar_directional_pairs','partner':partner,'partner_error':partner_error,
                        'e_group':full['e_group'],'extra_beyond_partner':partner_error-full['e_group'],
                        'e_without':removed['e_group'],'removal_effect':removed['e_group']-full['e_group'],
                        'directional_extra_difference':reverse['e_group']-full['e_group']})
    if len(fitrecords)!=(end-start+1)*100:raise RuntimeError('Primary refit count mismatch')
    path.mkdir(parents=True,exist_ok=True)
    atomic_parquet(path/'fits.parquet',pd.DataFrame(fitrecords))
    atomic_parquet(path/'split_metrics.parquet',pd.DataFrame(splitrecords))
    atomic_npz(path/'multiplicities.npz',replicates=np.arange(start,end+1),question_ids=np.asarray(dataset.values.question_ids),multiplicities=np.stack(draws))
    atomic_json(path/'meta.json',{'status':'complete','input_fingerprint':dataset.fingerprint,'replicate_start':start,'replicate_end':end,
        'fit_count':len(fitrecords),'split_metric_rows':len(splitrecords),'wall_seconds':time.perf_counter()-began,
        'artifact_hashes':{n:sha(path/n) for n in ('fits.parquet','split_metrics.parquet','multiplicities.npz')},'completed_at':utc()})
    return str(path)


def run_primary(dataset,tables,workers):
    global _BDATA,_PRIMARY_JOBS
    _BDATA=dataset
    directions={tuple(x) for x in dataset.config['uncertainty']['primary_refit']['removal_directions']}
    targets=set(dataset.config['uncertainty']['primary_refit']['group_targets'])
    _PRIMARY_JOBS=[j for j in dataset.jobs if j['panel']=='expanded11' and j['fit_loss']=='mse' and
        ((j['role']=='group' and j['target'] in targets) or (j['role']=='removal' and (j['target'],j['removed_peer']) in directions))]
    if len(_PRIMARY_JOBS)!=100:raise RuntimeError('Primary refit inventory must be 100 per replicate')
    cfg=dataset.config['uncertainty']['primary_refit'];count=cfg['replicates'];chunk=cfg['chunk_size']
    bounds=[(s,min(s+chunk-1,count)) for s in range(1,count+1,chunk)]
    root=dataset.root/'analysis'
    started=time.perf_counter();first=_primary_chunk(bounds[0]);firstmeta=json.loads((Path(first)/'meta.json').read_text())
    atomic_json(root/'primary_refit_timing_pilot.json',{'chunk':str(first),'replicates':chunk,'fit_count':firstmeta['fit_count'],
        'wall_seconds':firstmeta['wall_seconds'],'serial_projection_seconds':firstmeta['wall_seconds']*len(bounds),
        'reuse':'first prespecified 20-replicate chunk reused; no outcome-dependent sampling'})
    paths=[first]
    if workers==1:
        for b in bounds[1:]:paths.append(_primary_chunk(b))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            futures=[pool.submit(_primary_chunk,b) for b in bounds[1:]]
            for future in as_completed(futures):
                paths.append(future.result())
                atomic_json(root/'primary_refit_progress.json',{'complete_chunks':len(paths),'total_chunks':len(bounds),'updated_at':utc()})
    omissions=[]
    descriptors=contrast_groups(_primary_tables(dataset,tables),primary=True,omissions=omissions)
    values=np.empty((count,len(descriptors)));long=[]
    fits=0
    for path_text in sorted(paths):
        path=Path(path_text);meta=json.loads((path/'meta.json').read_text());fits+=meta['fit_count']
        frame=pd.read_parquet(path/'split_metrics.parquet')
        for replicate,part in frame.groupby('replicate',sort=True):
            for index,descriptor in enumerate(descriptors):
                group=part[part.table==descriptor['table']]
                for key in GROUP_KEYS:
                    if key in group and descriptor[key]!='':group=group[group[key].fillna('')==descriptor[key]]
                if len(group)!=10:raise RuntimeError('Primary replicate contrast lacks ten splits')
                metric_values=group[descriptor['metric']].to_numpy(dtype=float)
                value=float(np.mean(metric_values))
                if not np.isfinite(value) and descriptor['metric']!='relative_gamma_percent':raise RuntimeError('Undefined primary replicate absolute metric')
                values[int(replicate)-1,index]=value
                long.append({**{k:v for k,v in descriptor.items() if k not in ('_rows','point')},'replicate':int(replicate),'value':value})
    if fits!=100000:raise RuntimeError('Primary bootstrap must perform exactly 100000 nontrivial fits')
    descriptors,values=filter_defined_relative(descriptors,values,omissions)
    retained={d['contrast_id'] for d in descriptors}
    long=[r for r in long if r['contrast_id'] in retained]
    atomic_json(root/'primary_relative_omissions.json',{'omissions':omissions,'absolute_contrasts_retained':True})
    atomic_csv(root/'primary_refit_intervals.csv',interval_frame(descriptors,values,'primary_refit'))
    atomic_parquet(root/'primary_refit_replicates.parquet',pd.DataFrame(long))
    atomic_json(root/'primary_refit_complete.json',{'status':'complete','input_fingerprint':dataset.fingerprint,'replicates':count,
        'fits':fits,'contrast_count':len(descriptors),'chunks':len(paths),'wall_seconds':time.perf_counter()-started,
        'artifact_hashes':{n:sha(root/n) for n in ['primary_refit_intervals.csv','primary_refit_replicates.parquet','primary_relative_omissions.json']+
            [str((Path(p)/name).relative_to(root)) for p in sorted(paths) for name in ('fits.parquet','split_metrics.parquet','multiplicities.npz','meta.json')]},'completed_at':utc()})
    return {'replicates':count,'fits':fits,'contrast_count':len(descriptors)}


class FrozenSampler:
    def __init__(self,dataset,records,mults):
        self.dataset=dataset;self.records={r['job_id']:r for r in records};self.mults=mults
        self.cache={};self.partner_cache={}
        self.lookup={(r['panel'],r['target'],r['family'],r['split_seed'],r['fit_loss'],r['role'],r['removed_peer']):r for r in records}
        self.weights={s:mults[:,ids]/mults[:,ids].sum(axis=1,keepdims=True) for s,ids in dataset.eval_indices.items()}

    def metric(self,job_id,name):
        key=(job_id,name)
        if key in self.cache:return self.cache[key]
        r=self.records[job_id]
        col={'e_group':'total_variation' if r['fit_loss']=='tv' else 'absolute_error',
             'e_single':'single_total_variation' if r['fit_loss']=='tv' else 'single_absolute_error',
             'mse_group':'squared_error','mse_single':'single_squared_error',
             'group_top_option_agreement':'top_option_agreement','single_top_option_agreement':'single_top_option_agreement'}[name]
        _,path=job_paths(self.dataset.root,job_id)
        with np.load(path,allow_pickle=False) as a:loss=a[col]@self.dataset.values.masses[r['family']]
        sampled=self.weights[r['split_seed']]@loss
        self.cache[key]=sampled;return sampled

    def partner(self,row):
        pair=tuple(sorted((row['target'],row['partner'])))
        key=(*pair,row['family'],row['split_seed'],row['fit_loss']=='tv')
        if key not in self.partner_cache:
            loss=direct_partner_losses(self.dataset,*pair,row['family'],row['fit_loss'])@self.dataset.values.masses[row['family']]
            self.partner_cache[key]=self.weights[row['split_seed']]@loss[self.dataset.eval_indices[row['split_seed']]]
        return self.partner_cache[key]

    def row_value(self,table,row,metric):
        get=lambda jid,name='e_group':self.metric(jid,name)
        if table in ('scalar_main_split_results','vector_main_split_results'):
            if metric=='gamma':return get(row['job_id'],'e_single')-get(row['job_id'])
            if metric=='relative_gamma_percent':
                single=get(row['job_id'],'e_single')
                return np.divide(100*(single-get(row['job_id'])),single,out=np.full_like(single,np.nan),where=single>1e-14)
            if metric=='agreement_gain':return get(row['job_id'],'group_top_option_agreement')-get(row['job_id'],'single_top_option_agreement')
            return get(row['job_id'],metric)
        if table=='scalar_removal_split_results':
            if metric=='e_full':return get(row['full_job_id'])
            if metric=='e_without':return get(row['removed_job_id'])
            return get(row['removed_job_id'])-get(row['full_job_id'])
        if table in ('scalar_directional_pairs','vector_directional_removals'):
            full=get(row['full_job_id'])
            if metric=='e_group':return full
            if metric=='e_without':return get(row['removed_job_id'])
            if metric=='partner_error':return self.partner(row)
            if metric=='extra_beyond_partner':return self.partner(row)-full
            if metric=='removal_effect':return get(row['removed_job_id'])-full
            reverse=self.lookup[('expanded11',row['partner'],row['family'],row['split_seed'],row['fit_loss'],'group',None)]
            return get(reverse['job_id'])-full
        if table in ('scalar_roster_expansion','vector_roster_expansion'):
            if metric=='comparator_error':return get(row['comparator_job_id'])
            if metric=='expanded_error':return get(row['expanded_job_id'])
            return get(row['comparator_job_id'])-get(row['expanded_job_id'])
        raise ValueError(table)


def run_secondary(dataset,records,tables):
    started=time.perf_counter();root=dataset.root/'analysis'
    cfg=dataset.config['uncertainty']['secondary_frozen_fit'];count=cfg['replicates']
    draws=np.stack([multiplicities(dataset,r,cfg['seed']) for r in range(1,count+1)])
    sampler=FrozenSampler(dataset,records,draws)
    omissions=[]
    descriptors=contrast_groups(tables,omissions=omissions)
    values=np.empty((count,len(descriptors)),dtype=np.float64)
    chunks=root/'checkpoints/secondary_frozen_fit';chunks.mkdir(parents=True,exist_ok=True)
    # Save each fixed contrast block across all shared draws. This retains
    # completed descriptive work without repeatedly loading every point fit.
    for start in range(0,len(descriptors),50):
        stop=min(start+50,len(descriptors));block=descriptors[start:stop]
        path=chunks/f'contrasts_{start:04d}_{stop-1:04d}.npz';meta=path.with_suffix('.json')
        ids=[d['contrast_id'] for d in block]
        if meta.exists():
            record=json.loads(meta.read_text())
            if record['input_fingerprint']!=dataset.fingerprint or record['contrast_ids']!=ids or record['sha256']!=sha(path):
                raise RuntimeError('Frozen-fit chunk identity/checksum mismatch')
            with np.load(path,allow_pickle=False) as existing:
                if existing['values'].shape!=(count,stop-start):raise RuntimeError('Frozen-fit chunk shape mismatch')
                values[:,start:stop]=existing['values']
        else:
            for index,d in enumerate(block,start):
                values[:,index]=np.mean([sampler.row_value(d['table'],row,d['metric']) for row in d['_rows']],axis=0)
            atomic_npz(path,contrast_ids=np.asarray(ids),replicates=np.arange(1,count+1),values=values[:,start:stop])
            atomic_json(meta,{'input_fingerprint':dataset.fingerprint,'contrast_ids':ids,'replicates':count,
                'seed':cfg['seed'],'sha256':sha(path),'completed_at':utc()})
        atomic_json(root/'secondary_progress.json',{'complete_contrasts':stop,'total_contrasts':len(descriptors),'updated_at':utc()})
    descriptors,values=filter_defined_relative(descriptors,values,omissions)
    atomic_json(root/'secondary_relative_omissions.json',{'omissions':omissions,'absolute_contrasts_retained':True})
    atomic_csv(root/'secondary_descriptive_intervals.csv',interval_frame(descriptors,values,'secondary_frozen_fit'))
    atomic_npz(root/'secondary_frozen_fit_replicates.npz',replicates=np.arange(1,count+1),
        contrast_ids=np.asarray([d['contrast_id'] for d in descriptors]),values=values,
        question_ids=np.asarray(dataset.values.question_ids),multiplicities=draws)
    atomic_json(root/'secondary_complete.json',{'status':'complete','input_fingerprint':dataset.fingerprint,'replicates':count,
        'contrast_count':len(descriptors),'wall_seconds':time.perf_counter()-started,
        'artifact_hashes':{n:sha(root/n) for n in ['secondary_descriptive_intervals.csv','secondary_frozen_fit_replicates.npz','secondary_relative_omissions.json']+
            [str(p.relative_to(root)) for p in sorted(chunks.iterdir()) if p.suffix in ('.json','.npz')]},'completed_at':utc()})
    return {'replicates':count,'contrast_count':len(descriptors)}


def _completed(root,name,fingerprint):
    p=root/name
    if not p.exists():return False
    state=json.loads(p.read_text())
    if state['input_fingerprint']!=fingerprint:raise RuntimeError('Bootstrap completion identity mismatch')
    for name,digest in state['artifact_hashes'].items():
        if sha(root/name)!=digest:raise RuntimeError('Completed bootstrap artifact changed: '+name)
    return state


def run_bootstrap(repo_root,plan_root,output_root,cache_path=None,workers=4,resume=False):
    dataset=load_dataset(repo_root,plan_root,output_root,cache_path);root=dataset.root/'analysis'
    # Bootstrap is always a continuation of the frozen point-analysis identity.
    register_identity(dataset,resume=True)
    if not _completed(root,'point_complete.json',dataset.fingerprint):raise RuntimeError('Complete point analysis required')
    lock=(root/'.bootstrap.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another bootstrap writer owns this run')
    try:
        records=all_records(dataset);tables=make_tables(dataset,records)
        primary=_completed(root,'primary_refit_complete.json',dataset.fingerprint)
        if not primary:primary=run_primary(dataset,tables,max(1,min(int(workers),4)))
        secondary=_completed(root,'secondary_complete.json',dataset.fingerprint)
        if not secondary:secondary=run_secondary(dataset,records,tables)
        result={'status':'complete','input_fingerprint':dataset.fingerprint,'primary':primary,'secondary':secondary,'completed_at':utc()}
        atomic_json(root/'bootstrap_complete.json',result);return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
