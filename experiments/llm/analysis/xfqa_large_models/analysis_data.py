"""XfqA data/identity adapters; imports the unchanged audited XWVG loader."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib,importlib.util,inspect,json,sys,tempfile
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[4]
for _relative in ['experiments/llm/analysis/xwvg_reference_selection/src','experiments/llm/analysis/v23/src']:
    _path=str(REPO/_relative)
    if _path not in sys.path:sys.path.insert(0,_path)
from bca_reference_selection.data import load_inputs
from bca_reference_selection.splits import fixed_splits,split_manifest
from pier_llm_v23.core import common_bootstrap_multiplicities
from bca_reference_selection.fit import fit_tv_simplex


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def json_text(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def json_hash(value):return hashlib.sha256(json_text(value).encode()).hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_parquet(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    frame.to_parquet(tmp,index=False)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    frame.to_csv(tmp,index=False,float_format='%.17g')
    os.replace(tmp,path)


def import_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def revisions(lock):
    entries=lock.get('models',lock.get('new_models',lock.get('resolved_models',[])))
    if isinstance(entries,dict):
        return {k:(v if isinstance(v,str) else v.get('revision',v.get('commit',v.get('sha')))) for k,v in entries.items()}
    return {v.get('id',v.get('model_id')):v.get('revision',v.get('commit',v.get('sha'))) for v in entries}


@dataclass
class Dataset:
    config:dict
    values:object
    scalar:dict
    splits:dict
    fit_indices:dict
    eval_indices:dict
    jobs:list
    identity:dict
    fingerprint:str
    root:Path

    @property
    def model_ids(self):return self.values.model_ids


def load_dataset(repo_root,plan_root,output_root,cache_path=None,*,allow_baseline=False):
    repo,plan,root=(Path(x).absolute() for x in (repo_root,plan_root,output_root))
    config=json.loads((plan/'config/experiment.json').read_text())
    original=json.loads((plan/'config/original_xwvg_input_config.json').read_text())
    questions=root/'inputs/mmlu_pro_selected_560.jsonl'
    if not questions.exists():questions=plan/'seed_inputs/mmlu_pro_selected_560.jsonl'
    if cache_path is None:
        candidates=[root/'data/permutation_averaged_semantic_scores.parquet',root/'inputs/merged_averaged_probabilities.parquet',root/'inputs/merged_averaged_semantic_scores.parquet',root/'scores/merged_averaged_semantic_scores.parquet']
        matches=[p for p in candidates if p.exists()]
        if len(matches)!=1:raise ValueError('Pass cache_path explicitly: a unique complete merged cache was not found')
        cache=matches[0]
    else:cache=Path(cache_path).absolute()
    if not cache.exists():raise FileNotFoundError(cache)
    lockpath=root/'model_lock.json'
    if not lockpath.exists():
        lockpath=root/'run/model_lock.json'
    lock=json.loads(lockpath.read_text()) if lockpath.exists() else {}
    resolved=revisions(lock)
    roster=config['original_roster'] if allow_baseline else config['expanded_roster']
    if not allow_baseline:
        for model in config['new_models']:
            revision=resolved.get(model['id'])
            if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower()):
                raise ValueError('Missing frozen new-model commit: '+model['id'])
    derived=dict(original)
    derived.update(model_roster=roster,model_revisions={**config['original_model_revisions'],**resolved},
        expected_primary_cache_rows=len(roster)*560*7,primary_protocol='xfqa_balanced_full_vector_endpoints')
    derived['families']=config['families']
    values=load_inputs(cache,questions,derived)
    gold=values.questions['answer_index'].to_numpy(dtype=int)
    scalar={f:np.take_along_axis(p,np.broadcast_to(gold[None,:,None,None],p.shape[:3]+(1,)),axis=3)[...,0]
            for f,p in values.probabilities.items()}
    splits=fixed_splits(values.questions,config['split_seeds'])
    frozen_splits=root/'inputs/splits.json'
    if frozen_splits.exists():
        supplied=json.loads(frozen_splits.read_text())
        supplied={int(r['seed']):r for r in supplied} if isinstance(supplied,list) else {int(k):v for k,v in supplied.items()}
        if set(supplied)!=set(splits):raise RuntimeError('Preflight split inventory mismatch')
        for seed,split in splits.items():
            if set(supplied[seed]['fitting_ids'])!=set(split.fitting_ids) or set(supplied[seed]['evaluation_ids'])!=set(split.evaluation_ids):
                raise RuntimeError('Original split reconstruction differs from checked plan helper')
    lookup={q:i for i,q in enumerate(values.question_ids)}
    fit={s:np.array(sorted(lookup[q] for q in split.fitting_ids)) for s,split in splits.items()}
    evaluate={s:np.array(sorted(lookup[q] for q in split.evaluation_ids)) for s,split in splits.items()}
    builder=import_file(plan/'tools/build_analysis_jobs.py','xfqa_job_inventory')
    jobs=builder.build_jobs(config)
    if allow_baseline:jobs=[j for j in jobs if j['panel']=='baseline8']
    reused_functions=[load_inputs,fixed_splits,fit_tv_simplex,common_bootstrap_multiplicities]
    source_files=list(Path(__file__).parent.glob('analysis*.py'))+[Path(inspect.getfile(f)) for f in reused_functions]+[plan/'tools/build_analysis_jobs.py']
    import scipy,pyarrow
    identity={'schema':'xfqa_analysis_v1','cache_sha256':sha(cache),'questions_sha256':sha(questions),
        'config_sha256':sha(plan/'config/experiment.json'),'original_input_config_sha256':sha(plan/'config/original_xwvg_input_config.json'),
        'model_lock_sha256':sha(lockpath) if lockpath.exists() else None,'source_hashes':{str(p):sha(p) for p in sorted(source_files)},
        'imported_module_paths':{f.__module__:str(Path(inspect.getfile(f)).absolute()) for f in reused_functions},
        'model_order':list(values.model_ids),'question_order':list(values.question_ids),'split_ids':split_manifest(values.questions,config['split_seeds']),
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__},
        'synthetic_fixture':False,'scope':'baseline8_only' if allow_baseline else 'complete_expanded_experiment'}
    fp=json_hash(identity)
    return Dataset(config,values,scalar,splits,fit,evaluate,jobs,
        {'identity':identity,'cache_path':str(cache),'questions_path':str(questions),'config_path':str(plan/'config/experiment.json'),
         'model_lock_path':str(lockpath),'input_fingerprint':fp,'completion_state':'inputs_validated'},fp,root)


def multiplicities(dataset,replicate,seed):
    result=common_bootstrap_multiplicities(dataset.values.questions,int(replicate),int(seed))
    out=np.array([result[q] for q in dataset.values.question_ids],dtype=np.int16)
    if out.sum()!=560 or np.any(out<0):raise RuntimeError('Invalid question draw')
    for ids in (dataset.fit_indices,dataset.eval_indices):
        for side in ids.values():
            if out[side].sum()==0:raise RuntimeError('Empty bootstrap split side; replicate rejected without replacing draw')
    return out


def register_identity(dataset,resume=False):
    root=dataset.root/'analysis';root.mkdir(parents=True,exist_ok=True)
    p=root/'analysis_manifest.json'
    if p.exists():
        existing=json.loads(p.read_text())
        if existing['input_fingerprint']!=dataset.fingerprint:
            raise RuntimeError('Analysis identity changed; use a fresh output directory')
        if not resume:raise FileExistsError('Analysis exists; use --resume with identical source/input identity')
    else:
        atomic_json(p,{**dataset.identity,'created_at':utc()})
        atomic_json(root/'split_ids.json',dataset.identity['identity']['split_ids'])
        atomic_json(root/'derived_input_config.json',{'model_order':list(dataset.model_ids),'configured_revisions':dataset.values.report['configured_model_revisions'],'loader_report':dataset.values.report})
        with (root/'analysis_jobs.jsonl').open('x') as f:
            for job in dataset.jobs:f.write(json_text(job)+'\n')
    return root
