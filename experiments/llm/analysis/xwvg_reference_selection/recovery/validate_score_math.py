#!/usr/bin/env python3
"""Strict all-pilot GPU comparison of candidate-only and historical log-softmax.

Reads baseline scores and metadata; writes separate validation evidence only.
The comparator never changes the scorer, its completion markers, or its inputs.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from score_endpoints import (atomic_json, canonical, file_sha256, import_historical_scorer,
                            read_jsonl, runtime_versions, scan_completed, slug, utc_now,
                            validate_design)


def normalize(value):
    if isinstance(value,np.ndarray):return [normalize(v) for v in value.tolist()]
    if isinstance(value,np.generic):return normalize(value.item())
    if isinstance(value,dict):
        result={k:normalize(v) for k,v in value.items()}
        # Nullable Parquet track columns load as float64 (0.0, 1.0, 2.0).
        # Restore this schema-declared integer metadata field only. Probability
        # and likelihood values are never rounded, cast, or tolerance-matched.
        track=result.get('track')
        if isinstance(track,float) and np.isfinite(track) and track.is_integer():
            result['track']=int(track)
        return result
    if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
    if isinstance(value,float) and np.isnan(value):return None
    return value


def compare_rows(left,right):
    left,right=normalize(left),normalize(right)
    details=[]
    if len(left)!=len(right):
        return {'exact':False,'errors':[{'kind':'row_count','left':len(left),'right':len(right)}]}
    for a,b in zip(left,right,strict=True):
        if canonical(a)!=canonical(b):
            for field in sorted(set(a)|set(b)):
                if canonical(a.get(field))!=canonical(b.get(field)):
                    detail={'prompt_id':a.get('prompt_id'),'field':field,'baseline':a.get(field),'candidate':b.get(field)}
                    try:
                        x,y=np.asarray(a[field],dtype=float),np.asarray(b[field],dtype=float)
                        if x.shape==y.shape:
                            detail['max_absolute_difference']=float(np.max(np.abs(x-y)))
                    except (TypeError,ValueError):pass
                    details.append(detail)
    import hashlib
    return {'exact':not details,'baseline_sha256':hashlib.sha256(canonical(left)).hexdigest(),
            'candidate_sha256':hashlib.sha256(canonical(right)).hexdigest(),'errors':details}


def timed(function,model,tokenizer,rows,rendered,device):
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    result=function(model,tokenizer,rows,rendered,device=device)
    torch.cuda.synchronize(device)
    seconds=time.perf_counter()-started
    return result,{'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                   'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                   'prompt_count':len(rows),'candidate_continuations':sum(r['option_count'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    repo,root,out=(p.resolve() for p in (args.source_repo,args.run_root,args.output_root))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use a fresh comparison output directory')
    original=import_historical_scorer(repo)
    spec=importlib.util.spec_from_file_location('xwvg_score_math',HERE/'score_math.py')
    optimized=importlib.util.module_from_spec(spec);spec.loader.exec_module(optimized)
    import torch
    torch.cuda.set_device(torch.device(args.device))
    records=json.loads((repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json').read_text())['models']
    manifest=root/'data/permutations/score_execution_manifest.jsonl'
    if not manifest.is_file():manifest=root/'data/permutations/score_semantic_manifest.jsonl'
    prompts=read_jsonl(manifest);prompt_by_id={r['prompt_id']:r for r in prompts}
    source_files=[Path(original.__file__),HERE/'score_endpoints.py',HERE/'score_math.py',Path(__file__)]
    baseline_files=[p for p in (root/'status').glob('fresh_score_*') if p.is_file()]
    baseline_files += [p for p in (root/'outputs/raw_scores').glob('*/shard_*') if p.is_file()]
    original_hashes={str(p):file_sha256(p) for p in baseline_files}
    summary={'started_at':utc_now(),'status':'running','models':[],
             'source_sha256':{str(p):file_sha256(p) for p in source_files},
             'baseline_artifact_sha256':original_hashes,'versions':runtime_versions(),
             'device':args.device,'gpu_name':torch.cuda.get_device_name(args.device),
             'acceptance_rule':'Exact complete-row canonical JSON equality on every saved pilot batch for every model. No numerical tolerance.',
             'storage_metadata_normalization':'Nullable Parquet track float values are restored to exact integer metadata; no score values are transformed.',
             'warmup_policy':'One first-pilot-batch call per implementation, excluded from measured comparison; subsequent backend order alternates by batch index.'}
    atomic_json(out/'summary.json',summary)
    failure=False
    for record in records:
        model_id=record['id'];name=slug(model_id)
        report={'model_id':model_id,'model_revision':record['revision'],'status':'running','batches':[]}
        model=None;tokenizer=None
        try:
            baseline_identity=json.loads((root/'status'/f'fresh_score_{name}_identity.json').read_text())
            pilot=json.loads((root/'status'/f'fresh_score_{name}_pilot.json').read_text())
            design=json.loads((root/'status'/f'fresh_score_{name}_pilot_design.json').read_text())
            identity=baseline_identity['input_fingerprint']
            if pilot['input_fingerprint']!=identity or design['input_fingerprint']!=identity:
                raise ValueError('Pilot provenance differs')
            if pilot['status']!='complete':raise ValueError('Baseline pilot incomplete')
            rendered_path=root/'data/permutations/rendered'/f'{name}.jsonl'
            if file_sha256(rendered_path)!=baseline_identity['rendered_prompt_sha256']:
                raise ValueError('Rendered prompt checksum differs from baseline identity')
            rendered=read_jsonl(rendered_path)
            validate_design(prompts,rendered)
            rendered_by_id={r['prompt_id']:r for r in rendered}
            completed,metas=scan_completed(root/'outputs/raw_scores'/name,identity,prompt_by_id,rendered_by_id,
                                           model_id,record['revision'],file_sha256(manifest))
            if not set(design['prompt_ids'])<=completed:raise ValueError('Baseline pilot score rows incomplete')
            batches=[b['prompt_ids'] for b in pilot['batches']]
            flat=[p for batch in batches for p in batch]
            if len(flat)!=len(set(flat)) or set(flat)!=set(design['prompt_ids']):
                raise ValueError('Saved pilot batches do not cover the fixed pilot exactly once')
            stored={}
            for path in (root/'outputs/raw_scores'/name).glob('shard_*.parquet'):
                for row in pd.read_parquet(path).to_dict('records'):
                    if row['prompt_id'] in flat:stored[row['prompt_id']]=row
            download=json.loads((root/'status/model_downloads'/f'{name}.json').read_text())
            record=dict(record);record['local_cache_path']=download['snapshot']
            record['model_adapter']=baseline_identity['model_adapter']
            print(json.dumps({'model_id':model_id,'status':'loading','pilot_rows':len(flat)}),flush=True)
            started=time.perf_counter();model,tokenizer=original._load_model(record,args.device)
            report['model_load_seconds']=time.perf_counter()-started
            first=[prompt_by_id[p] for p in batches[0]]
            _,warm_original=timed(original.score_prompt_batch,model,tokenizer,first,rendered_by_id,args.device)
            _,warm_candidate=timed(optimized.score_prompt_batch,model,tokenizer,first,rendered_by_id,args.device)
            report['warmup']={'baseline':warm_original,'candidate':warm_candidate}
            for index,ids in enumerate(batches):
                batch=[prompt_by_id[p] for p in ids]
                calls=[('baseline',original.score_prompt_batch),('candidate',optimized.score_prompt_batch)]
                if index%2:calls.reverse()
                scores={};timings={}
                for backend,function in calls:
                    scores[backend],timings[backend]=timed(function,model,tokenizer,batch,rendered_by_id,args.device)
                comparison=compare_rows(scores['baseline'],scores['candidate'])
                expected=[{key:stored[row['prompt_id']][key] for key in row} for row in scores['baseline']]
                cached=compare_rows(expected,scores['baseline'])
                result={'batch_index':index,'prompt_ids':ids,'backend_order':[b for b,_ in calls],
                        'timings':timings,'baseline_vs_candidate':comparison,'cached_vs_repeated_baseline':cached}
                report['batches'].append(result)
                atomic_json(out/f'{name}.json',report)
                if not comparison['exact'] or not cached['exact']:
                    failure=True
                    print(json.dumps({'model_id':model_id,'status':'EXACT_EQUALITY_FAILED','batch_index':index,
                                      'candidate_errors':comparison['errors'],'baseline_repeat_errors':cached['errors']}),flush=True)
            report['exact_candidate_equality']=all(b['baseline_vs_candidate']['exact'] for b in report['batches'])
            report['exact_baseline_repeat']=all(b['cached_vs_repeated_baseline']['exact'] for b in report['batches'])
            report['prompt_count']=len(flat)
            report['candidate_continuations']=sum(prompt_by_id[p]['option_count'] for p in flat)
            report['steady_baseline_seconds']=sum(b['timings']['baseline']['seconds'] for b in report['batches'])
            report['steady_candidate_seconds']=sum(b['timings']['candidate']['seconds'] for b in report['batches'])
            report['steady_speedup']=report['steady_baseline_seconds']/report['steady_candidate_seconds']
            report['peak_allocated_baseline_bytes']=max(b['timings']['baseline']['peak_allocated_bytes'] for b in report['batches'])
            report['peak_allocated_candidate_bytes']=max(b['timings']['candidate']['peak_allocated_bytes'] for b in report['batches'])
            report['status']='passed' if report['exact_candidate_equality'] and report['exact_baseline_repeat'] else 'failed_exact_equality'
        except Exception as exc:
            report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
            failure=True
            print(json.dumps({'model_id':model_id,'status':'ERROR','error':repr(exc)}),flush=True)
        finally:
            if model is not None:del model
            if tokenizer is not None:del tokenizer
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(args.device)
        atomic_json(out/f'{name}.json',report)
        summary['models'].append({k:v for k,v in report.items() if k not in ('batches','traceback')})
        atomic_json(out/'summary.json',summary)
        print(json.dumps({k:report.get(k) for k in ('model_id','status','prompt_count','steady_baseline_seconds','steady_candidate_seconds','steady_speedup')}),flush=True)
    changed=[path for path,h in original_hashes.items() if file_sha256(Path(path))!=h]
    summary.update(completed_at=utc_now(),baseline_artifacts_unchanged=not changed,changed_artifacts=changed,
                   status='failed' if failure or changed else 'passed',
                   strict_all_eight_models_passed=not failure and not changed and len(summary['models'])==8)
    atomic_json(out/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'strict_all_eight_models_passed':summary['strict_all_eight_models_passed']}),flush=True)
    return int(summary['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
