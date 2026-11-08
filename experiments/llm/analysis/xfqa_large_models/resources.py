"""Finalize measured resource accounting from completed immutable proofs.

Forward timers from concurrently running models are aggregate measured work,
not campaign elapsed time, exclusive GPU-hours, or hardware utilization. Pilot
scores reused by production are counted once; repeatability forwards are extra.
"""
from __future__ import annotations
import csv,fcntl,hashlib,io,json,math,os
from datetime import datetime,timezone
from pathlib import Path

EXPECTED={'models':3,'prompts_per_model':36946,'candidate_sequences_per_model':357196,
          'point_fits':5840,'primary_replicates':1000,'primary_fits':100000,'secondary_replicates':2000}
SCHEMA='xfqa_measured_resources_v1'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _publish(path,payload,*,replace_initial=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace_initial:
        if path.read_bytes()!=payload:raise ValueError('Refusing to overwrite different completed resource artifact: '+str(path))
        return
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid measured resource quantity: '+name)
    return value


def _close(left,right,name):
    if not math.isclose(float(left),float(right),rel_tol=1e-12,abs_tol=1e-7):
        raise ValueError('Resource timer differs from immutable batch evidence: '+name)


class Proofs:
    def __init__(self,root):self.root=root;self.hashes={}
    def path(self,value):
        p=Path(value)
        if not p.is_absolute():p=self.root/p
        if p.is_symlink() or not p.resolve().is_relative_to(self.root):
            raise ValueError('Resource proof escapes the current run: '+str(p))
        if not p.is_file():raise FileNotFoundError('Required completed resource proof missing: '+str(p))
        return p
    def add(self,path,expected=None):
        p=self.path(path);d=digest(p)
        if expected is not None and d!=expected:raise ValueError('Resource proof checksum mismatch: '+str(p))
        self.hashes[str(p.relative_to(self.root))]=d
        return p
    def read(self,path,status=None):
        p=self.add(path);record=json.loads(p.read_text())
        if status is not None and record.get('status')!=status:
            raise ValueError('Resource accounting requires completed stage: '+str(p))
        return record


def _model_summary(proofs,locked):
    root=proofs.root;model=locked['id'];slug=model.replace('/','--')
    prefix=f'status/model_scoring/{slug}'
    complete=proofs.read(prefix+'_complete.json','complete')
    pilot=proofs.read(prefix+'_pilot.json','complete')
    identity=proofs.read(prefix+'_identity.json')
    fp=complete['input_fingerprint']
    if complete.get('phase')!='score' or pilot.get('phase')!='pilot':raise ValueError('Wrong model execution stage')
    if any(r.get('model_id')!=model or r.get('model_revision')!=locked['revision'] or r.get('input_fingerprint')!=fp for r in (complete,pilot,identity)):
        raise ValueError('Model resource proof identities differ')
    if complete.get('measurement_origin')!='fresh_endpoint_scoring' or complete.get('frozen_original_score_function') is not True:
        raise ValueError('Resource accounting requires actual prescribed scoring')
    if (complete['completed_rows']!=EXPECTED['prompts_per_model'] or complete['candidate_continuations']!=EXPECTED['candidate_sequences_per_model']):
        raise ValueError('Model scoring inventory is incomplete')
    if complete.get('pilot_report_sha256')!=proofs.hashes[prefix+'_pilot.json']:
        raise ValueError('Completed model does not bind its pilot proof')
    if pilot.get('repeatability',{}).get('passed') is not True or pilot.get('worst_case_probe_passed') is not True:
        raise ValueError('Pilot and maximum-workload checks must pass')
    if any(r['hardware']['gpu_uuid']!=complete['hardware']['gpu_uuid'] for r in (pilot,identity)):
        raise ValueError('Physical model device differs between pilot and score')
    directory=root/'outputs/raw_scores'/slug
    batches=[];pilot_batches=[];score_batches=[];seen=set();shards=0;oom_events=0
    metadata_paths=sorted(directory.glob('shard_*.meta.json'))
    if not metadata_paths:raise ValueError('No committed scoring metadata for '+model)
    for meta_path in metadata_paths:
        meta=proofs.read(meta_path)
        stem=meta_path.name.removesuffix('.meta.json')
        marker=proofs.read(directory/(stem+'.complete.json'))
        if marker.get('complete') is not True or marker.get('input_fingerprint')!=fp or meta.get('input_fingerprint')!=fp:
            raise ValueError('Uncommitted or differently identified score shard')
        if meta['model_id']!=model or meta['model_revision']!=locked['revision'] or marker['sha256']!=meta['sha256'] or marker['row_count']!=meta['row_count']:
            raise ValueError('Scoring shard resource proof mismatch')
        proofs.add(directory/(stem+'.parquet'),meta['sha256'])
        ids=meta['prompt_ids']
        if len(ids)!=meta['row_count'] or len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or inconsistent committed prompt inventory')
        observed=[pid for batch in meta['execution']['batches'] for pid in batch['prompt_ids']]
        if observed!=ids:raise ValueError('Batch timers do not identify exactly the committed shard prompts')
        seen.update(ids);shards+=1
        stage=meta['execution']['stage']
        if stage not in ('pilot','score'):raise ValueError('Unrecognized committed scoring stage')
        shard_hardware=meta['execution'].get('hardware')
        if shard_hardware is not None and shard_hardware['gpu_uuid']!=complete['hardware']['gpu_uuid']:
            raise ValueError('Committed scoring batch used a different physical GPU')
        current=meta['execution']['batches']
        for batch in current:
            _number(batch['seconds'],'batch.seconds')
            _number(batch['candidate_continuations'],'batch.candidate_continuations')
            if len(batch['prompt_ids'])!=batch['batch_size']:raise ValueError('Timing batch size differs from prompt inventory')
        batches.extend(current)
        (pilot_batches if stage=='pilot' else score_batches).extend(current)
        oom_events+=len(meta['execution'].get('oom_events',[]))
    if len(seen)!=EXPECTED['prompts_per_model'] or shards!=complete['shards']:
        raise ValueError('Resource shard inventory differs from completed model')
    sequences=sum(b['candidate_continuations'] for b in batches)
    if sequences!=EXPECTED['candidate_sequences_per_model']:raise ValueError('Candidate timer inventory incomplete')
    retained=math.fsum(b['seconds'] for b in batches)
    pilot_seconds=math.fsum(b['seconds'] for b in pilot_batches)
    score_seconds=math.fsum(b['seconds'] for b in score_batches)
    _close(retained,complete['committed_inference_seconds'],'all retained forwards')
    _close(pilot_seconds,pilot['committed_inference_seconds'],'retained pilot forwards')
    if sum(len(b['prompt_ids']) for b in pilot_batches)!=pilot['completed_rows']:
        raise ValueError('Pilot rows must occur exactly once among retained forwards')
    repeat=pilot['repeat_artifact'];proofs.add(repeat['path'],repeat['sha256'])
    repeat_batches=repeat['batches']
    repeat_ids=[pid for b in repeat_batches for pid in b['prompt_ids']]
    retained_pilot_ids={pid for batch in pilot_batches for pid in batch['prompt_ids']}
    if len(repeat_ids)!=repeat['rows'] or len(set(repeat_ids))!=len(repeat_ids) or set(repeat_ids)!=retained_pilot_ids:
        raise ValueError('Invalid extra repeatability-forward inventory')
    repeat_seconds=math.fsum(_number(b['seconds'],'repeat.seconds') for b in repeat_batches)
    all_telemetry=batches+repeat_batches
    peak_alloc=max(_number(b['peak_allocated_bytes'],'peak_allocated_bytes') for b in all_telemetry)
    peak_reserved=max(_number(b['peak_reserved_bytes'],'peak_reserved_bytes') for b in all_telemetry)
    observed_gpu_free=[_number(b['resources_after']['gpu_free_bytes'],'gpu_free_bytes') for b in all_telemetry]
    observed_cpu_free=[_number(b['resources_after']['cpu_available_bytes'],'cpu_available_bytes') for b in all_telemetry]
    observed_rss=[_number(b['resources_after']['process_rss_bytes'],'process_rss_bytes') for b in all_telemetry]
    minimum_gpu=min(observed_gpu_free+[_number(r['minimum_gpu_free_bytes'],'minimum_gpu_free_bytes') for r in (pilot,complete)])
    minimum_cpu=min(observed_cpu_free+[_number(r['minimum_cpu_available_bytes'],'minimum_cpu_available_bytes') for r in (pilot,complete)])
    gpu_reserve=_number(identity['gpu_reserve_bytes'],'gpu_reserve_bytes')
    cpu_reserve=_number(identity['cpu_reserve_bytes'],'cpu_reserve_bytes')
    if any(r['gpu_reserve_bytes']!=gpu_reserve or r['cpu_reserve_bytes']!=cpu_reserve for r in (pilot,complete)):
        raise ValueError('Resource reserve differs across the frozen model identity')
    if minimum_gpu<gpu_reserve or minimum_cpu<cpu_reserve:raise ValueError('Completed execution violated its declared memory reserve')
    last_failure=root/(prefix+'_last_failure.json')
    failure=None
    if last_failure.exists():
        failure_proof=proofs.read(last_failure)
        failure={k:failure_proof.get(k) for k in ('phase','error_type','status')}
    return {'model_id':model,'model_revision':locked['revision'],'input_fingerprint':fp,
        'hardware':complete['hardware'],'actual_model_class':complete['actual_model_class'],
        'completed_prompts':len(seen),'candidate_continuations':sequences,'committed_shards':shards,
        'retained_score_forward_seconds':retained,'retained_pilot_forward_seconds':pilot_seconds,
        'retained_formal_phase_forward_seconds':score_seconds,
        'extra_repeatability_forward_seconds':repeat_seconds,'extra_repeatability_prompts':len(repeat_ids),
        'pilot_invocation_wall_seconds':_number(pilot['wall_seconds'],'pilot wall_seconds'),
        'final_score_invocation_wall_seconds':_number(complete['wall_seconds'],'score wall_seconds'),
        'pilot_load_seconds':_number(pilot['load_seconds'],'pilot load_seconds'),
        'final_score_load_seconds':_number(complete['load_seconds'],'score load_seconds'),
        'peak_torch_allocated_bytes':peak_alloc,'peak_torch_reserved_bytes':peak_reserved,
        'minimum_gpu_free_bytes':minimum_gpu,'gpu_reserve_bytes':gpu_reserve,'minimum_gpu_reserve_margin_bytes':minimum_gpu-gpu_reserve,
        'minimum_cpu_available_bytes_during_scoring':minimum_cpu,'cpu_reserve_bytes':cpu_reserve,
        'minimum_cpu_reserve_margin_bytes_during_scoring':minimum_cpu-cpu_reserve,
        'maximum_observed_process_rss_bytes_after_forwards':max(observed_rss),
        'committed_shard_reported_oom_events':oom_events,'last_recorded_failed_attempt':failure,
        'timing_scope':'Retained pilot and formal shard forwards counted once. Repeatability forwards are extra. Stage/load timers describe the successful pilot and final completing score invocation, not all interrupted/failed attempts.',
        'memory_scope':'Post-load Torch allocation/reservation peaks and sampled after-forward RSS; minimum free GPU/available CPU includes successful pilot and final score watchdog observations. Historical unrecorded load peaks and interrupted-attempt watchdog minima are not inferred.'}


def _csv_bytes(models,cpu):
    fields=['kind','name','retained_score_forward_seconds','extra_repeatability_forward_seconds',
        'pilot_invocation_wall_seconds','final_score_invocation_wall_seconds','analysis_phase_wall_seconds',
        'parallel_fit_routine_wall_seconds_sum','peak_torch_allocated_bytes','peak_torch_reserved_bytes',
        'minimum_gpu_free_bytes','gpu_reserve_bytes','minimum_cpu_available_bytes_during_scoring','cpu_reserve_bytes','notes']
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    for m in models:
        row={k:m[k] for k in fields if k in m}
        row.update(kind='model',name=m['model_id'],notes=m['timing_scope']);writer.writerow(row)
    for name,record in cpu['stages'].items():
        writer.writerow({'kind':'cpu_analysis','name':name,'analysis_phase_wall_seconds':record['wall_seconds'],
            'parallel_fit_routine_wall_seconds_sum':record.get('parallel_fit_routine_wall_seconds_sum',''),
            'notes':'Elapsed stage wall time; fitting-routine sums overlap across workers and are not CPU usage hours.'})
    return out.getvalue().encode()


def finalize_resources(output_root):
    root=Path(output_root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.resource_summary.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another resource-summary publisher owns this run')
    try:
        proofs=Proofs(root);config=proofs.read('plan/config/experiment.json');model_lock=proofs.read('model_lock.json','complete')
        if len(model_lock['models'])!=EXPECTED['models'] or [m['id'] for m in model_lock['models']]!=[m['id'] for m in config['new_models']]:
            raise ValueError('Resource summary requires all three declared locked models')
        models=[_model_summary(proofs,model) for model in model_lock['models']]
        stages={name:proofs.read('analysis/'+filename,'complete') for name,filename in (
            ('point','point_complete.json'),('primary_refit','primary_refit_complete.json'),('secondary_frozen_fit','secondary_complete.json'))}
        bootstrap=proofs.read('analysis/bootstrap_complete.json','complete')
        manifest=proofs.read('analysis/analysis_manifest.json')
        fp=stages['point']['input_fingerprint']
        if (any(s['input_fingerprint']!=fp for s in [*stages.values(),bootstrap,manifest])
                or manifest['identity'].get('synthetic_fixture') is not False
                or manifest['identity'].get('scope')!='complete_expanded_experiment'):
            raise ValueError('Resource summary requires one complete empirical analysis identity')
        if (stages['point']['point_fit_count']!=EXPECTED['point_fits'] or
            stages['primary_refit']['replicates']!=EXPECTED['primary_replicates'] or
            stages['primary_refit']['fits']!=EXPECTED['primary_fits'] or
            stages['secondary_frozen_fit']['replicates']!=EXPECTED['secondary_replicates']):
            raise ValueError('Required analysis work is incomplete')
        cpu_stages={name:{'wall_seconds':_number(record['wall_seconds'],name+' wall_seconds')} for name,record in stages.items()}
        cpu_stages['point']['parallel_fit_routine_wall_seconds_sum']=_number(stages['point']['sum_fit_seconds'],'point summed fit timers')
        cpu_stages['point']['workers']=stages['point']['workers']
        cpu={'stages':cpu_stages,'analysis_phase_wall_seconds_sum':math.fsum(s['wall_seconds'] for s in cpu_stages.values()),
            'memory_peak_bytes':None,'memory_peak_status':'not_recorded','memory_peak_scope':'CPU analysis RSS peaks are not present in the frozen analysis-stage proofs; no peak is inferred from initial available memory.',
            'timing_scope':'Sum of recorded point, primary-refit and secondary-resampling phase elapsed times, excluding scoring/download/report/replay/package. Point fit timer sum is concurrent routine wall time, not process CPU time.'}
        summary_path=root/'resource_summary.json'
        existing=json.loads(summary_path.read_text()) if summary_path.exists() else None
        initial_path=root/'runtime/initial_resource_observations.json'
        if existing and existing.get('status')=='complete':
            if existing.get('schema')!=SCHEMA:raise ValueError('Refusing to replace an unknown completed resource summary')
            initial_bytes=initial_path.read_bytes();initial=json.loads(initial_bytes)
        else:
            initial_bytes=summary_path.read_bytes() if summary_path.exists() else json_bytes({'status':'not_recorded','note':'No initial resource observation file existed.'})
            initial=json.loads(initial_bytes)
            if initial_path.exists() and initial_path.read_bytes()!=initial_bytes:raise ValueError('Initial resource observation snapshot differs')
        proofs.hashes['runtime/initial_resource_observations.json']=hashlib.sha256(initial_bytes).hexdigest()
        source_sha=digest(Path(__file__))
        work={'retained_score_forward_seconds_sum':math.fsum(m['retained_score_forward_seconds'] for m in models),
              'extra_repeatability_forward_seconds_sum':math.fsum(m['extra_repeatability_forward_seconds'] for m in models),
              'successful_terminal_invocation_wall_seconds_sum':math.fsum(m['pilot_invocation_wall_seconds']+m['final_score_invocation_wall_seconds'] for m in models),
              'physical_gpu_count':len({m['hardware']['gpu_uuid'] for m in models}),
              'interpretation':'These per-model timers can overlap across GPUs. Their sums measure recorded work, not campaign elapsed wall time, exclusive GPU-hours, or GPU utilization.',
              'excluded_costs':['model downloading and rendering','uncommitted or unsuccessful forwards','earlier failed/interrupted invocation overhead not retained in terminal reports','reporting, independent replay and packaging']}
        body={'schema':SCHEMA,'status':'complete','input_fingerprint':fp,'initial_observations':initial,
            'initial_observations_scope':'Original pre-scoring observations of the GPU scoring host identified in this record; these are not observations or memory peaks of the local CPU analysis host.',
            'models':models,'forward_work':work,'cpu_analysis':cpu,'input_proof_hashes':proofs.hashes,
            'resource_producer_sha256':source_sha,'scope':'Measured completed scoring and numerical-analysis resource accounting; global experiment completion is a separate gate.'}
        csv_payload=_csv_bytes(models,cpu);csv_sha=hashlib.sha256(csv_payload).hexdigest()
        body['artifacts']={'resource_summary.csv':{'sha256':csv_sha,'bytes':len(csv_payload)}}
        if existing and existing.get('status')=='complete':
            expected={k:v for k,v in existing.items() if k not in ('created_utc','content_sha256')}
            if expected!=body or existing.get('content_sha256')!=content_hash(expected):
                raise ValueError('Completed resource inputs or summary changed; refusing to overwrite')
            if not (root/'resource_summary.csv').is_file() or digest(root/'resource_summary.csv')!=csv_sha:
                raise ValueError('Completed resource CSV changed; refusing to overwrite')
            return existing
        _publish(initial_path,initial_bytes)
        _publish(root/'resource_summary.csv',csv_payload)
        result={**body,'created_utc':datetime.now(timezone.utc).isoformat(),'content_sha256':content_hash(body)}
        _publish(summary_path,json_bytes(result),replace_initial=True)
        return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
