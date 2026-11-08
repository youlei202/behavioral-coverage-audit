#!/usr/bin/env python3
"""Schedule the fixed eight reference scorers over explicitly supplied SSH slots.

Inputs and score outputs must share the same absolute filesystem paths on every
host. Stop the old serial campaign and all independent scorers before starting;
--exclusive-run acknowledges that they will not be started concurrently. This
controller never edits the frozen scorer and never mixes scoring backends.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

from score_endpoints import SOURCE_FILES, canonical, runtime_versions

HERE=Path(__file__).resolve().parent
EXPECTED_MODELS=8
EXPECTED_ROWS=36946
PROCESS_NAMES={'score_endpoints.py','score_selected_endpoints.py','run_campaign.py','run_distributed.py',
               'validate_remote_reference.py','shared_scoring_runtime.py'}


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def slug(model): return re.sub(r'[^A-Za-z0-9._-]+','--',model).strip('-')


def load_slots(path):
    records=load(path)
    if not isinstance(records,list) or not records:raise ValueError('Slots JSON must be a nonempty list')
    slots=[];seen=set()
    for index,row in enumerate(records):
        if not isinstance(row,dict):raise ValueError('Every slot must be an object')
        host=row.get('host');port=row.get('port',22 if host else None)
        if host is not None and (not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9_.@:\[\]-]+',host) or host.startswith('-')):
            raise ValueError('Invalid SSH host')
        if host is not None and (not isinstance(port,int) or isinstance(port,bool) or not 1<=port<=65535):raise ValueError('Invalid SSH port')
        if host is None and port is not None:raise ValueError('A local slot must use null/omitted port')
        device=row.get('device');python=row.get('python')
        if not isinstance(device,str) or not re.fullmatch(r'cuda:\d+',device):raise ValueError('Each slot needs cuda:N device')
        if not isinstance(python,str) or not Path(python).is_absolute() or '\x00' in python:raise ValueError('Slot python must be an absolute interpreter path')
        key=(host,port,device)
        if key in seen:raise ValueError('Duplicate host/port/device slot')
        seen.add(key)
        shared=row.get('allow_shared',False)
        if not isinstance(shared,bool):raise ValueError('allow_shared must be an explicit boolean')
        limits={key:row.get(key) for key in ['memory_budget_mib','reserve_mib']}
        batch=row.get('batch_size',1 if shared else None)
        if shared:
            if any(not isinstance(value,int) or isinstance(value,bool) or value<=0 for value in limits.values()):
                raise ValueError('Shared slots require positive integer memory_budget_mib and reserve_mib')
            if limits['memory_budget_mib']>71680 or limits['reserve_mib']<16384:
                raise ValueError('Shared slots allow at most 71680 MiB budget and at least 16384 MiB reserve')
            if batch!=1 or isinstance(batch,bool):raise ValueError('Shared slots require fixed batch_size=1')
        elif any(value is not None for value in limits.values()) or batch is not None:
            raise ValueError('Memory limits and batch_size require explicit allow_shared=true')
        allowed=row.get('allowed_models')
        if allowed is not None and (not isinstance(allowed,list) or not allowed or
                any(not isinstance(model,str) or not model for model in allowed) or len(set(allowed))!=len(allowed)):
            raise ValueError('allowed_models must be a nonempty list of distinct model IDs')
        proofs=row.get('shared_pilot_proofs',{})
        if (not isinstance(proofs,dict) or (proofs and not shared) or
            any(not isinstance(model,str) or not isinstance(path,str) or not Path(path).is_absolute()
                for model,path in proofs.items())):
            raise ValueError('shared_pilot_proofs requires shared mode and model IDs mapped to absolute proof paths')
        slots.append({'id':f'slot_{index:02d}','host':host,'port':port,'device':device,'python':python,
                      'allow_shared':shared,'batch_size':batch,'allowed_models':allowed,'shared_pilot_proofs':proofs,**limits})
    return slots


def transport(slot, command, ssh_key):
    """No shell for local commands; one correctly quoted command for SSH exec."""
    if slot['host'] is None:return list(command)
    if ssh_key is None:raise ValueError('Remote slots require --ssh-key')
    return ['ssh','-T','-i',str(ssh_key),'-p',str(slot['port']),
            '-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=20',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4',slot['host'],
            'exec '+shlex.join([str(x) for x in command])]


def gpu_snapshot(device):
    """Read all users' GPU allocations; never stop or alter another process."""
    def query(fields,kind):
        result=subprocess.run(['nvidia-smi',f'--query-{kind}={fields}',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
        return [list(map(str.strip,row)) for row in csv.reader(result.stdout.splitlines()) if row]
    rows=query('index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,pci.bus_id,memory.free','gpu')
    pci_order=sorted(rows,key=lambda row:tuple(int(part,16) for part in re.split(r'[:.]',row[7])))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    index=int(device.split(':')[1])
    if visible is not None:
        mapping=visible.split(',')
        if index>=len(mapping):raise ValueError('Selected device is outside CUDA_VISIBLE_DEVICES')
        selector=mapping[index].strip()
        if len(rows)>1 and selector.isdigit() and os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':
            raise ValueError('Ambiguous GPU enumeration: use explicit GPU UUIDs in CUDA_VISIBLE_DEVICES')
        matches=([pci_order[int(selector)]] if selector.isdigit() and int(selector)<len(pci_order)
                 else [row for row in rows if row[1]==selector])
    else:
        # CUDA may enumerate heterogeneous devices differently from nvidia-smi.
        if len(rows)>1 and os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':
            raise ValueError('Multiple GPUs require CUDA_DEVICE_ORDER=PCI_BUS_ID or explicit CUDA_VISIBLE_DEVICES')
        matches=[pci_order[index]] if index<len(pci_order) else []
    if len(matches)!=1:raise ValueError('Cannot uniquely map selected CUDA device to GPU UUID')
    row=matches[0]
    processes=[{'pid':int(p[1]),'process_name':p[2]} for p in
               query('gpu_uuid,pid,process_name','compute-apps') if p[0]==row[1]]
    result={'index':int(row[0]),'uuid':row[1],'name':row[2],'memory_total_mib':int(row[3]),
            'memory_used_mib':int(row[4]),'utilization_percent':int(row[5]),'driver_version':row[6],
            'pci_bus_id':row[7],'memory_free_mib':int(row[8]),'compute_processes':processes}
    return result


def inspect_gpu(device,*,allow_shared=False,memory_budget_mib=None,reserve_mib=None):
    result=gpu_snapshot(device)
    if allow_shared:
        if not memory_budget_mib or not reserve_mib or result['memory_free_mib']<memory_budget_mib+reserve_mib:
            raise RuntimeError(f'Shared GPU cannot provide the requested cap plus reserve: {result}')
    elif result['compute_processes'] or result['memory_used_mib']>1024 or result['utilization_percent']>=5:
        raise RuntimeError(f'Selected GPU is occupied; refusing to launch: {result}')
    return result


def shared_cli(values):
    if not values.get('allow_shared'):return []
    return ['--allow-shared','--memory-budget-mib',str(values['memory_budget_mib']),
            '--reserve-mib',str(values['reserve_mib'])]


def resource_limits(args):
    return {'allow_shared':getattr(args,'allow_shared',False),
            'memory_budget_mib':getattr(args,'memory_budget_mib',None),'reserve_mib':getattr(args,'reserve_mib',None)}


def slot_accepts(slot,job):
    allowed=slot.get('allowed_models')
    return allowed is None or job['model_id'] in allowed


def check_shared_reserve(device,reserve_mib,supervisor,status_path):
    """Stop only our supervised child when the shared device loses its reserve."""
    try:
        snapshot=gpu_snapshot(device)
        record={'checked_utc':now(),'reserve_mib':reserve_mib,'gpu':snapshot,'status':'sufficient'}
        if snapshot['memory_free_mib']<reserve_mib:
            record['status']='reserve_breached'
    except Exception as exc:
        record={'checked_utc':now(),'reserve_mib':reserve_mib,'status':'monitor_failed','error':repr(exc)}
    try:save(status_path,record)
    except Exception:
        supervisor.request_stop();return False
    if record['status']!='sufficient':
        supervisor.request_stop();return False
    return True


def _argument(command,name):
    for index,value in enumerate(command):
        if value==name and index+1<len(command):return command[index+1]
        if value.startswith(name+'='):return value.split('=',1)[1]
    return None


def active_processes(root,ignore_pid=None,ignore_host=None):
    """Inspect this user's Linux processes without exposing full command lines."""
    root=Path(root).resolve();result=[];unreadable=[]
    if not Path('/proc').is_dir():raise RuntimeError('Linux /proc is required for exclusive-process verification')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        pid=int(entry.name)
        if pid==os.getpid() or (pid==ignore_pid and platform.node()==ignore_host):continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            command=(entry/'cmdline').read_bytes().decode(errors='replace').split('\x00')
            names={Path(value).name for value in command if value}
            if not names & PROCESS_NAMES or '--internal-probe' in command:continue
            run_root=_argument(command,'--run-root')
            if run_root is None:continue
            value=Path(run_root)
            if not value.is_absolute():value=(entry/'cwd').resolve()/value
            if value.resolve()!=root:continue
            parent_pid=int((entry/'stat').read_text().rsplit(')',1)[1].split()[1])
            result.append({'pid':pid,'program':sorted(names&PROCESS_NAMES),
                           'parent_pid':parent_pid,
                           'model_id':_argument(command,'--model-id'),
                           'owner_token':_argument(command,'--owner-token'),
                           'worker':'--internal-worker' in command,'hostname':platform.node()})
        except FileNotFoundError:continue
        except PermissionError:unreadable.append(pid)
        except ProcessLookupError:continue
    return {'processes':result,'unreadable_owned_pids':unreadable}


class DirectoryLock:
    """Atomic shared-filesystem lock; a crashed owner's lock is never auto-stolen."""
    def __init__(self,path,owner):self.path=Path(path);self.owner=owner;self.acquired=False
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.path.mkdir()
        except FileExistsError:raise RuntimeError(f'Lock already exists: {self.path}; stop old jobs and use --recover-stale-locks only after inspection')
        save(self.path/'owner.json',self.owner);self.acquired=True
    def release(self):
        if self.acquired and load(self.path/'owner.json').get('token')==self.owner['token']:
            (self.path/'owner.json').unlink();self.path.rmdir();self.acquired=False


class Supervisor:
    """Forward stop signals to local workers/SSH; committed score shards survive."""
    def __init__(self):self.stop=threading.Event();self.lock=threading.RLock();self.processes=set()
    def request_stop(self,*_):
        self.stop.set()
        with self.lock:
            for process in list(self.processes):
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
    def run(self,command,output=None):
        if self.stop.is_set():return 130
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT if output else None,start_new_session=True)
        with self.lock:self.processes.add(process)
        try:
            if self.stop.is_set():
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            return process.wait()
        finally:
            with self.lock:self.processes.discard(process)


def plan_jobs(repo,root,model_root):
    roster=load(repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json')['models']
    if len(roster)!=EXPECTED_MODELS or len({r['id'] for r in roster})!=EXPECTED_MODELS:raise ValueError('Exactly the original eight distinct models are required')
    jobs=[]
    for index,record in enumerate(roster):
        name=slug(record['id']);download=load(root/f'status/model_downloads/{name}.json')
        snapshot=Path(download['snapshot']).resolve()
        if (download.get('model_id')!=record['id'] or download.get('revision')!=record['revision']
            or download.get('tokenizer_only') is not False or download.get('historical_file_identity_verified') is not True
            or snapshot.name!=record['revision'] or not snapshot.is_relative_to(model_root)
            or not (snapshot/'config.json').is_file()):raise ValueError(f'Pinned shared checkpoint is incomplete: {record["id"]}')
        pilot_path=root/f'status/fresh_score_{name}_pilot.json'
        pilot=load(pilot_path) if pilot_path.is_file() else {}
        estimate=float(pilot.get('projection_full_scoring_seconds') or 0)
        if not 0<=estimate<float('inf'):raise ValueError('Invalid fixed pilot time projection')
        completed=0
        for sidecar in (root/'outputs/raw_scores'/name).glob('shard_*.meta.json'):
            marker=sidecar.with_suffix('').with_suffix('.complete.json')
            if marker.is_file():
                meta,commit=load(sidecar),load(marker)
                if commit.get('complete') is True and commit.get('sha256')==meta.get('sha256'):
                    completed+=int(meta.get('row_count',0))
        remaining=estimate*max(0,EXPECTED_ROWS-completed)/EXPECTED_ROWS
        jobs.append({'model_id':record['id'],'revision':record['revision'],'snapshot':str(snapshot),
                     'roster_index':index,'estimated_remaining_seconds':remaining,
                     'committed_rows_for_timing_only':completed,
                     'pilot_batch_size':int(pilot.get('final_batch_size') or record['batch_size'])})
    return sorted(jobs,key=lambda job:(-job['estimated_remaining_seconds'],job['roster_index']))


def worker_command(args,slot,job,owner_token):
    command=[slot['python'],str(HERE/'run_distributed.py'),'--internal-worker',
            '--source-repo',str(args.source_repo),'--run-root',str(args.run_root),
            '--model-root',str(args.model_root),'--model-id',job['model_id'],
            '--device',slot['device'],'--pilot-batch-size',str(job['pilot_batch_size']),
            '--owner-token',owner_token]
    command.extend(shared_cli(slot))
    if job['model_id'] in slot.get('shared_pilot_proofs',{}):
        command.extend(['--shared-pilot-proof',slot['shared_pilot_proofs'][job['model_id']]])
    if slot['host'] is not None and not slot.get('allow_shared'):command.append('--verify-remote')
    return command


def schedule(jobs,slots,execute,record,stop=None):
    """Each model is popped exactly once; a failed model is never silently retried."""
    if len(jobs)!=EXPECTED_MODELS or len({j['model_id'] for j in jobs})!=EXPECTED_MODELS:
        raise ValueError('Exactly eight unique scoring jobs are required')
    if not slots or len({s['id'] for s in slots})!=len(slots):raise ValueError('Nonempty unique slots are required')
    if any(not any(slot_accepts(slot,job) for slot in slots) for job in jobs):
        raise ValueError('Slot model allowlists do not cover all eight required models')
    pending=list(jobs);results={};active={}
    stop=stop or threading.Event()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        def submit(slot):
            if not pending or stop.is_set():return
            selected=next((i for i,job in enumerate(pending) if slot_accepts(slot,job)),None)
            if selected is None:return
            job=pending.pop(selected)
            record(job,slot,'running',None)
            active[pool.submit(execute,job,slot)]=(job,slot)
        for slot in slots:submit(slot)
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in sorted(done,key=lambda f:active[f][1]['id']):
                job,slot=active.pop(future)
                try:code=int(future.result());error=None
                except Exception as exc:code=1;error=repr(exc)
                results[job['model_id']]={'exit_code':code,'slot':slot['id'],'error':error}
                record(job,slot,'complete' if code==0 else 'failed',results[job['model_id']])
                # A dropped SSH connection can leave its remote worker alive.
                # Quarantine every failed slot for the rest of this invocation.
                if code==0:submit(slot)
    for job in pending:results[job['model_id']]={'exit_code':130,'slot':None,'error':'stopped or no healthy slots remain'}
    return results


def finish_barrier(jobs,results,finish):
    expected={job['model_id'] for job in jobs}
    if len(jobs)!=EXPECTED_MODELS or len(expected)!=EXPECTED_MODELS or set(results)!=expected or any(r['exit_code']!=0 for r in results.values()):
        raise RuntimeError('Scoring incomplete: assembly/analysis barrier remains closed')
    return finish()


def worker_conflicts(processes,owner,model_id):
    """Other models may run only as children of this controller's workers."""
    controlled={p['pid']:p for p in processes if p['worker'] and p['owner_token']==owner['token']}
    conflicts=[]
    for process in processes:
        if process['model_id']==model_id:
            conflicts.append(process);continue
        if 'run_campaign.py' in process['program']:
            conflicts.append(process);continue
        if 'run_distributed.py' in process['program']:
            if process['pid'] in controlled:continue
            if process['pid']==owner['pid'] and process['hostname']==owner['hostname']:continue
        elif {'score_endpoints.py','validate_remote_reference.py','shared_scoring_runtime.py'} & set(process['program']):
            parent=controlled.get(process['parent_pid'])
            if parent and parent['model_id']==process['model_id']:continue
        conflicts.append(process)
    return conflicts


def probe(args):
    root=args.run_root.resolve()
    if load(args.nonce_file).get('nonce')!=args.nonce:raise ValueError('Shared-filesystem nonce is not visible')
    driver=args.source_repo/'experiments/llm/analysis/xwvg_reference_selection/recovery/score_endpoints.py'
    if digest(driver)!=args.driver_sha256 or digest(Path(__file__))!=args.controller_sha256:raise ValueError('Remote scorer/controller source differs')
    jobs=plan_jobs(args.source_repo,root,args.model_root.resolve())
    processes=active_processes(root,args.controller_pid,args.controller_host)
    gpu=inspect_gpu(args.device,**resource_limits(args))
    versions=runtime_versions()
    for job in jobs:
        identity_path=root/f'status/fresh_score_{slug(job["model_id"])}_identity.json'
        if identity_path.is_file() and load(identity_path).get('versions')!=versions:
            raise ValueError(f'Runtime versions cannot resume existing identity: {job["model_id"]}')
    print(json.dumps({'hostname':platform.node(),'python':sys.executable,'python_version':platform.python_version(),
                      'uid':os.getuid(),'runtime_versions':versions,'platform':platform.platform(),
                      'source_sha256':{path:digest(args.source_repo/path) for path in SOURCE_FILES},
                      'driver_sha256':args.driver_sha256,'controller_sha256':args.controller_sha256,
                      'device':args.device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'cuda_device_order':os.environ.get('CUDA_DEVICE_ORDER'),'gpu':gpu,
                      'resource_limits':resource_limits(args),
                      'shared_filesystem_verified':True,'models_visible':len(jobs),**processes}),flush=True)
    return 0


def run_shared_stages(args,job,supervisor,state,state_path):
    """Measure batch-one feasibility, then run the unchanged scorer under a cap."""
    name=slug(args.model_id);root=args.run_root.resolve()
    runtime=HERE/'shared_scoring_runtime.py';driver=HERE/'score_endpoints.py'
    identity=load(root/f'status/fresh_score_{name}_identity.json')
    directory=root/f'status/distributed_shared/{args.owner_token}'
    cached=getattr(args,'shared_pilot_proof',None)
    pilot_path=Path(cached).expanduser().resolve() if cached else directory/f'{name}_pilot.json'
    state['shared_execution']={'wrapper_sha256':digest(runtime),'baseline_driver_sha256':digest(driver),
        'memory_budget_mib':args.memory_budget_mib,'reserve_mib':args.reserve_mib,'batch_size':1,
        'cross_batch_bitwise_equality_claimed':False,'proofs':[]}
    for stage in ['pilot','score']:
        gpu=inspect_gpu(args.device,**resource_limits(args));state['gpu']=gpu
        output=pilot_path if stage=='pilot' else directory/f'{name}_{stage}.json'
        command=[sys.executable,str(runtime),'--source-repo',str(args.source_repo),'--run-root',str(root),
                 '--model-id',args.model_id,'--model-path',job['snapshot'],'--device',args.device,
                 '--memory-budget-mib',str(args.memory_budget_mib),'--reserve-mib',str(args.reserve_mib),
                 '--stage',stage,'--output',str(output)]
        if stage=='score':command.extend(['--pilot-proof',str(pilot_path)])
        reused=stage=='pilot' and cached is not None
        state.update(current_stage=f'shared_{stage}',current_command=None if reused else command);save(state_path,state)
        print(json.dumps({'distributed_model':args.model_id,'stage':f'shared_{stage}',
                          'batch_size':1,'reused_shared_pilot':reused}),flush=True)
        started=time.perf_counter();code=0 if reused else supervisor.run(command)
        state['stages'].append({'stage':f'shared_{stage}','exit_code':code,
                               'seconds':time.perf_counter()-started,'reused_shared_pilot':reused})
        if code:raise RuntimeError(f'Shared {stage} failed with exit {code}')
        proof=load(output)
        checksum=hashlib.sha256(canonical({k:v for k,v in proof.items() if k!='proof_sha256'})).hexdigest()
        if (proof.get('status')!=('passed' if stage=='pilot' else 'complete')
            or proof.get('proof_sha256')!=checksum
            or proof.get('model_id')!=args.model_id or proof.get('model_revision')!=job['revision']
            or proof.get('input_fingerprint')!=identity['input_fingerprint']
            or proof.get('wrapper_sha256')!=digest(runtime) or proof.get('baseline_driver_sha256')!=digest(driver)
            or proof.get('memory_budget_mib')!=args.memory_budget_mib or proof.get('reserve_mib')!=args.reserve_mib
            or proof.get('batch_size')!=1 or proof.get('hostname')!=platform.node()
            or proof.get('gpu_uuid')!=gpu['uuid'] or proof.get('driver_version')!=gpu['driver_version']
            or proof.get('hardware')!=identity['hardware'] or proof.get('versions')!=identity['versions']
            or proof.get('immutable_inputs_sources_unchanged') is not True):
            raise ValueError('Shared execution proof has mismatched source/input/resource/node identity')
        if stage=='pilot' and (proof.get('prompt_count')!=42 or proof.get('finite_all_outputs') is not True
                or proof.get('worst_case_probe_passed') is not True
                or proof.get('baseline_score_outputs_written') is not False
                or proof.get('exact_equality_to_old_batches_claimed') is not False):
            raise ValueError('Shared fixed42 plus worst-case feasibility pilot did not pass')
        state['shared_execution']['proofs'].append({'stage':stage,'path':str(output),'sha256':digest(output)})
        save(state_path,state)


def worker(args):
    root=args.run_root.resolve();owner=load(root/'status/distributed_controller.lock/owner.json')
    if owner.get('token')!=args.owner_token:raise ValueError('Worker controller ownership token differs')
    jobs=plan_jobs(args.source_repo,root,args.model_root.resolve())
    matches=[job for job in jobs if job['model_id']==args.model_id]
    if len(matches)!=1:raise ValueError('Worker model is outside the fixed roster')
    job=matches[0];name=slug(args.model_id)
    metadata={'token':args.owner_token,'pid':os.getpid(),'hostname':platform.node(),'model_id':args.model_id,'started_utc':now()}
    lease=DirectoryLock(root/f'status/distributed_model_locks/{name}.lock',metadata);lease.acquire()
    supervisor=Supervisor()
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,supervisor.request_stop)
    state_path=root/f'status/distributed_workers/{name}.json'
    state={**metadata,'status':'running','device':args.device,'python':sys.executable,
           'runtime_versions':runtime_versions(),'controller_sha256':digest(Path(__file__)),
           'resource_limits':resource_limits(args),'stages':[]}
    save(state_path,state)
    watch_done=threading.Event()
    cancel_path=root/f'status/distributed_cancellations/{args.owner_token}/{name}.json'
    reserve_status=root/f'status/distributed_shared/{args.owner_token}/{name}_reserve_monitor.json'
    def watch_controller():
        # Shared cancellation reaches a remote worker even after SSH disconnects.
        while not watch_done.wait(2):
            try:
                current=load(root/'status/distributed_controller.lock/owner.json')
                if current.get('token')==args.owner_token and not cancel_path.exists():
                    if getattr(args,'allow_shared',False) and not check_shared_reserve(
                            args.device,args.reserve_mib,supervisor,reserve_status):return
                    continue
            except (OSError,json.JSONDecodeError):pass
            supervisor.request_stop();return
    monitor=threading.Thread(target=watch_controller,daemon=True);monitor.start()
    try:
        inspection=active_processes(root)
        if inspection['unreadable_owned_pids']:raise RuntimeError('Cannot verify owned processes before scoring')
        conflicts=worker_conflicts(inspection['processes'],owner,args.model_id)
        if conflicts:raise RuntimeError(f'Conflicting scorer/campaign processes: {conflicts}')
        driver=args.source_repo/'experiments/llm/analysis/xwvg_reference_selection/recovery/score_endpoints.py'
        if args.verify_remote and not getattr(args,'allow_shared',False):
            gpu=inspect_gpu(args.device)
            proof_path=root/f'status/distributed_remote_proofs/{args.owner_token}/{name}.json'
            command=[sys.executable,str(HERE/'validate_remote_reference.py'),'--source-repo',str(args.source_repo),
                     '--run-root',str(root),'--model-id',args.model_id,'--model-path',job['snapshot'],
                     '--device',args.device,'--output',str(proof_path)]
            state.update(current_stage='remote_reference_proof',current_command=command);save(state_path,state)
            code=supervisor.run(command)
            if code:raise RuntimeError(f'Remote fixed-pilot replay failed with exit {code}')
            proof=load(proof_path);identity=load(root/f'status/fresh_score_{name}_identity.json')
            proof_hash=hashlib.sha256(canonical({k:v for k,v in proof.items() if k!='proof_sha256'})).hexdigest()
            if (proof.get('status')!='passed' or proof.get('model_id')!=args.model_id
                or proof.get('model_revision')!=job['revision'] or proof.get('prompt_count')!=42
                or proof.get('input_fingerprint')!=identity.get('input_fingerprint')
                or proof.get('exact_reference_equality') is not True
                or proof.get('baseline_artifacts_unchanged') is not True
                or proof.get('immutable_inputs_sources_unchanged') is not True
                or proof.get('proof_sha256')!=proof_hash
                or proof.get('source_sha256',{}).get('validate_remote_reference.py')!=digest(HERE/'validate_remote_reference.py')
                or proof.get('source_sha256',{}).get('score_endpoints.py')!=digest(driver)
                or proof.get('versions')!=identity.get('versions')
                or proof.get('hardware')!=identity.get('hardware')
                or proof.get('hostname')!=platform.node() or proof.get('gpu_uuid')!=gpu['uuid']
                or proof.get('driver_version')!=gpu['driver_version']):
                raise ValueError('Remote fixed-pilot proof is incomplete or mismatched')
            state['remote_reference_proof']={'path':str(proof_path),'sha256':digest(proof_path)}
            save(state_path,state)
        if getattr(args,'allow_shared',False):
            state['reserve_monitor_status']=str(reserve_status)
            run_shared_stages(args,job,supervisor,state,state_path)
        for stage in ([] if getattr(args,'allow_shared',False) else ['pilot','score']):
            gpu=inspect_gpu(args.device)
            state['gpu']=gpu
            batch=args.pilot_batch_size
            if stage=='score':
                pilot=load(root/f'status/fresh_score_{name}_pilot.json')
                if pilot.get('status')!='complete':raise ValueError('Measured reference pilot incomplete')
                batch=int(pilot['final_batch_size'])
            command=[sys.executable,str(driver),'--source-repo',str(args.source_repo),
                     '--run-root',str(root),'--model-id',args.model_id,'--model-path',job['snapshot'],
                     '--device',args.device,'--batch-size',str(batch),'--stage',stage,'--resume']
            state.update(current_stage=stage,current_command=command);save(state_path,state)
            print(json.dumps({'distributed_model':args.model_id,'stage':stage,'status':'started','batch_size':batch}),flush=True)
            started=time.perf_counter();code=supervisor.run(command)
            state['stages'].append({'stage':stage,'exit_code':code,'seconds':time.perf_counter()-started})
            if code:raise RuntimeError(f'{args.model_id} {stage} failed with exit {code}')
        complete=load(root/f'status/fresh_score_{name}_complete.json')
        identity=load(root/f'status/fresh_score_{name}_identity.json')
        if (complete.get('status')!='complete' or complete.get('completed_rows')!=EXPECTED_ROWS
            or complete.get('model_id')!=args.model_id or complete.get('model_revision')!=job['revision']
            or complete.get('input_fingerprint')!=identity.get('input_fingerprint')
            or identity.get('driver_sha256')!=digest(driver) or 'scoring_backend' in identity):
            raise ValueError('Complete reference scoring status is invalid')
        state.update(status='complete',completed_utc=now(),input_fingerprint=identity['input_fingerprint'],
                     hardware=identity['hardware'],source_sha256=identity['source_sha256'],
                     driver_sha256=identity['driver_sha256']);save(state_path,state)
        return 0
    except Exception as exc:
        state.update(status='interrupted' if supervisor.stop.is_set() else 'failed',error=repr(exc),ended_utc=now());save(state_path,state)
        raise
    finally:
        watch_done.set();monitor.join();supervisor.request_stop();lease.release()


def _remove_stale_lock(path,observed_hosts,expected_owner):
    owner=load(path/'owner.json')
    if owner!=expected_owner:raise ValueError(f'Lock owner changed after preflight began: {path}')
    if owner.get('hostname') not in observed_hosts:raise ValueError(f'Cannot verify stale lock owner host: {owner.get("hostname")}')
    if set(p.name for p in path.iterdir())!={'owner.json'}:raise ValueError(f'Unexpected lock-directory contents: {path}')
    # Called only after every contacted host reports no conflicting processes.
    (path/'owner.json').unlink();path.rmdir()


def controller(args):
    for name in ['source_repo','run_root','model_root','questions','analysis_output']:
        setattr(args,name,getattr(args,name).expanduser().resolve())
    # Resolving a venv's symlinked interpreter would silently escape the venv.
    args.analysis_python=args.analysis_python.expanduser().absolute()
    if args.ssh_key:args.ssh_key=args.ssh_key.expanduser().resolve()
    slots=load_slots(args.slots);jobs=plan_jobs(args.source_repo,args.run_root,args.model_root)
    roster_ids={job['model_id'] for job in jobs}
    for slot in slots:
        if slot['allowed_models'] is not None and not set(slot['allowed_models'])<=roster_ids:
            raise ValueError('A slot allowlist contains a model outside the fixed eight')
        if not set(slot['shared_pilot_proofs'])<=roster_ids:
            raise ValueError('A shared proof map contains a model outside the fixed eight')
    if any(not any(slot_accepts(slot,job) for slot in slots) for job in jobs):
        raise ValueError('Slot model allowlists do not cover all eight required models')
    if args.plan_only:
        print(json.dumps({'mode':'read_only_plan_no_ssh_no_gpu','slots':slots,'jobs':jobs,
             'assignment_policy':'longest fixed-pilot projected remaining workload first; next job goes to first free slot',
             'command_previews':[transport(next(s for s in slots if slot_accepts(s,job)),
                                worker_command(args,next(s for s in slots if slot_accepts(s,job)),job,'OWNER_TOKEN'),args.ssh_key)
                                for job in jobs]},indent=2));return 0
    if not args.exclusive_run:raise ValueError('--exclusive-run is required: stop old campaign/scorers and do not restart them concurrently')
    token=uuid.uuid4().hex
    owner={'token':token,'pid':os.getpid(),'hostname':platform.node(),'started_utc':now()}
    status=args.run_root/'status';status.mkdir(parents=True,exist_ok=True)
    lock_path=status/'distributed_controller.lock'
    def current_locks():
        return ({lock_path} if lock_path.exists() else set()) | set((status/'distributed_model_locks').glob('*.lock'))
    stale_locks={path:load(path/'owner.json') for path in current_locks()} if args.recover_stale_locks else {}
    nonce_path=status/f'distributed_nonce_{token}.json';save(nonce_path,{'nonce':token})
    probes=[]
    try:
        for slot in slots:
            command=[slot['python'],str(HERE/'run_distributed.py'),'--internal-probe',
                     '--source-repo',str(args.source_repo),'--run-root',str(args.run_root),'--model-root',str(args.model_root),
                     '--device',slot['device'],'--nonce-file',str(nonce_path),'--nonce',token,
                     '--controller-pid',str(os.getpid()),'--controller-host',platform.node(),
                     '--driver-sha256',digest(HERE/'score_endpoints.py'),'--controller-sha256',digest(Path(__file__))]
            command.extend(shared_cli(slot))
            result=subprocess.run(transport(slot,command,args.ssh_key),capture_output=True,text=True,check=False,timeout=90)
            if result.returncode:raise RuntimeError(f'Preflight failed for {slot["id"]}: {result.stderr[-2000:]}')
            report=json.loads(result.stdout);report['slot']=slot['id'];probes.append(report)
            if report['processes'] or report['unreadable_owned_pids']:raise RuntimeError(f'Pre-existing or unverifiable scorer processes on {slot["id"]}: {report}')
        local=active_processes(args.run_root)
        if local['processes'] or local['unreadable_owned_pids']:raise RuntimeError(f'Local legacy campaign/scorer must be stopped: {local}')
    finally:
        nonce_path.unlink(missing_ok=True)
    actual_slots=[p['gpu']['uuid'] for p in probes]
    if len(set(actual_slots))!=len(actual_slots):raise ValueError('Two requested slots resolve to the same host/device')
    if args.recover_stale_locks:
        hosts={p['hostname'] for p in probes}|{platform.node()}
        if current_locks()!=set(stale_locks):raise ValueError('Lock set changed after preflight began')
        for path,expected in stale_locks.items():_remove_stale_lock(path,hosts,expected)
    lease=DirectoryLock(lock_path,owner);lease.acquire()
    supervisor=Supervisor()
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,supervisor.request_stop)
    logroot=args.run_root/'logs/distributed'/token;logroot.mkdir(parents=True,exist_ok=True)
    state={'schema_version':'xwvg_distributed_reference_campaign_v1',**owner,'status':'running','scorer':'reference',
           'controller_sha256':digest(Path(__file__)),'reference_driver_sha256':digest(HERE/'score_endpoints.py'),
           'shared_runtime_sha256':digest(HERE/'shared_scoring_runtime.py') if any(s['allow_shared'] for s in slots) else None,
           'slots':slots,'preflight':probes,'schedule':jobs,
           'assignment_policy':'longest fixed-pilot projected remaining workload first; no scientific outcomes inspected',
           'jobs':{},'log_root':str(logroot)}
    state_path=status/'distributed_campaign.json';state_lock=threading.Lock();save(state_path,state)
    def record(job,slot,phase,result):
        with state_lock:
            state['jobs'][job['model_id']]={'status':phase,'slot':slot['id'],'updated_utc':now(),
                'result':result,'worker_status':str(status/f'distributed_workers/{slug(job["model_id"])}.json')}
            if phase=='failed':state.setdefault('quarantined_slots',[]).append(slot['id'])
            save(state_path,state)
        print(json.dumps({'model_id':job['model_id'],'slot':slot['id'],'status':phase}),flush=True)
    def execute(job,slot):
        command=transport(slot,worker_command(args,slot,job,token),args.ssh_key)
        with (logroot/f'{slug(job["model_id"])}.log').open('a') as log:
            log.write(json.dumps({'started_utc':now(),'command':command})+'\n');log.flush()
            code=supervisor.run(command,log)
        if code:
            save(status/f'distributed_cancellations/{token}/{slug(job["model_id"])}.json',
                 {'requested_utc':now(),'reason':'worker transport or stage failed','exit_code':code})
        return code
    def finish():
        command=[str(args.analysis_python),str(HERE/'run_campaign.py'),'--source-repo',str(args.source_repo),
                 '--run-root',str(args.run_root),'--questions',str(args.questions),'--analysis-output',str(args.analysis_output),
                 '--analysis-python',str(args.analysis_python),'--phase','finish','--scorer','reference']
        with state_lock:state.update(status='finishing',finish_command=command);save(state_path,state)
        with (logroot/'finish.log').open('a') as log:return supervisor.run(command,log)
    try:
        results=schedule(jobs,slots,execute,record,supervisor.stop)
        state['results']=results;save(state_path,state)
        code=finish_barrier(jobs,results,finish)
        if code:raise RuntimeError(f'Assembly/analysis failed with exit {code}; see {logroot}/finish.log')
        state.update(status='complete',completed_utc=now());save(state_path,state);return 0
    except Exception as exc:
        state.update(status='interrupted' if supervisor.stop.is_set() else 'failed',error=repr(exc),ended_utc=now());save(state_path,state)
        raise
    finally:
        supervisor.request_stop();lease.release()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True,help='Shared parent of the pinned models--*/snapshots trees')
    parser.add_argument('--slots',type=Path)
    parser.add_argument('--ssh-key',type=Path)
    parser.add_argument('--questions',type=Path)
    parser.add_argument('--analysis-output',type=Path)
    parser.add_argument('--analysis-python',type=Path)
    parser.add_argument('--exclusive-run',action='store_true',help='Acknowledge no concurrent legacy campaign or independent scorer')
    parser.add_argument('--recover-stale-locks',action='store_true',help='Remove old controller/model locks only after all owner hosts pass no-process preflight')
    parser.add_argument('--plan-only',action='store_true',help='Read local metadata and display plan; no SSH/GPU/lock changes')
    parser.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--internal-probe',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--verify-remote',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--allow-shared',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--memory-budget-mib',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--reserve-mib',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--shared-pilot-proof',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--model-id',help=argparse.SUPPRESS)
    parser.add_argument('--owner-token',help=argparse.SUPPRESS)
    parser.add_argument('--device',default='cuda:0',help=argparse.SUPPRESS)
    parser.add_argument('--pilot-batch-size',type=int,default=4,help=argparse.SUPPRESS)
    parser.add_argument('--nonce-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--nonce',help=argparse.SUPPRESS)
    parser.add_argument('--controller-pid',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--controller-host',help=argparse.SUPPRESS)
    parser.add_argument('--driver-sha256',help=argparse.SUPPRESS)
    parser.add_argument('--controller-sha256',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.internal_worker and args.internal_probe:parser.error('Internal modes are mutually exclusive')
    if args.allow_shared:
        if args.memory_budget_mib is None or args.reserve_mib is None or min(args.memory_budget_mib,args.reserve_mib)<=0:
            parser.error('Shared mode requires positive memory budget and reserve')
        if args.memory_budget_mib>71680 or args.reserve_mib<16384:
            parser.error('Shared mode allows at most 71680 MiB budget and at least 16384 MiB reserve')
        if args.verify_remote:parser.error('Shared feasibility and idle exact-replay gates are separate explicit modes')
    if not args.internal_worker and not args.internal_probe:
        for name in ['slots','questions','analysis_output','analysis_python']:
            if getattr(args,name) is None:parser.error(f'--{name.replace("_","-")} is required')
    try:
        return probe(args) if args.internal_probe else worker(args) if args.internal_worker else controller(args)
    except Exception as exc:
        print(json.dumps({'status':'distributed_campaign_failed','error':repr(exc)}),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
