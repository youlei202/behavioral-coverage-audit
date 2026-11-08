"""CPU-only orchestration checks: no SSH, CUDA access, model loading, or downloads."""
from collections import Counter
from contextlib import redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

COMPONENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMPONENT/'recovery'))
import run_distributed as distributed
import run_campaign as campaign


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    repo=root/'repo';run=root/'run';models=root/'models';roster=[]
    for i in range(8):
        record={'id':f'org/model-{i}','revision':str(i)*40,'batch_size':8}
        roster.append(record)
        snapshot=models/f'models--org--model-{i}/snapshots'/record['revision']
        write(snapshot/'config.json',{})
        name=distributed.slug(record['id'])
        write(run/f'status/model_downloads/{name}.json',{
            'model_id':record['id'],'revision':record['revision'],'snapshot':str(snapshot),
            'tokenizer_only':False,'historical_file_identity_verified':True})
        write(run/f'status/fresh_score_{name}_pilot.json',{
            'projection_full_scoring_seconds':100*(i+1),'final_batch_size':4})
    write(repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json',{'models':roster})
    return repo,run,models,roster


class DistributedChecks(unittest.TestCase):
    def test_ssh_command_roundtrips_every_argument_without_interpolation(self):
        slot={'host':'ucloud@ssh.cloud.sdu.dk','port':2536}
        args=['/shared dir/python',"/x/it's a script.py",'--questions','/x/$(touch bad); `whoami`\n*.jsonl']
        command=distributed.transport(slot,args,Path('/home/test/key with spaces'))
        self.assertEqual(shlex.split(command[-1]),['exec',*args])
        self.assertEqual(command[-2],slot['host'])
        self.assertIn('BatchMode=yes',command)
        self.assertIn('StrictHostKeyChecking=accept-new',command)
        self.assertEqual(command[command.index('-i')+1],'/home/test/key with spaces')
        self.assertEqual(distributed.transport({'host':None},args,None),args)

    def test_slot_validation_supports_confirmed_users_and_rejects_duplicates(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'slots.json'
            slot={'host':'leiyo@ssh.cloud.sdu.dk','port':2702,'device':'cuda:0','python':'/shared/venv/bin/python'}
            write(path,[slot]);self.assertEqual(distributed.load_slots(path)[0]['host'],slot['host'])
            for records in [[slot,slot],[{**slot,'host':'-oProxyCommand=bad'}],[{**slot,'python':'python'}],
                            [{**slot,'host':None,'port':22}],[{**slot,'port':True}]]:
                write(path,records)
                with self.assertRaises(ValueError):distributed.load_slots(path)

    def test_shared_slots_require_explicit_limits_and_batch_one(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'slots.json'
            slot={'host':'leiyo@ssh.cloud.sdu.dk','port':2064,'device':'cuda:0','python':'/venv/bin/python',
                  'allow_shared':True,'memory_budget_mib':71680,'reserve_mib':16384,'batch_size':1,
                  'allowed_models':['org/model-1','org/model-2']}
            write(path,[slot]);actual=distributed.load_slots(path)[0]
            self.assertEqual(actual['memory_budget_mib'],71680);self.assertTrue(actual['allow_shared'])
            for changes in [{'allow_shared':'yes'},{'allow_shared':False},{'memory_budget_mib':0},
                            {'reserve_mib':None},{'memory_budget_mib':80000},{'reserve_mib':1024},
                            {'batch_size':4},{'allowed_models':[]}]:
                write(path,[{**slot,**changes}])
                with self.assertRaises(ValueError):distributed.load_slots(path)

    def test_fixed_pilot_priority_and_committed_progress_are_compute_only(self):
        with TemporaryDirectory() as directory:
            repo,run,models,_=fixture(Path(directory))
            shard=run/'outputs/raw_scores/org--model-7/shard_00000'
            write(shard.with_suffix('.meta.json'),{'row_count':distributed.EXPECTED_ROWS,'sha256':'commit'})
            write(shard.with_suffix('.complete.json'),{'complete':True,'sha256':'commit'})
            jobs=distributed.plan_jobs(repo,run,models)
            self.assertEqual([j['model_id'] for j in jobs],[f'org/model-{i}' for i in [6,5,4,3,2,1,0,7]])
            self.assertEqual(jobs[-1]['estimated_remaining_seconds'],0)
            self.assertTrue(all(j['pilot_batch_size']==4 for j in jobs))
            bad=run/'status/model_downloads/org--model-0.json'
            data=distributed.load(bad);data['revision']='wrong';write(bad,data)
            with self.assertRaisesRegex(ValueError,'checkpoint'):distributed.plan_jobs(repo,run,models)

    def test_unique_assignment_with_per_slot_mutual_exclusion(self):
        jobs=[{'model_id':str(i)} for i in range(8)];slots=[{'id':str(i)} for i in range(3)]
        seen=[];active=set();guard=threading.Lock();events=[]
        def execute(job,slot):
            with guard:
                self.assertNotIn(slot['id'],active);active.add(slot['id']);seen.append(job['model_id'])
            time.sleep(.005)
            with guard:active.remove(slot['id'])
            return 0
        results=distributed.schedule(jobs,slots,execute,lambda *args:events.append(args))
        self.assertEqual(Counter(seen),Counter(str(i) for i in range(8)))
        self.assertEqual(len(events),16)
        finish=Mock(return_value=0)
        self.assertEqual(distributed.finish_barrier(jobs,results,finish),0)
        finish.assert_called_once_with()

    def test_ssh_failure_quarantines_slot_and_closes_assembly_barrier(self):
        jobs=[{'model_id':str(i)} for i in range(8)];calls=[]
        def execute(job,slot):
            calls.append((job['model_id'],slot['id']))
            return 255 if slot['id']=='bad' else 0
        results=distributed.schedule(jobs,[{'id':'bad'},{'id':'good'}],execute,lambda *_:None)
        self.assertEqual(sum(slot=='bad' for _,slot in calls),1)
        self.assertEqual(Counter(model for model,_ in calls),Counter(str(i) for i in range(8)))
        finish=Mock()
        with self.assertRaisesRegex(RuntimeError,'barrier'):distributed.finish_barrier(jobs,results,finish)
        finish.assert_not_called()

    def test_no_healthy_slots_marks_pending_jobs_unscheduled(self):
        jobs=[{'model_id':str(i)} for i in range(8)];execute=Mock(return_value=255)
        results=distributed.schedule(jobs,[{'id':'bad'}],execute,lambda *_:None)
        self.assertEqual(execute.call_count,1)
        self.assertEqual(len(results),8)
        self.assertEqual(sum(r['slot'] is None for r in results.values()),7)
        with self.assertRaises(ValueError):distributed.schedule(jobs[:-1]+[jobs[0]],[{'id':'s'}],execute,lambda *_:None)

    def test_shared_allowlist_never_steals_ineligible_model(self):
        jobs=[{'model_id':str(i)} for i in range(8)]
        slots=[{'id':'shared','allowed_models':['2','5']},{'id':'local','allowed_models':None}]
        assigned=[]
        def execute(job,slot):
            assigned.append((job['model_id'],slot['id']))
            if slot['id']=='local':time.sleep(.005)
            return 0
        results=distributed.schedule(jobs,slots,execute,lambda *_:None)
        self.assertEqual(Counter(model for model,_ in assigned),Counter(str(i) for i in range(8)))
        self.assertEqual({model for model,slot in assigned if slot=='shared'},{'2','5'})
        self.assertTrue(all(result['exit_code']==0 for result in results.values()))
        with self.assertRaisesRegex(ValueError,'allowlists'):
            distributed.schedule(jobs,slots[:1],execute,lambda *_:None)

    def test_controller_lock_is_exclusive_and_foreign_owner_is_not_removed(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'lock'
            first=distributed.DirectoryLock(path,{'token':'first'});first.acquire()
            second=distributed.DirectoryLock(path,{'token':'second'})
            with self.assertRaisesRegex(RuntimeError,'Lock already exists'):second.acquire()
            second.release();self.assertEqual(distributed.load(path/'owner.json')['token'],'first')
            first.release();self.assertFalse(path.exists())

    def test_stale_recovery_never_removes_a_changed_or_unverified_owner(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'lock';owner={'token':'old','hostname':'node'}
            write(path/'owner.json',owner)
            with self.assertRaisesRegex(ValueError,'owner host'):
                distributed._remove_stale_lock(path,{'another-node'},owner)
            write(path/'owner.json',{**owner,'token':'new'})
            with self.assertRaisesRegex(ValueError,'owner changed'):
                distributed._remove_stale_lock(path,{'node'},owner)
            self.assertTrue(path.is_dir())

    def test_process_guard_accepts_only_this_controllers_different_model_children(self):
        owner={'token':'ours','pid':1,'hostname':'node'}
        def process(pid,program,model=None,worker=False,token=None,parent=0):
            return {'pid':pid,'program':[program],'model_id':model,'worker':worker,
                    'owner_token':token,'parent_pid':parent,'hostname':'node'}
        allowed=[process(1,'run_distributed.py'),process(2,'run_distributed.py','other',True,'ours'),
                 process(3,'score_endpoints.py','other',parent=2)]
        self.assertEqual(distributed.worker_conflicts(allowed,owner,'target'),[])
        blocked=[process(4,'score_endpoints.py','independent'),process(5,'run_campaign.py'),
                 process(6,'run_distributed.py','foreign',True,'theirs')]
        self.assertEqual(distributed.worker_conflicts(allowed+blocked,owner,'target'),blocked)

    def test_busy_gpu_owned_by_any_user_is_rejected(self):
        gpu='0, GPU-test, NVIDIA B200, 183359, 155215, 100, 580.95.05, 00000000:01:00.0, 28144\n'
        replies=[SimpleNamespace(stdout=gpu),SimpleNamespace(stdout='GPU-test, 1234, vllm\n')]
        with patch.dict(os.environ,{},clear=True),patch.object(distributed.subprocess,'run',side_effect=replies) as run:
            with self.assertRaisesRegex(RuntimeError,'GPU is occupied'):distributed.inspect_gpu('cuda:0')
        self.assertEqual(run.call_count,2)

    def test_gpu_uuid_mapping_and_idle_metadata(self):
        gpus=('0, GPU-zero, NVIDIA B200, 183359, 0, 0, 580.95.05, 00000000:02:00.0, 183359\n'
              '1, GPU-one, NVIDIA B200, 183359, 0, 0, 580.95.05, 00000000:01:00.0, 183359\n')
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-one'},clear=True),patch.object(
                distributed.subprocess,'run',side_effect=[SimpleNamespace(stdout=gpus),SimpleNamespace(stdout='')]):
            result=distributed.inspect_gpu('cuda:0')
        self.assertEqual(result['uuid'],'GPU-one');self.assertEqual(result['index'],1)
        with patch.dict(os.environ,{'CUDA_DEVICE_ORDER':'PCI_BUS_ID'},clear=True),patch.object(
                distributed.subprocess,'run',side_effect=[SimpleNamespace(stdout=gpus),SimpleNamespace(stdout='')]):
            self.assertEqual(distributed.inspect_gpu('cuda:0')['uuid'],'GPU-one')

    def test_shared_mode_accepts_other_job_only_with_budget_plus_reserve(self):
        snapshot={'memory_free_mib':110000,'memory_used_mib':73359,'utilization_percent':100,
                  'compute_processes':[{'pid':900,'process_name':'other-user-vllm'}]}
        with patch.object(distributed,'gpu_snapshot',return_value=snapshot):
            with self.assertRaisesRegex(RuntimeError,'occupied'):distributed.inspect_gpu('cuda:0')
            self.assertEqual(distributed.inspect_gpu('cuda:0',allow_shared=True,memory_budget_mib=71680,reserve_mib=16384),snapshot)
        with patch.object(distributed,'gpu_snapshot',return_value={**snapshot,'memory_free_mib':80000}):
            with self.assertRaisesRegex(RuntimeError,'cap plus reserve'):
                distributed.inspect_gpu('cuda:0',allow_shared=True,memory_budget_mib=71680,reserve_mib=16384)

    def test_reserve_monitor_stops_only_supervised_child_and_fails_closed(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'monitor.json';supervisor=Mock()
            snapshot={'memory_free_mib':16000,'compute_processes':[{'pid':900,'process_name':'other-user-vllm'}]}
            with patch.object(distributed,'gpu_snapshot',return_value=snapshot),patch.object(distributed.os,'killpg') as kill:
                self.assertFalse(distributed.check_shared_reserve('cuda:0',16384,supervisor,output))
            supervisor.request_stop.assert_called_once_with();kill.assert_not_called()
            self.assertEqual(distributed.load(output)['status'],'reserve_breached')
            supervisor.reset_mock()
            with patch.object(distributed,'gpu_snapshot',side_effect=TimeoutError('nvidia-smi timeout')):
                self.assertFalse(distributed.check_shared_reserve('cuda:0',16384,supervisor,output))
            supervisor.request_stop.assert_called_once_with()
            self.assertEqual(distributed.load(output)['status'],'monitor_failed')

    def test_remote_workers_require_reference_replay(self):
        args=SimpleNamespace(source_repo=Path('/repo'),run_root=Path('/run'),model_root=Path('/models'))
        job={'model_id':'model','pilot_batch_size':4};slot={'python':'/venv/bin/python','device':'cuda:0','host':'user@host'}
        self.assertIn('--verify-remote',distributed.worker_command(args,slot,job,'token'))
        self.assertNotIn('--verify-remote',distributed.worker_command(args,{**slot,'host':None},job,'token'))
        shared={**slot,'allow_shared':True,'memory_budget_mib':71680,'reserve_mib':16384}
        command=distributed.worker_command(args,shared,job,'token')
        self.assertIn('--allow-shared',command);self.assertNotIn('--verify-remote',command)
        self.assertEqual(command[command.index('--memory-budget-mib')+1],'71680')
        command=distributed.worker_command(args,{**shared,'shared_pilot_proofs':{'model':'/shared/proof.json'}},job,'token')
        self.assertEqual(command[command.index('--shared-pilot-proof')+1],'/shared/proof.json')

    def test_remote_worker_proves_then_pilots_then_scores_with_measured_batch(self):
        with TemporaryDirectory() as directory:
            repo,run,models,roster=fixture(Path(directory));model=roster[0];name=distributed.slug(model['id'])
            owner={'token':'ours','pid':1234,'hostname':'controller'}
            write(run/'status/distributed_controller.lock/owner.json',owner)
            driver=repo/'experiments/llm/analysis/xwvg_reference_selection/recovery/score_endpoints.py'
            driver.parent.mkdir(parents=True);driver.write_text('# synthetic frozen reference\n')
            identity={'input_fingerprint':'identity','driver_sha256':distributed.digest(driver),
                      'hardware':{'name':'synthetic GPU'},'source_sha256':{},'versions':{'fixture':'version'}}
            write(run/f'status/fresh_score_{name}_identity.json',identity)
            write(run/f'status/fresh_score_{name}_complete.json',{'status':'complete','completed_rows':36946,
                'model_id':model['id'],'model_revision':model['revision'],'input_fingerprint':'identity'})
            args=SimpleNamespace(run_root=run,source_repo=repo,model_root=models,model_id=model['id'],
                owner_token='ours',device='cuda:0',pilot_batch_size=4,verify_remote=True)
            commands=[]
            def execute(_supervisor,command,output=None):
                commands.append(command)
                if '--output' in command:
                    proof={
                        'status':'passed','model_id':model['id'],'model_revision':model['revision'],
                        'prompt_count':42,'input_fingerprint':'identity','exact_reference_equality':True,
                        'baseline_artifacts_unchanged':True,'immutable_inputs_sources_unchanged':True,
                        'hostname':distributed.platform.node(),'gpu_uuid':'GPU-test','driver_version':'driver',
                        'source_sha256':{'score_endpoints.py':distributed.digest(driver),
                            'validate_remote_reference.py':distributed.digest(COMPONENT/'recovery/validate_remote_reference.py')},
                        'versions':identity['versions'],'hardware':identity['hardware']}
                    proof['proof_sha256']=hashlib.sha256(distributed.canonical(proof)).hexdigest()
                    write(Path(command[command.index('--output')+1]),proof)
                elif command[command.index('--stage')+1]=='pilot':
                    write(run/f'status/fresh_score_{name}_pilot.json',{'status':'complete','final_batch_size':2})
                return 0
            with patch.object(distributed.Supervisor,'run',execute),patch.object(distributed.signal,'signal'),patch.object(
                    distributed,'active_processes',return_value={'processes':[],'unreadable_owned_pids':[]}),patch.object(
                    distributed,'inspect_gpu',return_value={'uuid':'GPU-test','driver_version':'driver'}),redirect_stdout(io.StringIO()):
                self.assertEqual(distributed.worker(args),0)
            self.assertEqual(len(commands),3)
            self.assertTrue(commands[0][1].endswith('validate_remote_reference.py'))
            self.assertEqual([c[c.index('--stage')+1] for c in commands[1:]],['pilot','score'])
            self.assertEqual([c[c.index('--batch-size')+1] for c in commands[1:]],['4','2'])
            self.assertTrue(all('--resume' in c for c in commands[1:]))
            status=distributed.load(run/f'status/distributed_workers/{name}.json')
            self.assertEqual(status['status'],'complete');self.assertIn('sha256',status['remote_reference_proof'])
            self.assertFalse((run/f'status/distributed_model_locks/{name}.lock').exists())

    def test_shared_stages_use_capped_wrapper_and_require_stress_proof(self):
        with TemporaryDirectory() as directory:
            repo,run,models,roster=fixture(Path(directory));model=roster[0];name=distributed.slug(model['id'])
            identity={'input_fingerprint':'identity','hardware':{'name':'fixture'},'versions':{'version':'fixture'}}
            write(run/f'status/fresh_score_{name}_identity.json',identity)
            args=SimpleNamespace(source_repo=repo,run_root=run,model_id=model['id'],owner_token='controller',
                device='cuda:0',allow_shared=True,memory_budget_mib=71680,reserve_mib=16384)
            job={'snapshot':str(models/'snapshot'),'revision':model['revision']}
            gpu={'uuid':'GPU-shared','driver_version':'driver'}
            state={'stages':[]};state_path=run/'status/worker.json';commands=[];stress_ok=True
            def execute(command):
                commands.append(command);stage=command[command.index('--stage')+1]
                proof={'status':'passed' if stage=='pilot' else 'complete','model_id':model['id'],
                    'model_revision':model['revision'],'input_fingerprint':'identity','batch_size':1,
                    'memory_budget_mib':71680,'reserve_mib':16384,'hostname':distributed.platform.node(),
                    'gpu_uuid':'GPU-shared','driver_version':'driver','hardware':identity['hardware'],
                    'versions':identity['versions'],'immutable_inputs_sources_unchanged':True,
                    'wrapper_sha256':distributed.digest(COMPONENT/'recovery/shared_scoring_runtime.py'),
                    'baseline_driver_sha256':distributed.digest(COMPONENT/'recovery/score_endpoints.py'),
                    'prompt_count':42 if stage=='pilot' else 36946,'finite_all_outputs':True,
                    'worst_case_probe_passed':stress_ok,'baseline_score_outputs_written':False,
                    'exact_equality_to_old_batches_claimed':False}
                proof['proof_sha256']=hashlib.sha256(distributed.canonical(proof)).hexdigest()
                write(Path(command[command.index('--output')+1]),proof)
                return 0
            supervisor=Mock();supervisor.run.side_effect=execute
            with patch.object(distributed,'inspect_gpu',return_value=gpu),redirect_stdout(io.StringIO()):
                distributed.run_shared_stages(args,job,supervisor,state,state_path)
            self.assertEqual([c[c.index('--stage')+1] for c in commands],['pilot','score'])
            self.assertTrue(all(c[1].endswith('shared_scoring_runtime.py') for c in commands))
            self.assertTrue(all(c[c.index('--memory-budget-mib')+1]=='71680' for c in commands))
            self.assertIn('--pilot-proof',commands[1]);self.assertNotIn('--batch-size',commands[1])
            self.assertEqual(state['shared_execution']['batch_size'],1)
            self.assertFalse(state['shared_execution']['cross_batch_bitwise_equality_claimed'])
            cached=Path(state['shared_execution']['proofs'][0]['path']);cached_hash=distributed.digest(cached)
            args.shared_pilot_proof=cached;commands.clear()
            with patch.object(distributed,'inspect_gpu',return_value=gpu),redirect_stdout(io.StringIO()):
                distributed.run_shared_stages(args,job,supervisor,{'stages':[]},state_path)
            self.assertEqual(len(commands),1)
            self.assertEqual(commands[0][commands[0].index('--stage')+1],'score')
            self.assertEqual(commands[0][commands[0].index('--pilot-proof')+1],str(cached))
            self.assertEqual(distributed.digest(cached),cached_hash)
            args.shared_pilot_proof=None
            commands.clear();stress_ok=False
            with patch.object(distributed,'inspect_gpu',return_value=gpu),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'worst-case'):
                    distributed.run_shared_stages(args,job,supervisor,{'stages':[]},state_path)
            self.assertEqual(len(commands),1)  # a failed feasibility gate never launches full scoring

    def test_preexisting_campaign_refuses_before_any_worker_launch(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);repo,run,models,_=fixture(root)
            slot_file=root/'slots.json';write(slot_file,[{'host':None,'device':'cuda:0','python':sys.executable}])
            args=SimpleNamespace(source_repo=repo,run_root=run,model_root=models,questions=root/'questions',
                analysis_output=root/'analysis',analysis_python=Path(sys.executable),ssh_key=None,
                slots=slot_file,plan_only=False,exclusive_run=True,recover_stale_locks=False)
            report={'processes':[{'program':['run_campaign.py'],'pid':42}],'unreadable_owned_pids':[]}
            with patch.object(distributed.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(report))),patch.object(
                    distributed.Supervisor,'run') as worker:
                with self.assertRaisesRegex(RuntimeError,'Pre-existing'):distributed.controller(args)
            worker.assert_not_called()
            self.assertFalse((run/'status/distributed_controller.lock').exists())

    def test_plan_only_never_launches_ssh_gpu_or_changes_status(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);repo,run,models,_=fixture(root)
            slot_file=root/'slots.json';write(slot_file,[{'host':'ucloud@host','port':2253,'device':'cuda:0','python':'/venv/bin/python'}])
            args=SimpleNamespace(source_repo=repo,run_root=run,model_root=models,questions=root/'questions',
                analysis_output=root/'analysis',analysis_python=Path('/venv/bin/python'),ssh_key=root/'ssh key',
                slots=slot_file,plan_only=True)
            before=sorted(str(p) for p in root.rglob('*'))
            with patch.object(distributed.subprocess,'run') as external,patch.object(distributed,'inspect_gpu') as gpu,redirect_stdout(io.StringIO()) as output:
                self.assertEqual(distributed.controller(args),0)
            external.assert_not_called();gpu.assert_not_called()
            self.assertEqual(len(json.loads(output.getvalue())['jobs']),8)
            self.assertEqual(before,sorted(str(p) for p in root.rglob('*')))

    def test_finish_keeps_symlinked_analysis_venv_interpreter(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);repo,run,_,_=fixture(root)
            executable=root/'venv/bin/python';executable.parent.mkdir(parents=True);executable.symlink_to(sys.executable)
            argv=['run_campaign.py','--source-repo',str(repo),'--run-root',str(run),'--questions',str(root/'questions.jsonl'),
                  '--analysis-output',str(root/'analysis'),'--analysis-python',str(executable),'--phase','finish']
            with patch.object(sys,'argv',argv),patch.object(campaign.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute,patch.dict(os.environ),redirect_stdout(io.StringIO()):
                self.assertEqual(campaign.main(),0)
            commands=[call.args[0] for call in execute.call_args_list]
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[1][0],str(executable))
            self.assertNotEqual(commands[1][0],str(executable.resolve()))
            self.assertIn('--input-provenance',commands[1]);self.assertIn('--resume',commands[1])


if __name__=='__main__':unittest.main()
