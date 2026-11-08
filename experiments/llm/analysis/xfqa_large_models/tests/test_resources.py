"""Measured-work accounting controls; isolated fake completion proofs only."""
from __future__ import annotations
import json,sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resources


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(resources.json_bytes(value))


def batch(prompt,seconds):
    return {'prompt_ids':[prompt],'batch_size':1,'seconds':seconds,'candidate_continuations':3,
        'peak_allocated_bytes':200,'peak_reserved_bytes':300,
        'resources_after':{'gpu_free_bytes':600,'gpu_total_bytes':1000,'cpu_available_bytes':2000,'process_rss_bytes':50}}


@pytest.fixture
def completed_run(tmp_path,monkeypatch):
    """Tiny fake timer files; no scorer, numerical fitting, or real outputs."""
    root=tmp_path/'isolated_run'
    monkeypatch.setitem(resources.EXPECTED,'prompts_per_model',4)
    monkeypatch.setitem(resources.EXPECTED,'candidate_sequences_per_model',12)
    models=[{'id':f'test/model{i}','revision':str(i)*40} for i in range(3)]
    write(root/'model_lock.json',{'status':'complete','models':models})
    write(root/'plan/config/experiment.json',{'new_models':models})
    initial={'status':'preparation_complete_scoring_not_started','requested_node':{'port':2813},
        'observed_hostname':'fake-gpu-host','cpu_available_bytes':3000}
    write(root/'resource_summary.json',initial)
    for i,model in enumerate(models):
        slug=model['id'].replace('/','--');prefix=root/f'status/model_scoring/{slug}'
        hardware={'hostname':'fake-gpu-host','gpu_uuid':f'GPU-{i}','device':f'cuda:{i}'}
        identity={'model_id':model['id'],'model_revision':model['revision'],'input_fingerprint':f'model-{i}',
            'hardware':hardware,'gpu_reserve_bytes':100,'cpu_reserve_bytes':200}
        write(Path(str(prefix)+'_identity.json'),identity)
        repeat_path=root/f'validation/pilot_repeats/{slug}.parquet'
        repeat_path.parent.mkdir(parents=True,exist_ok=True);repeat_path.write_bytes(b'fake repeated scores')
        common={**identity,'status':'complete','actual_model_class':'FakeModel',
            'measurement_origin':'fresh_endpoint_scoring','frozen_original_score_function':True,
            'minimum_gpu_free_bytes':500,'minimum_cpu_available_bytes':1900}
        pilot={**common,'phase':'pilot','completed_rows':1,'candidate_continuations':3,'shards':1,
            'committed_inference_seconds':1.,'wall_seconds':10.,'load_seconds':2.,
            'repeatability':{'passed':True},'worst_case_probe_passed':True,
            'repeat_artifact':{'path':str(repeat_path),'sha256':resources.digest(repeat_path),'rows':1,'batches':[batch('p0',1.5)]}}
        pilot_path=Path(str(prefix)+'_pilot.json');write(pilot_path,pilot)
        score={**common,'phase':'score','completed_rows':4,'candidate_continuations':12,'shards':2,
            'committed_inference_seconds':10.,'wall_seconds':40.,'load_seconds':3.,
            'pilot_report_sha256':resources.digest(pilot_path)}
        write(Path(str(prefix)+'_complete.json'),score)
        raw=root/f'outputs/raw_scores/{slug}'
        raw.mkdir(parents=True,exist_ok=True)
        for j,(stage,batches) in enumerate([('pilot',[batch('p0',1.)]),('score',[batch(f'p{k}',3.) for k in range(1,4)])]):
            path=raw/f'shard_{j:05d}.parquet';path.write_bytes(f'fake {stage} scores'.encode())
            ids=[p for b in batches for p in b['prompt_ids']]
            meta={'input_fingerprint':identity['input_fingerprint'],'model_id':model['id'],
                'model_revision':model['revision'],'prompt_ids':ids,'row_count':len(ids),
                'sha256':resources.digest(path),'execution':{'stage':stage,'batches':batches,'oom_events':[]}}
            write(path.with_suffix('.meta.json'),meta)
            write(path.with_suffix('.complete.json'),{'complete':True,'input_fingerprint':identity['input_fingerprint'],
                'row_count':len(ids),'sha256':resources.digest(path)})
    common={'status':'complete','input_fingerprint':'analysis-fp'}
    write(root/'analysis/point_complete.json',{**common,'point_fit_count':5840,'wall_seconds':5.,'sum_fit_seconds':12.,'workers':4})
    write(root/'analysis/primary_refit_complete.json',{**common,'replicates':1000,'fits':100000,'wall_seconds':3.})
    write(root/'analysis/secondary_complete.json',{**common,'replicates':2000,'wall_seconds':2.})
    write(root/'analysis/bootstrap_complete.json',common)
    write(root/'analysis/analysis_manifest.json',{'input_fingerprint':'analysis-fp',
        'identity':{'synthetic_fixture':False,'scope':'complete_expanded_experiment'}})
    return root,initial


def test_parallel_timers_are_aggregate_work_and_reused_pilot_counted_once(completed_run):
    root,initial=completed_run
    summary=resources.finalize_resources(root)
    assert summary['status']=='complete' and summary['initial_observations']==initial
    assert json.loads((root/'runtime/initial_resource_observations.json').read_text())==initial
    work=summary['forward_work']
    assert work['retained_score_forward_seconds_sum']==30.  # Includes three pilot seconds once.
    assert work['extra_repeatability_forward_seconds_sum']==4.5
    assert work['successful_terminal_invocation_wall_seconds_sum']==150.
    assert work['physical_gpu_count']==3
    assert 'not campaign elapsed wall time, exclusive GPU-hours' in work['interpretation']
    assert 'gpu_hours' not in work and 'campaign_elapsed_seconds' not in work
    assert 'GPU scoring host' in summary['initial_observations_scope']
    assert 'not observations or memory peaks of the local CPU analysis host' in summary['initial_observations_scope']
    assert summary['cpu_analysis']['analysis_phase_wall_seconds_sum']==10.
    assert summary['cpu_analysis']['stages']['point']['parallel_fit_routine_wall_seconds_sum']==12.
    assert summary['cpu_analysis']['memory_peak_bytes'] is None
    assert summary['cpu_analysis']['memory_peak_status']=='not_recorded'
    assert all(m['minimum_gpu_reserve_margin_bytes']==400 for m in summary['models'])
    assert all(m['peak_torch_reserved_bytes']==300 for m in summary['models'])
    assert summary['input_proof_hashes']['runtime/initial_resource_observations.json']==resources.digest(root/'runtime/initial_resource_observations.json')
    assert summary['artifacts']['resource_summary.csv']['sha256']==resources.digest(root/'resource_summary.csv')


@pytest.mark.parametrize('missing',['status/model_scoring/test--model0_complete.json','analysis/primary_refit_complete.json','analysis/bootstrap_complete.json'])
def test_incomplete_stage_does_not_publish_or_replace_initial(completed_run,missing):
    root,_=completed_run;initial=(root/'resource_summary.json').read_bytes()
    (root/missing).unlink()
    with pytest.raises(FileNotFoundError,match='Required completed resource proof missing'):
        resources.finalize_resources(root)
    assert (root/'resource_summary.json').read_bytes()==initial
    assert not (root/'runtime/initial_resource_observations.json').exists()
    assert not (root/'resource_summary.csv').exists()


@pytest.mark.parametrize('corruption',['partial_replicates','synthetic','bad_batch_timer','reserve_violation'])
def test_partial_or_invalid_evidence_does_not_publish(completed_run,corruption):
    root,_=completed_run;initial=(root/'resource_summary.json').read_bytes()
    if corruption=='partial_replicates':
        path=root/'analysis/primary_refit_complete.json';v=json.loads(path.read_text());v['replicates']=999
    elif corruption=='synthetic':
        path=root/'analysis/analysis_manifest.json';v=json.loads(path.read_text());v['identity']['synthetic_fixture']=True
    elif corruption=='bad_batch_timer':
        path=root/'outputs/raw_scores/test--model0/shard_00001.meta.json';v=json.loads(path.read_text());v['execution']['batches'][0]['seconds']=5.
    else:
        path=root/'status/model_scoring/test--model0_complete.json';v=json.loads(path.read_text());v['minimum_gpu_free_bytes']=50
    write(path,v)
    with pytest.raises(ValueError):resources.finalize_resources(root)
    assert (root/'resource_summary.json').read_bytes()==initial
    assert not (root/'resource_summary.csv').exists()


def test_completion_resume_is_exact_noop(completed_run):
    root,_=completed_run;first=resources.finalize_resources(root)
    paths=[root/'resource_summary.json',root/'resource_summary.csv',root/'runtime/initial_resource_observations.json']
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
    assert resources.finalize_resources(root)==first
    assert {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}==before


@pytest.mark.parametrize('artifact',['input','csv','summary','initial'])
def test_completed_publication_rejects_changed_evidence_and_outputs(completed_run,artifact):
    root,_=completed_run;resources.finalize_resources(root)
    if artifact=='input':
        path=root/'status/model_scoring/test--model0_complete.json';v=json.loads(path.read_text());v['wall_seconds']=41.;write(path,v)
    elif artifact=='csv':
        (root/'resource_summary.csv').write_bytes(b'changed')
    elif artifact=='initial':
        path=root/'runtime/initial_resource_observations.json';v=json.loads(path.read_text());v['cpu_available_bytes']=4000;write(path,v)
    else:
        path=root/'resource_summary.json';v=json.loads(path.read_text());v['forward_work']['retained_score_forward_seconds_sum']=999;write(path,v)
    before=(root/'resource_summary.json').read_bytes()
    with pytest.raises(ValueError,match='refusing to overwrite'):resources.finalize_resources(root)
    assert (root/'resource_summary.json').read_bytes()==before
