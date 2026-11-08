"""Driver boundary and empirical-completion gates, with isolated fake proofs.

These tests never invoke a scorer or write to any actual experiment output.
The fake status fixtures exercise the coordinator's trust boundaries; the
independent numerical verifier has its own response/weight replay tests.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,sys
import pytest

COMPONENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMPONENT))
spec=importlib.util.spec_from_file_location('xfqa_driver_under_test',COMPONENT/'run.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

MODELS=['Qwen/Qwen2.5-32B-Instruct','Qwen/QwQ-32B','google/gemma-3-27b-it']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def arguments(tmp_path,**changes):
    repo=tmp_path/'repo';out=repo/'outputs/xfqa_large_models/fixture';plan=out/'plan'
    (out/'status').mkdir(parents=True)
    write(plan/'config/experiment.json',{'new_models':[{'id':m} for m in MODELS]})
    return SimpleNamespace(repo_root=repo,output_root=out,plan_root=plan,resume=True,cache=None,workers=4,
        placement=None,device='cuda:0',scoring_python=sys.executable,analysis_python=sys.executable,
        model_id=None,batch_size=1,gpu_reserve_gib=16,cpu_reserve_gib=8,model_cache=out/'cache/models',
        execution_manifest=None,**changes)


def argv(args,phase='all'):
    return ['--repo-root',str(args.repo_root),'--plan-root',str(args.plan_root),'--output-root',str(args.output_root),'--phase',phase,'--resume']


def test_all_executes_each_phase_in_order_then_finalizes(tmp_path,monkeypatch):
    args=arguments(tmp_path);events=[]
    monkeypatch.setattr(driver,'run_phase',lambda a,p:events.append(p))
    monkeypatch.setattr(driver,'finalize',lambda a:events.append('finalize'))
    assert driver.main(argv(args))==0
    assert events==driver.PHASES+['finalize']
    for phase in driver.PHASES:
        assert json.loads((args.output_root/f'status/driver_{phase}.json').read_text())['status']=='complete'


@pytest.mark.parametrize('failed_phase',['bootstrap','package'])
def test_phase_failure_stops_downstream_and_never_writes_complete(tmp_path,monkeypatch,failed_phase):
    args=arguments(tmp_path);events=[]
    def phase(_args,name):
        events.append(name)
        if name==failed_phase:raise RuntimeError('deliberate isolated test failure')
    monkeypatch.setattr(driver,'run_phase',phase)
    monkeypatch.setattr(driver,'finalize',lambda a:pytest.fail('Finalization must not follow a failed phase'))
    with pytest.raises(RuntimeError,match='deliberate'):
        driver.main(argv(args))
    assert events==driver.PHASES[:driver.PHASES.index(failed_phase)+1]
    assert not (args.output_root/'status/complete.json').exists()
    status=json.loads((args.output_root/f'status/driver_{failed_phase}.json').read_text())
    assert status['status']=='failed' and status['error_type']=='RuntimeError'


@pytest.mark.parametrize('phase',['analyze','bootstrap'])
@pytest.mark.parametrize('explicit',[False,True])
def test_analysis_phases_pass_actual_assembled_cache_or_explicit_override(tmp_path,monkeypatch,phase,explicit):
    args=arguments(tmp_path);calls=[]
    if explicit:args.cache=tmp_path/'external_complete_cache.parquet'
    module=SimpleNamespace(**{phase:lambda *a,**k:calls.append((a,k))})
    monkeypatch.setitem(sys.modules,'analysis',module)
    driver.run_phase(args,phase)
    expected=args.cache if explicit else args.output_root/'data/permutation_averaged_semantic_scores.parquet'
    assert calls[0][1]['cache_path']==expected
    assert calls[0][1]['workers']==4 and calls[0][1]['resume'] is True


def test_alternate_cpu_environment_executes_real_child_phase(tmp_path,monkeypatch):
    args=arguments(tmp_path);calls=[];other=tmp_path/'different-venv/bin/python'
    monkeypatch.setattr(driver,'execute',lambda a,c,n:calls.append((c,n)))
    monkeypatch.setattr(driver,'run_phase',lambda *a:pytest.fail('CPU phase must use selected interpreter'))
    assert driver.main(argv(args,'analyze')+['--analysis-python',str(other)])==0
    command,name=calls[0]
    assert command[0]==str(other) and command[command.index('--phase')+1]=='analyze'
    assert name=='cpu-analyze'


def test_all_rejects_single_model_and_old_output_root(tmp_path):
    args=arguments(tmp_path)
    with pytest.raises(SystemExit):driver.main(argv(args)+['--model-id',MODELS[0]])
    bad=argv(args,'preflight');index=bad.index('--output-root');bad[index+1]=str(args.repo_root/'outputs/xwvg_reference_selection/old')
    with pytest.raises(SystemExit):driver.main(bad)
    assert not (args.repo_root/'outputs/xwvg_reference_selection/old').exists()


def test_placement_rejects_duplicate_addresses_and_physical_gpus(tmp_path):
    args=arguments(tmp_path);args.placement=tmp_path/'placement.json'
    records=[{'model_id':m,'device':f'cuda:{i}','expected_gpu_uuid':f'GPU-{i}'} for i,m in enumerate(MODELS)]
    write(args.placement,{'slots':records})
    assert len(driver.slots(args,[{'id':m} for m in MODELS]))==3
    records[1]['device']='cuda:0';write(args.placement,{'slots':records})
    with pytest.raises(ValueError,match='Duplicate GPU'):driver.slots(args,[{'id':m} for m in MODELS])
    records[1]['device']='cuda:1';records[1]['expected_gpu_uuid']='GPU-0';write(args.placement,{'slots':records})
    with pytest.raises(ValueError,match='Duplicate GPU'):driver.slots(args,[{'id':m} for m in MODELS])


def test_incomplete_uuid_list_cannot_disable_known_duplicate_detection(tmp_path):
    args=arguments(tmp_path);args.placement=tmp_path/'placement.json'
    records=[{'model_id':m,'device':f'cuda:{i}','expected_gpu_uuid':f'GPU-{i}'} for i,m in enumerate(MODELS)]
    records[1]['expected_gpu_uuid']='GPU-0';records[2].pop('expected_gpu_uuid')
    write(args.placement,{'slots':records})
    with pytest.raises(ValueError):driver.slots(args,[{'id':m} for m in MODELS])


def complete_fixture(tmp_path):
    args=arguments(tmp_path);root=args.output_root;fp='isolated-unit-fixture-fingerprint'
    models=[{'id':m,'revision':str(i)*40} for i,m in enumerate(MODELS,1)]
    write(root/'model_lock.json',{'status':'complete','models':models})
    for model in models:
        write(root/f'status/model_scoring/{model["id"].replace("/","--")}_complete.json',
            {'status':'complete','model_revision':model['revision'],'completed_rows':36946,'candidate_continuations':357196})
    write(root/'status/assembly_complete.json',{'status':'complete','real_three_model_scoring_complete':True,
        'new_rotation_rows':110838,'new_averaged_vectors':11760,'merged_averaged_vectors':43120})
    write(root/'analysis/analysis_manifest.json',{'input_fingerprint':fp,'identity':{'synthetic_fixture':False,'scope':'complete_expanded_experiment'}})
    artifact=root/'analysis/fixture-proof.txt';artifact.write_text('isolated test coordinator evidence only')
    inventory={artifact.name:driver.sha(artifact)}
    write(root/'analysis/point_complete.json',{'status':'complete','input_fingerprint':fp,'point_fit_count':5840,'artifact_hashes':inventory})
    write(root/'analysis/primary_refit_complete.json',{'status':'complete','input_fingerprint':fp,'replicates':1000,'fits':100000,'artifact_hashes':inventory})
    write(root/'analysis/secondary_complete.json',{'status':'complete','input_fingerprint':fp,'replicates':2000,'artifact_hashes':inventory})
    write(root/'analysis/bootstrap_complete.json',{'status':'complete','input_fingerprint':fp})
    counts={'new_rotation_rows':110838,'merged_vectors':43120,'point_fits':5840,'per_question_rows':6540800,
        'primary_refit_replicates':1000,'primary_refit_fits':100000,'secondary_replicates':2000}
    write(root/'validation/numerical_replay.json',{'status':'passed','input_fingerprint':fp,'counts':counts})
    required=['REPORT.md','XfqA_RESPONSE_DRAFT.md','METHODS_INSERT.md','RESULTS_INSERT.md','LIMITATIONS_AND_DEVIATIONS.md','headline_evidence.json']
    required += [f'figures/{name}.{extension}' for name in ['large_target_coverage','large_pair_directionality','roster_expansion'] for extension in ['pdf','png']]
    report_artifacts=[]
    for name in required:
        p=root/'analysis'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('not a scientific result: isolated coordinator fixture')
        report_artifacts.append({'path':str(p.relative_to(root)),'sha256':driver.sha(p)})
    write(root/'status/report_complete.json',{'status':'complete','input_fingerprint':fp,'artifacts':report_artifacts})
    archive=root/'unit-fixture.zip';archive.write_bytes(b'archive verification is mocked by a coordinator fixture')
    write(root/'status/package_complete.json',{'status':'verified','input_fingerprint':fp,
        'numerical_replay_sha256':driver.sha(root/'validation/numerical_replay.json'),
        'report_complete_sha256':driver.sha(root/'status/report_complete.json'),
        'zip_crc_verified':True,'bundle_hashes_verified':True,'portable_replay_verified':True,
        'archive_path':str(archive),'archive_sha256':driver.sha(archive)})
    return args


def modify(path,**changes):
    value=json.loads(path.read_text());value.update(changes);write(path,value)


def test_finalization_is_idempotent_with_verified_complete_proofs(tmp_path):
    args=complete_fixture(tmp_path)
    first=driver.finalize(args);second=driver.finalize(args)
    assert first==second and first['status']=='complete'
    assert first['counts']['primary_refit_fits']==100000


@pytest.mark.parametrize('mutation',['synthetic','partial_scope','partial_scoring','partial_bootstrap','different_identity','tampered_report','tampered_analysis','unverified_package','tampered_archive'])
def test_finalization_rejects_partial_synthetic_or_changed_evidence(tmp_path,mutation):
    args=complete_fixture(tmp_path);root=args.output_root
    if mutation in ('synthetic','partial_scope'):
        p=root/'analysis/analysis_manifest.json';v=json.loads(p.read_text())
        if mutation=='synthetic':v['identity']['synthetic_fixture']=True
        else:v['identity']['scope']='baseline8_only'
        write(p,v)
    elif mutation=='partial_scoring':
        modify(root/f'status/model_scoring/{MODELS[0].replace("/","--")}_complete.json',completed_rows=36945)
    elif mutation=='partial_bootstrap':modify(root/'analysis/primary_refit_complete.json',replicates=999)
    elif mutation=='different_identity':modify(root/'analysis/secondary_complete.json',input_fingerprint='another-run')
    elif mutation=='tampered_report':(root/'analysis/XfqA_RESPONSE_DRAFT.md').write_text('changed after report verification')
    elif mutation=='tampered_analysis':(root/'analysis/fixture-proof.txt').write_text('changed after point computation')
    elif mutation=='unverified_package':modify(root/'status/package_complete.json',portable_replay_verified=False)
    elif mutation=='tampered_archive':(root/'unit-fixture.zip').write_bytes(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)):driver.finalize(args)
    assert not (root/'status/complete.json').exists()


def test_equals_phase_syntax_delegates_each_cpu_phase_without_recursing_all(tmp_path,monkeypatch):
    args=arguments(tmp_path);calls=[];direct=[];other=tmp_path/'analysis-venv/bin/python'
    monkeypatch.setattr(driver,'execute',lambda a,c,n:calls.append((c,n)))
    monkeypatch.setattr(driver,'run_phase',lambda a,p:direct.append(p))
    monkeypatch.setattr(driver,'finalize',lambda a:None)
    command=argv(args);index=command.index('--phase');command[index:index+2]=['--phase=all']
    assert driver.main(command+['--analysis-python',str(other)])==0
    assert direct==driver.PHASES[:6]
    assert len(calls)==5
    for (child,name),phase in zip(calls,driver.PHASES[6:]):
        assert name=='cpu-'+phase and child[0]==str(other)
        phase_values=[]
        for i,value in enumerate(child):
            if value=='--phase':phase_values.append(child[i+1])
            elif value.startswith('--phase='):phase_values.append(value.split('=',1)[1])
        assert phase_values==[phase]


def test_duplicate_gpu_identity_is_case_and_prefix_insensitive(tmp_path):
    args=arguments(tmp_path);args.placement=tmp_path/'placement.json'
    records=[{'model_id':m,'device':f'cuda:{i}','expected_gpu_uuid':f'GPU-abc{i}'} for i,m in enumerate(MODELS)]
    records[1]['expected_gpu_uuid']='ABC0'
    write(args.placement,{'slots':records})
    with pytest.raises(ValueError,match='Duplicate GPU'):driver.slots(args,[{'id':m} for m in MODELS])
