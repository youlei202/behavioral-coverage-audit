"""CPU-only acceptance/identity checks; constructed proof never authorizes GPU work."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

COMPONENT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(COMPONENT/'recovery'))
import score_endpoints as reference
import score_selected_endpoints as selected


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def proof_fixture(root):
    """A synthetic eight-model proof for validation tests, never scoring."""
    baseline=root/'baseline';out=root/'evidence';destination=root/'selected'
    roster=json.loads((REPO/'experiments/llm/analysis/v23/configs/resolved_models_v23.json').read_text())['models']
    models=[];artifacts={};identities={}
    for model in roster:
        name=reference.slug(model['id'])
        ident={'model_id':model['id'],'model_revision':model['revision'],'precision':'bfloat16',
               'candidate_definition':'summed_full_continuation_log_likelihood','model_adapter':model['model_adapter'],
               'model_inventory':{'revision':model['revision'],'files':{'synthetic':True}},
               'semantic_manifest_sha256':'manifest','rendered_prompt_sha256':'rendered',
               'source_sha256':{p:reference.file_sha256(REPO/p) for p in reference.SOURCE_FILES},
               'hardware':{'name':'synthetic B200','capability':[10,0],'total_memory_bytes':100,'torch_cuda_version':'fixture'},
               'versions':{'synthetic':'fixture'}}
        identity=hashlib.sha256(reference.canonical(ident)).hexdigest()
        ident={'input_fingerprint':identity,**ident};identities[model['id']]=ident
        identity_path=baseline/f'status/fresh_score_{name}_identity.json'
        write(identity_path,ident)
        ids=[f'{name}:pilot{i}' for i in range(42)]
        grouped=[ids[i:i+4] for i in range(0,42,4)]
        design_path=baseline/f'status/fresh_score_{name}_pilot_design.json'
        pilot_path=baseline/f'status/fresh_score_{name}_pilot.json'
        write(design_path,{'input_fingerprint':identity,'prompt_ids':ids,'outcome_blind':True,'row_count':42})
        write(pilot_path,{'input_fingerprint':identity,'status':'complete','pilot_measured_prompt_count':42,
                          'batches':[{'prompt_ids':g} for g in grouped]})
        for p in [identity_path,design_path,pilot_path]:artifacts[str(p)]=reference.file_sha256(p)
        summary={'model_id':model['id'],'model_revision':model['revision'],'status':'passed',
                 'exact_candidate_equality':True,'exact_baseline_repeat':True,'prompt_count':42}
        equal={'exact':True,'baseline_sha256':'a'*64,'candidate_sha256':'a'*64,'errors':[]}
        detail={**summary,'batches':[{'prompt_ids':g,'baseline_vs_candidate':equal,
                    'cached_vs_repeated_baseline':equal,
                    'timings':{b:{'prompt_count':len(g),'seconds':1.0} for b in ['baseline','candidate']}}
                    for g in grouped]}
        write(out/f'{name}.json',detail);models.append(summary)
    paths=[REPO/'experiments/llm/analysis/v23/src/pier_llm_v23/inference.py',
           COMPONENT/'recovery/score_endpoints.py',COMPONENT/'recovery/score_math.py',
           COMPONENT/'recovery/validate_score_math.py']
    proof={'status':'passed','strict_all_eight_models_passed':True,'baseline_artifacts_unchanged':True,
           'changed_artifacts':[],'models':models,'versions':{'synthetic':'fixture'},'gpu_name':'synthetic B200',
           'source_sha256':{str(p):reference.file_sha256(p) for p in paths},
           'baseline_artifact_sha256':artifacts}
    path=out/'summary.json';write(path,proof)
    return path,destination,identities


class SelectedDriverChecks(unittest.TestCase):
    def test_run_is_direct_baseline_copy_except_explicit_reviewed_backend_changes(self):
        def function(path,name):
            tree=ast.parse(path.read_text())
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        actual=function(COMPONENT/'recovery/score_selected_endpoints.py','run')
        original=function(COMPONENT/'recovery/score_endpoints.py','run')
        class StripExplicitBackend(ast.NodeTransformer):
            removed_acceptance=removed_identity=removed_load=changed_call=0
            def visit_Assign(self,node):
                names=[t.id for t in node.targets if isinstance(t,ast.Name)]
                if names==['acceptance'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='validate_acceptance_report':
                    self.removed_acceptance+=1;return None
                if names==['selected_backend'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load_selected_backend':
                    self.removed_load+=1;return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                call=node.value
                if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and isinstance(call.func.value,ast.Name) and call.func.value.id=='identity_record'
                    and call.func.attr=='update' and len(call.args)==1 and isinstance(call.args[0],ast.Call)
                    and isinstance(call.args[0].func,ast.Name) and call.args[0].func.id=='selected_backend_identity'):
                    self.removed_identity+=1;return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='selected_backend' and node.func.attr=='score_prompt_batch'):
                    self.changed_call+=1;node.func.value.id='scorer'
                return self.generic_visit(node)
        cleaner=StripExplicitBackend();actual=cleaner.visit(actual)
        self.assertEqual((cleaner.removed_acceptance,cleaner.removed_identity,cleaner.removed_load,cleaner.changed_call),(1,1,1,1))
        self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(original,include_attributes=False))

    def test_all_eight_exact_proof_binds_explicit_backend_and_code_hashes(self):
        with TemporaryDirectory() as directory:
            path,destination,identities=proof_fixture(Path(directory))
            accepted=selected.validate_acceptance_report(path,REPO,destination)
            model=next(iter(identities));record=deepcopy(identities[model])
            backend=selected.selected_backend_identity(accepted,record,SimpleNamespace(model_id=model))['scoring_backend']
            self.assertTrue(backend['all_eight_models_exact_gpu_proof'])
            self.assertEqual(backend['name'],'selected_positions')
            self.assertEqual(backend['acceptance_report_sha256'],reference.file_sha256(path))
            self.assertEqual(backend['selected_driver_sha256'],reference.file_sha256(COMPONENT/'recovery/score_selected_endpoints.py'))
            self.assertEqual(len(backend['acceptance_model_proofs']),8)
            self.assertNotIn('batch_size',backend)  # measured execution choice, not a scientific parameter

    def test_running_partial_and_failed_proofs_are_rejected(self):
        with TemporaryDirectory() as directory:
            path,destination,_=proof_fixture(Path(directory));original=json.loads(path.read_text())
            for change in [{'status':'running'}, {'strict_all_eight_models_passed':False},
                           {'baseline_artifacts_unchanged':False}, {'models':original['models'][:-1]}]:
                write(path,{**original,**change})
                with self.assertRaises(ValueError):selected.validate_acceptance_report(path,REPO,destination)

    def test_changed_code_and_changed_baseline_artifacts_are_rejected(self):
        with TemporaryDirectory() as directory:
            path,destination,_=proof_fixture(Path(directory));original=json.loads(path.read_text())
            changed=deepcopy(original);changed['source_sha256'][str(COMPONENT/'recovery/score_math.py')]='changed'
            write(path,changed)
            with self.assertRaisesRegex(ValueError,'source changed'):selected.validate_acceptance_report(path,REPO,destination)
            write(path,original)
            artifact=Path(next(iter(original['baseline_artifact_sha256'])))
            artifact.write_text(artifact.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Baseline artifact changed'):selected.validate_acceptance_report(path,REPO,destination)

    def test_missing_case_or_nonexact_complete_row_proof_is_rejected(self):
        with TemporaryDirectory() as directory:
            path,destination,_=proof_fixture(Path(directory))
            model=json.loads(path.read_text())['models'][0]['model_id']
            detail_path=path.parent/f'{reference.slug(model)}.json';original=json.loads(detail_path.read_text())
            bad=deepcopy(original);bad['batches']=bad['batches'][:-1];write(detail_path,bad)
            with self.assertRaisesRegex(ValueError,'every fixed pilot batch'):
                selected.validate_acceptance_report(path,REPO,destination)
            bad=deepcopy(original);bad['batches'][0]['baseline_vs_candidate']['candidate_sha256']='b'*64;write(detail_path,bad)
            with self.assertRaisesRegex(ValueError,'hashes disagree'):
                selected.validate_acceptance_report(path,REPO,destination)

    def test_baseline_root_and_write_symlink_escape_are_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);path,destination,_=proof_fixture(root)
            with self.assertRaisesRegex(ValueError,'disjoint new run root'):
                selected.validate_acceptance_report(path,REPO,root/'baseline')
            destination.mkdir();(destination/'status').symlink_to(root/'baseline/status',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'escapes its run root'):
                selected.validate_acceptance_report(path,REPO,destination)

    def test_tested_runtime_model_and_rendering_are_required(self):
        with TemporaryDirectory() as directory:
            path,destination,identities=proof_fixture(Path(directory))
            accepted=selected.validate_acceptance_report(path,REPO,destination)
            model=next(iter(identities));original=identities[model];args=SimpleNamespace(model_id=model)
            for key,value in [('model_revision','other'),('rendered_prompt_sha256','other'),
                              ('versions',{'other':'version'}),('hardware',{'name':'other'})]:
                record=deepcopy(original);record[key]=value
                with self.assertRaises(ValueError):selected.selected_backend_identity(accepted,record,args)


if __name__=='__main__':unittest.main()
