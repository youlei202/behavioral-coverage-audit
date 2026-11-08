"""Recovery integrity checks that do not require checkpoints or model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

RECOVERY = Path(__file__).resolve().parents[1]/'recovery/prepare_models.py'
spec=importlib.util.spec_from_file_location('prepare_models', RECOVERY)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryModelChecks(unittest.TestCase):
    def test_tokenizer_metadata_cannot_replace_complete_weight_state(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            record={'id':'maker/model','revision':'revision'}
            snapshot=root/'snapshots/revision'
            snapshot.mkdir(parents=True)
            full=module.download_status_path(root,record['id'],False)
            tokenizer=module.download_status_path(root,record['id'],True)
            self.assertNotEqual(full,tokenizer)
            state={'model_id':record['id'],'revision':'revision','snapshot':str(snapshot),'tokenizer_only':True}
            module.dump(tokenizer,state)
            self.assertEqual(module.load_download_state(root,record),state)
            complete={**state,'tokenizer_only':False}
            module.dump(full,complete)
            self.assertEqual(module.load_download_state(root,record),complete)
            module.dump(tokenizer,{**state,'note':'separate refresh'})
            self.assertEqual(json.loads(full.read_text()),complete)

    def test_state_rejects_wrong_revision(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            record={'id':'maker/model','revision':'wanted'}
            module.dump(module.download_status_path(root,record['id'],True),
                        {'model_id':'maker/model','revision':'wrong','snapshot':str(root)})
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                module.load_download_state(root,record)

    def test_granite_date_override_and_wrapper_hash(self):
        message='Current Date: August 28, 2026'
        expected=hashlib.sha256(message.encode()).hexdigest()
        class Tokenizer:
            chat_template='{{ strftime_now("%B %d, %Y") }}'
            def apply_chat_template(self,messages,**kwargs):
                self.date=kwargs['strftime_now']('%B %d, %Y')
                return '<|start_of_role|>system<|end_of_role|>Current Date: '+self.date+'<|end_of_text|>user'
        tokenizer=Tokenizer()
        self.assertEqual(module.frozen_system_message(tokenizer,{'v23_frozen_dynamic_system_message_sha256':expected}),message)
        self.assertEqual(tokenizer.date,'August 28, 2026')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            module.frozen_system_message(tokenizer,{'v23_frozen_dynamic_system_message_sha256':'incorrect'})

    def test_cached_rendering_requires_full_hash_count_and_token_length(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'rendered.jsonl'
            path.write_text(json.dumps({'prompt_id':'q','rendered_prompt':'text','prompt_token_count':3})+'\n')
            record={'id':'model','revision':'rev','v23_rendered_prompt_sha256':module.sha(path),
                    'v23_rendered_prompt_count':1,'v23_maximum_prompt_tokens':3,
                    'v23_frozen_dynamic_system_message_sha256':None}
            state=module.cached_rendering_status(path,record,'snapshot')
            self.assertTrue(state['historical_rendered_bytes_verified'])
            self.assertTrue(state['reused'])
            with self.assertRaisesRegex(ValueError,'token length'):
                module.cached_rendering_status(path,{**record,'v23_maximum_prompt_tokens':4},'snapshot')
            path.write_text(path.read_text().replace('text','changed'))
            with self.assertRaisesRegex(ValueError,'historical digest'):
                module.cached_rendering_status(path,record,'snapshot')

    def test_download_filters_weights_and_verifies_each_file(self):
        with TemporaryDirectory() as directory:
            repo=Path(directory)/'repo';root=Path(directory)/'run';cache=Path(directory)/'cache'
            snapshot=cache/'snapshots/rev';snapshot.mkdir(parents=True)
            (snapshot/'tokenizer.json').write_text('original-tokenizer')
            payload=(snapshot/'tokenizer.json').read_bytes()
            inventory=[{'path':'tokenizer.json','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()},
                       {'path':'model.safetensors','bytes':99,'sha256':'not-downloaded'}]
            manifest=repo/'experiments/llm/manifests/v2/data/manifests/models/maker--model.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'model_id':'maker/model','revision':'rev','file_inventory':inventory}))
            calls=[]
            def snapshot_download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)
            hub=types.SimpleNamespace(snapshot_download=snapshot_download)
            with patch.dict('sys.modules',{'huggingface_hub':hub}):
                module.download(repo,root,cache,{'id':'maker/model','revision':'rev'},True)
                self.assertEqual(calls[0]['allow_patterns'],['tokenizer.json'])
                self.assertTrue(module.download_status_path(root,'maker/model',True).is_file())
                self.assertFalse(module.download_status_path(root,'maker/model',False).exists())
                (snapshot/'tokenizer.json').write_text('changed')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    module.download(repo,root,cache,{'id':'maker/model','revision':'rev'},True)


if __name__=='__main__':unittest.main()
