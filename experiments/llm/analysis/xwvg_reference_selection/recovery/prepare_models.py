"""Download pinned original files and reproduce historical model prompt wrappers."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n'); tmp.replace(path)


def download_status_path(root, model_id, tokenizer_only=False):
    category = 'model_tokenizers' if tokenizer_only else 'model_downloads'
    return Path(root)/'status'/category/f'{model_id.replace("/", "--")}.json'


def load_download_state(root, record):
    # Tokenizer metadata is sufficient for CPU rendering and must never overwrite
    # or masquerade as the complete weight-download status.
    for tokenizer_only in (False, True):
        path = download_status_path(root, record['id'], tokenizer_only)
        if path.is_file():
            state = json.loads(path.read_text())
            if state['model_id'] != record['id'] or state['revision'] != record['revision']:
                raise ValueError(f'Download-state model identity mismatch: {path}')
            snapshot = Path(state['snapshot'])
            if snapshot.name != record['revision'] or not snapshot.is_dir():
                raise ValueError(f'Download-state snapshot mismatch: {path}')
            return state
    raise FileNotFoundError(f'No verified model or tokenizer snapshot for {record["id"]}')


def records(repo,model_id):
    rows=json.loads((repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json').read_text())['models']
    return [r for r in rows if model_id is None or r['id']==model_id]


def download(repo,root,cache,record,tokenizer_only=False):
    from huggingface_hub import snapshot_download
    slug=record['id'].replace('/','--')
    compatibility=json.loads((repo/'experiments/llm/manifests/v2/data/manifests/models'/f'{slug}.json').read_text())
    if compatibility['model_id'] != record['id'] or compatibility['revision'] != record['revision']:
        raise ValueError('Historical compatibility manifest model identity mismatch')
    inventory=compatibility['file_inventory']
    # Mistral's consolidated.safetensors is an alternative format, not read by
    # the pinned transformers sharded loader. Keep only the loaded weight set.
    selected=[x for x in inventory if x['path']!='consolidated.safetensors' and
              not (tokenizer_only and x['path'].endswith('.safetensors'))]
    start=time.perf_counter()
    snapshot=Path(snapshot_download(repo_id=record['id'],revision=record['revision'],cache_dir=str(cache),
                                  allow_patterns=[x['path'] for x in selected],max_workers=4))
    if snapshot.name!=record['revision']: raise ValueError('Resolved snapshot revision differs')
    checks=[]
    for item in selected:
        path=snapshot/item['path']
        actual=sha(path)
        if path.stat().st_size!=item['bytes'] or actual!=item['sha256']:
            raise ValueError(f'Historical model-file identity mismatch: {record["id"]}/{item["path"]}')
        checks.append({'path':item['path'],'bytes':item['bytes'],'sha256':actual})
    result={'model_id':record['id'],'revision':record['revision'],'snapshot':str(snapshot),
            'historical_file_identity_verified':True,'tokenizer_only':tokenizer_only,'files':checks,
            'seconds':time.perf_counter()-start,'unused_alternative_weight_files_omitted':['consolidated.safetensors'] if any(x['path']=='consolidated.safetensors' for x in inventory) else []}
    dump(download_status_path(root, record['id'], tokenizer_only),result)
    print(json.dumps({'model_id':record['id'],'status':'download_verified','snapshot':str(snapshot),'seconds':result['seconds']}),flush=True)
    return result


def frozen_system_message(tokenizer, record):
    """Freeze Granite's historical date before extracting its original wrapper."""
    template = str(getattr(tokenizer, 'chat_template', '') or '')
    if 'strftime_now' not in template:
        if record.get('v23_frozen_dynamic_system_message_sha256') is not None:
            raise ValueError('Expected historical dynamic wrapper is absent')
        return None
    sample = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'placeholder'}], tokenize=False,
        add_generation_prompt=True,
        strftime_now=lambda fmt: datetime(2026, 8, 28).strftime(fmt))
    prefix = '<|start_of_role|>system<|end_of_role|>'
    suffix = '<|end_of_text|>'
    if prefix not in sample or suffix not in sample.split(prefix, 1)[1]:
        raise ValueError('Historical Granite system-wrapper delimiters are absent')
    frozen = sample.split(prefix, 1)[1].split(suffix, 1)[0]
    if hashlib.sha256(frozen.encode()).hexdigest() != record['v23_frozen_dynamic_system_message_sha256']:
        raise ValueError('Frozen historical Granite system-message hash mismatch')
    return frozen


def cached_rendering_status(destination, record, snapshot):
    if sha(destination) != record['v23_rendered_prompt_sha256']:
        raise ValueError('Existing rendered cache differs from historical digest')
    count = maximum = 0
    with Path(destination).open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            maximum = max(maximum, row['prompt_token_count'])
    if count != record['v23_rendered_prompt_count'] or maximum != record['v23_maximum_prompt_tokens']:
        raise ValueError('Existing rendered cache count/maximum token length differs')
    return {'model_id': record['id'], 'revision': record['revision'], 'path': str(destination),
            'sha256': record['v23_rendered_prompt_sha256'], 'historical_rendered_bytes_verified': True,
            'prompt_count': count, 'maximum_prompt_tokens': maximum,
            'frozen_system_sha256': record.get('v23_frozen_dynamic_system_message_sha256'),
            'seconds': 0.0, 'snapshot': str(snapshot), 'reused': True}


def render(repo,root,record,snapshot):
    sys.path.insert(0,str(repo/'experiments/llm/analysis/v23/src'))
    from pier_llm_v23.prepare import _load_tokenizer,_render_prompt,_candidate_manifest
    from pier_llm_v23.utils import atomic_write_jsonl,read_jsonl,sha256_text
    slug=record['id'].replace('/','--')
    source=root/'data/permutations/score_semantic_manifest.jsonl'
    destination=root/'data/permutations/rendered'/f'{slug}.jsonl'
    expected=record['v23_rendered_prompt_sha256']
    if destination.exists():
        result = cached_rendering_status(destination, record, snapshot)
        status_path = root/'status/model_rendering'/f'{slug}.json'
        if status_path.is_file():
            previous = json.loads(status_path.read_text())
            if previous.get('sha256') == expected and previous.get('revision') == record['revision']:
                result['seconds'] = previous.get('seconds', 0.0)
        dump(status_path, result)
        print(json.dumps({'model_id': record['id'], 'status': 'existing_historical_match'}), flush=True)
        return result
    tokenizer=_load_tokenizer(snapshot)
    template=str(getattr(tokenizer,'chat_template','') or '')
    if sha256_text(template)!=record['v23_chat_template_sha256']: raise ValueError('Chat template mismatch')
    for count,old in record['v23_candidate_sequences_by_option_count'].items():
        if _candidate_manifest(tokenizer,int(count))!=old: raise ValueError('Candidate label tokens mismatch')
    frozen=frozen_system_message(tokenizer, record)
    rows=[]; maximum=0; start=time.perf_counter()
    for prompt in read_jsonl(source):
        text,tokens=_render_prompt(tokenizer,prompt['canonical_content'],frozen)
        maximum=max(maximum,len(tokens))
        rows.append({'schema_version':'pier_v23_rendered_prompt_v1','prompt_id':prompt['prompt_id'],
                     'canonical_content_sha256':prompt['canonical_content_sha256'],
                     'rendered_prompt':text,'rendered_prompt_sha256':sha256_text(text),'prompt_token_count':len(tokens)})
    if len(rows)!=record['v23_rendered_prompt_count']: raise ValueError('Rendered prompt count mismatch')
    if maximum!=record['v23_maximum_prompt_tokens']: raise ValueError('Rendered maximum token count mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.checking.jsonl')
    atomic_write_jsonl(temporary,rows)
    actual=sha(temporary)
    if actual!=expected:
        raise ValueError(f'Rendered bytes differ from historical wrapper: {record["id"]}; actual={actual}, expected={expected}. Staged file retained.')
    temporary.replace(destination)
    result={'model_id':record['id'],'revision':record['revision'],'path':str(destination),'sha256':actual,
            'historical_rendered_bytes_verified':True,'prompt_count':len(rows),'maximum_prompt_tokens':maximum,
            'frozen_system_sha256':sha256_text(frozen) if frozen is not None else None,
            'seconds':time.perf_counter()-start,'snapshot':str(snapshot)}
    dump(root/'status/model_rendering'/f'{slug}.json',result)
    print(json.dumps({'model_id':record['id'],'status':'rendered_historical_match','seconds':result['seconds']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['download','render','all'])
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--model-id')
    parser.add_argument('--tokenizer-only',action='store_true')
    args=parser.parse_args()
    repo=args.source_repo.resolve();root=args.run_root.resolve();cache=args.cache_dir.resolve()
    rows=records(repo,args.model_id)
    if not rows: raise ValueError('Model not in fixed roster')
    failures=[]
    for record in rows:
        try:
            if args.stage in ['download','all']:
                state=download(repo,root,cache,record,args.tokenizer_only)
            else:
                state=load_download_state(root,record)
            if args.stage in ['render','all']:render(repo,root,record,Path(state['snapshot']))
        except Exception as exc:
            error={'model_id':record['id'],'error_type':type(exc).__name__,'error':str(exc)}
            failures.append(error);print(json.dumps(error),flush=True)
    if failures:
        dump(root/'status'/f'model_{args.stage}{"_tokenizer_only" if args.tokenizer_only else ""}_failures.json',failures)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
