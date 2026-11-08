#!/usr/bin/env python3
"""Pinned XfqA inputs, native rendering, original scoring and complete assembly.

Only this new component writes into the XfqA run. Historical implementations
are imported read-only; their fixed eight-model orchestration is never invoked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

EXECUTION_SHA = '183b4fb684cec5d44224ab385c87de3904d97b239b4590608edaff2f5288542b'
SEMANTIC_SHA = '872051ccdfa272e1e5c90ea4d7257e3c6572bbe2565dd1a7883940c34f931048'
EXPECTED_SOURCES = {
    'experiments/llm/analysis/v23/src/pier_llm_v23/inference.py': '0bcd17ba4b3939f257c16e35b78d7ce03239b10442ca761fa3bc225fa2271a09',
    'experiments/llm/analysis/v23/src/pier_llm_v23/prepare.py': '0f5b8eabcdf9e19f86beeae22c1cdb04f0e35f47ae24e9fb283796688f265803',
    'experiments/llm/inference/v2/src/pier_llm/data_prep.py': '1f8f86b978a1009f350251901005b47b4ff92b741b629932f6197adae28d7bac',
}
PROBABILITY_COLUMN = 'permutation_averaged_semantic_probabilities'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def slug(model_id):
    return model_id.replace('/', '--')


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if path.exists() and immutable:
        if load(path) != value:
            raise ValueError(f'Immutable record differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_parquet(path, frame, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    frame.to_parquet(tmp, index=False)
    if immutable and path.exists():
        if sha(path) != sha(tmp):
            raise ValueError(f'Existing parquet differs: {path}')
        tmp.unlink()
    else:
        with tmp.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def copy_exact(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha(source) != expected:
        raise ValueError(f'Unexpected source bytes: {source}')
    if destination.exists():
        if sha(destination) != expected:
            raise ValueError(f'Existing staged input differs: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
        shutil.copyfile(source, tmp)
        if sha(tmp) != expected:
            raise ValueError('Input copy checksum failure')
        os.replace(tmp, destination)
        destination.chmod(0o444)
    return {'path': str(destination), 'sha256': expected, 'bytes': destination.stat().st_size}


def original(repo):
    repo = Path(repo).resolve()
    path = repo / 'experiments/llm/analysis/xwvg_reference_selection/recovery/score_endpoints.py'
    name = '_xfqa_original_score_endpoints'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError('Original helper imported from a different repository')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_roots(args):
    args.repo_root = Path(args.repo_root).expanduser().resolve()
    args.plan_root = Path(args.plan_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    allowed = args.repo_root / 'outputs/xfqa_large_models'
    if args.output_root == allowed or not args.output_root.is_relative_to(allowed):
        raise ValueError('New measurements must use outputs/xfqa_large_models/<run_id>')
    for name, expected in EXPECTED_SOURCES.items():
        if sha(args.repo_root / name) != expected:
            raise ValueError(f'Historical source differs; review before scoring: {name}')
    return load(args.plan_root / 'config/experiment.json')


def prepare(args):
    config = validate_roots(args)
    root, plan, repo = args.output_root, args.plan_root, args.repo_root
    expected = load(plan / 'evidence/seed_provenance.json')['inputs']
    inputs = {name: copy_exact(plan / 'seed_inputs' / name, root / 'inputs' / name, record['sha256'])
              for name, record in expected.items()}
    existing = repo / 'outputs/xwvg_reference_selection/20261107_recovery/fresh_endpoints/data/permutations'
    supplied = getattr(args, 'execution_manifest', None)
    execution = Path(supplied) if supplied else existing / 'score_execution_manifest.jsonl'
    if not execution.is_file():
        raise FileNotFoundError(f'Exact full semantic execution manifest required: {execution}; regenerate the complete pinned V2/V2.3 design, not just the 560 questions')
    execution_info = copy_exact(execution, root / 'data/permutations/score_execution_manifest.jsonl', EXECUTION_SHA)
    semantic = execution.with_name('score_semantic_manifest.jsonl')
    semantic_info = copy_exact(semantic, root / 'data/permutations/score_semantic_manifest.jsonl', SEMANTIC_SHA)
    rows = read_rows(execution)
    # The original validator checks all metadata and cyclic groups. A synthetic
    # identity rendering is used only to validate the model-independent design.
    rendered = [{'prompt_id': r['prompt_id'], 'canonical_content_sha256': r['canonical_content_sha256'],
                 'rendered_prompt': r['canonical_content'], 'rendered_prompt_sha256': r['canonical_content_sha256'],
                 'prompt_token_count': 1} for r in rows]
    design = original(repo).validate_design(rows, rendered)
    if any(not r.get('inference_required', True) or r.get('reuse_rotation0', False) for r in rows):
        raise ValueError('Execution manifest must score all rotations without reuse')
    result = {'status': 'complete', 'inputs': inputs, 'execution_manifest': execution_info,
              'semantic_manifest': semantic_info, 'design': design,
              'historical_source_hashes': EXPECTED_SOURCES,
              'protocol_sha256': sha(plan / 'config/experiment.json')}
    atomic_json(root / 'status/prepared_inputs.json', result, immutable=True)
    return result


def resolve(args):
    validate_roots(args)
    command = [sys.executable, str(args.plan_root / 'tools/resolve_model_revisions.py'),
               '--config', str(args.plan_root / 'config/experiment.json'),
               '--output', str(args.output_root / 'model_lock.json')]
    if (args.output_root / 'model_lock.json').exists():
        command.append('--resume-missing')
    subprocess.run(command, check=True)
    return load(args.output_root / 'model_lock.json')


def model_record(args):
    config = validate_roots(args)
    lock = load(args.output_root / 'model_lock.json')
    if lock['protocol_sha256'] != sha(args.plan_root / 'config/experiment.json'):
        raise ValueError('Pinned model lock belongs to another protocol')
    ids = [r['id'] for r in config['new_models']]
    if args.model_id not in ids:
        raise ValueError('Only the three prescribed new model IDs may be scored')
    records = [r for r in lock['models'] if r['id'] == args.model_id]
    if len(records) != 1:
        raise ValueError(f'New model revision is not locked: {args.model_id}')
    return records[0]


def state_path(args, kind):
    return args.output_root / 'status' / kind / f'{slug(args.model_id)}.json'


def selected_asset_names(names, weight_index=None):
    names = set(names)
    assets = {n for n in names if '/' not in n and (
        n.endswith('.json') or n.endswith('.jinja') or n in ('tokenizer.model', 'spiece.model', 'merges.txt', 'vocab.txt'))}
    # Index alternatives and unused serialization formats are not downloaded.
    assets = {n for n in assets if not n.endswith('.index.json')}
    if weight_index is not None:
        assets.add('model.safetensors.index.json')
        assets.update(weight_index['weight_map'].values())
    elif 'model.safetensors' in names:
        assets.add('model.safetensors')
    else:
        raise ValueError('Official checkpoint has no supported complete safetensors weight set')
    if not assets <= names:
        raise ValueError(f'Checkpoint index references missing assets: {sorted(assets-names)}')
    if 'config.json' not in assets or 'tokenizer_config.json' not in assets:
        raise ValueError('Model config and native tokenizer config are required')
    return sorted(assets)


def physical_stamp(path):
    st = Path(path).stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino, 'device': st.st_dev}


def verified_snapshot(args, *, full_hash=False):
    record = model_record(args)
    status = load(state_path(args, 'model_downloads'))
    if status['model_id'] != record['id'] or status['revision'] != record['revision']:
        raise ValueError('Snapshot model/revision differs from lock')
    path = Path(status['snapshot'])
    if path.name != record['revision']:
        raise ValueError('Snapshot directory must end in the pinned commit')
    for entry in status['files']:
        asset = path / entry['path']
        # Full hashes are checked on download. Subsequent local reads may reuse
        # that check only while the physical inode/size/mtime identity is intact.
        if full_hash or physical_stamp(asset) != entry['physical_stamp']:
            if sha(asset) != entry['sha256']:
                raise ValueError(f'Pinned model asset changed: {entry["path"]}')
    return path, status


def download(args):
    record = model_record(args)
    destination = state_path(args, 'model_downloads')
    if destination.exists():
        snapshot, state = verified_snapshot(args)
        return state
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    cache = Path(getattr(args, 'model_cache', None) or args.output_root / 'cache/models')
    api = HfApi()
    info = api.model_info(record['id'], revision=record['revision'], files_metadata=True)
    if str(info.sha) != record['revision']:
        raise ValueError('HF model metadata moved away from the frozen commit')
    remote = {s.rfilename: s for s in info.siblings}
    index = None
    if 'model.safetensors.index.json' in remote:
        index = load(hf_hub_download(record['id'], 'model.safetensors.index.json',
                                    revision=record['revision'], cache_dir=str(cache)))
    selected = selected_asset_names(remote, index)
    before = time.perf_counter()
    snapshot = Path(snapshot_download(record['id'], revision=record['revision'], cache_dir=str(cache),
                                     allow_patterns=selected, max_workers=4))
    if snapshot.name != record['revision']:
        raise ValueError('Downloaded snapshot differs from pinned revision')
    config = load(snapshot / 'config.json')
    if config.get('model_type') != record['expected_model_type']:
        raise ValueError('Unexpected official model type')
    files = []
    for name in selected:
        asset = snapshot / name
        actual = sha(asset)
        metadata = remote[name]
        lfs = getattr(metadata, 'lfs', None)
        upstream_sha = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        if upstream_sha and actual != upstream_sha:
            raise ValueError(f'Official LFS digest mismatch: {name}')
        if getattr(metadata, 'size', None) is not None and asset.stat().st_size != metadata.size:
            raise ValueError(f'Official file size mismatch: {name}')
        if not upstream_sha and getattr(metadata, 'blob_id', None):
            content = asset.read_bytes()
            git_sha = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            if git_sha != metadata.blob_id:
                raise ValueError(f'Official Git blob mismatch: {name}')
        files.append({'path': name, 'sha256': actual, 'bytes': asset.stat().st_size,
                      'upstream_lfs_sha256': upstream_sha, 'upstream_git_blob': getattr(metadata, 'blob_id', None),
                      'physical_stamp': physical_stamp(asset)})
    state = {'status': 'complete', 'model_id': record['id'], 'revision': record['revision'],
             'snapshot': str(snapshot), 'files': files, 'seconds': time.perf_counter()-before,
             'downloaded_official_snapshot': True, 'quantized': False,
             'model_lock_sha256': sha(args.output_root / 'model_lock.json'),
             'weight_bytes': sum(r['bytes'] for r in files if r['path'].endswith('.safetensors'))}
    atomic_json(destination, state, immutable=True)
    return state


def render(args):
    record = model_record(args)
    snapshot, inventory = verified_snapshot(args)
    helper = original(args.repo_root)
    helper.import_historical_scorer(args.repo_root)
    from pier_llm_v23.prepare import _load_tokenizer, _render_prompt, _candidate_manifest
    prompts = read_rows(args.output_root / 'data/permutations/score_execution_manifest.jsonl')
    dest = args.output_root / 'data/permutations/rendered' / f'{slug(record["id"])}.jsonl'
    report_path = state_path(args, 'model_rendering')
    if report_path.exists():
        state = load(report_path)
        if state['revision'] != record['revision'] or state['sha256'] != sha(dest):
            raise ValueError('Rendered cache identity differs')
        return state
    if dest.exists():
        raise ValueError('Rendered file has no immutable completed report; inspect interrupted preparation')
    tokenizer = _load_tokenizer(snapshot)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError('Pinned native chat template must be a nonempty string')
    candidates = {n: _candidate_manifest(tokenizer, n) for n in sorted({int(r['option_count']) for r in prompts})}
    examples = {}
    rendered = []
    start = time.perf_counter()
    for row in prompts:
        text, tokens = _render_prompt(tokenizer, row['canonical_content'])
        prefix, suffix = text.split(row['canonical_content'], 1)
        if record['id'] == 'Qwen/QwQ-32B' and '<think>' not in suffix:
            raise ValueError('QwQ native generation prefix no longer contains its reasoning opening')
        if record['id'] == 'Qwen/QwQ-32B' and '</think>' in suffix:
            raise ValueError('QwQ prefix unexpectedly closes reasoning before the fixed candidate boundary')
        item = {'schema_version': 'pier_v23_rendered_prompt_v1', 'prompt_id': row['prompt_id'],
                'canonical_content_sha256': row['canonical_content_sha256'], 'rendered_prompt': text,
                'rendered_prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompt_token_count': len(tokens)}
        rendered.append(item)
        key = f'{row["semantic_condition"]}/{row["option_count"]}'
        if key not in examples:
            examples[key] = {'prompt_id': row['prompt_id'], 'option_count': row['option_count'],
                             'semantic_condition': row['semantic_condition'], 'native_prefix': prefix,
                             'native_suffix': suffix, 'rendered_sha256': item['rendered_prompt_sha256'],
                             'candidate_boundary_samples': [suffix + c['continuation'] for c in candidates[int(row['option_count'])]],
                             'prompt_token_count': len(tokens)}
    design = helper.validate_design(prompts, rendered)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        for row in rendered:
            handle.write(canonical(row) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dest)
    state = {'status': 'complete', 'model_id': record['id'], 'revision': record['revision'],
             'path': str(dest), 'sha256': sha(dest), 'snapshot': str(snapshot),
             'native_chat_template': template, 'native_chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
             'candidate_sequences_by_option_count': candidates, 'boundary_examples': examples,
             'native_reasoning_prefix_preserved': record['id'] == 'Qwen/QwQ-32B',
             'rendered_prompt_count': len(rendered), 'maximum_prompt_tokens': max(r['prompt_token_count'] for r in rendered),
             'design': design, 'seconds': time.perf_counter()-start,
             'model_inventory_sha256': sha(state_path(args, 'model_downloads')),
             'execution_manifest_sha256': EXECUTION_SHA}
    atomic_json(report_path, state, immutable=True)
    return state


@contextmanager
def exclusive_model_lock(root, model_id):
    import fcntl
    path = Path(root) / 'status/locks' / f'{slug(model_id)}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another XfqA worker owns this model: {model_id}') from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'hostname': platform.node(), 'model_id': model_id}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def maximum_workload(prompts, rendered, candidates):
    def workload(row):
        n = int(row['option_count'])
        lengths = candidates[str(n)] if str(n) in candidates else candidates[n]
        return n * (int(rendered[row['prompt_id']]['prompt_token_count']) + max(c['token_count'] for c in lengths))
    row = min(prompts, key=lambda r: (-workload(r), r['prompt_id']))
    return {'prompt_id': row['prompt_id'], 'candidate_expanded_tokens': workload(row),
            'formula': 'option_count * (prompt_token_count + max_candidate_token_count)',
            'outcome_blind': True, 'tie_break': 'lexicographically first prompt_id'}


class ResourceWatchdog:
    """Watch actual free resources and terminate only this owned worker."""
    def __init__(self, observe, gpu_reserve, cpu_reserve, failure_path, *, terminate=os._exit, interval=2):
        self.observe, self.gpu_reserve, self.cpu_reserve = observe, gpu_reserve, cpu_reserve
        self.failure_path, self.terminate, self.interval = failure_path, terminate, interval
        self.stop_event = threading.Event()
        self.thread = None
        self.minimum_gpu_free = None
        self.minimum_cpu_free = None

    def check(self):
        values = self.observe()
        self.minimum_gpu_free = min(self.minimum_gpu_free or values['gpu_free_bytes'], values['gpu_free_bytes'])
        self.minimum_cpu_free = min(self.minimum_cpu_free or values['cpu_available_bytes'], values['cpu_available_bytes'])
        if values['gpu_free_bytes'] < self.gpu_reserve or values['cpu_available_bytes'] < self.cpu_reserve:
            raise RuntimeError(f'XfqA resource reserve breached: {values}')
        return values

    def start(self):
        self.check()
        def monitor():
            while not self.stop_event.wait(self.interval):
                try:
                    self.check()
                except Exception as exc:
                    atomic_json(self.failure_path, {'status': 'failed', 'reason': str(exc),
                                'pid': os.getpid(), 'hostname': platform.node(),
                                'gpu_reserve_bytes': self.gpu_reserve, 'cpu_reserve_bytes': self.cpu_reserve})
                    self.terminate(86)
                    return
        self.thread = threading.Thread(target=monitor, name='xfqa-resource-reserve', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


def compare_repeat(reference, repeated, thresholds):
    import numpy as np
    if [r['prompt_id'] for r in reference] != [r['prompt_id'] for r in repeated]:
        raise ValueError('Repeatability comparison prompt identities differ')
    max_ll = max_prob = 0.0
    tv = []
    mismatches = 0
    for a, b in zip(reference, repeated):
        ll_a, ll_b = (np.asarray(r['candidate_sequence_log_likelihoods']) for r in (a, b))
        p_a, p_b = (np.asarray(r['semantic_remapped_probabilities']) for r in (a, b))
        max_ll = max(max_ll, float(np.max(np.abs(ll_a-ll_b))))
        max_prob = max(max_prob, float(np.max(np.abs(p_a-p_b))))
        tv.append(float(np.abs(p_a-p_b).sum()/2))
        mismatches += int(np.argmax(p_a) != np.argmax(p_b))
    result = {'rows': len(reference), 'max_abs_continuation_loglikelihood_difference': max_ll,
              'max_abs_candidate_probability_difference': max_prob, 'mean_pilot_tv_difference': float(np.mean(tv)),
              'top_option_mismatches': mismatches, 'bitwise_equal': max_ll == max_prob == 0}
    result['passed'] = all(result[k] <= thresholds[k] for k in (
        'max_abs_continuation_loglikelihood_difference', 'max_abs_candidate_probability_difference',
        'mean_pilot_tv_difference', 'top_option_mismatches'))
    return result


def model_status(args, kind):
    return args.output_root / 'status/model_scoring' / f'{slug(args.model_id)}_{kind}.json'


def normalized_gpu_uuid(value):
    value = str(value).lower()
    if value.startswith('gpu-'):
        value = value[4:]
    return 'GPU-' + value


def score(args):
    """One original BF16 model on one device; pilot rows are reused verbatim."""
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    config = validate_roots(args)
    record = model_record(args)
    lock = load(args.output_root / 'model_lock.json')
    if lock.get('status') != 'complete' or len(lock['models']) != 3:
        raise ValueError('Freeze all three revisions before any new-model scoring')
    if int(args.batch_size) != 1:
        raise ValueError('This measured original-scoring route is fixed at prompt batch size one')
    if args.gpu_reserve_gib < 1 or args.cpu_reserve_gib < 1:
        raise ValueError('Concrete positive GPU and CPU memory reserves are required')
    snapshot, download_state = verified_snapshot(args)
    helper = original(args.repo_root)
    manifest_path = args.output_root / 'data/permutations/score_execution_manifest.jsonl'
    if sha(manifest_path) != EXECUTION_SHA:
        raise ValueError('Execution manifest changed')
    rendering = load(state_path(args, 'model_rendering'))
    rendered_path = args.output_root / 'data/permutations/rendered' / f'{slug(args.model_id)}.jsonl'
    if rendering['revision'] != record['revision'] or sha(rendered_path) != rendering['sha256']:
        raise ValueError('Frozen new-model rendering changed')
    prompts, rendered_rows = read_rows(manifest_path), read_rows(rendered_path)
    design = helper.validate_design(prompts, rendered_rows)
    by_id = {r['prompt_id']: r for r in prompts}
    rendered = {r['prompt_id']: r for r in rendered_rows}
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('An explicitly selected CUDA device is required for real scoring')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    raw_uuid = str(getattr(properties, 'uuid', ''))
    if not raw_uuid:
        raise RuntimeError('CUDA device UUID is required to prevent ambiguous placement')
    gpu_uuid = normalized_gpu_uuid(raw_uuid)
    expected_uuid = getattr(args, 'expected_gpu_uuid', None)
    if expected_uuid and gpu_uuid != normalized_gpu_uuid(expected_uuid):
        raise ValueError(f'Selected GPU differs from the assigned slot: {gpu_uuid}')
    hardware = {'hostname': platform.node(), 'gpu_name': properties.name, 'gpu_uuid': gpu_uuid,
                'device': str(device), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'total_memory_bytes': properties.total_memory, 'capability': [properties.major, properties.minor],
                'torch_cuda_version': torch.version.cuda,
                'driver_version': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]}
    source_names = set(EXPECTED_SOURCES) | set(helper.SOURCE_FILES)
    identity_record = {'schema': 'xfqa_full_continuation_scoring_v1', 'model_id': record['id'],
        'model_revision': record['revision'], 'model_lock_sha256': sha(args.output_root / 'model_lock.json'),
        'protocol_sha256': sha(args.plan_root / 'config/experiment.json'), 'precision': 'bfloat16',
        'candidate_definition': 'summed_full_continuation_log_likelihood', 'prompt_batch_size': 1,
        'semantic_manifest_sha256': EXECUTION_SHA, 'rendered_prompt_sha256': rendering['sha256'],
        'native_chat_template_sha256': rendering['native_chat_template_sha256'],
        'candidate_sequences_by_option_count': rendering['candidate_sequences_by_option_count'],
        'model_inventory_sha256': sha(state_path(args, 'model_downloads')),
        'model_inventory': {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in download_state['files']},
        'versions': helper.runtime_versions(), 'hardware': hardware,
        'source_sha256': {name: sha(args.repo_root / name) for name in sorted(source_names)},
        'original_driver_sha256': sha(Path(helper.__file__)), 'new_adapter_sha256': sha(Path(__file__)),
        'gpu_reserve_bytes': int(args.gpu_reserve_gib*1024**3), 'cpu_reserve_bytes': int(args.cpu_reserve_gib*1024**3),
        'measurement_origin': 'fresh_endpoint_scoring', 'all_rotations_scored': True,
        'free_text_generation': False, 'quantization': False, 'truncation': False,
        'model_adapter': 'image_text_to_text' if record['expected_model_type'] == 'gemma3' else 'causal_lm'}
    identity = digest(identity_record)
    identity_path = model_status(args, 'identity')
    directory = args.output_root / 'outputs/raw_scores' / slug(args.model_id)
    with exclusive_model_lock(args.output_root, args.model_id):
        if identity_path.exists() and not args.resume:
            raise ValueError('Existing model scoring requires --resume with the identical fingerprint')
        atomic_json(identity_path, {'input_fingerprint': identity, **identity_record}, immutable=True)
        complete_ids, metas = helper.scan_completed(directory, identity, by_id, rendered, args.model_id, record['revision'], EXECUTION_SHA)
        pilot_ids = helper.fixed_pilot_ids(prompts, rendered)
        if len(pilot_ids) != 42:
            raise ValueError('The frozen pilot must contain exactly 42 prompts')
        probe = maximum_workload(prompts, rendered, rendering['candidate_sequences_by_option_count'])
        selected_pilot = set(pilot_ids) | {probe['prompt_id']}
        pilot_design = {'input_fingerprint': identity, 'prompt_ids': pilot_ids, 'maximum_workload_probe': probe,
                        'outcome_blind': True, 'fixed_pilot_rows': 42, 'formal_unique_pilot_rows': len(selected_pilot)}
        atomic_json(model_status(args, 'pilot_design'), pilot_design, immutable=True)
        pilot_path, complete_path = model_status(args, 'pilot'), model_status(args, 'complete')
        prior_pilot = load(pilot_path) if pilot_path.exists() else None
        valid_pilot = (prior_pilot is not None and prior_pilot.get('status') == 'complete'
                       and prior_pilot.get('input_fingerprint') == identity
                       and prior_pilot.get('repeatability', {}).get('passed') is True
                       and prior_pilot.get('worst_case_probe_passed') is True
                       and selected_pilot <= complete_ids)
        if valid_pilot:
            artifact = prior_pilot.get('repeat_artifact', {})
            if not artifact.get('path') or sha(artifact['path']) != artifact.get('sha256'):
                raise ValueError('Pilot repeatability artifact is missing or changed')
            repeated = pd.read_parquet(artifact['path']).to_dict('records')
            helper.validate_score_rows(repeated, by_id, rendered, args.model_id, record['revision'])
            if len(repeated) != len(selected_pilot) or {r['prompt_id'] for r in repeated} != selected_pilot:
                raise ValueError('Pilot repeatability artifact coverage differs')
        if args.phase == 'score' and not valid_pilot:
            raise ValueError('Complete fixed pilot, repeatability and maximum-workload proof required before production')
        if args.phase == 'pilot' and valid_pilot:
            return prior_pilot
        if args.phase == 'score' and complete_path.exists():
            previous = load(complete_path)
            if previous.get('input_fingerprint') != identity or complete_ids != set(by_id) or previous.get('status') != 'complete':
                raise ValueError('Completed scoring proof or committed coverage differs')
            return previous
        selected = selected_pilot if args.phase == 'pilot' else set(by_id)
        todo = [r for r in prompts if r['prompt_id'] in selected and r['prompt_id'] not in complete_ids]
        gpu_reserve = identity_record['gpu_reserve_bytes']
        cpu_reserve = identity_record['cpu_reserve_bytes']
        def observe():
            free, total = torch.cuda.mem_get_info(device)
            return {'gpu_free_bytes': int(free), 'gpu_total_bytes': int(total),
                    'cpu_available_bytes': int(psutil.virtual_memory().available),
                    'process_rss_bytes': int(psutil.Process().memory_info().rss)}
        monitor = ResourceWatchdog(observe, gpu_reserve, cpu_reserve, model_status(args, 'resource_failure'))
        before_load = monitor.check()
        budget = int(before_load['gpu_free_bytes'] - gpu_reserve - 1024**3)
        if budget < download_state['weight_bytes'] + 8*1024**3:
            raise RuntimeError('Available GPU memory cannot hold BF16 weights, reserve and an initial 8 GiB workspace')
        if before_load['cpu_available_bytes'] < cpu_reserve + download_state['weight_bytes']:
            raise RuntimeError('Insufficient available CPU RAM for conservative checkpoint loading reserve')
        torch.cuda.set_per_process_memory_fraction(budget / properties.total_memory, device)
        monitor.start()
        started = time.perf_counter()
        model = tokenizer = None
        timings = []
        pending, pending_timing = [], []
        actual_class = None
        load_seconds = 0.0
        repeatability = None
        repeat_artifact = None
        load_record = {**record, 'local_cache_path': str(snapshot), 'model_adapter': identity_record['model_adapter']}
        scorer = helper.import_historical_scorer(args.repo_root)
        def forward(current):
            monitor.check()
            torch.cuda.synchronize(device)
            began = time.perf_counter()
            rows = scorer.score_prompt_batch(model, tokenizer, current, rendered, device=str(device))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-began
            for row in rows:
                row.update(model_id=record['id'], model_revision=record['revision'],
                           score_origin='fresh_endpoint_scoring', manifest_sha256=EXECUTION_SHA)
            helper.validate_score_rows(rows, by_id, rendered, record['id'], record['revision'])
            if [r['prompt_id'] for r in rows] != [r['prompt_id'] for r in current]:
                raise ValueError('Original scorer changed the requested prompt order')
            memory = monitor.check()
            timing = {'prompt_ids': [r['prompt_id'] for r in rows], 'seconds': elapsed, 'batch_size': len(rows),
                      'candidate_continuations': sum(int(r['option_count']) for r in rows),
                      'candidate_tokens_including_prompt': sum(int(r['option_count'])*int(r['prompt_token_count']) + sum(r['candidate_token_counts']) for r in rows),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'resources_after': memory}
            return rows, timing
        try:
            if todo or args.phase == 'pilot':
                began = time.perf_counter()
                model, tokenizer = scorer._load_model(load_record, str(device))
                load_seconds = time.perf_counter()-began
                actual_class = type(model).__name__
                if actual_class != record['expected_model_class'] or model.dtype != torch.bfloat16:
                    raise ValueError(f'Unexpected model class or precision: {actual_class}, {model.dtype}')
                if any(p.device.type != 'cuda' or p.device.index != torch.cuda.current_device() for p in model.parameters()):
                    raise ValueError('Unexpected CPU/disk/off-device parameters; full BF16 placement required')
                if getattr(model, 'is_quantized', False):
                    raise ValueError('Quantized models are outside the locked scoring protocol')
                torch.cuda.reset_peak_memory_stats(device)
                atomic_json(model_status(args, 'loaded'), {'input_fingerprint': identity, 'actual_model_class': actual_class,
                    'hf_device_map': {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
                    'dtype': str(model.dtype), 'load_seconds': load_seconds, 'resources': observe()})
                for prompt in todo:
                    rows, timing = forward([prompt])
                    pending.extend(rows)
                    pending_timing.append(timing)
                    timings.append(timing)
                    if len(pending) >= int(config['scoring']['shard_prompts']):
                        helper.publish_shard(directory, pending, identity, EXECUTION_SHA, model_id=record['id'], revision=record['revision'],
                            execution={'stage': args.phase, 'requested_batch_size': 1, 'effective_batch_size': 1,
                                       'batches': pending_timing, 'oom_events': [], 'hardware': hardware})
                        complete_ids.update(r['prompt_id'] for r in pending)
                        pending, pending_timing = [], []
                        print(json.dumps({'model_id': record['id'], 'stage': args.phase, 'completed_rows': len(complete_ids), 'required_rows': len(by_id)}), flush=True)
                if pending:
                    helper.publish_shard(directory, pending, identity, EXECUTION_SHA, model_id=record['id'], revision=record['revision'],
                        execution={'stage': args.phase, 'requested_batch_size': 1, 'effective_batch_size': 1,
                                   'batches': pending_timing, 'oom_events': [], 'hardware': hardware})
                    complete_ids.update(r['prompt_id'] for r in pending)
                    pending, pending_timing = [], []
                if args.phase == 'pilot':
                    reference = {}
                    for path in sorted(directory.glob('shard_*.parquet')):
                        if path.with_suffix('.complete.json').exists():
                            reference.update({r['prompt_id']: r for r in pd.read_parquet(path).to_dict('records')})
                    repeated, repeat_timing = [], []
                    repeat_ids = pilot_ids + ([] if probe['prompt_id'] in pilot_ids else [probe['prompt_id']])
                    for pid in repeat_ids:
                        rows, timing = forward([by_id[pid]])
                        repeated.extend(rows)
                        repeat_timing.append(timing)
                    repeatability = compare_repeat([reference[pid] for pid in repeat_ids], repeated,
                        config['scoring']['alternative_backend_pilot_thresholds'])
                    repeat_path = args.output_root / 'validation/pilot_repeats' / f'{slug(record["id"])}.parquet'
                    atomic_parquet(repeat_path, pd.DataFrame(repeated))
                    repeat_artifact = {'path': str(repeat_path), 'sha256': sha(repeat_path), 'rows': len(repeated), 'batches': repeat_timing}
                    if not repeatability['passed']:
                        atomic_json(model_status(args, 'repeatability_failure'), repeatability)
                        raise ValueError('Original scorer repeatability exceeded fixed pre-outcome thresholds')
        except BaseException as exc:
            # Commit only fully validated successful forwards; no incomplete
            # CUDA operation or unvalidated row can become a completed shard.
            if pending:
                helper.publish_shard(directory, pending, identity, EXECUTION_SHA, model_id=record['id'], revision=record['revision'],
                    execution={'stage': args.phase, 'requested_batch_size': 1, 'effective_batch_size': 1,
                               'batches': pending_timing, 'oom_events': [], 'interrupted_after_valid_rows': True, 'hardware': hardware})
            atomic_json(model_status(args, 'last_failure'), {'status': 'failed', 'phase': args.phase, 'input_fingerprint': identity,
                        'error_type': type(exc).__name__, 'error': str(exc), 'truncation_or_quantization_applied': False})
            raise
        finally:
            monitor.stop()
            if model is not None:
                del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        complete_ids, metas = helper.scan_completed(directory, identity, by_id, rendered, record['id'], record['revision'], EXECUTION_SHA)
        if not selected <= complete_ids:
            raise ValueError('Scoring stage incomplete after committed-shard readback')
        if args.phase == 'score' and (len(complete_ids) != config['scoring']['required_prompts_per_model']
                or sum(len(b['prompt_ids']) for m in metas for b in m['execution']['batches']) != len(complete_ids)):
            raise ValueError('Final score/timing inventory differs from the prescribed prompt count')
        all_batches = [b for m in metas for b in m['execution']['batches']]
        total_sequences = sum(b['candidate_continuations'] for b in all_batches)
        if args.phase == 'score' and total_sequences != config['scoring']['required_candidate_sequences_per_model']:
            raise ValueError('Complete candidate-continuation inventory differs')
        report = {'status': 'complete', 'phase': args.phase, 'model_id': record['id'], 'model_revision': record['revision'],
                  'input_fingerprint': identity, 'completed_rows': len(complete_ids), 'new_rows': len(todo),
                  'candidate_continuations': total_sequences, 'design': design, 'hardware': hardware,
                  'actual_model_class': actual_class or load(model_status(args, 'loaded'))['actual_model_class'],
                  'batch_size': 1, 'load_seconds': load_seconds, 'wall_seconds': time.perf_counter()-started,
                  'committed_inference_seconds': sum(b['seconds'] for b in all_batches),
                  'new_inference_seconds': sum(b['seconds'] for b in timings), 'shards': len(metas),
                  'minimum_gpu_free_bytes': monitor.minimum_gpu_free, 'minimum_cpu_available_bytes': monitor.minimum_cpu_free,
                  'gpu_reserve_bytes': gpu_reserve, 'cpu_reserve_bytes': cpu_reserve,
                  'allocator_cap_bytes': budget, 'formal_pilot_ids': pilot_ids,
                  'maximum_workload_probe': probe, 'worst_case_probe_passed': probe['prompt_id'] in complete_ids,
                  'repeatability': repeatability if args.phase == 'pilot' else prior_pilot['repeatability'],
                  'repeat_artifact': repeat_artifact if args.phase == 'pilot' else prior_pilot['repeat_artifact'],
                  'pilot_report_sha256': None if args.phase == 'pilot' else sha(pilot_path),
                  'frozen_original_score_function': True, 'measurement_origin': 'fresh_endpoint_scoring',
                  'semantic_manifest_sha256': EXECUTION_SHA, 'created_utc': helper.utc_now()}
        atomic_json(pilot_path if args.phase == 'pilot' else complete_path, report, immutable=True)
        return report


def derived_input_config(args, *, new_only=False):
    config = load(args.plan_root / 'config/experiment.json')
    result = load(args.plan_root / 'config/original_xwvg_input_config.json')
    locked = {r['id']: r['revision'] for r in load(args.output_root / 'model_lock.json')['models']}
    models = [r['id'] for r in config['new_models']] if new_only else config['expanded_roster']
    result['model_roster'] = models
    result['model_revisions'] = {m: (locked[m] if m in locked else config['original_model_revisions'][m]) for m in models}
    result['expected_primary_cache_rows'] = len(models)*560*7
    return result


def assemble(args):
    import pandas as pd
    config = validate_roots(args)
    root = args.output_root
    complete_path = root / 'status/assembly_complete.json'
    if complete_path.exists():
        existing = load(complete_path)
        for item in existing['artifacts'].values():
            if sha(item['path']) != item['sha256']:
                raise ValueError('Existing completed assembly artifact changed')
        return existing
    helper = original(args.repo_root)
    manifests = root / 'data/permutations/score_execution_manifest.jsonl'
    if sha(manifests) != EXECUTION_SHA:
        raise ValueError('Execution manifest changed before assembly')
    prompts = read_rows(manifests)
    by_id = {r['prompt_id']: r for r in prompts}
    lock = load(root / 'model_lock.json')
    if lock['status'] != 'complete' or len(lock['models']) != 3:
        raise ValueError('All three model revisions are required for assembly')
    rotation_frames, records, shard_inventory = [], [], []
    saved_model_id = getattr(args, 'model_id', None)
    try:
        for record in lock['models']:
            args.model_id = record['id']
            identity_path = model_status(args, 'identity')
            identity = load(identity_path)
            fingerprint = identity.pop('input_fingerprint')
            if digest(identity) != fingerprint or identity['model_revision'] != record['revision'] or identity['model_id'] != record['id']:
                raise ValueError('Scoring identity checksum/revision differs')
            if identity['new_adapter_sha256'] != sha(Path(__file__)):
                raise ValueError('Scoring adapter changed after measurement')
            if identity['original_driver_sha256'] != sha(Path(helper.__file__)):
                raise ValueError('Original shard/scoring helper changed after measurement')
            for name, expected in identity['source_sha256'].items():
                if sha(args.repo_root / name) != expected:
                    raise ValueError(f'Original source changed after measurement: {name}')
            complete = load(model_status(args, 'complete'))
            if complete.get('status') != 'complete' or complete['input_fingerprint'] != fingerprint:
                raise ValueError('Missing complete real score proof')
            if complete.get('pilot_report_sha256') != sha(model_status(args, 'pilot')):
                raise ValueError('Measured pilot report changed after scoring')
            path = root / 'data/permutations/rendered' / f'{slug(record["id"])}.jsonl'
            if sha(path) != identity['rendered_prompt_sha256']:
                raise ValueError('Rendered input changed before assembly')
            rendered = {r['prompt_id']: r for r in read_rows(path)}
            helper.validate_design(prompts, list(rendered.values()))
            directory = root / 'outputs/raw_scores' / slug(record['id'])
            ids, metas = helper.scan_completed(directory, fingerprint, by_id, rendered, record['id'], record['revision'], EXECUTION_SHA)
            if ids != set(by_id):
                raise ValueError(f'Incomplete real scoring coverage: {record["id"]}')
            for meta_file in sorted(directory.glob('shard_*.meta.json')):
                data = meta_file.with_suffix('').with_suffix('.parquet')
                marker = data.with_suffix('.complete.json')
                if not marker.exists():
                    continue
                frame = pd.read_parquet(data)
                rotation_frames.append(frame)
                shard_inventory.append({'path': str(data), 'sha256': sha(data), 'rows': len(frame),
                                        'metadata_path': str(meta_file), 'metadata_sha256': sha(meta_file),
                                        'commit_path': str(marker), 'commit_sha256': sha(marker)})
            records.append({'model_id': record['id'], 'revision': record['revision'], 'input_fingerprint': fingerprint,
                            'identity_path': str(identity_path), 'identity_sha256': sha(identity_path),
                            'complete_path': str(model_status(args, 'complete')), 'complete_sha256': sha(model_status(args, 'complete')),
                            'rows': len(ids), 'rendered_sha256': sha(path)})
    finally:
        args.model_id = saved_model_id
    raw = pd.concat(rotation_frames, ignore_index=True)
    raw = raw.sort_values(['model_id', 'base_question_id', 'semantic_condition', 'permutation_id']).reset_index(drop=True)
    if len(raw) != config['expected_outputs']['new_rotation_rows'] or raw.duplicated(['model_id','prompt_id']).any():
        raise ValueError('Invalid total new rotation count or duplicate identity')
    if int(raw['option_count'].sum()) != config['expected_outputs']['new_candidate_sequences']:
        raise ValueError('New continuation sequence count differs')
    raw_path = root / 'data/new_rotation_scores.parquet'
    atomic_parquet(raw_path, raw, immutable=True)
    source = args.repo_root / 'experiments/llm/analysis/xwvg_reference_selection/src'
    sys.path.insert(0, str(source))
    from bca_reference_selection.data import aggregate_rotations, load_inputs
    questions = root / 'inputs/mmlu_pro_selected_560.jsonl'
    new_path = root / 'data/new_permutation_averaged_semantic_scores.parquet'
    new_config = derived_input_config(args, new_only=True)
    if new_path.exists():
        # Recompute into a new temporary derivative to validate interrupted
        # publication without replacing any previous probability bytes.
        candidate = new_path.with_name(f'.new_average_validation.{os.getpid()}.parquet')
        report = aggregate_rotations(raw_path, candidate, questions, new_config)
        if sha(candidate) != sha(new_path):
            raise ValueError('Existing new averaged cache differs from validated rotations')
        candidate.unlink()
    else:
        report = aggregate_rotations(raw_path, new_path, questions, new_config)
    old_path = root / 'inputs/permutation_averaged_semantic_scores.parquet'
    if sha(old_path) != config['baseline_cache']['sha256'] or sha(questions) != config['questions']['sha256']:
        raise ValueError('Frozen seed input changed')
    old = pd.read_parquet(old_path)
    new = pd.read_parquet(new_path)
    merged = pd.concat([old, new], ignore_index=True)
    if len(merged) != config['expected_outputs']['merged_averaged_vectors']:
        raise ValueError('Merged model roster has an incorrect row count')
    merged_path = root / 'data/permutation_averaged_semantic_scores.parquet'
    atomic_parquet(merged_path, merged, immutable=True)
    merged_config = derived_input_config(args)
    config_path = root / 'data/eleven_model_input_config.json'
    atomic_json(config_path, merged_config, immutable=True)
    loaded = load_inputs(merged_path, questions, merged_config)
    # Pandas concat is required to preserve every original cell, including
    # object/list vectors. Compare directly before any tensor normalization.
    pd.testing.assert_frame_equal(pd.read_parquet(merged_path).iloc[:len(old)][old.columns].reset_index(drop=True), old, check_dtype=False)
    provenance = {'schema': 'xfqa_measurement_provenance_v1', 'status': 'complete',
                  'old_measurements': {'path': str(old_path), 'sha256': sha(old_path), 'rows': len(old),
                                       'origin': 'fresh_endpoint_scoring_for_XWVG_20261107', 'rescored': False},
                  'new_measurements': {'rows': len(new), 'rotation_rows': len(raw), 'origin': 'fresh_endpoint_scoring',
                                       'all_cyclic_rotations_including_zero': True, 'models': records},
                  'questions': {'path': str(questions), 'sha256': sha(questions)},
                  'execution_manifest_sha256': EXECUTION_SHA, 'semantic_manifest_sha256': SEMANTIC_SHA,
                  'model_lock_sha256': sha(root / 'model_lock.json'), 'protocol_sha256': sha(args.plan_root / 'config/experiment.json'),
                  'raw_shards': shard_inventory, 'scoring_definition': 'original full continuation summed log likelihood; BF16; native templates; no generation',
                  'new_scoring_adapter_sha256': sha(Path(__file__))}
    provenance_path = root / 'provenance/measurement_provenance.json'
    atomic_json(provenance_path, provenance, immutable=True)
    atomic_json(root / 'validation/assembly_inputs.json', {'passed': True, 'new_averaging': report,
                'merged_validation': loaded.report, 'old_cells_unchanged': True})
    paths = {'new_rotations': raw_path, 'new_averaged': new_path, 'merged_averaged': merged_path,
             'questions': questions, 'baseline': old_path, 'input_config': config_path, 'provenance': provenance_path}
    result = {'status': 'complete', 'real_three_model_scoring_complete': True,
              'new_rotation_rows': len(raw), 'new_averaged_vectors': len(new), 'merged_averaged_vectors': len(merged),
              'artifacts': {name: {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in paths.items()},
              'input_fingerprint': digest({name: sha(path) for name, path in paths.items()})}
    atomic_json(complete_path, result, immutable=True)
    return result


def run_phase(args):
    validate_roots(args)
    operations = {'prepare': prepare, 'resolve': resolve, 'download': download, 'render': render,
                  'pilot': score, 'score': score, 'assemble': assemble}
    if args.phase not in operations:
        raise NotImplementedError(f'Phase implementation is being prepared: {args.phase}')
    result = operations[args.phase](args)
    print(json.dumps({'phase': args.phase, 'model_id': getattr(args, 'model_id', None),
                      'status': result.get('status', 'complete')}, sort_keys=True), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--plan-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'resolve', 'download', 'render', 'pilot', 'score', 'assemble'], required=True)
    parser.add_argument('--model-id')
    parser.add_argument('--model-cache', type=Path)
    parser.add_argument('--execution-manifest', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-gpu-uuid')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu-reserve-gib', type=float, default=16)
    parser.add_argument('--cpu-reserve-gib', type=float, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    run_phase(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
