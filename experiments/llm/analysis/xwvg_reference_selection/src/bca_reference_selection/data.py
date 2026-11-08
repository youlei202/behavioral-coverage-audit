"""Strict, portable adapters for complete V2.3 semantic probability vectors.

Semantic vector coordinates are original option order, as defined by the
historical score producer. No historical orchestration is imported or run.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import numpy as np
import pandas as pd

QUESTION_SHA256 = '9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6'
OPTION_COUNTS = {3: 1, 4: 33, 5: 2, 6: 1, 7: 3, 8: 22, 9: 50, 10: 448}
PROBABILITY_COLUMN = 'permutation_averaged_semantic_probabilities'
KEYS = ['model_id', 'base_question_id', 'semantic_condition']
METADATA = ['category', 'condition', 'family', 'dose', 'track', 'option_count', 'gold_semantic_option']
CONDITION_CODES = {'irrelevant_context': 'I', 'content_deletion': 'D'}


def fingerprint(path: Path | str) -> dict:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


@dataclass
class Inputs:
    probabilities: dict[str, np.ndarray]  # [model, question, condition, option]
    mask: np.ndarray                     # [question, condition, option]
    question_ids: tuple[str, ...]
    categories: tuple[str, ...]
    model_ids: tuple[str, ...]
    masses: dict[str, np.ndarray]
    questions: pd.DataFrame
    report: dict


def load_questions(path: Path | str, config: dict) -> tuple[pd.DataFrame, dict]:
    identity = fingerprint(path)
    matched = identity['sha256'] == QUESTION_SHA256
    fixture = config.get('synthetic_fixture', False)
    if not matched and not fixture:
        raise ValueError(f'Original selected-question SHA-256 mismatch: {identity["path"]}; expected {QUESTION_SHA256}, found {identity["sha256"]}')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    required = {'base_question_id', 'category', 'options', 'answer_index'}
    if required - set(frame):
        raise ValueError(f'Selected questions lack fields: {sorted(required - set(frame))}')
    if frame[list(required)].isna().any().any():
        raise ValueError('Null selected-question metadata')
    frame['base_question_id'] = frame['base_question_id'].astype(str)
    if frame['base_question_id'].duplicated().any():
        raise ValueError('Duplicate selected-question IDs')
    counts = frame.groupby('category').size().to_dict()
    if len(frame) != config['question_count'] or len(counts) != config['category_count'] or set(counts.values()) != {config['questions_per_category']}:
        raise ValueError(f'Selected-question/category counts mismatch: {counts}')
    frame['option_count'] = frame['options'].map(len)
    for row in frame.itertuples(index=False):
        if not isinstance(row.options, list) or not 2 <= len(row.options) <= 10:
            raise ValueError(f'Invalid option list: {row.base_question_id}')
        if int(row.answer_index) != row.answer_index or not 0 <= row.answer_index < len(row.options):
            raise ValueError(f'Invalid reference answer: {row.base_question_id}')
    distribution = dict(Counter(frame['option_count']))
    if not fixture and distribution != OPTION_COUNTS:
        raise ValueError(f'Historical option-count distribution mismatch: {distribution}')
    frame = frame.sort_values('base_question_id').reset_index(drop=True)
    return frame, {**identity, 'historical_sha256': QUESTION_SHA256, 'historical_sha256_match': matched,
                   'synthetic_fixture': fixture, 'category_counts': counts, 'option_count_distribution': distribution}


def _vector(value: object, n: int, config: dict, report: dict) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    policy = config['preprocessing']
    if v.ndim != 1 or len(v) < n or len(v) > 10 or not np.isfinite(v).all():
        raise ValueError('Malformed/nonfinite probability vector')
    padding_max = float(np.max(np.abs(v[n:]))) if len(v) > n else 0.
    report['max_padding_abs'] = max(report.get('max_padding_abs', 0.), padding_max)
    if np.any(np.abs(v[n:]) > policy['padding_tolerance']):
        raise ValueError('Nonzero invalid option padding')
    v = v[:n].copy()
    bounds = policy['probability_bounds_tolerance']
    if v.min() < -bounds or v.max() > 1 + bounds:
        raise ValueError('Probability bounds violated')
    drift = abs(v.sum() - 1.)
    if drift > policy['probability_sum_tolerance']:
        raise ValueError(f'Probability normalization violated: {drift}')
    corrected = np.clip(v, 0., 1.)
    corrected /= corrected.sum()
    repair = float(np.abs(corrected - v).sum())
    report['max_probability_sum_error'] = max(report.get('max_probability_sum_error', 0.), float(drift))
    report['max_probability_repair_l1'] = max(report.get('max_probability_repair_l1', 0.), repair)
    report['roundoff_corrected_vectors'] = report.get('roundoff_corrected_vectors', 0) + int(repair > 0)
    report['validated_vectors'] = report.get('validated_vectors', 0) + 1
    return corrected


def _integer(value: object, name: str) -> int:
    if pd.isna(value) or not np.isfinite(value) or int(value) != value:
        raise ValueError(f'Invalid integer field {name}: {value}')
    return int(value)


def _validate_metadata(row: dict, question: dict, config: dict) -> tuple[str, int]:
    if row['category'] != question['category']:
        raise ValueError(f'Category mismatch: {row["base_question_id"]}')
    n = _integer(row['option_count'], 'option_count')
    gold = _integer(row['gold_semantic_option'], 'gold_semantic_option')
    if n != len(question['options']) or gold != int(question['answer_index']):
        raise ValueError(f'Option count/reference answer mismatch: {row["base_question_id"]}')
    if row['condition'] == 'clean':
        track = -1 if pd.isna(row['track']) else _integer(row['track'], 'track')
        if row['family'] != 'clean' or float(row['dose']) != 0 or track != -1 or row['semantic_condition'] != 'C0':
            raise ValueError('Malformed clean condition; expected C0, family clean, dose 0, track null/-1')
        return 'clean', -1
    family = row['family']
    if family not in config['families'] or row['condition'] != family:
        raise ValueError(f'Unexpected stress family: {family}')
    track = _integer(row['track'], 'track')
    if track not in range(config['families'][family]['stress_track_count']):
        raise ValueError(f'Unexpected stress track: {track}')
    if float(row['dose']) != float(config['families'][family]['doses'][-1]):
        raise ValueError(f'Incorrect stress dose: {row["dose"]}')
    if row['semantic_condition'] != f'{CONDITION_CODES[family]}{track}':
        raise ValueError('Semantic-condition identity conflicts with family/track')
    return family, track


def _check_revision(frame: pd.DataFrame, config: dict) -> str:
    if 'model_revision' not in frame:
        return 'not_present_in_averaged_cache; configured revisions recorded but not independently verified by this cache'
    for model, group in frame.groupby('model_id'):
        if group['model_revision'].isna().any() or set(group['model_revision']) != {config['model_revisions'][model]}:
            raise ValueError(f'Model revision mismatch: {model}')
    return 'all_row_revisions_match_configured_roster'


def load_inputs(cache: Path | str, questions: Path | str, config: dict) -> Inputs:
    selected, question_report = load_questions(questions, config)
    frame = pd.read_parquet(cache)
    report = {'passed': False, 'cache': {**fingerprint(cache), 'historical_digest_available': False,
               'identity_scope': 'current-input fingerprint; no historical cache checksum retained'},
              'questions': question_report, 'protocol': config['primary_protocol'],
              'probability_policy': 'clip bounds-roundoff and normalize only within configured tolerances; record every changed vector'}
    required = set(KEYS + METADATA + [PROBABILITY_COLUMN])
    if required - set(frame):
        raise ValueError(f'Full-vector cache lacks columns: {sorted(required - set(frame))}')
    if frame[KEYS].isna().any().any():
        raise ValueError('Null semantic identity key')
    frame['base_question_id'] = frame['base_question_id'].astype(str)
    if len(frame) != config['expected_primary_cache_rows'] or frame.duplicated(KEYS).any():
        raise ValueError('Primary cache row count or duplicate semantic keys invalid')
    model_ids = tuple(config['model_roster'])
    if set(frame['model_id']) != set(model_ids):
        raise ValueError('Model roster mismatch')
    report['revision_verification'] = _check_revision(frame, config)
    report['configured_model_revisions'] = config['model_revisions']
    question_ids = tuple(selected['base_question_id'])
    if set(frame['base_question_id']) != set(question_ids):
        raise ValueError('Cache question IDs do not equal original selected sample')
    qmap = selected.set_index('base_question_id').to_dict('index')
    qindex = {q: i for i, q in enumerate(question_ids)}
    mindex = {m: i for i, m in enumerate(model_ids)}
    nmax = int(selected['option_count'].max())
    shape = (len(model_ids), len(selected), 4, nmax)
    values = {f: np.zeros(shape, dtype=np.float64) for f in config['families']}
    present = {f: np.zeros(shape[:3], dtype=bool) for f in config['families']}
    mask = np.arange(nmax)[None, None, :] < selected['option_count'].to_numpy()[:, None, None]
    mask = np.broadcast_to(mask, (len(selected), 4, nmax)).copy()
    for row in frame.to_dict('records'):
        q = qmap[row['base_question_id']]
        family, track = _validate_metadata(row, q, config)
        n, gold = int(row['option_count']), int(row['gold_semantic_option'])
        vector = _vector(row[PROBABILITY_COLUMN], n, config, report)
        if 'permutation_averaged_gold_probability' in row:
            scalar = float(row['permutation_averaged_gold_probability'])
            if not np.isfinite(scalar) or abs(scalar - vector[gold]) > config['preprocessing']['probability_sum_tolerance']:
                raise ValueError('Saved gold probability differs from full-vector gold coordinate')
        if 'permutation_averaged_semantic_argmax' in row and _integer(row['permutation_averaged_semantic_argmax'], 'argmax') != int(vector.argmax()):
            raise ValueError('Saved argmax differs from semantic probability vector')
        mi, qi = mindex[row['model_id']], qindex[row['base_question_id']]
        for dest in (values if family == 'clean' else [family]):
            ci = 0 if family == 'clean' else track + 1
            if present[dest][mi, qi, ci]:
                raise ValueError('Duplicate observation after semantic alignment')
            values[dest][mi, qi, ci, :n] = vector
            present[dest][mi, qi, ci] = True
    if not all(p.all() for p in present.values()):
        raise ValueError('Missing model/question/condition after semantic alignment')
    masses = {f: np.array([s['clean_mass']] + [s['stress_track_mass']] * 3, dtype=float)
              for f, s in config['families'].items()}
    if any(not np.isclose(m.sum(), 1., atol=1e-14, rtol=0) for m in masses.values()):
        raise ValueError('Endpoint masses must sum to one')
    report.update(passed=True, cache_rows=len(frame), tensor_shape=list(shape),
                  question_order=list(question_ids), model_order=list(model_ids),
                  semantic_option_identity='original selected-question options order; validated option count and gold index',
                  condition_order={f: ['C0'] + [f'{CONDITION_CODES[f]}{t}' for t in range(3)] for f in values},
                  condition_masses={f: m.tolist() for f, m in masses.items()},
                  original_clean_track='null normalized to -1; clean stored once per family',
                  valid_option_coordinates_per_family=int(mask.sum()), duplicate_keys=0,
                  missing_observations=0)
    return Inputs(values, mask, question_ids, tuple(selected['category']), model_ids, masses, selected, report)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    frame.to_parquet(temporary, index=False)
    with temporary.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(temporary, path)


def aggregate_rotations(rotations_cache: Path | str, output_cache: Path | str,
                        questions: Path | str, config: dict) -> dict:
    """CPU-only complete cyclic average; existing input files are never written."""
    source, destination = Path(rotations_cache), Path(output_cache)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Aggregation requires a fresh output cache path distinct from input')
    selected, question_report = load_questions(questions, config)
    frame = pd.read_parquet(source)
    required = set(KEYS + METADATA + ['permutation_id', 'semantic_remapped_probabilities', 'model_revision', 'schema_version', 'prompt_id'])
    if required - set(frame):
        raise ValueError(f'Rotation cache lacks fields: {sorted(required - set(frame))}')
    if frame[KEYS].isna().any().any():
        raise ValueError('Null rotation identity key')
    frame['base_question_id'] = frame['base_question_id'].astype(str)
    if set(frame['model_id']) != set(config['model_roster']):
        raise ValueError('Rotation model roster mismatch')
    _check_revision(frame, config)
    qmap = selected.set_index('base_question_id').to_dict('index')
    if set(frame['base_question_id']) != set(qmap):
        raise ValueError('Rotation question IDs mismatch')
    expected = len(config['model_roster']) * 7 * int(selected['option_count'].sum())
    if len(frame) != expected:
        raise ValueError(f'Rotation row count mismatch: expected {expected}, got {len(frame)}')
    if 'schema_version' in frame and not frame['schema_version'].eq('pier_v23_candidate_score_v1').all():
        raise ValueError('Rotation score schema mismatch')
    if 'prompt_id' in frame and frame.duplicated(['model_id', 'prompt_id']).any():
        raise ValueError('Duplicate rotation prompt identity')
    report = {'source': fingerprint(source), 'historical_digest_available': False,
              'questions': question_report, 'source_rows': len(frame),
              'semantic_remapping': 'already semantic; no second remapping applied'}
    rows = []
    for keys, group in frame.groupby(KEYS, sort=True):
        first = group.iloc[0].to_dict()
        _validate_metadata(first, qmap[first['base_question_id']], config)
        n = int(first['option_count'])
        for column in METADATA:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f'Conflicting rotation metadata {column}: {keys}')
        rotations = [_integer(v, 'permutation_id') for v in group['permutation_id']]
        if sorted(rotations) != list(range(n)):
            raise ValueError(f'Incomplete or duplicate cyclic rotation set: {keys}')
        for rotation, prompt_id in zip(rotations, group['prompt_id']):
            expected_prompt_id = f'{keys[1]}__{keys[2]}__cyclic_r{rotation:02d}'
            if prompt_id != expected_prompt_id:
                raise ValueError(f'Rotation prompt identity mismatch: {prompt_id}')
        vectors = [_vector(v, n, config, report) for v in group.sort_values('permutation_id')['semantic_remapped_probabilities']]
        average = np.stack(vectors).mean(axis=0)
        gold = int(first['gold_semantic_option'])
        rows.append({**{key: first[key] for key in KEYS + METADATA},
                     'model_revision': first['model_revision'],
                     PROBABILITY_COLUMN: average.tolist(),
                     'permutation_averaged_gold_probability': float(average[gold]),
                     'permutation_averaged_semantic_argmax': int(average.argmax())})
    averaged = pd.DataFrame(rows)
    if len(averaged) != config['expected_primary_cache_rows']:
        raise ValueError('Averaged row count mismatch')
    _atomic_parquet(averaged, destination)
    report['averaged_validation'] = load_inputs(destination, questions, config).report
    report['output'] = fingerprint(destination)
    report['complete_cyclic_sets'] = len(rows)
    report['passed'] = True
    return report


SEMANTIC_MANIFEST_SHA256 = '872051ccdfa272e1e5c90ea4d7257e3c6572bbe2565dd1a7883940c34f931048'
V23_INFERENCE_MANIFEST_SHA256 = '065d65f94e51a4897159511f67720bdc793309b237b2dd92e5dae00e6ec933a3'
V2_PROMPT_MANIFEST_SHA256 = '199bd992d10009ac04bcdf212e13023f08ddc6d2f2abeaa26597855f5c2b5b3a'


def reconstruct_from_shards(v2_root: Path | str, v23_root: Path | str, output_cache: Path | str,
                            questions: Path | str, config: dict) -> dict:
    """Rebuild rotation 0 from V2 and combine saved V2.3 rotations, CPU only.

    Verifies sidecar checksums, schemas, roster revisions, exact prompt coverage,
    and visible/semantic mappings against the archived semantic prompt manifest.
    The only writes are the new output cache and its adjacent rotation derivative.
    """
    v2_root, v23_root, output_cache = Path(v2_root), Path(v23_root), Path(output_cache)
    rotation_output = output_cache.with_name('reconstructed_permutation_wise_semantic_scores.parquet')
    if output_cache.exists() or rotation_output.exists():
        raise ValueError('Shard reconstruction requires fresh derivative output paths')
    selected, question_report = load_questions(questions, config)
    manifest_path = v23_root / 'data/permutations/score_semantic_manifest.jsonl'
    manifest_identity = fingerprint(manifest_path)
    if manifest_identity['sha256'] != SEMANTIC_MANIFEST_SHA256 and not config.get('synthetic_fixture', False):
        raise ValueError('Historical semantic prompt manifest checksum mismatch')
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    prompts = {str(row['prompt_id']): row for row in manifest_rows}
    if len(prompts) != len(manifest_rows):
        raise ValueError('Duplicate semantic manifest prompt IDs')
    qmap = selected.set_index('base_question_id').to_dict('index')
    rotations0 = {}
    for prompt in prompts.values():
        qid = str(prompt['base_question_id'])
        if qid not in qmap:
            raise ValueError('Semantic manifest includes unknown question')
        check = {**prompt, 'gold_semantic_option': prompt['semantic_answer_index']}
        _validate_metadata(check, qmap[qid], config)
        n, rotation = int(prompt['option_count']), _integer(prompt['permutation_id'], 'permutation_id')
        if not 0 <= rotation < n:
            raise ValueError('Invalid manifest rotation')
        s2v = [(i + rotation) % n for i in range(n)]
        v2s = [(i - rotation) % n for i in range(n)]
        if prompt['semantic_to_visible'] != s2v or prompt['visible_to_semantic'] != v2s:
            raise ValueError('Manifest semantic mapping conflicts with cyclic rotation')
        if prompt['options'] != [qmap[qid]['options'][i] for i in v2s]:
            raise ValueError('Manifest option identity differs from original question options')
        if prompt['prompt_id'] != f'{qid}__{prompt["semantic_condition"]}__cyclic_r{rotation:02d}':
            raise ValueError('Manifest prompt ID does not encode its semantic identity')
        if rotation == 0:
            source = str(prompt['source_v2_prompt_id'])
            if source in rotations0:
                raise ValueError('Duplicate rotation-0 source prompt')
            rotations0[source] = prompt
    sources = []
    rows = []
    vector_report = {}
    for root, schema, manifest_field, manifest_digest, is_v2 in [
        (v2_root, 'pier_raw_label_scores_v2', 'prompt_manifest_sha256', V2_PROMPT_MANIFEST_SHA256, True),
        (v23_root, 'pier_v23_candidate_score_v1', 'manifest_sha256', V23_INFERENCE_MANIFEST_SHA256, False),
    ]:
        paths = sorted((root / 'outputs/raw_scores').glob('*/shard_*.parquet'))
        if not paths:
            raise FileNotFoundError(f'Missing score shards: {root}/outputs/raw_scores/*/shard_*.parquet')
        seen = set()
        for path in paths:
            sidecar = path.with_suffix('.meta.json')
            meta = json.loads(sidecar.read_text())
            identity = fingerprint(path)
            if meta.get('sha256') != identity['sha256'] or meta.get('schema_version') != schema:
                raise ValueError(f'Shard checksum/schema mismatch: {path}')
            model = meta.get('model_id')
            if model not in config['model_roster'] or meta.get('model_revision') != config['model_revisions'][model]:
                raise ValueError(f'Shard model/revision mismatch: {path}')
            if meta.get(manifest_field) != manifest_digest and not config.get('synthetic_fixture', False):
                raise ValueError(f'Shard historical prompt manifest checksum mismatch: {path}')
            frame = pd.read_parquet(path)
            if len(frame) != meta['row_count'] or not frame['schema_version'].eq(schema).all() or not frame['model_id'].eq(model).all():
                raise ValueError(f'Shard row schema/count/model mismatch: {path}')
            _check_revision(frame, config)
            sources.append({**identity, 'sidecar': fingerprint(sidecar)})
            for source in frame.to_dict('records'):
                key = (model, source['prompt_id'])
                if key in seen:
                    raise ValueError(f'Duplicate shard prompt: {key}')
                seen.add(key)
                if is_v2 and source['prompt_id'] not in rotations0:
                    continue
                prompt = rotations0[source['prompt_id']] if is_v2 else prompts.get(source['prompt_id'])
                if prompt is None or (not is_v2 and int(prompt['permutation_id']) == 0):
                    raise ValueError(f'Unexpected new-rotation prompt: {key}')
                for column in ['base_question_id', 'category', 'condition', 'family', 'dose', 'track']:
                    a, b = source[column], prompt[column]
                    if column == 'base_question_id':
                        a, b = str(a), str(b)
                    if not ((pd.isna(a) and pd.isna(b)) or a == b):
                        raise ValueError(f'Shard/manifest {column} mismatch: {key}')
                n = int(prompt['option_count'])
                semantic = _vector(source['semantic_probabilities' if is_v2 else 'semantic_remapped_probabilities'], n, config, vector_report)
                visible = _vector(source['probabilities' if is_v2 else 'visible_label_probabilities'], n, config, vector_report)
                mapping_name = 'presented_to_original' if is_v2 else 'visible_to_semantic'
                if list(source[mapping_name]) != prompt['visible_to_semantic']:
                    raise ValueError(f'Shard semantic mapping mismatch: {key}')
                remapped = np.empty(n)
                remapped[np.asarray(prompt['visible_to_semantic'])] = visible
                if not np.allclose(semantic, remapped, atol=config['preprocessing']['probability_sum_tolerance'], rtol=0):
                    raise ValueError(f'Saved semantic vector differs from remapped visible vector: {key}')
                gold = source['semantic_answer_index' if is_v2 else 'gold_semantic_option']
                if gold != prompt['semantic_answer_index']:
                    raise ValueError('Shard gold identity differs from semantic manifest')
                if not is_v2:
                    for column in ['semantic_condition', 'permutation_id', 'option_count']:
                        if source[column] != prompt[column]:
                            raise ValueError(f'Shard/manifest {column} mismatch')
                    if source['canonical_prompt_hash'] != prompt['canonical_content_sha256']:
                        raise ValueError('Shard canonical prompt hash mismatch')
                rows.append({**{field: prompt[field] for field in KEYS[1:] + METADATA if field != 'gold_semantic_option'},
                             'model_id': model, 'model_revision': config['model_revisions'][model],
                             'schema_version': 'pier_v23_candidate_score_v1', 'prompt_id': prompt['prompt_id'],
                             'permutation_id': prompt['permutation_id'], 'gold_semantic_option': int(gold),
                             'semantic_remapped_probabilities': semantic.tolist(),
                             'source_v2_prompt_id': prompt['source_v2_prompt_id'],
                             'score_origin': 'V2_reused_rotation0' if is_v2 else 'V2.3_saved_rotation'})
    rotation_frame = pd.DataFrame(rows)
    expected_keys = {(model, prompt) for model in config['model_roster'] for prompt in prompts}
    actual_keys = list(zip(rotation_frame['model_id'], rotation_frame['prompt_id']))
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError('Reconstructed rotations have missing/duplicate/unexpected model/prompt identities')
    _atomic_parquet(rotation_frame, rotation_output)
    result = aggregate_rotations(rotation_output, output_cache, questions, config)
    result.update(route='validated_original_score_shards', shard_sources=sources,
                  semantic_manifest=manifest_identity, question_identity=question_report,
                  shard_probability_validation=vector_report)
    return result
