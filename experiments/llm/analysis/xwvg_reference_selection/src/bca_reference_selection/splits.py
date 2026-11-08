"""Original subject-stratified SHA-256 question half-splits (V2.2/V2.3)."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Iterable
import pandas as pd


def stable_u64(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256('\x1f'.join(map(str, parts)).encode()).digest()[:8], 'big')


@dataclass(frozen=True)
class Split:
    seed: int
    fitting_ids: frozenset[str]
    evaluation_ids: frozenset[str]


def fixed_split(selected: pd.DataFrame, seed: int) -> Split:
    unique = selected[['base_question_id', 'category']].drop_duplicates().copy()
    unique['base_question_id'] = unique['base_question_id'].astype(str)
    if unique['base_question_id'].duplicated().any():
        raise ValueError('Question assigned to multiple categories')
    fitting, evaluation = set(), set()
    for category, group in unique.groupby('category', sort=True):
        ids = sorted(group['base_question_id'], key=lambda q: (stable_u64(seed, category, q), q))
        if len(ids) % 2:
            raise ValueError(f'Odd category size: {category}')
        fitting.update(ids[:len(ids)//2])
        evaluation.update(ids[len(ids)//2:])
    if fitting & evaluation or fitting | evaluation != set(unique['base_question_id']):
        raise AssertionError('Split leaked or lost question identities')
    return Split(int(seed), frozenset(fitting), frozenset(evaluation))


def fixed_splits(selected: pd.DataFrame, seeds: Iterable[int]) -> dict[int, Split]:
    return {int(seed): fixed_split(selected, int(seed)) for seed in seeds}


def split_manifest(selected: pd.DataFrame, seeds: Iterable[int]) -> dict:
    categories = dict(zip(selected['base_question_id'].astype(str), selected['category']))
    return {str(seed): {'fitting_ids': sorted(split.fitting_ids),
                        'evaluation_ids': sorted(split.evaluation_ids),
                        'fitting_categories': pd.Series([categories[q] for q in split.fitting_ids]).value_counts().to_dict(),
                        'evaluation_categories': pd.Series([categories[q] for q in split.evaluation_ids]).value_counts().to_dict(),
                        'overlap_count': len(split.fitting_ids & split.evaluation_ids)}
            for seed, split in fixed_splits(selected, seeds).items()}
