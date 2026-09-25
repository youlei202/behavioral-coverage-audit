from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    project_root,
    stable_u64,
    write_jsonl,
)

LABELS = tuple("ABCDEFGHIJ")
WORD_RE = re.compile(r"\S+")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def load_experiment_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "configs" / "experiment.json").read_text(encoding="utf-8"))


def canonical_prompt(question: str, options: list[str]) -> str:
    option_lines = "\n".join(f"({LABELS[i]}) {text}" for i, text in enumerate(options))
    return (
        "You are answering a multiple-choice question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Return only the label of the best option.\n"
        "Final answer:"
    )


def _answer_index(row: dict[str, Any], option_count: int) -> int:
    for key in ("answer_index", "answer_idx", "label"):
        value = row.get(key)
        if isinstance(value, (int, np.integer)):
            result = int(value)
            if 0 <= result < option_count:
                return result
    value = row.get("answer")
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < option_count:
        return int(value)
    if isinstance(value, str):
        token = value.strip().upper().strip("()")
        if token in LABELS[:option_count]:
            return LABELS.index(token)
        if token.isdigit() and 0 <= int(token) < option_count:
            return int(token)
    raise ValueError(f"Cannot resolve answer index from keys: {sorted(row)}")


def normalize_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        options_raw = row.get("options") or row.get("choices")
        if not question or not isinstance(options_raw, (list, tuple)):
            raise ValueError(f"Malformed dataset row {index}")
        options = [str(value).strip() for value in options_raw]
        if not 2 <= len(options) <= len(LABELS) or any(not value for value in options):
            raise ValueError(f"Invalid options in dataset row {index}")
        category = str(row.get("category") or row.get("subject") or "").strip()
        if not category:
            raise ValueError(f"Missing category in dataset row {index}")
        source_id = row.get("question_id") or row.get("id") or row.get("src") or index
        base_id = str(source_id)
        if base_id in seen_ids:
            base_id = f"{base_id}--{index}"
        seen_ids.add(base_id)
        answer_index = _answer_index(row, len(options))
        normalized.append(
            {
                "base_question_id": base_id,
                "dataset_index": index,
                "category": category,
                "question": question,
                "options": options,
                "answer_index": answer_index,
                "answer_label": LABELS[answer_index],
            }
        )
    return normalized


def stratified_fixed_sample(
    rows: list[dict[str, Any]], per_category: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        candidates = sorted(
            grouped[category],
            key=lambda row: (
                stable_u64(seed, category, row["base_question_id"]),
                row["base_question_id"],
            ),
        )
        if len(candidates) < per_category:
            raise ValueError(f"Category {category!r} has only {len(candidates)} rows")
        selected.extend(candidates[:per_category])
    return sorted(selected, key=lambda row: (row["category"], row["base_question_id"]))


def _normalized_contains(haystack: str, needle: str) -> bool:
    haystack_norm = " ".join(haystack.casefold().split())
    needle_norm = " ".join(needle.casefold().split())
    return bool(needle_norm) and needle_norm in haystack_norm


def _source_spans(candidates: list[dict[str, Any]], max_words: int) -> tuple[list[str], list[dict[str, Any]]]:
    words: list[str] = []
    spans: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_words = WORD_RE.findall(candidate["question"])
        if not candidate_words:
            continue
        start = len(words)
        words.extend(candidate_words)
        end = len(words)
        spans.append(
            {
                "source_base_question_id": candidate["base_question_id"],
                "source_category": candidate["category"],
                "start_word": start,
                "end_word": end,
            }
        )
        if len(words) >= max_words:
            break
    return words[:max_words], spans


def build_irrelevant_context(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    doses: list[int],
    tracks: int,
    seed: int,
) -> list[dict[str, Any]]:
    max_words = max(doses)
    corpus = [row["question"] for row in selected] + [row["question"] for row in pool]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(corpus)
    selected_matrix = matrix[: len(selected)]
    pool_matrix = matrix[len(selected) :]
    similarities = (selected_matrix @ pool_matrix.T).toarray()
    records: list[dict[str, Any]] = []
    for selected_index, target in enumerate(selected):
        eligible: list[int] = []
        for pool_index, candidate in enumerate(pool):
            if candidate["category"] == target["category"]:
                continue
            if any(_normalized_contains(candidate["question"], option) for option in target["options"]):
                continue
            eligible.append(pool_index)
        if not eligible:
            raise ValueError(f"No eligible distractors for {target['base_question_id']}")
        ordered_by_similarity = sorted(
            eligible,
            key=lambda idx: (float(similarities[selected_index, idx]), pool[idx]["base_question_id"]),
        )
        low_count = min(len(ordered_by_similarity), max(256, math.ceil(len(ordered_by_similarity) * 0.25)))
        low_similarity = ordered_by_similarity[:low_count]
        used_sources: set[str] = set()
        for track in range(tracks):
            hashed = sorted(
                low_similarity,
                key=lambda idx: (
                    stable_u64(seed, target["base_question_id"], track, pool[idx]["base_question_id"]),
                    pool[idx]["base_question_id"],
                ),
            )
            preferred = [idx for idx in hashed if pool[idx]["base_question_id"] not in used_sources]
            fallback = [idx for idx in hashed if pool[idx]["base_question_id"] in used_sources]
            words, spans = _source_spans([pool[idx] for idx in preferred + fallback], max_words)
            if len(words) < max_words:
                raise ValueError(
                    f"Only {len(words)} irrelevant-context words for {target['base_question_id']} track {track}"
                )
            for span in spans:
                if span["start_word"] < max_words:
                    used_sources.add(span["source_base_question_id"])
            for dose in doses:
                if dose == 0:
                    continue
                block_words = words[:dose]
                block = " ".join(block_words)
                source_subset = [span for span in spans if span["start_word"] < dose]
                records.append(
                    {
                        "base_question_id": target["base_question_id"],
                        "category": target["category"],
                        "family": "irrelevant_context",
                        "track": track,
                        "requested_words": dose,
                        "achieved_words": len(block_words),
                        "source_items": source_subset,
                        "context_block": block,
                        "intervened_question": f"Irrelevant context:\n{block}\n\nQuestion:\n{target['question']}",
                    }
                )
    return records


def eligible_deletion_positions(text: str, stopwords: set[str]) -> list[tuple[int, int, str]]:
    url_spans = [match.span() for match in URL_RE.finditer(text)]
    result: list[tuple[int, int, str]] = []
    for match in ALPHA_WORD_RE.finditer(text):
        start, end = match.span()
        token = match.group(0)
        if sum(character.isalpha() for character in token) < 3:
            continue
        if token.casefold() in stopwords:
            continue
        if any(start < url_end and end > url_start for url_start, url_end in url_spans):
            continue
        if any(character.isdigit() for character in token):
            continue
        result.append((start, end, token))
    return result


def _replace_positions(text: str, positions: list[tuple[int, int, str]], deleted: set[int]) -> str:
    chunks: list[str] = []
    cursor = 0
    for position_index, (start, end, _token) in enumerate(positions):
        chunks.append(text[cursor:start])
        chunks.append("___" if position_index in deleted else text[start:end])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def build_content_deletion(
    selected: list[dict[str, Any]],
    doses: list[float],
    tracks: int,
    seed: int,
    stopwords: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in selected:
        positions = eligible_deletion_positions(target["question"], stopwords)
        for track in range(tracks):
            ordering = sorted(
                range(len(positions)),
                key=lambda idx: (
                    stable_u64(seed, target["base_question_id"], track, idx, positions[idx][2].casefold()),
                    idx,
                ),
            )
            for dose in doses:
                if dose == 0:
                    continue
                delete_count = min(len(positions), int(math.floor(len(positions) * dose + 0.5)))
                deleted_indices = sorted(ordering[:delete_count])
                intervened = _replace_positions(target["question"], positions, set(deleted_indices))
                records.append(
                    {
                        "base_question_id": target["base_question_id"],
                        "category": target["category"],
                        "family": "content_deletion",
                        "track": track,
                        "requested_fraction": dose,
                        "achieved_fraction": delete_count / len(positions) if positions else 0.0,
                        "eligible_token_count": len(positions),
                        "deleted_positions": deleted_indices,
                        "deleted_tokens": [positions[idx][2] for idx in deleted_indices],
                        "track_seed": stable_u64(seed, target["base_question_id"], track),
                        "intervened_question": intervened,
                    }
                )
    return records


def build_option_permutations(
    selected: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in selected:
        identity = tuple(range(len(target["options"])))
        permutations: set[tuple[int, ...]] = set()
        attempt = 0
        while len(permutations) < count:
            candidate = list(identity)
            rng = random.Random(stable_u64(seed, target["base_question_id"], "perm", attempt))
            rng.shuffle(candidate)
            permutation = tuple(candidate)
            attempt += 1
            if permutation != identity:
                permutations.add(permutation)
            if attempt > 10_000:
                raise RuntimeError(f"Cannot create permutations for {target['base_question_id']}")
        for permutation_index, permutation in enumerate(sorted(permutations)):
            presented_options = [target["options"][original] for original in permutation]
            presented_gold = permutation.index(target["answer_index"])
            records.append(
                {
                    "base_question_id": target["base_question_id"],
                    "category": target["category"],
                    "family": "option_permutation",
                    "permutation_index": permutation_index,
                    "presented_to_original": list(permutation),
                    "original_to_presented": [permutation.index(index) for index in identity],
                    "presented_options": presented_options,
                    "presented_answer_index": presented_gold,
                    "presented_answer_label": LABELS[presented_gold],
                }
            )
    return records


def _prompt_row(
    target: dict[str, Any],
    prompt_id: str,
    condition: str,
    question: str,
    options: list[str],
    answer_index: int,
    **metadata: Any,
) -> dict[str, Any]:
    content = canonical_prompt(question, options)
    return {
        "prompt_id": prompt_id,
        "base_question_id": target["base_question_id"],
        "category": target["category"],
        "condition": condition,
        "question": question,
        "options": options,
        "option_labels": list(LABELS[: len(options)]),
        "answer_index": answer_index,
        "answer_label": LABELS[answer_index],
        "semantic_answer_index": target["answer_index"],
        "semantic_answer_label": LABELS[target["answer_index"]],
        "canonical_content": content,
        "canonical_content_sha256": __import__("hashlib").sha256(content.encode()).hexdigest(),
        **metadata,
    }


def build_prompt_manifest(
    selected: list[dict[str, Any]],
    irrelevant: list[dict[str, Any]],
    deletion: list[dict[str, Any]],
    permutations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = {row["base_question_id"]: row for row in selected}
    prompts: list[dict[str, Any]] = []
    for target in selected:
        prompts.append(
            _prompt_row(
                target,
                f"{target['base_question_id']}__clean",
                "clean",
                target["question"],
                target["options"],
                target["answer_index"],
                family="clean",
                dose=0,
                track=None,
                presented_to_original=list(range(len(target["options"]))),
            )
        )
    for row in irrelevant:
        target = targets[row["base_question_id"]]
        dose = row["requested_words"]
        track = row["track"]
        prompts.append(
            _prompt_row(
                target,
                f"{target['base_question_id']}__irrelevant_context__t{track}__d{dose}",
                "irrelevant_context",
                row["intervened_question"],
                target["options"],
                target["answer_index"],
                family="irrelevant_context",
                dose=dose,
                track=track,
                presented_to_original=list(range(len(target["options"]))),
                achieved_dose=row["achieved_words"],
            )
        )
    for row in deletion:
        target = targets[row["base_question_id"]]
        dose = row["requested_fraction"]
        track = row["track"]
        dose_token = str(dose).replace(".", "p")
        prompts.append(
            _prompt_row(
                target,
                f"{target['base_question_id']}__content_deletion__t{track}__d{dose_token}",
                "content_deletion",
                row["intervened_question"],
                target["options"],
                target["answer_index"],
                family="content_deletion",
                dose=dose,
                track=track,
                presented_to_original=list(range(len(target["options"]))),
                achieved_dose=row["achieved_fraction"],
            )
        )
    for row in permutations:
        target = targets[row["base_question_id"]]
        permutation_index = row["permutation_index"]
        prompts.append(
            _prompt_row(
                target,
                f"{target['base_question_id']}__option_permutation__p{permutation_index}",
                "option_permutation",
                target["question"],
                row["presented_options"],
                row["presented_answer_index"],
                family="option_permutation",
                dose=permutation_index,
                track=None,
                permutation_index=permutation_index,
                presented_to_original=row["presented_to_original"],
            )
        )
    return sorted(prompts, key=lambda row: row["prompt_id"])


def build_generation_manifest(
    selected: list[dict[str, Any]], prompts: list[dict[str, Any]], per_category: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subset = stratified_fixed_sample(selected, per_category, seed)
    prompt_index = {row["prompt_id"]: row for row in prompts}
    generation: list[dict[str, Any]] = []
    for target in subset:
        base_id = target["base_question_id"]
        wanted = [
            f"{base_id}__clean",
            f"{base_id}__irrelevant_context__t0__d512",
            f"{base_id}__content_deletion__t0__d0p4",
        ]
        for scoring_prompt_id in wanted:
            prompt = prompt_index[scoring_prompt_id]
            generation.append(
                {
                    "generation_prompt_id": f"generation__{scoring_prompt_id}",
                    "scoring_prompt_id": scoring_prompt_id,
                    "base_question_id": base_id,
                    "category": target["category"],
                    "condition": prompt["condition"],
                    "canonical_content": prompt["canonical_content"],
                    "option_labels": prompt["option_labels"],
                    "answer_index": prompt["answer_index"],
                    "semantic_answer_index": prompt["semantic_answer_index"],
                }
            )
    return subset, sorted(generation, key=lambda row: row["generation_prompt_id"])


def validate_artifacts(
    selected: list[dict[str, Any]],
    irrelevant: list[dict[str, Any]],
    deletion: list[dict[str, Any]],
    permutations: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    generation: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_base = config["expected_categories"] * config["base_questions_per_category"]
    expected_generation_questions = (
        config["expected_categories"] * config["generation_questions_per_category"]
    )
    if len(selected) != expected_base:
        errors.append(f"base question count {len(selected)} != {expected_base}")
    category_counts = Counter(row["category"] for row in selected)
    if len(category_counts) != config["expected_categories"]:
        errors.append(f"category count {len(category_counts)} != {config['expected_categories']}")
    if any(count != config["base_questions_per_category"] for count in category_counts.values()):
        errors.append(f"unbalanced category counts: {category_counts}")
    if len(prompts) != config["scoring_prompts_per_model"]:
        errors.append(f"prompt count {len(prompts)} != {config['scoring_prompts_per_model']}")
    if len(generation) != config["generation_prompts_per_model"]:
        errors.append(f"generation count {len(generation)} != {config['generation_prompts_per_model']}")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        errors.append("duplicate prompt IDs")
    if len({row["generation_prompt_id"] for row in generation}) != len(generation):
        errors.append("duplicate generation prompt IDs")
    selected_by_id = {row["base_question_id"]: row for row in selected}
    for row in prompts:
        target = selected_by_id[row["base_question_id"]]
        if row["semantic_answer_index"] != target["answer_index"]:
            errors.append(f"semantic gold changed for {row['prompt_id']}")
        if not row["canonical_content"].strip():
            errors.append(f"empty prompt {row['prompt_id']}")
        mapping = row["presented_to_original"]
        if sorted(mapping) != list(range(len(target["options"]))):
            errors.append(f"invalid semantic mapping {row['prompt_id']}")
        remapped_options = [row["options"][mapping.index(index)] for index in range(len(mapping))]
        if remapped_options != target["options"]:
            errors.append(f"answer options changed for {row['prompt_id']}")
    irrel_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in irrelevant:
        irrel_groups[(row["base_question_id"], row["track"])].append(row)
        if row["achieved_words"] > row["requested_words"]:
            errors.append(f"context exceeds dose for {row['base_question_id']}")
    for key, group in irrel_groups.items():
        ordered = sorted(group, key=lambda row: row["requested_words"])
        previous: list[str] = []
        for row in ordered:
            current = WORD_RE.findall(row["context_block"])
            if current[: len(previous)] != previous:
                errors.append(f"non-nested context for {key}")
            previous = current
    deletion_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deletion:
        deletion_groups[(row["base_question_id"], row["track"])].append(row)
    for key, group in deletion_groups.items():
        previous_deleted: set[int] = set()
        for row in sorted(group, key=lambda item: item["requested_fraction"]):
            current_deleted = set(row["deleted_positions"])
            if not previous_deleted.issubset(current_deleted):
                errors.append(f"non-nested deletion for {key}")
            previous_deleted = current_deleted
    tracks_by_base: dict[str, dict[int, str]] = defaultdict(dict)
    for row in irrelevant:
        if row["requested_words"] == max(config["irrelevant_context_doses"]):
            tracks_by_base[row["base_question_id"]][row["track"]] = row["context_block"]
    identical_context_tracks = sum(
        len(set(track_map.values())) != len(track_map) for track_map in tracks_by_base.values()
    )
    if identical_context_tracks:
        errors.append(f"{identical_context_tracks} questions have identical irrelevant-context tracks")
    generation_questions = {row["base_question_id"] for row in generation}
    if len(generation_questions) != expected_generation_questions:
        errors.append("generation subset question count mismatch")
    result = {
        "passed": not errors,
        "errors": errors,
        "base_question_count": len(selected),
        "category_counts": dict(sorted(category_counts.items())),
        "irrelevant_context_record_count": len(irrelevant),
        "content_deletion_record_count": len(deletion),
        "option_permutation_record_count": len(permutations),
        "scoring_prompt_count": len(prompts),
        "generation_prompt_count": len(generation),
    }
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return result


def build_from_rows(
    raw_rows: list[dict[str, Any]], root: Path, dataset_revision: str
) -> dict[str, Any]:
    config = load_experiment_config(root)
    normalized = normalize_dataset_rows(raw_rows)
    selected = stratified_fixed_sample(
        normalized, config["base_questions_per_category"], config["sample_seed"]
    )
    selected_ids = {row["base_question_id"] for row in selected}
    pool = [row for row in normalized if row["base_question_id"] not in selected_ids]
    stopwords = {
        line.strip().casefold()
        for line in (root / "configs" / "stopwords_en.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    irrelevant = build_irrelevant_context(
        selected,
        pool,
        config["irrelevant_context_doses"],
        config["irrelevant_context_tracks"],
        config["sample_seed"],
    )
    deletion = build_content_deletion(
        selected,
        config["content_deletion_doses"],
        config["content_deletion_tracks"],
        config["sample_seed"],
        stopwords,
    )
    permutations = build_option_permutations(
        selected, config["option_permutations"], config["sample_seed"]
    )
    prompts = build_prompt_manifest(selected, irrelevant, deletion, permutations)
    generation_subset, generation = build_generation_manifest(
        selected,
        prompts,
        config["generation_questions_per_category"],
        config["generation_seed"],
    )
    validation = validate_artifacts(
        selected, irrelevant, deletion, permutations, prompts, generation, config
    )

    files: dict[str, list[dict[str, Any]]] = {
        "data/mmlu_pro_selected_560.jsonl": selected,
        "data/mmlu_pro_generation_subset_140.jsonl": generation_subset,
        "data/interventions/irrelevant_context_v2.jsonl": irrelevant,
        "data/interventions/content_deletion_v2.jsonl": deletion,
        "data/interventions/option_permutation_control.jsonl": permutations,
        "data/manifests/scoring_prompts.jsonl": prompts,
        "data/manifests/generation_prompts.jsonl": generation,
    }
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for relative, rows in files.items():
        count, digest = write_jsonl(root / relative, rows)
        hashes[relative] = digest
        counts[relative] = count
        atomic_write_text(root / "data" / "manifests" / f"{Path(relative).name}.sha256", f"{digest}  {relative}\n")
    source_manifest = {
        "schema_version": "pier_mmlu_pro_source_v2",
        "dataset_id": config["dataset_id"],
        "dataset_revision": dataset_revision,
        "split": config["dataset_split"],
        "raw_row_count": len(raw_rows),
        "normalized_row_count": len(normalized),
        "selected_id_sha256": __import__("hashlib").sha256(
            canonical_json_bytes(sorted(selected_ids))
        ).hexdigest(),
        "counts": counts,
        "hashes": hashes,
        "validation": validation,
    }
    atomic_write_json(root / "data" / "manifests" / "dataset_source.json", source_manifest)
    return source_manifest


def prepare(root: Path) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    config = load_experiment_config(root)
    api = HfApi()
    info = api.dataset_info(config["dataset_id"])
    revision = info.sha
    if revision is None:
        raise RuntimeError("Hugging Face did not return an immutable dataset revision")
    dataset = load_dataset(
        config["dataset_id"],
        split=config["dataset_split"],
        revision=revision,
        cache_dir=str(root / "hf_cache" / "datasets"),
    )
    rows = [dict(row) for row in dataset]
    return build_from_rows(rows, root, revision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    manifest = prepare(args.root.resolve())
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
