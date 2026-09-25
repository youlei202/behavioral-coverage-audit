from __future__ import annotations

from pier_llm_v23.utils import load_json, project_root, read_jsonl, sha256_file


def test_full_manifest_counts_are_derived_from_option_counts() -> None:
    root = project_root()
    status_path = root / "status/prompt_manifest_status.json"
    if not status_path.is_file():
        return
    status = load_json(status_path)
    semantic = list(read_jsonl(root / "data/permutations/score_semantic_manifest.jsonl"))
    inference = list(read_jsonl(root / "data/permutations/score_inference_manifest.jsonl"))
    generation = list(read_jsonl(root / "data/generation/long_generation_manifest.jsonl"))
    assert len(semantic) == status["score_all_rotation_rows_per_model"]
    assert len(inference) == status["score_new_inference_rows_per_model"]
    assert len(generation) == status["generation_rows_per_model"]
    assert sha256_file(root / "data/permutations/score_semantic_manifest.jsonl") == status[
        "score_manifest_sha256"
    ]


def test_rotation_zero_is_exact_source_prompt() -> None:
    root = project_root()
    path = root / "data/permutations/score_semantic_manifest.jsonl"
    if not path.is_file():
        return
    rows = [row for row in read_jsonl(path) if row["permutation_id"] == 0]
    assert len(rows) == 560 * 7
    for row in rows:
        assert row["rotation0_reuse"] is True
        assert row["canonical_content_sha256"] == row[
            "source_v2_canonical_content_sha256"
        ]
        assert row["semantic_to_visible"] == list(range(row["option_count"]))
        assert row["visible_to_semantic"] == list(range(row["option_count"]))
