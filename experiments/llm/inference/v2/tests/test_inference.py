from __future__ import annotations

from types import SimpleNamespace

import torch

from pier_llm.inference import (
    GENERATION_SCHEMA,
    SCORE_SCHEMA,
    _run_batched_with_oom_backoff,
    _validate_shard,
    _write_shard,
    score_prompt_batch,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(character) + 2 for character in text]}

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value - 2) for value in ids)


class UniformModel:
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool) -> object:
        del attention_mask, use_cache
        logits = torch.zeros((*input_ids.shape, 256), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def test_score_schema_and_semantic_remapping() -> None:
    prompt = {
        "prompt_id": "p1",
        "base_question_id": "q1",
        "category": "c",
        "condition": "option_permutation",
        "family": "option_permutation",
        "dose": 0,
        "track": None,
        "permutation_index": 0,
        "option_labels": ["A", "B", "C"],
        "presented_to_original": [2, 0, 1],
        "answer_index": 1,
        "semantic_answer_index": 0,
    }
    rows = score_prompt_batch(
        UniformModel(), CharacterTokenizer(), [prompt], {"p1": "Prompt:"}, device="cpu"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == SCORE_SCHEMA
    assert abs(sum(row["probabilities"]) - 1) < 1e-12
    assert row["semantic_probabilities"] == row["probabilities"]
    assert row["candidate_token_counts"] == [4, 4, 4]


def test_shard_checksum_and_resume_validation(tmp_path) -> None:
    record = {"id": "model", "revision": "abc"}
    rows = [
        {
            "schema_version": SCORE_SCHEMA,
            "model_id": "model",
            "model_revision": "abc",
            "prompt_id": f"p{index}",
            "value": index,
        }
        for index in range(3)
    ]
    data_path, metadata_path = _write_shard(
        tmp_path,
        tmp_path / "raw",
        0,
        rows,
        schema=SCORE_SCHEMA,
        record=record,
        prompt_manifest_sha256="manifest",
        prompt_key="prompt_id",
    )
    assert _validate_shard(
        data_path,
        metadata_path,
        schema=SCORE_SCHEMA,
        model_id="model",
        revision="abc",
        prompt_manifest_sha256="manifest",
        expected_prompt_ids=["p0", "p1", "p2"],
    )
    assert not _validate_shard(
        data_path,
        metadata_path,
        schema=SCORE_SCHEMA,
        model_id="model",
        revision="wrong",
        prompt_manifest_sha256="manifest",
        expected_prompt_ids=["p0", "p1", "p2"],
    )


def test_cuda_oom_batch_backoff_preserves_order(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def limited_batch(prompts: list[dict[str, int]]) -> list[dict[str, int]]:
        if len(prompts) > 2:
            raise torch.OutOfMemoryError("synthetic CUDA OOM")
        return prompts

    prompts = [{"index": index} for index in range(7)]
    rows, effective_batch_size, events = _run_batched_with_oom_backoff(
        limited_batch,
        prompts,
        batch_size=4,
        device="cuda:0",
        model_id="test/model",
        phase="scoring",
    )

    assert rows == prompts
    assert effective_batch_size == 2
    assert [(event["previous_batch_size"], event["new_batch_size"]) for event in events] == [
        (4, 2)
    ]


def test_generation_shard_uses_generation_prompt_id(tmp_path) -> None:
    record = {"id": "model", "revision": "abc"}
    rows = [
        {
            "schema_version": GENERATION_SCHEMA,
            "model_id": "model",
            "model_revision": "abc",
            "generation_prompt_id": f"g{index}",
            "generated_text": "(A)",
        }
        for index in range(3)
    ]
    data_path, metadata_path = _write_shard(
        tmp_path,
        tmp_path / "generation",
        0,
        rows,
        schema=GENERATION_SCHEMA,
        record=record,
        prompt_manifest_sha256="generation-manifest",
        prompt_key="generation_prompt_id",
    )

    assert _validate_shard(
        data_path,
        metadata_path,
        schema=GENERATION_SCHEMA,
        model_id="model",
        revision="abc",
        prompt_manifest_sha256="generation-manifest",
        expected_prompt_ids=["g0", "g1", "g2"],
    )
