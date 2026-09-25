from __future__ import annotations

from pathlib import Path

from pier_llm_v23.inference import SCORE_SCHEMA, validate_shard, write_shard


def test_shard_completion_checksum_and_resume(tmp_path: Path) -> None:
    record = {"id": "test/model", "revision": "abc"}
    rows = [
        {
            "schema_version": SCORE_SCHEMA,
            "model_id": record["id"],
            "prompt_id": f"p{index}",
            "value": float(index),
        }
        for index in range(3)
    ]
    data, metadata, completion = write_shard(
        tmp_path,
        tmp_path / "output",
        0,
        rows,
        schema=SCORE_SCHEMA,
        record=record,
        manifest_sha256="manifest",
        prompt_key="prompt_id",
    )
    expected = [row["prompt_id"] for row in rows]
    assert completion.is_file()
    assert validate_shard(
        data,
        metadata,
        schema=SCORE_SCHEMA,
        model_id=record["id"],
        revision=record["revision"],
        manifest_sha256="manifest",
        expected_prompt_ids=expected,
        prompt_key="prompt_id",
    )
    with data.open("ab") as handle:
        handle.write(b"corrupt")
    assert not validate_shard(
        data,
        metadata,
        schema=SCORE_SCHEMA,
        model_id=record["id"],
        revision=record["revision"],
        manifest_sha256="manifest",
        expected_prompt_ids=expected,
        prompt_key="prompt_id",
    )

