from __future__ import annotations

import pandas as pd

from pier_llm_reanalysis_v22.bootstrap import (
    shard_needs_recompute,
    write_bootstrap_shard,
)


def test_resume_preserves_valid_shard_and_rejects_corruption(tmp_path) -> None:
    path = tmp_path / "target__family__0000_0001.parquet"
    frame = pd.DataFrame({"replicate": [0, 1], "trackwise_delta": [0.1, 0.2]})
    write_bootstrap_shard(
        path,
        frame,
        analysis_type="endpoint",
        target="target/model",
        family="family",
        replicate_start=0,
        replicate_end=1,
    )
    expected = {
        "analysis_type": "endpoint",
        "target": "target/model",
        "family": "family",
        "replicate_start": 0,
        "replicate_end": 1,
    }
    assert shard_needs_recompute(path, **expected) is False
    path.write_bytes(b"corrupt")
    assert shard_needs_recompute(path, **expected) is True
