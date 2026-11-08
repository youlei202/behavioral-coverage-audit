"""Candidate-position-only FP32 log-softmax, with the historical forward intact.

This is a separately validated potential acceleration. It is NOT activated by
score_endpoints.py. Before adopting it on GPU, compare exact complete output
rows against the historical function on all fixed eight-model pilot cases.
Vocabulary-row kernel selection can depend on shapes/backends, so mathematical
equivalence and CPU equality alone do not establish real-model GPU equality.

The setup and result schema below are copied unchanged from V2.3; only the
post-forward probability extraction and host-transfer path differ. There is no
prefix reuse, length normalization, logits_to_keep, prompt truncation, or dtype
change. The candidate continuation probabilities remain summed token log scores.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

from pier_llm_v23.inference import (
    SCORE_SCHEMA, _candidate_token_ids, _padded_batch,
)
from pier_llm_v23.core import remap_visible_to_semantic


def score_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    rendered_by_id: dict[str, dict[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    sequences: list[list[int]] = []
    descriptors: list[tuple[int, int, list[int], int]] = []
    prompt_lengths: list[int] = []
    candidates_by_count: dict[int, list[list[int]]] = {}
    for prompt_index, prompt in enumerate(prompts):
        rendered_record = rendered_by_id[prompt["prompt_id"]]
        if rendered_record["canonical_content_sha256"] != prompt["canonical_content_sha256"]:
            raise ValueError(f"Rendered/canonical hash mismatch: {prompt['prompt_id']}")
        prompt_ids = [
            int(value)
            for value in tokenizer(rendered_record["rendered_prompt"], add_special_tokens=False)[
                "input_ids"
            ]
        ]
        if len(prompt_ids) != int(rendered_record["prompt_token_count"]):
            raise ValueError(f"Prompt token count drift: {prompt['prompt_id']}")
        prompt_lengths.append(len(prompt_ids))
        option_count = int(prompt["option_count"])
        candidates = candidates_by_count.setdefault(
            option_count, _candidate_token_ids(tokenizer, prompt["option_labels"])
        )
        for candidate_index, candidate_ids in enumerate(candidates):
            sequences.append(prompt_ids + candidate_ids)
            descriptors.append((prompt_index, candidate_index, candidate_ids, len(prompt_ids)))
    input_ids, attention_mask = _padded_batch(sequences, tokenizer.pad_token_id, device)
    # Keep model.forward, padded inputs and precision exactly as in V2.3.
    # Log-softmax is independent across vocabulary rows. Only continuation
    # prediction positions are needed; prompt-only positions are never scored.
    sequence_indices: list[int] = []
    prediction_positions: list[int] = []
    continuation_tokens: list[int] = []
    for sequence_index, descriptor in enumerate(descriptors):
        _, _, candidate_ids, prompt_length = descriptor
        for offset, token_id in enumerate(candidate_ids):
            sequence_indices.append(sequence_index)
            prediction_positions.append(prompt_length + offset - 1)
            continuation_tokens.append(token_id)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        row_index = torch.tensor(sequence_indices, dtype=torch.long, device=outputs.logits.device)
        position_index = torch.tensor(prediction_positions, dtype=torch.long, device=outputs.logits.device)
        token_index = torch.tensor(continuation_tokens, dtype=torch.long, device=outputs.logits.device)
        # Select before converting to float32. The model still returns the full
        # unchanged logits; logits_to_keep or prefix-cache changes are not used.
        selected_logits = outputs.logits[row_index, position_index, :].float()
        selected_log_probabilities = torch.log_softmax(selected_logits, dim=-1)
        chosen_values = selected_log_probabilities.gather(1, token_index[:, None]).squeeze(1)
        # One bulk host transfer replaces one CUDA synchronization per token.
        host_log_probabilities = chosen_values.cpu().tolist()
    del outputs, selected_logits, selected_log_probabilities, chosen_values
    del row_index, position_index, token_index, input_ids, attention_mask
    scores: list[list[float]] = [
        [math.nan] * int(prompt["option_count"]) for prompt in prompts
    ]
    candidate_lengths: list[list[int]] = [
        [0] * int(prompt["option_count"]) for prompt in prompts
    ]
    probability_offset = 0
    for sequence_index, descriptor in enumerate(descriptors):
        prompt_index, candidate_index, candidate_ids, prompt_length = descriptor
        score = 0.0
        # Preserve the original Python-float additions in the original token
        # order; a torch/numpy reduction can change the accumulated rounding.
        for _ in candidate_ids:
            score += float(host_log_probabilities[probability_offset])
            probability_offset += 1
        scores[prompt_index][candidate_index] = score
        candidate_lengths[prompt_index][candidate_index] = len(candidate_ids)
    if probability_offset != len(host_log_probabilities):
        raise AssertionError("Continuation probability indexing is inconsistent")

    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        log_likelihoods = np.asarray(scores[prompt_index], dtype=np.float64)
        shifted = np.exp(log_likelihoods - float(np.max(log_likelihoods)))
        visible = shifted / shifted.sum()
        semantic = remap_visible_to_semantic(visible, prompt["visible_to_semantic"])
        visible_argmax = int(np.argmax(visible))
        semantic_argmax = int(prompt["visible_to_semantic"][visible_argmax])
        rows.append(
            {
                "schema_version": SCORE_SCHEMA,
                "prompt_id": prompt["prompt_id"],
                "source_v2_prompt_id": prompt["source_v2_prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "semantic_condition": prompt["semantic_condition"],
                "condition": prompt["condition"],
                "family": prompt["family"],
                "dose": float(prompt["dose"]),
                "track": prompt["track"],
                "permutation_id": int(prompt["permutation_id"]),
                "option_count": int(prompt["option_count"]),
                "semantic_to_visible": prompt["semantic_to_visible"],
                "visible_to_semantic": prompt["visible_to_semantic"],
                "candidate_continuations": prompt["candidate_continuations"],
                "candidate_sequence_log_likelihoods": log_likelihoods.tolist(),
                "visible_label_probabilities": visible.tolist(),
                "semantic_remapped_probabilities": semantic.tolist(),
                "argmax_visible_index": visible_argmax,
                "argmax_visible_label": prompt["option_labels"][visible_argmax],
                "argmax_semantic_option": semantic_argmax,
                "gold_semantic_option": int(prompt["semantic_answer_index"]),
                "gold_semantic_probability": float(semantic[int(prompt["semantic_answer_index"])]),
                "prompt_token_count": prompt_lengths[prompt_index],
                "candidate_token_counts": candidate_lengths[prompt_index],
                "canonical_prompt_hash": prompt["canonical_content_sha256"],
                "rendered_prompt_hash": rendered_by_id[prompt["prompt_id"]][
                    "rendered_prompt_sha256"
                ],
            }
        )
    return rows

