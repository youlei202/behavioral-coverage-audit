from __future__ import annotations

from pier_llm.data_prep import (
    build_content_deletion,
    build_irrelevant_context,
    build_option_permutations,
    build_prompt_manifest,
    canonical_prompt,
)


def _row(index: int, category: str, question: str) -> dict[str, object]:
    return {
        "base_question_id": f"q{index}",
        "dataset_index": index,
        "category": category,
        "question": question,
        "options": ["Alpha choice", "Beta choice", "Gamma choice", "Delta choice"],
        "answer_index": index % 4,
        "answer_label": "ABCD"[index % 4],
    }


def test_canonical_prompt_is_exact() -> None:
    rendered = canonical_prompt("What is tested?", ["One", "Two"])
    assert rendered.startswith("You are answering a multiple-choice question.\n\nQuestion:\n")
    assert "(A) One\n(B) Two" in rendered
    assert rendered.endswith("Final answer:")


def test_nested_interventions_and_semantic_permutations_are_deterministic() -> None:
    selected = [
        _row(0, "cat-a", "Which scientific principle explains the observed system behavior?"),
        _row(1, "cat-b", "Which historical process best accounts for the stated outcome?"),
    ]
    pool = [
        _row(
            index + 2,
            f"pool-{index % 3}",
            f"Background passage {index} discusses unrelated astronomy geology literature and "
            "economic policy through several independent descriptive statements.",
        )
        for index in range(30)
    ]
    irrelevant_a = build_irrelevant_context(selected, pool, [0, 4, 8], 3, 123)
    irrelevant_b = build_irrelevant_context(selected, pool, [0, 4, 8], 3, 123)
    assert irrelevant_a == irrelevant_b
    for base_id in ("q0", "q1"):
        for track in range(3):
            group = [
                row
                for row in irrelevant_a
                if row["base_question_id"] == base_id and row["track"] == track
            ]
            short, long = sorted(group, key=lambda row: row["requested_words"])
            assert long["context_block"].split()[:4] == short["context_block"].split()

    stopwords = {"which", "the", "for", "and", "is", "a"}
    deletion = build_content_deletion(selected, [0.0, 0.2, 0.4], 3, 321, stopwords)
    for base_id in ("q0", "q1"):
        for track in range(3):
            group = sorted(
                (
                    row
                    for row in deletion
                    if row["base_question_id"] == base_id and row["track"] == track
                ),
                key=lambda row: row["requested_fraction"],
            )
            assert set(group[0]["deleted_positions"]).issubset(group[1]["deleted_positions"])

    permutations = build_option_permutations(selected, 3, 999)
    assert permutations == build_option_permutations(selected, 3, 999)
    assert len(permutations) == 6
    for permutation in permutations:
        mapping = permutation["presented_to_original"]
        assert mapping != list(range(4))
        assert mapping[permutation["presented_answer_index"]] == selected[
            int(permutation["base_question_id"][1:])
        ]["answer_index"]
    prompts = build_prompt_manifest(selected, irrelevant_a, deletion, permutations)
    assert len(prompts) == 2 + 12 + 12 + 6
    assert len({row["prompt_id"] for row in prompts}) == len(prompts)
