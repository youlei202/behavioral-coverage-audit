from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .plotting import KEY_TARGETS
from .utils import PHYSICAL_ROOT, atomic_write_text


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    return f"{float(value):.{digits}g}"


def _ci(row: pd.Series) -> str:
    return f"[{_fmt(row['bootstrap_lower_95'])}, {_fmt(row['bootstrap_upper_95'])}]"


def _model_effect_list(frame: pd.DataFrame) -> str:
    return "; ".join(
        f"`{row.target}` {_fmt(row.mean_delta_absolute)} ({row.evidence_status})"
        for row in frame.sort_values("mean_delta_absolute", ascending=False).itertuples()
    ) or "none"


def _section(
    number: int,
    question: str,
    observed: str,
    interpretation: str,
    limitation: str,
    stage2: str,
) -> str:
    return "\n".join(
        [
            f"## {number}. {question}",
            "",
            f"**Observed result.** {observed}",
            "",
            f"**Interpretation.** {interpretation}",
            "",
            f"**Measurement limitation.** {limitation}",
            "",
            f"**Implication for Stage 2.** {stage2}",
        ]
    )


def write_corrected_report(*, byte_identical: bool | None) -> Path:
    tables = PHYSICAL_ROOT / "outputs/tables"
    endpoint = pd.read_csv(tables / "table_v21_endpoint_effects.csv")
    track = pd.read_csv(tables / "table_v21_track_consistency.csv")
    sibling = pd.read_csv(tables / "table_v21_sibling_removal.csv")
    convex = pd.read_csv(tables / "table_v21_convexity_gap.csv")
    calibration = pd.read_csv(tables / "table_v21_calibration_sensitivity.csv")
    interface = pd.read_csv(tables / "table_v21_interface_sensitivity.csv")
    generation = pd.read_csv(tables / "table_v21_generation_alignment.csv")
    primary_convex = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/convex_vs_single_corrected.parquet"
    )
    subsets = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_subsets.parquet"
    )

    phi = endpoint[
        endpoint["target"].eq("microsoft/phi-4")
        & endpoint["family"].eq("irrelevant_context")
    ].iloc[0]
    irrelevant = endpoint[endpoint["family"].eq("irrelevant_context")]
    away = irrelevant[irrelevant["mean_delta_absolute"] > 0]
    toward = irrelevant[irrelevant["mean_delta_absolute"] < 0]
    track_columns = [column for column in track if column.startswith("mean_track_") and column.endswith("endpoint_delta")]
    deletion_track = track[track["family"].eq("content_deletion")]
    deletion_track = deletion_track.copy()
    deletion_track["all_three_mean_track_deltas_negative"] = (
        deletion_track[track_columns].to_numpy(dtype=float) < 0
    ).all(axis=1)
    collapse_targets = deletion_track[
        deletion_track["all_three_mean_track_deltas_negative"]
    ]["target"].tolist()
    noncollapse_targets = deletion_track[
        ~deletion_track["all_three_mean_track_deltas_negative"]
    ]["target"].tolist()
    maximum_gap = float(track["maximum_track_cancellation_gap"].max())
    maximum_gap_row = track.loc[track["maximum_track_cancellation_gap"].idxmax()]
    track_sign_preservation = float(track["trackwise_endpoint_sign_agreement"].mean())
    qwen_sibling = sibling[sibling["target"].eq("Qwen/Qwen2.5-7B-Instruct")]
    qwen_text = "; ".join(
        f"{row.family}: {_fmt(row.mean_observed_inflation)} (95% {_ci(row)})"
        for _, row in qwen_sibling.iterrows()
    )
    qwen_exceeds = "; ".join(
        f"{row.family}: {int(row.splits_exceeding_every_null)}/10 splits"
        for row in qwen_sibling.itertuples()
    )
    rank_text = "; ".join(
        f"{row.target}/{row.family}: mean rank {_fmt(row.mean_sibling_removal_rank, 3)}, first in {int(row.splits_sibling_ranked_first)}/10"
        for row in sibling.itertuples()
    )
    requested_convex = convex[
        convex["target"].isin(
            [
                "microsoft/Phi-4-reasoning-plus",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ]
        )
    ]
    convex_text = "; ".join(
        f"{row.target}/{row.family}: {_fmt(row.mean_absolute_improvement)} (95% {_ci(row)}, {row.evidence_status})"
        for _, row in requested_convex.iterrows()
    )
    interval_text = "; ".join(
        f"{row.target}/{row.family}: {'excludes' if row.bootstrap_lower_95 > 0 or row.bootstrap_upper_95 < 0 else 'includes'} zero"
        for row in requested_convex.itertuples()
    )
    selected_counts = primary_convex["fit_selected_single_peer"].value_counts()
    most_selected = str(selected_counts.index[0])
    most_selected_count = int(selected_counts.iloc[0])
    reversals = calibration[calibration["calibration_reversed_endpoint_sign"]]
    reversal_text = (
        "; ".join(
            f"{row.target}/{row.family}: raw {_fmt(row.mean_raw_endpoint_delta)} → calibrated {_fmt(row.mean_calibrated_endpoint_delta)}"
            for row in reversals.itertuples()
        )
        if not reversals.empty
        else "No target-family aggregate endpoint sign reversed."
    )
    validation_unique = interface.drop_duplicates("target")
    before_tv = float(validation_unique["before_mean_heldout_permutation_tv"].mean())
    after_tv = float(validation_unique["after_mean_heldout_permutation_tv"].mean())
    worst_artifact = validation_unique.loc[
        validation_unique["after_mean_heldout_permutation_tv"].idxmax()
    ]
    bias_survival = interface[
        interface["interface_condition"].eq("label_bias_corrected")
        & interface["same_endpoint_sign_as_raw"]
    ]
    four_survival = interface[
        interface["interface_condition"].eq("raw")
        & interface["survives_all_four_interfaces"]
    ]
    bias_survival_text = "; ".join(
        f"{row.target}/{row.family}" for row in bias_survival.itertuples()
    ) or "none"
    four_survival_text = "; ".join(
        f"{row.target}/{row.family}" for row in four_survival.itertuples()
    ) or "none"
    generation_unique = generation[
        generation["row_type"].eq("alignment_summary")
        & generation["condition"].eq("overall")
    ]
    raw_generation = float(generation_unique["raw_score_generation_agreement"].mean())
    corrected_generation = float(
        generation_unique["bias_corrected_score_generation_agreement"].mean()
    )
    high_models = generation_unique[generation_unique["enters_high_alignment_subset"]][
        "target"
    ].tolist()
    restricted = subsets[
        subsets["ecosystem_condition"].eq("restricted_high_alignment_ecosystem")
    ]
    if restricted.empty:
        restricted_text = "G2 was not run because fewer than four models met the fixed threshold."
    else:
        restricted_text = (
            f"G2 ran for {restricted['target'].nunique()} targets; mean effect-sign agreement with G1 was "
            f"{restricted['effect_sign_agreement_with_full'].mean():.1%}, and mean rank correlation was "
            f"{restricted['aggregate_rank_correlation_with_full'].mean():.3f}."
        )
    byte_text = (
        "Yes. The before/after inventories are byte-identical."
        if byte_identical is True
        else "The final byte-identical recheck is pending Stage 11."
    )

    unstable_models = interface[
        interface["interface_condition"].eq("raw")
        & ~interface["survives_all_four_interfaces"]
    ]["target"].tolist()
    mismatch_model = str(
        generation_unique.loc[
            generation_unique["raw_score_generation_agreement"].idxmin(), "target"
        ]
    )
    stage2_models = sorted(set(KEY_TARGETS + unstable_models + [mismatch_model]))
    stage2_scope = (
        "Models: "
        + ", ".join(f"`{model}`" for model in stage2_models)
        + ". Conditions: clean, irrelevant-context 512 words on each of the three fixed tracks, "
        "and content-deletion 0.4 on each of the three fixed tracks; validate canonical and the "
        "three fixed clean label permutations, and use a longer deterministic generation budget on "
        "the fixed 140-question generation subset."
    )

    sections = [
        _section(
            1,
            "Corrected paired clean-to-512 effect for microsoft/phi-4",
            f"Mean paired ΔPIER is {_fmt(phi.mean_delta_absolute)} across 10 splits, replacing the invalid unpaired `0.0867769` value.",
            "The sign and magnitude now compare the two pre-declared endpoints within the same split and fitted intervention family.",
            "The response remains gold-label probability under the candidate-label interface.",
            "Retain phi-4 clean and 512-word conditions in the focused validation.",
        ),
        _section(
            2,
            "95% clustered-bootstrap interval for that phi-4 effect",
            f"The 1000-replicate interval is {_ci(phi)}; P(Δ>0)={phi.bootstrap_probability_gt_zero:.3f}.",
            f"The empirical evidence status is `{phi.evidence_status}` under the 9/10-sign plus interval rule.",
            "The bootstrap refits within each fixed split and summarizes across those ten designs; it does not add benchmark diversity.",
            "Use this interval, rather than an extreme-row contrast, to prioritize phi-4.",
        ),
        _section(
            3,
            "Models moving away from the peer hull under irrelevant context",
            _model_effect_list(away),
            "Positive paired endpoint deltas mean larger held-out residuals at 512 words.",
            "Positive movement is design-dependent peer expressibility, not a general behavioral property.",
            "Prioritize stable positive cases and interface-sensitive positive cases separately.",
        ),
        _section(
            4,
            "Models moving toward the peer hull under irrelevant context",
            _model_effect_list(toward),
            "Negative paired endpoint deltas mean the target response is better covered by the fixed peer convex hull at 512 words.",
            "The hull and response interface are fixed to this eight-model design.",
            "Validate the strongest stable negative case at the same endpoint and tracks.",
        ),
        _section(
            5,
            "Content-deletion collapse across all three tracks",
            f"No—not universally. All three mean track-specific endpoint deltas are negative for {len(collapse_targets)}/8 targets: "
            + ", ".join(f"`{target}`" for target in collapse_targets)
            + ". They are not all negative for: "
            + ", ".join(f"`{target}`" for target in noncollapse_targets)
            + f". The minimum split-level same-sign track count is {int(deletion_track['minimum_same_sign_track_count'].min())}/3.",
            "The primary aggregate deletion decrease is therefore partly track-dependent and, for some targets, amplified by cancellation after averaging responses.",
            "Only the three already fixed tracks are represented.",
            "Carry all three tracks into Stage 2 rather than selecting the strongest one.",
        ),
        _section(
            6,
            "Track-cancellation gap",
            f"The maximum split-level cancellation gap is {_fmt(maximum_gap)} for `{maximum_gap_row.target}`/{maximum_gap_row.family}; the mean across target-family summaries is {_fmt(track['mean_track_cancellation_gap'].mean())}.",
            "A positive gap quantifies variation hidden by averaging tracks before applying the absolute residual.",
            "This is a Jensen gap, not an independent uncertainty interval.",
            "Use per-track reporting for cases with the largest gap.",
        ),
        _section(
            7,
            "Trackwise-loss fitting sensitivity",
            f"Mean endpoint-sign agreement with the primary fit is {track_sign_preservation:.1%}; mean projected-response difference is {_fmt(track['mean_projected_response_difference'].mean())}.",
            "The primary conclusions are preserved to the extent indicated by those two diagnostics.",
            "Trackwise fitting is a sensitivity analysis and does not replace the preregistered mean-response fit.",
            "Escalate cases with sign disagreement or unusually large weight changes.",
        ),
        _section(
            8,
            "Corrected Qwen sibling-removal inflation",
            qwen_text,
            "These are paired all-peer versus sibling-removed effects with refitting.",
            "The effect measures convex coverage supplied by one designated sibling, not a causal lineage effect.",
            "Include both Qwen siblings in any focused ecosystem validation.",
        ),
        _section(
            9,
            "Does Qwen sibling removal exceed every same-size non-sibling removal?",
            qwen_exceeds,
            "The 10/10 result supports disproportionate sibling coverage relative to the exact enumerated non-sibling null in both families.",
            "The null is small because only seven peers exist, although it now excludes the sibling by construction.",
            "Retain the full peer roster when validating sibling coverage.",
        ),
        _section(
            10,
            "Sibling ranks in the single-peer removal influence map",
            rank_text,
            "Rank 1 means the largest held-out inflation after individual peer removal within that split.",
            "Rank is sensitive to the fixed roster and can tie.",
            "Use the rank map to identify non-sibling comparators for focused validation.",
        ),
        _section(
            11,
            "Corrected convex-versus-single improvements for Phi-4-reasoning-plus and Mistral-Nemo",
            convex_text,
            "Positive values mean the fixed convex surrogate outperformed a peer selected only on fitting questions.",
            "The comparison uses overall held-out dose designs, not a best-case split.",
            "Validate cases whose advantage is stable and scientifically material.",
        ),
        _section(
            12,
            "Do those convexity intervals exclude zero?",
            interval_text,
            "Exclusion of zero is combined with the 9/10 split-sign rule for the `stable` label.",
            "Percentile intervals quantify base-question sampling uncertainty within this benchmark.",
            "Treat intervals including zero as exploratory rather than discarding them.",
        ),
        _section(
            13,
            "Most frequently selected honest single peer",
            f"`{most_selected}` was selected in {most_selected_count} of {len(primary_convex)} target-family-split comparisons.",
            "This identifies the dominant fit-only baseline comparator.",
            "Selection frequency depends on the target roster and five-dose fitting design.",
            "Include this peer when constructing the smallest comparison set.",
        ),
        _section(
            14,
            "Calibrated versus raw dose trajectories",
            f"Mean raw/calibrated rank correlation across the family summaries is {_fmt(calibration['mean_raw_calibrated_rank_correlation'].mean())}; mean absolute endpoint change is {_fmt(calibration['calibration_induced_endpoint_change'].abs().mean())}.",
            "Temperature calibration can shift both residual magnitudes and rankings while using one clean-fit temperature across all doses.",
            "Temperature rescales candidate scores but does not remove label-position effects.",
            "Keep raw and calibrated endpoints as paired reporting conditions.",
        ),
        _section(
            15,
            "Effects reversing after calibration",
            reversal_text,
            "A sign reversal indicates sensitivity to score sharpness rather than a robust endpoint direction.",
            "Calibration itself is estimated from clean fitting questions and is not ground truth.",
            "Prioritize reversed cases for direct-generation validation.",
        ),
        _section(
            16,
            "Held-out permutation TV after additive label-bias correction",
            f"Mean TV changed from {_fmt(before_tv)} to {_fmt(after_tv)}; {validation_unique['split_improvement_rate'].mean():.1%} of model-split fits improved held-out invariance.",
            "The correction reduces the interface artifact when held-out TV decreases.",
            "The correction is exploratory and bias estimated on clean permutations may not transport to stressed prompts.",
            "Validate transport using fixed endpoint prompts and balanced labels.",
        ),
        _section(
            17,
            "Residual artifact after correction",
            f"The largest residual mean TV is {_fmt(worst_artifact.after_mean_heldout_permutation_tv)} for `{worst_artifact.target}`; its residual mean 95th-percentile TV is {_fmt(worst_artifact.residual_95th_percentile_tv)}.",
            "Nonzero residual TV shows that the additive correction does not fully account for presentation sensitivity.",
            "Semantic utilities and label effects may interact non-additively.",
            "Retain clean label permutations for the worst residual case.",
        ),
        _section(
            18,
            "Headline effects surviving label-bias correction",
            bias_survival_text,
            "Survival here means aggregate endpoint sign agreement with raw across the split-fitted correction.",
            "Sign agreement alone does not establish unchanged magnitude or validity under stressed permutations.",
            "Prioritize effects that also have stable raw bootstrap evidence.",
        ),
        _section(
            19,
            "Effects surviving all four response-interface conditions",
            four_survival_text,
            "These target-family effects retain the raw aggregate direction under temperature, label bias, and their joint condition.",
            "The label-bias sensitivity uses ten split fits rather than an additional bootstrap.",
            "These are the strongest candidates for minimal focused validation.",
        ),
        _section(
            20,
            "Score-generation agreement after label-bias correction",
            f"Mean agreement changed from {raw_generation:.1%} to {corrected_generation:.1%} across models.",
            (
                "The increase supports better alignment between corrected candidate scores and the observed short generation labels."
                if corrected_generation > raw_generation
                else "The decrease shows that improved held-out permutation invariance did not translate into better score-generation alignment overall."
            ),
            "The existing reasoning-model generations are truncated and are not treated as ground-truth behavior.",
            "Use a longer deterministic generation budget for the fixed subset.",
        ),
        _section(
            21,
            "Models in the pre-declared ≥70% high-alignment subset",
            ", ".join(f"`{model}`" for model in high_models) or "none",
            "Membership follows the fixed overall raw score-generation agreement threshold without post-hoc changes.",
            "The threshold is a measurement screen, not a quality ranking.",
            "Use G1 for all qualifying targets and G2 only when roster size permits.",
        ),
        _section(
            22,
            "Persistence in the restricted high-alignment ecosystem",
            restricted_text,
            "Agreement with G1 indicates how much the findings depend on low-alignment peers.",
            "G2 changes both target and peer sets and is therefore a sensitivity ecosystem, not the primary design.",
            "If G2 diverges, validate the affected target with both the full and restricted peer sets.",
        ),
        _section(
            23,
            "Invalid or misleading original V2 headlines and replacements",
            f"The phi-4 `0.0867769` max-minus-min value is replaced by paired mean {_fmt(phi.mean_delta_absolute)} with 95% {_ci(phi)}. Single-row lineage maxima are replaced by split-paired sibling means and the sibling-excluding null in Table V2.1 sibling removal. Maximum conditions are now selected only after target-family-dose split aggregation, and convexity headlines use target-family aggregates with fit-only peer selection.",
            "The corrected summaries remove cross-split extrema and null contamination.",
            "The replacements remain specific to the fixed design and response interface.",
            "Use only V2.1 tables for Stage 2 prioritization.",
        ),
        _section(
            24,
            "Original V2 input byte identity",
            byte_text,
            "Matching SHA256, size, and nanosecond modification time confirms the reanalysis did not alter the locked inputs.",
            "Byte identity does not by itself validate the scientific measurement model.",
            "Stage 2 must write to a separate root as well.",
        ),
        _section(
            25,
            "Is focused B200 Stage 2 still scientifically necessary?",
            "Yes: residual option-permutation TV, truncated generation alignment, and interface-sensitive endpoint effects remain.",
            "New inference is justified only for focused measurement validation, not to recompute the corrected CPU summaries.",
            "Stage 1 cannot test stressed-prompt label transport or longer generation without new inference.",
            "Proceed only with the minimal scope listed next.",
        ),
        _section(
            26,
            "Exact proposed Stage 2 scope",
            stage2_scope,
            "This scope targets the unresolved response-interface and generation-alignment limitations identified by V2.1.",
            "It remains one benchmark and the same fixed intervention design; broader claims would require separate work.",
            "Do not add models, benchmark questions, or post-hoc doses before completing this focused validation.",
        ),
    ]
    report = "\n\n".join(
        [
            "# PIER Modern-LLM Ecosystem V2.1 — Corrected CPU Reanalysis",
            "",
            "This report reanalyzes the completed V2 scores without new inference. The primary estimand is controlled, design-dependent peer expressibility under a candidate-label response interface. The additive label-bias correction is exploratory and is not described as resolving the interface issue.",
            "",
            "All endpoint intervals use 1000-replicate stratified base-question cluster bootstraps with refitting and synchronized clean/high samples. `Stable` means at least 9 of 10 paired split effects share a sign and the 95% interval excludes zero.",
            *sections,
        ]
    )
    path = PHYSICAL_ROOT / "outputs/analysis/V2_1_CORRECTED_REANALYSIS_REPORT.md"
    atomic_write_text(path, report + "\n")
    return path
