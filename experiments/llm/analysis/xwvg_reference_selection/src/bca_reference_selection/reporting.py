"""Tables, standalone figures and result-grounded prose for complete real-data runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .resample import INTERVAL_SCOPE, METHODS, frozen_fit_resampling


def summarize_strategy_results(strategy: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal split weighting within every cell and equal target/family macro weighting."""
    keys = ['target', 'family', 'split_seed', 'k', 'method']
    required = {*keys, 'evaluation_tv', 'top_option_agreement'}
    if not required <= set(strategy):
        raise ValueError(f'Missing strategy columns: {sorted(required - set(strategy))}')
    if strategy.duplicated(keys).any():
        raise ValueError('Duplicate strategy split results')
    expected = pd.MultiIndex.from_product([
        config['model_roster'], list(config['families']), config['split_seeds'], config['budgets'], METHODS
    ], names=keys)
    actual = pd.MultiIndex.from_frame(strategy[keys])
    if set(actual) != set(expected):
        raise ValueError('All target/family/split/budget/method results are required')
    metrics = strategy[['evaluation_tv', 'top_option_agreement']].to_numpy(dtype=float)
    if not np.isfinite(metrics).all() or (metrics < -1e-12).any() or (metrics > 1 + 1e-12).any():
        raise ValueError('Invalid strategy metrics')
    grouped = strategy.groupby(['target', 'family', 'k', 'method'], sort=False)[['evaluation_tv', 'top_option_agreement']].mean()
    result = grouped.unstack('method')
    result.columns = [f'{method}_{"tv" if metric == "evaluation_tv" else metric}' for metric, method in result.columns]
    result = result.reset_index()
    macro = result.groupby('k', sort=False).mean(numeric_only=True).reset_index()
    for frame in [result, macro]:
        for baseline, method in [('nearest', 'nearest_k'), ('random', 'uniform_random_expected')]:
            gain = frame[f'{method}_tv'] - frame.collective_tv
            frame[f'{baseline}_minus_collective_tv'] = gain
            frame[f'{baseline}_relative_tv_reduction_percent'] = np.divide(
                100 * gain, frame[f'{method}_tv'], out=np.full(len(frame), np.nan),
                where=frame[f'{method}_tv'].to_numpy() > 0,
            )
            frame[f'collective_minus_{baseline}_agreement'] = frame.collective_top_option_agreement - frame[f'{method}_top_option_agreement']
        frame['protocol'] = config['primary_protocol']
        frame['budget_role'] = frame.k.map({config['primary_budget']: 'primary', config['secondary_budget']: 'secondary'}).fillna('control_or_reference')
    return result, macro


def _peer_list(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(item) for item in value]


def selection_diagnostics(strategy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selected available peers count even if their fitted weight is zero."""
    records = []
    differences = []
    for (target, family, k), group in strategy.groupby(['target', 'family', 'k'], sort=False):
        for method in ('collective', 'nearest_k'):
            subset = group.loc[group.method == method]
            counts = {}
            for value in subset.selected_peers:
                for peer in _peer_list(value):
                    counts[peer] = counts.get(peer, 0) + 1
            for peer, count in sorted(counts.items()):
                records.append(dict(target=target, family=family, k=k, method=method, peer=peer,
                                    selected_splits=count, split_count=len(subset), selection_frequency=count / len(subset)))
        different = 0
        for _, split in group.groupby('split_seed'):
            collective = _peer_list(split.loc[split.method == 'collective', 'selected_peers'].iloc[0])
            nearest = _peer_list(split.loc[split.method == 'nearest_k', 'selected_peers'].iloc[0])
            different += set(collective) != set(nearest)
        nsplit = group.split_seed.nunique()
        differences.append(dict(target=target, family=family, k=k, different_selected_sets=different,
                                split_count=nsplit, different_selected_set_fraction=different / nsplit))
    return pd.DataFrame(records), pd.DataFrame(differences)


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value):
        if isinstance(value, (float, np.floating)):
            return 'undefined' if not np.isfinite(value) else f'{value:.6f}'
        return str(value).replace('|', '\\|')
    return '\n'.join([
        '| ' + ' | '.join(map(str, frame.columns)) + ' |',
        '| ' + ' | '.join('---' for _ in frame.columns) + ' |',
        *('| ' + ' | '.join(render(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)),
    ])


def _short_name(model: str) -> str:
    return model.split('/')[-1]


def make_figure(all_results: pd.DataFrame, macro: pd.DataFrame, intervals: pd.DataFrame, config: dict, output_root: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'collective': '#2166AC', 'nearest_k': '#D6604D', 'uniform_random_expected': '#666666'}
    labels = {'collective': 'Collective', 'nearest_k': 'Nearest k', 'uniform_random_expected': 'Uniform random (expected)'}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})
    budgets = config['budgets']
    macro = macro.set_index('k').loc[budgets]
    x = np.asarray(budgets, dtype=float)
    for mi, method in enumerate(METHODS):
        xpos = x + (mi - 1) * .065
        axes[0].plot(xpos, macro[f'{method}_tv'], marker='o', color=colors[method], label=labels[method])
        ci = intervals.loc[(intervals.scope == 'macro') & (intervals.metric == f'{method}_tv')].set_index('k').loc[budgets]
        axes[0].vlines(xpos, ci.lower, ci.upper, color=colors[method], linewidth=1)
    axes[0].set(xticks=x, xticklabels=budgets, xlabel='Available peer budget k', ylabel='Held-out total variation (lower is better)', title='(a) Macro-average response reconstruction')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis='y', alpha=.2)
    targets = config['model_roster']
    families = list(config['families'])
    for fi, family in enumerate(families):
        subset = all_results.loc[(all_results.k == config['primary_budget']) & (all_results.family == family)].set_index('target').loc[targets]
        ci = intervals.loc[(intervals.scope == 'target_family') & (intervals.k == config['primary_budget']) &
                           (intervals.family == family) & (intervals.metric == 'nearest_minus_collective_tv')].set_index('target').loc[targets]
        y = np.arange(len(targets)) + (fi - (len(families) - 1) / 2) * .25
        color = ['#2166AC', '#D6604D'][fi % 2]
        axes[1].scatter(subset.nearest_minus_collective_tv, y, color=color, s=24, label=family.replace('_', ' '))
        axes[1].hlines(y, ci.lower, ci.upper, color=color)
    axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set(yticks=np.arange(len(targets)), yticklabels=[_short_name(t) for t in targets],
                xlabel='Nearest k − collective TV (positive favors collective)', title=f'(b) Primary k={config["primary_budget"]}: all targets')
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis='x', alpha=.2)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=[0, .18, 1, 1])
    caption = (
        'Complete cyclic-permutation-averaged semantic option probabilities; clean mass 1/2, each of three maximum-stress tracks 1/6.\n'
        'Points average ten overlapping question splits. Intervals resample common questions with selections and fits fixed; they exclude\n'
        'selection/fitting uncertainty and do not treat the splits as independent replications. Only the displayed budgets were measured.'
    )
    fig.text(.02, .035, caption, fontsize=8, va='bottom')
    folder = Path(output_root) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'reference_selection.pdf', bbox_inches='tight')
    fig.savefig(folder / 'reference_selection.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _effect_phrase(gain: float) -> str:
    if gain > 1e-12:
        return 'lowered'
    if gain < -1e-12:
        return 'increased'
    return 'left unchanged'


def measurement_provenance_text(provenance: dict | None) -> str:
    """Describe only recorded execution conditions, independently of outcomes."""
    if not provenance or provenance.get('measurement_origin') != 'fresh_endpoint_scoring':
        return 'Input cache identity and provenance scope are recorded in the run manifest.'
    text = ('Probability inputs are fresh measurements from the pinned checkpoints and restored historical prompt design, '
            'not recovered historical score bytes. Full hardware, precision, source and input identities appear in the run manifest.')
    inputs = provenance.get('input_identity', {})
    execution = inputs.get('distributed_execution_provenance', {})
    shared = [row for row in execution.get('shared_batch_changes', [])
              if row.get('new_committed_row_count', 0) > 0 or row.get('new_committed_shard_count', 0) > 0]
    if not shared:
        return text  # Failed feasibility pilots alone are not shared score inputs.

    def numbers(values):
        values = [f'{value:g}' for value in sorted(set(values))]
        return ' and '.join(values) if len(values) < 3 else ', '.join(values[:-1]) + ' and ' + values[-1]

    shared_batches = {row['batch_size'] for row in shared if row.get('batch_size')}
    attributed = execution.get('shared_new_shard_attribution', {})
    retained_batches = {row['execution']['effective_batch_size'] for row in inputs.get('shards', [])
                        if row.get('data', {}).get('path') not in attributed
                        and row.get('execution', {}).get('effective_batch_size')}
    all_batches = {size for row in execution.get('committed_batch_sizes_by_model', {}).values()
                   for size in row.get('effective_batch_size', [])}
    if retained_batches and shared_batches:
        grouping = (f'combining retained pilot/local batch settings {numbers(retained_batches)} '
                    f'with shared-node batch settings {numbers(shared_batches)}')
    elif all_batches and shared_batches:
        grouping = (f'with recorded batch settings {numbers(all_batches)}, '
                    f'including shared-node batch settings {numbers(shared_batches)}')
    else:
        grouping = 'with the shared-node batch groupings recorded in the run manifest'
    hosts = {row['hostname'] for row in shared if row.get('hostname')}
    caps = {row['memory_budget_mib'] / 1024 for row in shared if row.get('memory_budget_mib')}
    reserves = {row['reserve_mib'] / 1024 for row in shared if row.get('reserve_mib')}
    resources = []
    if hosts:
        resources.append(f'{len(hosts)} shared node' + ('s' if len(hosts) != 1 else ''))
    if caps:
        resources.append(f'{numbers(caps)} GiB allocator cap' + ('s' if len(caps) != 1 else ''))
    if reserves:
        resources.append(f'{numbers(reserves)} GiB reserve' + ('s' if len(reserves) != 1 else ''))
    detail = ' (' + '; '.join(resources) + ')' if resources else ''
    scorer = ('The same original full-continuation scorer was used'
              if provenance.get('candidate_scoring') == 'summed_full_continuation_log_likelihood'
              else 'The recorded scoring procedure was used')
    precision = ' in BF16' if provenance.get('precision') == 'bfloat16' else ''
    return (text + f'\n\n{scorer}{precision}, {grouping}{detail}. '
            'No bitwise equivalence across these batch groupings is claimed.')


def write_results(
    strategy_results: pd.DataFrame,
    heldout_losses: pd.DataFrame,
    questions: pd.DataFrame,
    config: dict,
    output_root: Path,
    *,
    run_metadata: dict,
    command: str,
    complete: bool = True,
    synthetic: bool = False,
) -> dict:
    """Write final deliverables only after a complete real-data point analysis.

    This function deliberately does not write status/complete.json; the orchestrator
    must also verify candidate counts, numerical checks and input/source integrity.
    """
    if not complete or synthetic or run_metadata.get('synthetic', False):
        raise ValueError('Completed empirical reporting requires a complete real-data run')
    provenance = run_metadata.get('input_measurement_provenance')
    if (provenance and provenance.get('measurement_origin') == 'fresh_endpoint_scoring'
            and not (run_metadata.get('input_provenance_validation') or {}).get('verified')):
        raise ValueError('Fresh-measurement reporting requires cache-bound provenance validation')
    if 'solver_complete' in strategy_results and not strategy_results.solver_complete.all():
        raise ValueError('Cannot report an incomplete solver run')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results, macro = summarize_strategy_results(strategy_results, config)
    loss_means = heldout_losses.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean()
    strategies = strategy_results.set_index(['target', 'family', 'split_seed', 'k', 'method']).sort_index()
    loss_means = loss_means.reindex(strategies.index)
    if loss_means.isna().any().any() or not np.allclose(loss_means.to_numpy(), strategies[['evaluation_tv', 'top_option_agreement']].to_numpy(), atol=1e-12, rtol=0):
        raise ValueError('Saved question metrics do not reproduce strategy results')
    replicate, intervals, resampling = frozen_fit_resampling(heldout_losses, questions, config, output_root)
    counts = selection_diagnostics(strategy_results)
    frequencies, differences = counts
    main = all_results.loc[all_results.k.isin([config['primary_budget'], config['secondary_budget']])]
    for name, frame in [('main_results', main), ('all_budget_results', all_results), ('macro_results', macro),
                        ('frozen_fit_resampling', replicate), ('descriptive_intervals', intervals),
                        ('selection_frequencies', frequencies), ('selection_differences', differences)]:
        frame.to_csv(output_root / f'{name}.csv', index=False, float_format='%.12g')
    make_figure(all_results, macro, intervals, config, output_root)
    primary = macro.set_index('k').loc[config['primary_budget']]
    secondary = macro.set_index('k').loc[config['secondary_budget']]
    primary_cells = main.loc[main.k == config['primary_budget']]
    wins = int((primary_cells.nearest_minus_collective_tv > 1e-12).sum())
    losses = int((primary_cells.nearest_minus_collective_tv < -1e-12).sum())
    ties = len(primary_cells) - wins - losses
    selected = differences.loc[differences.k == config['primary_budget']]
    different = int(selected.different_selected_sets.sum())
    total = int(selected.split_count.sum())
    pi = intervals.loc[(intervals.scope == 'macro') & (intervals.k == config['primary_budget']) &
                       (intervals.metric == 'nearest_minus_collective_tv')].iloc[0]
    relative = primary.nearest_relative_tv_reduction_percent
    relative_text = 'undefined (zero nearest-k baseline)' if pd.isna(relative) else f'{relative:.3f}%'
    direction = _effect_phrase(primary.nearest_minus_collective_tv)
    cost = json.dumps(run_metadata.get('compute', run_metadata.get('timing', {})), indent=2, default=str)
    provenance = run_metadata.get('input_measurement_provenance')
    provenance_text = measurement_provenance_text(provenance)
    report = f'''# XWVG reference-selection experiment: measured results

{provenance_text}

All required target/family/split comparisons in the fixed protocol were completed. The primary budget is k={config['primary_budget']}; k={config['secondary_budget']} is secondary. This post-review supplementary analysis uses the already-studied question pool and is not an independent new benchmark.

At k={config['primary_budget']}, collective selection {direction} macro-average held-out TV relative to nearest-k: collective {primary.collective_tv:.6f}, nearest-k {primary.nearest_k_tv:.6f}, absolute nearest-minus-collective difference {primary.nearest_minus_collective_tv:.6f}. The relative reduction computed from macro errors is {relative_text}. The exact uniform-random-subset expected TV is {primary.uniform_random_expected_tv:.6f}, with random-minus-collective difference {primary.random_minus_collective_tv:.6f}.

At k={config['secondary_budget']}, collective TV is {secondary.collective_tv:.6f}, nearest-k TV is {secondary.nearest_k_tv:.6f}, and their difference is {secondary.nearest_minus_collective_tv:.6f}. The random expected TV is {secondary.uniform_random_expected_tv:.6f}, with random-minus-collective difference {secondary.random_minus_collective_tv:.6f}. These secondary comparisons do not change the headline budget.

The primary comparison favors collective selection in {wins} target/family cells, favors nearest-k in {losses}, and ties to 1e-12 in {ties}. All cells, including unfavorable effects, appear below. Differences are absolute TV units. Positive gains favor collective selection.

{_markdown_table(main[['target', 'family', 'k', 'collective_tv', 'nearest_k_tv', 'uniform_random_expected_tv', 'nearest_minus_collective_tv', 'random_minus_collective_tv']])}

At the primary budget, score-based top-option agreement is {primary.collective_top_option_agreement:.6f} for collective, {primary.nearest_k_top_option_agreement:.6f} for nearest-k, and {primary.uniform_random_expected_top_option_agreement:.6f} for the random expectation. Collective-minus-nearest agreement is {primary.collective_minus_nearest_agreement:.6f}. Agreement is a separate outcome and was not used to select subsets.

The two deterministic policies selected different available-peer sets in {different}/{total} primary target/family/split settings. `selection_frequencies.csv` records every observed selection frequency; selected available peers may have zero fitted weight. No unique coefficient attribution is claimed.

The primary gain's descriptive frozen-fit question-resampling interval is [{pi.lower:.6f}, {pi.upper:.6f}], using {config['uncertainty']['replicates']} common within-subject question resamples. {INTERVAL_SCOPE} No p-values or significance claims are derived from these intervals.

The measured interface consists of complete cyclic-permutation-averaged semantic option probabilities. Each question weights clean by 1/2 and each of its three maximum-stress realizations by 1/6. TV sums over valid option coordinates. The random comparator is the exact mean of subset metrics, not the metric of averaged predictions. The k=7 fit is a full-roster reference and is not guaranteed to minimize held-out error.

This result concerns selecting references to approximate the target's recorded response distributions within this design. It does not establish improved task accuracy, free-text substitution, safe target removal, latency or FLOP savings. The model roster and two stress families are fixed. Overlapping splits reuse the same 560-question pool, and fitting/selection uncertainty is excluded.

Recorded computational cost and resource settings:

```json
{cost}
```

Reproduce or resume using:

```bash
{command}
```

Exact machine-readable tables retain additional numerical precision. Input identities, source revision, package versions and split IDs are recorded in the run manifest and validation files. The run orchestrator writes its completion marker only after the separate numerical and integrity checks pass.
'''
    conclusion = (
        'Collective selection had lower observed macro-average TV in this fixed experiment. The absolute magnitude and complete target/family pattern determine the practical value of this difference within the tested response interface.'
        if primary.nearest_minus_collective_tv > 1e-12 else
        'These measured results do not show a macro-average improvement over nearest-k at the primary budget; we therefore do not claim a demonstrated advantage for this selection use.'
    )
    draft = f'''# Draft response to XWVG — completed supplementary analysis

{provenance_text}

To test a concrete use of the coverage information, we used it to choose a small, target-specific set of reference models whose fixed convex combination reproduces the target's recorded option-probability distributions on held-out questions. We compared exhaustive collective selection, the k individually nearest peers (followed by the same convex TV fit), and the exact expected performance of a uniformly chosen k-subset. The analysis covers all eight targets, both stress families and the ten original subject-stratified half-splits; k=2 was the fixed primary budget and k=3 was secondary.

At k=2, macro-average held-out TV was {primary.collective_tv:.6f} for collective selection, {primary.nearest_k_tv:.6f} for nearest-k and {primary.uniform_random_expected_tv:.6f} for uniform random selection. The nearest-minus-collective difference was {primary.nearest_minus_collective_tv:.6f} TV ({relative_text} relative reduction), and the random-minus-collective difference was {primary.random_minus_collective_tv:.6f}. At k=3, the corresponding TV errors were {secondary.collective_tv:.6f}, {secondary.nearest_k_tv:.6f} and {secondary.uniform_random_expected_tv:.6f}. At the primary budget, {wins} target/family cells favored collective selection and {losses} favored nearest-k, with {ties} tied to numerical reporting tolerance; the full table shows every case. Primary top-option agreement was {primary.collective_top_option_agreement:.6f}, {primary.nearest_k_top_option_agreement:.6f} and {primary.uniform_random_expected_top_option_agreement:.6f}, respectively. {conclusion}

The response interface uses complete semantic option distributions averaged across all cyclic option rotations, with mass 1/2 on clean and 1/6 on each of three maximum-stress realizations. The primary TV gain has a descriptive frozen-fit common-question resampling interval of [{pi.lower:.6f}, {pi.upper:.6f}] ({config['uncertainty']['replicates']} replicates). Selected subsets and fitted weights are fixed during resampling; these intervals exclude fitting and selection uncertainty and do not treat overlapping splits as independent replications. This is a post-review supplementary analysis of the existing question pool, not independent validation. The result addresses reference selection for this measured interface, rather than task accuracy, serving cost or safe replacement of the target.
'''
    (output_root / 'REPORT.md').write_text(report, encoding='utf-8')
    (output_root / 'XWVG_RESPONSE_DRAFT.md').write_text(draft, encoding='utf-8')
    return {'main_rows': len(main), 'all_budget_rows': len(all_results), 'macro_rows': len(macro),
            'replicate_rows': len(replicate), 'resampling': resampling}
