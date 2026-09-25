"""Typeset the completed canonical full-option audit. No fitting or resampling.
Reads the scalar-column CSV extracted losslessly from the frozen Parquet.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/v22/vector_audit';T=ROOT/'tables';Q=ROOT/'qa'
raw=D/'trackwise_vector_results.parquet'
expected='1604bc3b71ba193fea699dec26ec24965230a73435ba69588e9f1bf82e1a05a9'
assert hashlib.sha256(raw.read_bytes()).hexdigest()==expected
marker=json.loads((D/'stage_11_vector_sensitivity.json').read_text())
assert expected in marker['checksums'].values()
d=pd.read_csv(D/'trackwise_vector_scalar_columns.csv')
keys=['target','family','split_seed','weight_source']
assert len(d)==1600 and not d.duplicated(keys+['dose']).any()
assert d.groupby(keys).size().eq(5).all()
for c in ['vector_overall_total_variation','single_peer_overall_total_variation',
          'vector_convexity_improvement','vector_endpoint_delta','vector_sibling_removal_inflation']:
    assert d.groupby(keys)[c].nunique(dropna=False).eq(1).all()
assert np.allclose(d.vector_convexity_improvement,d.single_peer_overall_total_variation-d.vector_overall_total_variation,atol=2e-15,rtol=0)
assert np.allclose(d.groupby(keys).mean_total_variation.mean(),d.groupby(keys).vector_overall_total_variation.first(),atol=2e-15,rtol=0)
for f in ['mean_total_variation','top1_semantic_answer_agreement','correctness_agreement']:
    assert d[f].between(0,1).all()
x=d.drop_duplicates(keys)
v=x[x.weight_source=='vector_fitted']
models=['Qwen/Qwen2.5-7B-Instruct','TIGER-Lab/General-Reasoner-Qwen2.5-7B',
'allenai/OLMo-2-1124-13B-Instruct','google/gemma-3-12b-it',
'ibm-granite/granite-3.3-8b-instruct','microsoft/Phi-4-reasoning-plus',
'microsoft/phi-4','mistralai/Mistral-Nemo-Instruct-2407']
short=['Qwen2.5','General Reasoner','OLMo','Gemma','Granite','Phi-R','phi-4','Mistral']
ms=dict(zip(models,short));fams=['content_deletion','irrelevant_context']
rows=[]
for m in models:
    for f in fams:
        a=v[(v.target==m)&(v.family==f)]
        b=x[(x.target==m)&(x.family==f)&(x.weight_source=='scalar_fitted')]
        assert len(a)==len(b)==10
        s=float(a.single_peer_overall_total_variation.mean());g=float(a.vector_overall_total_variation.mean())
        rows.append(dict(target=m,family=f,single_tv=s,group_tv=g,gain=s-g,
            relative_gain_pct=100*(s-g)/s,positive_splits=int((a.vector_convexity_improvement>0).sum()),
            scalar_weight_tv=float(b.vector_overall_total_variation.mean()),
            group_tv_min=float(a.vector_overall_total_variation.min()),
            group_tv_max=float(a.vector_overall_total_variation.max()),
            sibling_inflation=float(a.vector_sibling_removal_inflation.mean()),
            sibling_positive_splits=int((a.vector_sibling_removal_inflation>0).sum()),
            endpoint_delta=float(a.vector_endpoint_delta.mean())))
z=pd.DataFrame(rows);z.to_csv(D/'vector_summary.csv',index=False,float_format='%.17g')
# Main table has preselected illustrative targets, including a change of sign across audit settings.
sel=[models[0],models[1],models[7],models[3]]
main=r'''\begin{table}[!ht]
\centering\small
\caption{\textbf{Coverage across the complete option distribution.} Mean held-out total variation over ten splits in the canonical-order, five-dose audit. The group minimizes vector squared error; the single peer is selected by fitting total variation. This response diagnostic uses a different design from the balanced, loss-matched results in Figure~\ref{fig:llm-coverage}. All models and removal effects appear in Appendix~\ref{app:vector}.}
\label{tab:vector-main}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Deletion} & \multicolumn{2}{c}{Irrelevant context}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Target & Single & Group & Single & Group\\
\midrule
'''
for m in sel:
    a=z[(z.target==m)&(z.family==fams[0])].iloc[0];b=z[(z.target==m)&(z.family==fams[1])].iloc[0]
    main+=f"{ms[m]} & {a.single_tv:.3f} & {a.group_tv:.3f} & {b.single_tv:.3f} & {b.group_tv:.3f}\\\\\n"
main+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
(T/'vector_main.tex').write_text(main)
# Full table: every target in both families, no unreported significance test.
full=r'''\begin{table}[!b]
\centering\small\setlength{\tabcolsep}{5pt}
\caption{\textbf{Complete-option reconstruction in the canonical audit.} Each cell averages ten fixed splits. D/I denote deletion/irrelevant context. Single denotes selection by fitting TV; transferred uses scalar-fit weights to predict every option; refitted minimizes vector squared error. $+$ is the number of splits with single TV greater than refitted-group TV, not an independent-replication test.}
\label{tab:vector-all}
\begin{tabular}{llrrrrr}
\toprule
Target & Family & Single TV & Transferred TV & Refitted TV & Gain (\%) & $+$\\
\midrule
'''
for _,a in z.iterrows():
    full+=f"{ms[a.target]} & {'D' if a.family==fams[0] else 'I'} & {a.single_tv:.4f} & {a.scalar_weight_tv:.4f} & {a.group_tv:.4f} & {a.relative_gain_pct:+.1f} & {int(a.positive_splits)}/10\\\\\n"
full+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
(T/'vector_all.tex').write_text(full)
sib=r'''\begin{table}[!ht]
\centering\small
\caption{\textbf{Sibling-removal effects for complete option vectors.} Increase in held-out TV after removing the declared sibling and refitting the vector squared-loss combination; means over ten fixed splits. The last column counts positive changes in the overlapping deletion/context splits; the full diagnostic is specified in Appendix~\ref{app:vector}.}
\label{tab:vector-sibling}
\begin{tabular}{llrrr}
\toprule
Target & Removed peer & Deletion & Context & Positive splits (D/I)\\
\midrule
'''
sibs={models[0]:models[1],models[1]:models[0],models[5]:models[6],models[6]:models[5]}
for m,p in sibs.items():
    a=z[(z.target==m)&(z.family==fams[0])].iloc[0];b=z[(z.target==m)&(z.family==fams[1])].iloc[0]
    sib+=f"{ms[m]} & {ms[p]} & {a.sibling_inflation:+.4f} & {b.sibling_inflation:+.4f} & {int(a.sibling_positive_splits)}/10, {int(b.sibling_positive_splits)}/10\\\\\n"
sib+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
(T/'vector_siblings.tex').write_text(sib)
(Q/'vector_evidence_checks.json').write_text(json.dumps(dict(
    raw_parquet_sha256=expected,stage_marker_matches=True,rows=1600,scalar_columns=23,
    unique_split_weight_groups=320,unique_vector_fit_groups=160,
    five_doses_per_group=True,group_gain_identity=True,overall_equal_dose_identity=True,
    no_new_inference=True,no_refitting=True,no_bootstrap=True,
    caveat='Canonical five-dose vector-square fit / TV-selected single; separate from balanced endpoint matched-loss inference.'),indent=2))
# Extend rather than replace the existing generated provenance.
header='\n## Complete-option-distribution tables\n'
path=ROOT/'FIGURE_PROVENANCE.md';text=path.read_text().split(header)[0]
text+=header+'\nSources under `data/v22/vector_audit/`:\n'
for name in ['trackwise_vector_results.parquet','stage_11_vector_sensitivity.json','trackwise_vector_scalar_columns.csv','sensitivities.py','data.py']:
    text+=f'- `{name}`; SHA256 `{hashlib.sha256((D/name).read_bytes()).hexdigest()}`\n'
text+='\nThe completed 1,600-row audit contains five dose rows for each target/family/split/weight-source combination. Scalar columns are exported at 17-digit precision. Existing overall statistics are deduplicated across dose before averaging ten splits. No fit, resampling, model inference, or new labels are computed. The transferred-weight and vector-refit paths are kept separate. Main and appendix tables are generated by `scripts/build_vector_evidence.py`.\n'
path.write_text(text)
print(z[['target','family','single_tv','group_tv','relative_gain_pct','sibling_inflation']].to_string(index=False))

# Manuscript assembly locations are kept separate from the historical panel filenames.
source_map=ROOT/'SOURCE_MAP.md'
marker='\n## Current manuscript assembly and tables\n'
text=source_map.read_text().split(marker)[0]+marker
text+='''
- Main Figure 1: two conceptual panels, assembled in Section 2.
- Main Figure 2: balanced reference-answer group/removal comparisons, assembled in Section 4.
- Main Figure 3: canonical dose trajectories and balanced endpoints, assembled in Section 5.
- Main Figure 4: exact vision stress/context, fitting geometry and residual localization, assembled in Section 6.
- Appendix Figure 5: interface endpoint comparison and intervention aggregation, Appendix C.4.
- Appendix Figure 6: all six separately fitted PCA views, Appendix D.2.
- Appendix Figure 7: archived forecasting scatter comparisons, Appendix D.4. Historical source filenames still begin with `fig5`; these are not a main-text downstream experiment.
- Appendix Figure 8: sentiment-classifier subgroup/control panels, Appendix F.

The complete-option tables are not images or new plots. `vector_main.tex` and `vector_siblings.tex` are main-text Tables 2 and 3 in Section 4.4; `vector_all.tex` remains in Appendix C.3. They are generated from the same completed canonical vector audit by `build_vector_evidence.py`. Table 1 contains exact-response controls. The other tables retain model identities, complete primary comparisons, generation readouts, and archived geometry support in the appendices. The main text occupies pages 1–9; references start on page 10.
'''
source_map.write_text(text)
