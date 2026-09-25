"""Regenerate the controls and checkpoint tables previously embedded in prose."""
import json
import re
import pandas as pd
from common import ROOT, require

def generate(stage):
    controls={r['control']:r for r in json.loads((stage/'data/v2/control_results.json').read_text())['controls']}
    template=(ROOT/'scripts/table_templates/exact_controls.tex').read_text()
    selected=[('Exact clone','exact_clone_gold_probability'),
              ('Mixture $(0.5,0.3,0.2)$','known_sparse_mixture_gold_probability'),
              ('Boundary mixture $(0.75,0.25)$','boundary_mixture_gold_probability'),
              ('Duplicated peer','duplicated_peer_ambiguity')]
    rows=[]
    def sci(value):
        mantissa,exponent=f'{value:.2e}'.split('e')
        return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
    for label,key in selected:
        row=controls[key]
        rows.append(label+' & '+sci(row['observed_mean_residual'])+' & '+sci(row['observed_max_residual'])+r'\\')
    result=re.sub(r'(\\midrule\n).*?(\n\\bottomrule)',lambda m:m.group(1)+'\n'.join(rows)+m.group(2),template,flags=re.S)
    require(result==template,'Control table differs from the authoritative manuscript display')
    (stage/'tables/exact_controls.tex').write_text(result)
    template=(ROOT/'scripts/table_templates/models.tex').read_text()
    models=pd.read_csv(stage/'data/v2/table1_ecosystem_manifest.csv').model.tolist()
    labels=['Qwen2.5','General Reasoner (GR)','phi-4','Phi-R','Mistral','OLMo','Granite','Gemma']
    require(len(models)==len(labels),'Model roster denominator changed')
    rows=[label+r' & \texttt{'+model+r'}\\' for label,model in zip(labels,models)]
    result=re.sub(r'(\\midrule\n).*?(\n\\bottomrule)',lambda m:m.group(1)+'\n'.join(rows)+m.group(2),template,flags=re.S)
    require(result==template,'Checkpoint table differs from the authoritative roster')
    (stage/'tables/models.tex').write_text(result)
