"""Serial single-device fresh scoring campaign followed by the locked analysis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--analysis-output',type=Path,required=True)
    parser.add_argument('--analysis-python',type=Path,required=True)
    parser.add_argument('--phase',choices=['pilot','score','finish','all'],default='all')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--scorer',choices=['reference','selected_positions'],default='reference')
    parser.add_argument('--acceptance-report',type=Path,
                        help='Required all-eight-model exact GPU comparison for selected_positions')
    parser.add_argument('--pilot-batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.pilot_batch_size < 1:
        parser.error('pilot-batch-size must be positive')
    if args.scorer == 'selected_positions' and args.acceptance_report is None:
        parser.error('selected_positions requires --acceptance-report')
    repo=args.source_repo.resolve();root=args.run_root.resolve()
    component=repo/'experiments/llm/analysis/xwvg_reference_selection'
    records=json.loads((repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json').read_text())['models']
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    state={'started_utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,
           'scorer':args.scorer,'pilot_batch_size':args.pilot_batch_size,
           'acceptance_report':str(args.acceptance_report.resolve()) if args.acceptance_report else None,
           'status':'running','completed':[]}
    state_path=root/'status/campaign.json'
    save(state_path,state)
    def command(cmd,name):
        state['current_step']=name;state['current_command']=cmd;save(state_path,state)
        print(json.dumps({'step':name,'status':'started'}),flush=True)
        start=time.perf_counter()
        with (logs/(name+'.log')).open('a') as log:
            log.write('\n--- invocation '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            state.update(status='failed',exit_code=result.returncode,failed_step=name)
            save(state_path,state)
            raise RuntimeError(f'{name} failed with exit {result.returncode}; see {logs/(name+".log")}')
        state['completed'].append({'step':name,'seconds':time.perf_counter()-start})
        save(state_path,state)
        print(json.dumps({'step':name,'status':'complete'}),flush=True)
    for stage in (['pilot','score'] if args.phase=='all' else [args.phase] if args.phase in ['pilot','score'] else []):
        for record in records:
            slug=record['id'].replace('/','--')
            downloaded=json.loads((root/'status/model_downloads'/f'{slug}.json').read_text())
            if downloaded['tokenizer_only'] or not downloaded['historical_file_identity_verified']:
                raise ValueError(f'Incomplete checkpoint: {record["id"]}')
            batch=args.pilot_batch_size
            if stage=='score':
                pilot=json.loads((root/'status'/f'fresh_score_{slug}_pilot.json').read_text())
                batch=pilot['final_batch_size']
            driver='score_endpoints.py' if args.scorer=='reference' else 'score_selected_endpoints.py'
            cmd=[sys.executable,str(component/'recovery'/driver),'--source-repo',str(repo),
                 '--run-root',str(root),'--model-id',record['id'],'--model-path',downloaded['snapshot'],
                 '--device',args.device,'--batch-size',str(batch),'--stage',stage,'--resume']
            if args.scorer=='selected_positions':
                cmd.extend(['--acceptance-report',str(args.acceptance_report.resolve())])
            command(cmd,stage+'_'+slug)
        if stage=='pilot':
            pilots=[json.loads((root/'status'/f'fresh_score_{r["id"].replace("/","--")}_pilot.json').read_text()) for r in records]
            save(root/'status/pilot_summary.json',{'model_count':len(pilots),'models':[
                {k:p[k] for k in ['model_id','completed_rows','scoring_seconds','load_seconds','final_batch_size','projection_full_scoring_seconds']} for p in pilots],
                'projected_serial_scoring_seconds':sum(p['projection_full_scoring_seconds'] for p in pilots),
                'note':'Fixed outcome-blind pilot. Projection excludes variable padding and I/O; no scientific comparisons inspected.'})
    if args.phase in ['finish','all']:
        command([sys.executable,str(component/'recovery/assemble_scores.py'),'--run-root',str(root),'--questions',str(args.questions.resolve())], 'assemble')
        import os
        # The analysis CLI is installed separately; its Python environment does
        # not load torch or modify the scoring environment.
        os.environ['PYTHONPATH']=str(component/'src')
        # Resolving a venv interpreter symlink would escape its environment.
        command([str(args.analysis_python.expanduser().absolute()),'-m','bca_reference_selection.pipeline','--repo-root',str(repo),
                 '--cache',str(root/'outputs/analysis/permutation_averaged_semantic_scores.parquet'),
                 '--questions',str(args.questions.resolve()),'--input-provenance',str(root/'status/measurement_provenance.json'),
                 '--output-root',str(args.analysis_output.resolve()),'--stage','all','--workers','4','--resume'], 'analysis')
    state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    save(state_path,state)
    return 0

if __name__=='__main__':raise SystemExit(main())
