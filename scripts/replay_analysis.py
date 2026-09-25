"""Replay cached split/bootstrap summaries without model loading, fitting or resampling."""
from pathlib import Path
import argparse
import importlib
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
from common import ROOT, frozen, read_tsv, require, sha256

def compare_csv(actual,expected):
    a=pd.read_csv(actual,float_precision='round_trip')
    b=pd.read_csv(expected,float_precision='round_trip')
    require(list(a.columns)==list(b.columns) and a.shape==b.shape,f'Analysis schema changed: {actual}')
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            require(np.allclose(a[col],b[col],atol=2e-15,rtol=0,equal_nan=True),f'Analysis values differ: {actual.name}/{col}')
        elif col.endswith('_interval'):
            require(np.allclose(np.array([json.loads(v) for v in a[col]]),np.array([json.loads(v) for v in b[col]]),
                                atol=2e-15,rtol=0,equal_nan=True),f'Intervals differ: {col}')
        else:
            require(a[col].fillna('').astype(str).tolist()==b[col].fillna('').astype(str).tolist(),f'Analysis labels differ: {actual.name}/{col}')

def replay(raw_root):
    from verify_inputs import verify
    verify()
    records=read_tsv(ROOT/'manifests/external_analysis_inputs.tsv')
    for row in records:
        path=raw_root/row['relative_path']
        require(path.is_file(),f'Missing cached analysis input: {path}')
        require(path.stat().st_size==int(row['size_bytes']) and sha256(path)==row['sha256'],f'Cached input SHA256/size mismatch: {path}')
    output=ROOT/'build/analysis'
    # No read-only upstream root may be used as a replay destination.
    require(not raw_root.resolve().is_relative_to(output.resolve()),'Raw root cannot be inside the output directory')
    for row in records:
        require(not (raw_root/row['relative_path']).resolve().is_relative_to(output.resolve()),'Input overlaps output')
    if output.exists():shutil.rmtree(output)
    output.mkdir(parents=True)
    for row in records:
        dest=output/row['staged_path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(raw_root/row['relative_path'],dest)
    v22=output/'v22';v23=output/'v23';v21=output/'v21'
    config=json.loads((v22/'configs/reanalysis_v22.json').read_text())
    config['physical_root']=config['logical_root']=str(v22)
    for key in ['v2','v21']:
        config['source_roots'][key]={'logical':str(output/key),'physical_alias':str(output/key)}
    (v22/'configs/reanalysis_v22.json').write_text(json.dumps(config,indent=2)+'\n')
    sys.path.insert(0,str(ROOT/'experiments/llm/analysis/v22/src'))
    utils=importlib.import_module('pier_llm_reanalysis_v22.utils')
    utils.PHYSICAL_ROOT=utils.LOGICAL_ROOT=v22
    reporting=importlib.import_module('pier_llm_reanalysis_v22.reporting')
    reporting.write_all_tables()
    checked=[]
    for name in sorted((v22/'outputs/tables').glob('table_*.csv')):
        compare_csv(name,frozen('data/v22/'+name.name))
        checked.append('v22/'+name.name)
    config=json.loads((v23/'configs/experiment_v23.json').read_text())
    config['source_roots']={k:str(output/k) for k in config['source_roots']}
    (v23/'configs/experiment_v23.json').write_text(json.dumps(config,indent=2)+'\n')
    sys.path.insert(0,str(ROOT/'experiments/llm/analysis/v23/src'))
    reporting23=importlib.import_module('pier_llm_v23.reporting')
    claims=reporting23.build_claim_locking(v23)
    reporting23.create_tables(v23,claims)
    files={'table_2_endpoint_claim_comparison.csv':'table_v23_endpoint_claim_comparison.csv',
           'table_3_sibling_coverage.csv':'table_v23_sibling_coverage.csv',
           'table_4_distributed_convex_coverage.csv':'table_v23_distributed_convex_coverage.csv',
           'table_5_generation_validation.csv':'table_v23_generation_validation.csv',
           'table_6_final_claim_locking_matrix.csv':'table_v23_claim_locking_matrix.csv'}
    for name,baseline in files.items():
        compare_csv(v23/'outputs/tables'/name,frozen('data/'+baseline))
        checked.append('v23/'+name)
    # Independently decode the original Parquet using a standard reader.
    pq=frozen('data/v22/vector_audit/trackwise_vector_results.parquet')
    baseline=frozen('data/v22/vector_audit/trackwise_vector_scalar_columns.csv')
    columns=pd.read_csv(baseline,nrows=0).columns
    exported=output/'vector_scalar_columns.csv'
    pd.read_parquet(pq)[columns].to_csv(exported,index=False,float_format='%.17g')
    compare_csv(exported,baseline)
    # Recheck the actual upstream bytes after all original reporting functions returned.
    for row in records:
        require(sha256(raw_root/row['relative_path'])==row['sha256'],f'Upstream was changed: {row["relative_path"]}')
    result=dict(status='passed',input_files=len(records),tables=checked,vector_export='verified_from_parquet',
                refitting='not_run',resampling='not_run',model_inference='not_run',
                scope='Cached fitted split outcomes and archived bootstrap distributions to tables; not a new score-to-fit campaign',
                output_root=str(output))
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Cached analysis replay passed: {len(checked)} LLM tables and independent vector export; upstream hashes unchanged')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=os.environ.get('BEHAVIORAL_COVERAGE_RAW_DATA'))
    args=parser.parse_args()
    if args.raw_root is None:
        parser.error('Set BEHAVIORAL_COVERAGE_RAW_DATA to the parent of the archived PIER experiment directories')
    replay(Path(args.raw_root))
