"""Verify frozen bytes, schemas, vector completion, and scientific controls."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import ROOT, read_tsv, require, sha256, frozen, owned_path

def verify(root=ROOT):
    records=read_tsv(root/'manifests/input_sha256.tsv')
    require(bool(records), 'Empty input manifest')
    require(len({r['relative_path'] for r in records})==len(records), 'Duplicate input paths')
    for row in records:
        path=owned_path(row['relative_path'],root)
        require(path.is_file(),f'Missing input: {path}')
        require('/generated/' not in row['relative_path'], 'Generated file classified as frozen evidence')
        require(path.stat().st_size==int(row['size_bytes']),f'Input size mismatch: {path}')
        require(sha256(path)==row['sha256'],f'Input SHA256 mismatch: {path}')
    for relative, expected in json.loads((root/'manifests/input_schemas.json').read_text()).items():
        with (root/relative).open(newline='') as handle:
            reader=csv.reader(handle)
            require(next(reader)==expected['columns'],f'CSV columns changed: {relative}')
            require(sum(1 for _ in reader)==expected['row_count'],f'CSV row count changed: {relative}')
    verify_vector(root)
    controls=json.loads(frozen('data/v2/control_results.json',root).read_text())
    require(controls['control_count']==len(controls['controls'])==7, 'Control denominator changed')
    for control in controls['controls']:
        require(control['passed'] is True, f'Archived control failed: {control["control"]}')
        # Use the archived gates, never a newly chosen scientific threshold.
        for field,threshold in [('observed_mean_residual','threshold_mean'),('observed_max_residual','threshold_max')]:
            if threshold in control:
                require(control[field]<=control[threshold],f'Control gate failed: {control["control"]} {field}')
    other=json.loads(frozen('data/v22/V2_2_CONTROLS.json',root).read_text())
    require(other['all_passed'] is True and other['control_count']==10, 'Trackwise controls changed')
    print(f'Input verification passed: {len(records)} files, vector audit, CSV schemas, and archived controls')
    return len(records)

def verify_vector(root=ROOT):
    base='data/v22/vector_audit/'
    raw=frozen(base+'trackwise_vector_results.parquet',root)
    marker=json.loads(frozen(base+'stage_11_vector_sensitivity.json',root).read_text())
    require(marker['success'] is True and marker['error'] is None,'Vector completion marker is unsuccessful')
    require(sha256(raw)==marker['checksums']['outputs/analysis/trackwise_vector_results.parquet'],
            'Vector checksum does not match its completion marker')
    require(pq.read_metadata(raw).num_rows==marker['row_counts']['vector_results']==1600,'Vector row count mismatch')
    frame=pd.read_csv(frozen(base+'trackwise_vector_scalar_columns.csv',root),float_precision='round_trip')
    require(frame.shape==(1600,23),'Scalar export shape mismatch')
    original=pd.read_parquet(raw)[frame.columns]
    for column in frame:
        if pd.api.types.is_numeric_dtype(original[column]):
            require(np.allclose(frame[column],original[column],atol=2e-15,rtol=0,equal_nan=True),
                    f'Scalar export differs from Parquet: {column}')
        else:
            require(frame[column].fillna('').astype(str).tolist()==original[column].fillna('').astype(str).tolist(),
                    f'Scalar export strings differ: {column}')
    keys=['target','family','split_seed','weight_source']
    require(not frame.duplicated(keys+['dose']).any(),'Duplicate vector design rows')
    require(frame.groupby(keys).size().eq(5).all(),'Vector groups do not contain five doses')
    require(frame.target.nunique()==8 and frame.split_seed.nunique()==10,'Vector target/split denominator changed')
    require(set(frame.weight_source)=={'vector_fitted','scalar_fitted'},'Vector weight sources conflated')
    require(set(frame.family)=={'content_deletion','irrelevant_context'},'Vector families changed')
    for family,doses in [('content_deletion',{0.,.1,.2,.3,.4}),('irrelevant_context',{0.,64.,128.,256.,512.})]:
        require(set(frame.loc[frame.family.eq(family),'dose'])==doses, f'Dose design changed: {family}')
    require(np.allclose(frame.vector_convexity_improvement,
                        frame.single_peer_overall_total_variation-frame.vector_overall_total_variation,
                        atol=2e-15,rtol=0),'Vector gain identity failed')
    require(np.allclose(frame.groupby(keys).mean_total_variation.mean(),
                        frame.groupby(keys).vector_overall_total_variation.first(),atol=2e-15,rtol=0),
                        'Equal-dose aggregation identity failed')

if __name__=='__main__':
    verify()
