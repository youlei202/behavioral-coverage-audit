import json
import numpy as np
import pandas as pd
import pytest
from common import ROOT, frozen
from verify_inputs import verify, verify_vector
from replay import numeric

def test_frozen_inputs_and_completed_vector_audit():
    assert verify()==77

def test_original_replay_outputs():
    numeric()

def test_actual_control_residuals_and_table_rounding():
    controls=json.loads(frozen('data/v2/control_results.json').read_text())['controls']
    displayed=(ROOT/'paper/tables/exact_controls.tex').read_text()
    for control in controls:
        assert control['passed']
        if 'threshold_mean' in control:
            assert control['observed_mean_residual']<=control['threshold_mean']
    assert '2.71\\times10^{-11}' in displayed
    assert 'Duplicated peer' in displayed

def test_independent_vector_summary_from_parquet():
    frame=pd.read_parquet(frozen('data/v22/vector_audit/trackwise_vector_results.parquet'))
    fit=frame[frame.weight_source.eq('vector_fitted')].drop_duplicates(['target','family','split_seed'])
    expected=fit.groupby(['target','family'])[['single_peer_overall_total_variation','vector_overall_total_variation']].mean()
    actual=pd.read_csv(ROOT/'data/generated/vector_summary.csv').set_index(['target','family'])
    for key,row in expected.iterrows():
        assert np.isclose(actual.loc[key,'single_tv'],row.single_peer_overall_total_variation,atol=2e-15,rtol=0)
        assert np.isclose(actual.loc[key,'group_tv'],row.vector_overall_total_variation,atol=2e-15,rtol=0)

def test_residual_localization_keeps_all_evaluation_inputs():
    for context,expected_overlap in [('texture_natural',2),('shape_bias',19)]:
        a=np.abs(np.load(frozen(f'data/original/artifacts/exp11/exp11_{context}_ConvNeXtTiny_residuals.npz'))['residuals'])
        b=np.abs(np.load(frozen(f'data/original/artifacts/exp11/exp11_{context}_ShapeResNet50_ShapeResNet_residuals.npz'))['residuals'])
        assert len(a)==len(b)==400
        indices_a=np.argsort(-a,kind='stable')[:20]
        indices_b=np.argsort(-b,kind='stable')[:20]
        assert len(set(indices_a)&set(indices_b))==expected_overlap
        cumulative=pd.read_csv(ROOT/f'data/generated/figures/residual_mass_{context}.csv')
        assert len(cumulative)==401
        np.testing.assert_allclose(cumulative.mass.to_numpy(),100*np.r_[0,np.cumsum(np.sort(a)[::-1])/a.sum()],rtol=0,atol=2e-13)

def test_traffic_keeps_infeasible_archived_coefficients():
    weights=pd.read_csv(frozen('data/original/tables/exp14_multicity_tabular_summary_peer_weights_long.csv'))
    # Recover the generator's 17-digit serialization without another parser rounding.
    points=pd.read_csv(ROOT/'data/generated/figures/traffic_all.csv',float_precision='round_trip')
    assert len(points)==31
    assert weights.Weight.min()==points.min_weight.min()
    assert weights.Weight.min()<0

def test_failed_vector_marker_is_rejected(tmp_path):
    # Build an isolated input view, then change only its marker; never edit frozen inputs.
    import shutil
    shutil.copytree(ROOT/'manifests',tmp_path/'manifests')
    mapping=json.loads((ROOT/'manifests/data_map.json').read_text())
    for old,new in mapping.items():
        if old.startswith('data/v22/vector_audit/'):
            dst=tmp_path/new;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/new,dst)
    marker=frozen('data/v22/vector_audit/stage_11_vector_sensitivity.json',tmp_path)
    record=json.loads(marker.read_text());record['success']=False;marker.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='unsuccessful'):
        verify_vector(tmp_path)
