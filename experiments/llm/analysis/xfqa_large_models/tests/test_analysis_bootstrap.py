from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
import pandas as pd
import pytest

COMPONENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMPONENT))
from analysis_data import Dataset,multiplicities,common_bootstrap_multiplicities,fixed_splits,json_hash,atomic_npz
from analysis import fit_job,job_paths,direct_partner_losses
from analysis_bootstrap import FrozenSampler,contrast_groups,interval_frame


def fixture_dataset(tmp_path):
    rng=np.random.default_rng(283)
    ids=tuple(f'q{i:03d}' for i in range(560))
    selected=pd.DataFrame({'base_question_id':ids,'category':np.repeat([f's{i:02d}' for i in range(14)],40)})
    model_ids=('a','b','c','d');values=rng.dirichlet(np.ones(3),size=(4,560,4))
    mask=np.ones((560,4,3),dtype=bool);masses={'f':np.array([.5,1/6,1/6,1/6])}
    inp=SimpleNamespace(model_ids=model_ids,question_ids=ids,questions=selected,categories=tuple(selected.category),
        probabilities={'f':values},mask=mask,masses=masses)
    splits=fixed_splits(selected,[20260828,20260829]);index={q:i for i,q in enumerate(ids)}
    fit={s:np.array(sorted(index[q] for q in sp.fitting_ids)) for s,sp in splits.items()}
    evaluation={s:np.array(sorted(index[q] for q in sp.evaluation_ids)) for s,sp in splits.items()}
    return Dataset({},inp,{'f':values[...,0]},splits,fit,evaluation,[],{},'synthetic-unit-test',tmp_path)


def job(seed=20260828,loss='mse',peers=('b','c','d')):
    x={'panel':'expanded11','target':'a','family':'f','split_seed':seed,'response':'reference_answer_probability' if loss!='tv' else 'complete_option_vector',
        'fit_loss':loss,'peers':list(peers),'role':'group','removed_peer':None}
    x['job_id']=json_hash(x);return x


def test_all_1000_multiplicity_draws_match_independent_legacy_stable_seed(tmp_path):
    d=fixture_dataset(tmp_path)
    for replicate in range(1,1001):
        text='\x1f'.join(map(str,(20261108,'common-global',replicate)))
        seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%(2**32)
        rng=np.random.default_rng(seed);expected={}
        for _,group in d.values.questions.groupby('category',sort=True):
            ids=sorted(group.base_question_id)
            sampled=rng.choice(ids,size=len(ids),replace=True)
            expected.update({q:int(np.sum(sampled==q)) for q in ids})
        actual=common_bootstrap_multiplicities(d.values.questions,replicate,20261108)
        assert expected==actual
        vec=multiplicities(d,replicate,20261108)
        assert sum(vec)==560 and all(sum(vec[i:i+40])==40 for i in range(0,560,40))


@pytest.mark.parametrize('loss',['mse','mae','tv'])
def test_refitting_and_frozen_weighting_use_separate_split_denominators(tmp_path,loss):
    d=fixture_dataset(tmp_path);j=job(loss=loss)
    fit,metrics,arrays=fit_job(d,j)
    draw=multiplicities(d,13,20261108)
    bfit,bmetrics,barrays=fit_job(d,j,draw)
    assert abs(np.sum(bfit['weights'])-1)<1e-12
    error='total_variation' if loss=='tv' else 'absolute_error'
    ids=d.eval_indices[j['split_seed']]
    expected=float(np.average(barrays[error]@d.values.masses['f'],weights=draw[ids]))
    assert abs(expected-bmetrics['e_group'])<1e-13
    _,path=job_paths(tmp_path,j['job_id']);atomic_npz(path,**arrays)
    record={**j,'peer_ids':json.dumps(j['peers'])}
    sampler=FrozenSampler(d,[record],np.stack([draw,multiplicities(d,14,20261108)]))
    frozen=sampler.metric(j['job_id'],'e_group')
    assert abs(frozen[0]-np.average(arrays[error]@d.values.masses['f'],weights=draw[ids]))<1e-13
    assert abs(frozen[0]-bmetrics['e_group'])>1e-8
    removal=job(loss=loss,peers=('b','c'))
    rfit,_,_=fit_job(d,removal,draw)
    assert rfit['train_objective']>=bfit['train_objective']-1e-7


def test_paired_symmetry_and_signed_directionality(tmp_path):
    d=fixture_dataset(tmp_path)
    for loss in ('mse','mae','tv'):
        np.testing.assert_array_equal(direct_partner_losses(d,'a','b','f',loss),direct_partner_losses(d,'b','a','f',loss))
    first=fit_job(d,job())[1]
    secondjob=job(peers=('a','c','d'));secondjob['target']='b'
    second=fit_job(d,secondjob)[1]
    pair=np.mean(direct_partner_losses(d,'a','b','f','mse')[d.eval_indices[20260828]]@d.values.masses['f'])
    assert abs(((pair-first['e_group'])-(pair-second['e_group']))-(second['e_group']-first['e_group']))<1e-15


def test_empty_bootstrap_side_fails_without_replacing_draw(tmp_path):
    d=fixture_dataset(tmp_path);draw=np.zeros(560);draw[d.fit_indices[20260828]]=1
    with pytest.raises(RuntimeError,match='Empty weighted split'):fit_job(d,job(),draw)


def test_zero_relative_denominator_omits_only_ratio_and_retains_absolute_contrasts():
    from analysis_bootstrap import filter_defined_relative
    records=[]
    for seed in range(10):
        records.append({'run_id':'synthetic_zero','panel':'expanded11','target':'clone','family':'f','fit_loss':'mse',
            'response':'reference_answer_probability','evaluation_metric':'mae','peer_ids':'["original"]',
            'input_fingerprint':'synthetic','split_seed':seed,'e_single':0.,'e_group':0.,'gamma':0.,
            'relative_gamma_percent':None,'mse_single':0.,'mse_group':0.})
    omitted=[]
    d=contrast_groups({'scalar_main_split_results':pd.DataFrame(records)},omissions=omitted)
    assert len(d)==5 and len(omitted)==1 and omitted[0]['metric']=='relative_gamma_percent'
    ratios=[{'metric':'relative_gamma_percent','contrast_id':'ratio'}, {'metric':'gamma','contrast_id':'absolute'}]
    omissions=[]
    kept,values=filter_defined_relative(ratios,np.array([[1.,0.],[np.nan,0.]]),omissions)
    assert kept==[ratios[1]] and values.shape==(2,1) and omissions[0]['undefined_replicates']==[2]
    with pytest.raises(RuntimeError,match='Nonfinite absolute'):
        filter_defined_relative([ratios[1]],np.array([[np.nan]]),[])


def test_atomic_point_resume_rejects_changed_identity_and_corrupt_loss(tmp_path,monkeypatch):
    import analysis
    d=fixture_dataset(tmp_path);j=job();monkeypatch.setattr(analysis,'_DATA',d)
    first=analysis._point_worker(j)
    again=analysis._point_worker(j)
    assert first==again
    with pytest.raises(RuntimeError,match='identity mismatch'):
        analysis.load_point(tmp_path,j,'different identity')
    _,arrays=job_paths(tmp_path,j['job_id'])
    with arrays.open('ab') as f:f.write(b'corrupted')
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        analysis.load_point(tmp_path,j,d.fingerprint)


def test_condition_loss_export_has_complete_identity_columns(tmp_path,monkeypatch):
    import analysis
    d=fixture_dataset(tmp_path);j=job();monkeypatch.setattr(analysis,'_DATA',d)
    record=analysis._point_worker(j)
    output=tmp_path/'condition_losses.parquet'
    rows=analysis.export_losses(d,[record],output)
    frame=pd.read_parquet(output)
    assert rows==1120 and len(frame)==1120
    metadata=['run_id','panel','target','family','split_seed','response','fit_loss','evaluation_metric','peer_ids','input_fingerprint','completion_state']
    assert set(metadata)<=set(frame)
    for key in metadata:assert set(frame[key])=={record[key]}
    np.testing.assert_allclose(frame.groupby('base_question_id').condition_mass.sum(),1.,atol=1e-15)
    assert frame.total_variation.isna().all() and frame.top_option_agreement.isna().all()
    assert abs(np.sum(frame.condition_mass*frame.absolute_error)/280-record['e_group'])<1e-13
