from pathlib import Path
from src.lambdarank_disk import requested_model


def test_fm_only_filter_excludes_unrelated_checkpoints():
    wanted={'fm_all_bpr'}
    assert requested_model('temporal_fm_fm_all_bpr',Path('fm_all_bpr'),wanted)
    assert not requested_model('temporal_fm_deepfm_all_bpr',Path('deepfm_all_bpr'),wanted)
    assert not requested_model('hofm4_jointbcd',Path('hofm4'),wanted)
    assert requested_model('any',Path('any'),set())


def test_launcher_requests_five_hundred_fm_candidates_and_two_threads():
    text=(Path(__file__).resolve().parents[1]/'scripts/fm_lambdarank_500.sh').read_text()
    assert '--models fm_all_bpr --pool 500 --threads 2' in text


def test_single_model_writes_baseline_and_reranker_comparison(tmp_path):
    import json
    import pytest
    pytest.importorskip('lightgbm')
    pd=pytest.importorskip('pandas')
    from src.lambdarank_disk import single_comparison
    for split,queries in [('train',20),('valid',4),('test',4)]:
        rows=[]
        for query in range(queries):
            for item in range(20):
                rows.append(dict(query_id=f'{split}_{query}',item_id=item,
                    label=int(item==query%20),total_relevant=1,
                    score_fm_all_bpr=float((item+query)%20)))
        pd.DataFrame(rows).to_csv(tmp_path/(split+'.csv'),index=False)
    protocol={key:'synthetic' for key in ('id','split_method','candidate_policy','label_rule','cohort_signature','cutoff')}
    protocol['dataset_id']='ml-32m'
    manifest=dict(dataset={'id':'ml-32m','name':'SYNTHETIC FIXTURE','version':'fixture'},
                  protocol=protocol,features_from_past_only=True,frozen_base_models=True,test_untouched_for_training=True)
    (tmp_path/'evaluation_manifest.json').write_text(json.dumps(manifest))
    single_comparison(tmp_path,'fm_all_bpr',2)
    frame=pd.read_csv(tmp_path/'comparison/comparison.csv')
    assert frame.variant.tolist()==['frozen_fm_all_bpr','lambdarank_fm_all_bpr']
    assert (tmp_path/'comparison/comparison.md').is_file()
    assert not (tmp_path/'comparison/lambdarank_ensemble.txt').exists()
