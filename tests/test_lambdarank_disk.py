import json
from pathlib import Path
import pytest
from src.lambdarank_disk import discover, choose_workspace


def checkpoint(root, name, metadata, weights=True):
    folder=root/'models/ml-32m/temporal_fm'/name
    folder.mkdir(parents=True)
    (folder/'metadata.json').write_text(json.dumps(metadata))
    if weights:
        (folder/'weights.pt').write_bytes(b'fixture')
    return folder


def test_discovery_uses_existing_checkpoint_pairs_only(tmp_path):
    checkpoint(tmp_path,'fm_all_bpr',{'kind':'fm','objective':'bpr'})
    checkpoint(tmp_path,'deepfm_all_bpr',{'kind':'deepfm','objective':'bpr'},weights=False)
    checkpoint(tmp_path,'unrelated',{'kind':'unknown','objective':'rating'})
    found, skipped=discover(tmp_path)
    assert len(found)==1
    assert found[0][2].name=='fm_all_bpr'
    assert len(skipped)==2
    assert not (tmp_path/'models/ml-32m/temporal_fm/deepfm_all_bpr/weights.pt').exists()


def test_no_checkpoints_stops_without_training(tmp_path):
    with pytest.raises(FileNotFoundError,match='No retraining was started'):
        choose_workspace(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_multiple_workspaces_require_explicit_selection(tmp_path):
    for root in (tmp_path,tmp_path/'artifacts/retrain_runs/run/code'):
        cache=root/'data/temporal/ml-32m'
        cache.mkdir(parents=True)
        (cache/'manifest.json').write_text('{}')
        checkpoint(root,'fm_all_bpr',{'kind':'fm','objective':'bpr'})
    with pytest.raises(ValueError,match='Multiple trained workspaces'):
        choose_workspace(tmp_path)
