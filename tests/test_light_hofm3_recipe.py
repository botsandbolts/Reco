from argparse import Namespace
import importlib.util
from pathlib import Path


def test_light_recipe_trains_only_required_family_with_requested_budget():
    spec=importlib.util.spec_from_file_location('fresh_light',Path(__file__).resolve().parents[1]/'scripts/retrain_all_ml32m.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=Namespace(dataset='ml-32m',backend='sentence',epochs=10,steps=200,threads=2,recipe='light_hofm3')
    stages=module.commands(args)
    assert [name for name,_ in stages]==['download','fm_deepfm','semi25','hofm3','preference_d','joint_hofm3']
    for name,argv in stages:
        assert argv[argv.index('--threads')+1]=='2'
        if name=='download': continue
        assert argv[argv.index('--epochs')+1]=='10'
        assert argv[argv.index('--steps-per-epoch')+1]=='200'
        assert argv[argv.index('--batch-size')+1]=='64'
    for name in ('semi25','hofm3','joint_hofm3'):
        argv=dict(stages)[name]
        assert argv[argv.index('--hard-negative-share')+1]=='.25'
        assert argv[argv.index('--hard-pool')+1]=='4'
    assert dict(stages)['fm_deepfm'][dict(stages)['fm_deepfm'].index('--models')+1]=='fm'
