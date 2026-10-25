"""Checks for the fresh training orchestrator; no internet/data download needed."""
from argparse import Namespace
import importlib.util
from pathlib import Path
import subprocess
import sys

SPEC = importlib.util.spec_from_file_location('fresh', Path(__file__).resolve().parents[1] / 'scripts/retrain_all_ml32m.py')
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)


def test_training_plan_respects_cpu_budget_and_dependency_order():
    args = Namespace(dataset='ml-32m', backend='sentence', threads=2, epochs=15, steps=800)
    stages = fresh.commands(args)
    names = [name for name, _ in stages]
    assert names.index('semi25') < names.index('hofm3') < names.index('joint_hofm4')
    assert names.index('preference_d') < names.index('joint_hofm3')
    assert names[-2:] == ['export', 'compare']
    assert not any('--force-retrain' in argv for _, argv in stages)
    for _, argv in stages:
        assert argv[argv.index('--threads') + 1] == '2'


def test_installer_produces_compatible_trainer_without_touching_root(tmp_path):
    fresh.install(tmp_path)
    trainer = (tmp_path / 'temporal_fm/rank_train.py').read_text()
    assert 'HOFM3_DEGREE3_EXTENSION_V1' in trainer
    assert 'hard_negative_share' in trainer
    # HOFM4 returns a report dictionary, which must not become an exit code.
    launcher = (tmp_path / 'fast_unified_hofm4.py').read_text()
    assert 'sys.exit(main(sys.argv[1:]))' not in launcher
    assert 'main(sys.argv[1:])' in launcher
    assert not (tmp_path / 'models').exists()


def test_plan_is_read_only(tmp_path):
    output = tmp_path / 'must_not_exist'
    result = subprocess.run([sys.executable, str(fresh.ROOT/'scripts/retrain_all_ml32m.py'),
                             '--plan', '--run-dir', str(output)], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'joint_hofm4' in result.stdout
    assert not output.exists()
