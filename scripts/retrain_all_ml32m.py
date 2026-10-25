"""Fresh, isolated training of the shipped Reco research model family.

Use --plan to inspect commands without downloads/training. Reuse --run-dir to
resume completed stages. Every new run starts without historical checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = [
 'research-patches/movielens32m_upgrade.zip',
 'research-patches/movie32m_csv_fix.zip',
 'research-patches/temporal_hybrid_fm_patch.zip',
 'research-patches/temporal_fm_ranking_patch.zip',
 'research-patches/temporal_fm_pairwise_ranking_patch.zip',
 'research-patches/temporal_fm_rating_aware_patch.zip',
 'research-patches/fm_serious_training_patch.zip',
 'research-patches/temporal_fm_regularization_patch.zip',
 'research-patches/fm_shared_comparison_patch.zip',
 'research-patches/temporal_fm_lambdamart_patch.zip',
 'reco_advanced_ranking_complete.zip',
 'temporal_fm_third_order_patch.zip',
 'reco_temporal_preferences_B_to_E_complete.zip',
 'reco_unified_hofm3_bcd_patch.zip',
]


def install(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        with zipfile.ZipFile(ROOT / archive) as package:
            for name in package.namelist():
                path = Path(name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archive path')
                if path.parts[0] not in ('scalable', 'temporal_fm', 'scripts'):
                    continue
                if not name.endswith('.py'):
                    continue
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(name))
    shutil.copytree(ROOT / 'src', destination / 'src', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('fast_unified_regularized.py', 'fast_unified_hofm4.py'):
        text = (ROOT / name).read_text()
        if name == 'fast_unified_hofm4.py':
            # The shipped launcher passes its result dict to sys.exit, which
            # reports exit 1 even after successful training. Normalize its CLI.
            text = text.replace('sys.exit(main(sys.argv[1:]))', 'main(sys.argv[1:])')
        (destination / name).write_text(text)
    # Apply the compatible surgical installers only AFTER the final trainer overlay.
    for script in ('install_semi_hard.py', 'install_hofm3.py'):
        options = (['--file', str(destination / 'temporal_fm/rank_train.py')]
                   if script == 'install_semi_hard.py' else ['--root', str(destination)])
        subprocess.run([sys.executable, str(destination / 'scripts' / script), *options], check=True)
    for path in destination.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')


def commands(a):
    common = ['--dataset', a.dataset, '--text-backend', a.backend,
              '--objective', 'bpr', '--profile', 'research', '--ablation', 'all',
              '--epochs', str(a.epochs), '--min-epochs', str(a.epochs),
              '--steps-per-epoch', str(a.steps), '--batch-size', '128',
              '--validation-users', '60', '--validation-mode', 'sampled',
              '--validation-candidates', '512', '--validate-every', '2',
              '--threads', str(a.threads), '--device', 'cpu', '--progress-every', '50']
    out = [('download', ['-m', 'scalable.prepare', '--dataset', a.dataset,
                         '--threads', str(a.threads)])]
    out.append(('fm_deepfm', ['-m', 'temporal_fm.rank_train', *common,
                              '--models', 'fm,deepfm']))
    out.append(('semi25', ['-m', 'temporal_fm.rank_train', *common, '--models', 'fm',
                           '--init-from', 'fm_all_bpr', '--run-label', 'semi25',
                           '--hard-negative-share', '.25', '--hard-pool', '8',
                           '--lr', '.00015', '--weight-decay', '.05', '--l2-factors', '.005']))
    out.append(('hofm3', ['-m', 'temporal_fm.rank_train', *common, '--models', 'fm',
                          '--init-from', 'fm_all_semi25_bpr', '--run-label', 'hofm3semi25',
                          '--fm-order', '3', '--third-rank', '32', '--third-l2', '.002',
                          '--hard-negative-share', '.25', '--hard-pool', '8',
                          '--lr', '.00015', '--weight-decay', '.05', '--l2-factors', '.005']))
    for stage in 'bcde':
        out.append(('preference_' + stage, ['-m', 'temporal_fm.preference_experiment',
            '--dataset', a.dataset, '--backend', a.backend, '--base-model', 'fm_all_semi25_bpr',
            '--stage', stage, '--run-label', 'jointBCD', '--epochs', str(a.epochs),
            '--steps-per-epoch', str(a.steps), '--eval-users', '60',
            '--threads', str(a.threads), '--max-candidates', '512']))
    for order in (3, 4):
        args = ['fast_unified_regularized.py' if order == 3 else 'fast_unified_hofm4.py',
            '--dataset', a.dataset, '--backend', a.backend,
            '--init-from', 'fm_all_hofm3semi25_bpr', '--run-label', f'fresh_hofm{order}',
            '--preference-checkpoint', 'auto', '--epochs', str(a.epochs),
            '--min-epochs', str(a.epochs), '--steps-per-epoch', str(a.steps),
            '--batch-size', '128', '--threads', str(a.threads), '--device', 'cpu',
            '--validation-users', '60', '--max-candidates', '512',
            '--weight-decay', '.12', '--third-l2', '.002',
            '--bcd-dropout', '.25', '--factor-l2', '.01', '--content-l2', '.002',
            '--bcd-l1', '.003', '--bcd-l2', '.005']
        if order == 4:
            args += ['--fourth-rank', '24', '--fourth-strength', '1.0',
                     '--fourth-dropout', '.20', '--fourth-factor-l2', '.001']
        out.append((f'joint_hofm{order}', args))
    out.append(('export', ['-m', 'src.export_fresh_ml32m', '--dataset', a.dataset,
                           '--backend', a.backend, '--threads', str(a.threads)]))
    out.append(('compare', ['-m', 'src.lambdarank_compare', '--top', '4',
                            '--threads', str(a.threads), '--output', 'artifacts/rerank/comparison']))
    if getattr(a, 'recipe', 'all') in ('light_hofm3', 'fm_lambdarank'):
        keep = {'download', 'fm_deepfm', 'semi25', 'hofm3', 'preference_d', 'joint_hofm3'}
        out = [(name, argv) for name, argv in out if name in keep]
        for name, argv in out:
            changes = {'--batch-size':'64', '--validation-users':'20',
                       '--eval-users':'20', '--validation-candidates':'256',
                       '--max-candidates':'256', '--hard-pool':'4'}
            if name == 'fm_deepfm':
                changes['--models'] = 'fm'
            if name == 'preference_d':
                argv += ['--batch-size', '64']
            if name == 'joint_hofm3':
                argv += ['--hard-negative-share', '.25', '--hard-pool', '4']
            for flag, value in changes.items():
                if flag in argv:
                    argv[argv.index(flag)+1] = value
    if getattr(a, 'recipe', 'all') == 'fm_lambdarank':
        out = [(name, argv) for name, argv in out if name in ('download', 'fm_deepfm')]
        out.append(('fm_lambdarank', ['-m', 'src.lambdarank_disk', '--workspace', '.',
            '--backend', a.backend, '--models', 'fm_all_bpr', '--pool', '500',
            '--threads', str(a.threads), '--output', 'artifacts/rerank/fm500']))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', default='artifacts/retrain_runs/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--dataset', choices=['ml-32m', 'ml-1m', 'ml-latest-small'], default='ml-32m')
    p.add_argument('--recipe', choices=['all', 'light_hofm3', 'fm_lambdarank'], default='all')
    p.add_argument('--backend', choices=['sentence', 'hash'], default='sentence')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--install-only', action='store_true')
    a = p.parse_args()
    if min(a.threads, a.epochs, a.steps) < 1:
        p.error('threads, epochs and steps must be positive')
    plan = commands(a)
    if a.plan:
        print(json.dumps(plan, indent=2)); return
    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'running.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit('Run is locked; check for a running process before removing ' + str(lock))
    try:
        config = {key: value for key, value in vars(a).items() if key not in ('plan', 'install_only', 'run_dir')}
        tracked = [*ARCHIVES, 'scripts/retrain_all_ml32m.py',
                   'src/export_fresh_ml32m.py', 'src/lambdarank_compare.py',
                   'fast_unified_regularized.py', 'fast_unified_hofm4.py']
        config['source_sha256'] = hashlib.sha256(''.join(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in tracked
        ).encode()).hexdigest()
        metadata = run / 'run_config.json'
        if metadata.exists() and json.loads(metadata.read_text()) != config:
            raise ValueError('Run configuration changed; choose a fresh --run-dir')
        metadata.write_text(json.dumps(config, indent=2))
        code = run / 'code'
        if not (run / 'installed.ok').exists():
            install(code)
            (run / 'installed.ok').touch()
        if a.install_only:
            return
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads),
                   MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        for stage, args in plan:
            marker = run / (stage + '.ok')
            if marker.exists():
                print('Completed earlier:', stage, flush=True); continue
            print('START', stage, flush=True)
            log = run / (stage + '.log')
            if log.exists():
                # Preserve partial checkpoints before restarting an unfinished
                # stage. Successful earlier stages are never moved.
                base = code / 'models' / a.dataset
                targets = {
                    'fm_deepfm': [base/'temporal_fm/fm_all_bpr', base/'temporal_fm/deepfm_all_bpr'],
                    'semi25': [base/'temporal_fm/fm_all_semi25_bpr'],
                    'hofm3': [base/'temporal_fm/fm_all_hofm3semi25_bpr'],
                    **{'preference_' + stage: [base/'temporal_preferences'/f'fm_all_semi25_bpr_{stage}_jointBCD'] for stage in 'bcde'},
                    **{f'joint_hofm{order}': [base/'unified_hofm3_bcd'/f'fm_all_hofm3semi25_bpr_fresh_hofm{order}'] for order in (3,4)},
                    'compare': [code/'artifacts/rerank/comparison'],
                    'fm_lambdarank': [code/'artifacts/rerank/fm500'],
                }
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                for target in targets.get(stage, []):
                    if target.exists():
                        target.rename(target.with_name(target.name + '_interrupted_' + stamp))
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, '-u', *args], cwd=code, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-6000:], flush=True)
                raise SystemExit(f'Stage {stage} failed; see {log}. Rerun with the SAME --run-dir after fixing it.')
            marker.touch()
            print('DONE', stage, flush=True)
        result = (code/'models'/a.dataset/'unified_hofm3_bcd/fm_all_hofm3semi25_bpr_fresh_hofm3/weights.pt'
                  if a.recipe == 'light_hofm3' else code/'artifacts/rerank/comparison/comparison.md')
        if a.recipe == 'fm_lambdarank':
            result = code/'artifacts/rerank/fm500/comparison/comparison.md'
        print('COMPLETE:', result, flush=True)
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
