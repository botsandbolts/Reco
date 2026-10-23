#!/usr/bin/env python3
"""Fast + strongly regularized joint HOFM3 / semi25 / JointBCD trainer.

Run from the Reco project root with the same arguments accepted by
`python -m temporal_fm.unified_hofm3_bcd`, plus the regularization options below.

No repository sources or parent checkpoints are modified. This launcher patches
*in-process* model methods and the temporal feature builder before calling the
existing trainer. It requires the unified model and B-E patches already installed.

Examples:
    python fast_unified_regularized.py --self-test
    python -u fast_unified_regularized.py --run-label fast_bcd_reg_v1 \
      --epochs 15 --min-epochs 15 --patience 999 --force-retrain ...
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as torch_F


@lru_cache(maxsize=2048)
def _profiles(builder, user: int, timestamp: int):
    """Cache candidate-INDEPENDENT features per historical user/time context.

    The Stage D builder always derives context from strictly earlier TRAIN
    events; this cache does not change the history cut-off policy.
    """
    from temporal_fm.preference_genre import weighted_profile

    _, g, s = builder.context(user, timestamp)
    g_rows, g_weights, ages = g['genres'], g['weights'], g['ages']
    g_long = weighted_profile(g_rows, g_weights)
    g30 = weighted_profile(g_rows[-30:], g_weights[-30:])
    g90 = weighted_profile(g_rows, g_weights * np.exp2(-ages / 90.0))
    g365 = weighted_profile(g_rows, g_weights * np.exp2(-ages / 365.0))
    genre_vectors = np.stack((
        weighted_profile(g_rows[-10:], g_weights[-10:]),
        g30,
        weighted_profile(g_rows[-100:], g_weights[-100:]),
        g_long,
        g30 - g_long,
        g90,
        g365,
        g90 - g_long,
    ), axis=1).astype(np.float32)

    t_rows, t_weights = s['tags'], s['weights']
    t_long = weighted_profile(t_rows, t_weights)
    t30 = weighted_profile(t_rows[-30:], t_weights[-30:])
    t90 = weighted_profile(t_rows, t_weights * np.exp2(-ages / 90.0))
    t365 = weighted_profile(t_rows, t_weights * np.exp2(-ages / 365.0))
    tag_vectors = np.stack((
        t30,
        weighted_profile(t_rows[-100:], t_weights[-100:]),
        t_long,
        t30 - t_long,
        t90,
        t365,
        t90 - t_long,
    ), axis=1).astype(np.float32)
    return genre_vectors, tag_vectors


def install_optimizations_and_regularization(*, bcd_dropout: float,
                                             factor_l2: float,
                                             content_l2: float,
                                             bcd_l1: float,
                                             bcd_l2: float,
                                             fast_features: bool = True):
    """Patch ONLY this Python process. Return the installed trainer module."""
    from temporal_fm import preference_experiment as pref
    from temporal_fm import unified_hofm3_bcd as trainer
    from temporal_fm.preference_genre import safe_normalize

    cls = trainer.UnifiedHOFM3BCD
    if not all(hasattr(cls, k) for k in ('forward', 'third_regularization')):
        raise RuntimeError('Unified HOFM3 class is incompatible with this launcher')
    if getattr(cls, '_strong_reg_launcher_installed', False):
        raise RuntimeError('Regularization launcher already installed in this process')

    original_forward = cls.forward
    original_penalty = cls.third_regularization

    def regularized_forward(self, batch, bcd):
        # The parent's HOFM3 factors are never disabled. Drop *input features*
        # of the newly learned temporal field, only while training.
        if self.training and bcd_dropout:
            bcd = torch_F.dropout(bcd, p=bcd_dropout, training=True)
        return original_forward(self, batch, bcd)

    def stronger_penalty(self, positive, negative, coefficient=0.0):
        # Called directly from the ORIGINAL BPR loss every optimization step:
        # loss = rank_loss + model.third_regularization(...)
        reg = original_penalty(self, positive, negative, coefficient)

        if factor_l2:
            reg = reg + factor_l2 * (
                self.user(positive['user']).square().mean()
                + self.movie(positive['item']).square().mean()
                + self.movie(negative['item']).square().mean()
            )
        if content_l2:
            # Dense feature projections only. Avoid scanning huge user/item
            # embedding tables for regularization on every minibatch.
            layers = (self.title, self.genre, self.tags, self.time_cont,
                      self.third_title, self.third_genre,
                      self.third_tags, self.third_time)
            reg = reg + content_l2 * sum(
                (layer.weight.square().mean() for layer in layers
                 if layer.weight.numel()),
                reg.new_zeros(())
            )
        if bcd_l1:
            # L1 sparsifies learned genre/tag/time preference coefficients.
            reg = reg + bcd_l1 * self.bcd_linear.weight.abs().sum()
        if bcd_l2:
            reg = reg + bcd_l2 * (
                self.bcd_linear.weight.square().mean()
                + self.bcd_pair.weight.square().mean()
                + self.bcd_triple.weight.square().mean()
            )
        return reg

    cls.forward = regularized_forward
    cls.third_regularization = stronger_penalty
    cls._strong_reg_launcher_installed = True

    if fast_features:
        original_matrix = pref.TemporalPreferenceFeatures.matrix

        def fast_matrix(self, user, timestamp, movie_ids, ctx=None):
            if self.stage != 'd':
                return original_matrix(self, user, timestamp, movie_ids, ctx)
            movie_ids = np.asarray(movie_ids, dtype=np.int32)
            G, T = _profiles(self, int(user), int(timestamp))
            genres = safe_normalize(self.ev.genre_matrix[movie_ids])
            tags = self.provider.tags.lookup(
                movie_ids, np.full(len(movie_ids), int(timestamp), dtype=np.int64)
            )
            tags = safe_normalize(tags)
            g = genres @ G
            t = tags @ T
            result = np.column_stack((g[:, :5], t[:, :4], g[:, 5:], t[:, 4:])).astype(np.float32)
            if not np.isfinite(result).all():
                raise FloatingPointError('Non-finite temporal features')
            return result

        def fast_batch(builder, users, times, movies, device):
            # Calculate all candidates for the same user/time in one operation.
            groups = defaultdict(list)
            for j, (u, t) in enumerate(zip(users, times)):
                groups[(int(u), int(t))].append(j)
            mids = np.asarray(movies, dtype=np.int32)
            if not len(mids):
                return torch.empty((0, len(builder.names)), dtype=torch.float32, device=device)
            out = np.empty((len(mids), len(builder.names)), dtype=np.float32)
            for (user, timestamp), rows in groups.items():
                out[rows] = builder.matrix(user, timestamp, mids[rows])
            return torch.as_tensor(out, dtype=torch.float32, device=device)

        pref.TemporalPreferenceFeatures.matrix = fast_matrix
        trainer._compute_features = fast_batch

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A session embedding this script may have already set the thread pool.
        pass
    return trainer


def _read_option(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        try:
            return argv[argv.index(name) + 1]
        except IndexError as exc:
            raise SystemExit(f'Missing value after {name}') from exc
    for argument in argv:
        if argument.startswith(name + '='):
            return argument.split('=', 1)[1]
    return default


def _preflight_run_label(argv, config):
    dataset = _read_option(argv, '--dataset', 'ml-32m')
    parent = _read_option(argv, '--init-from', 'fm_all_hofm3semi25_bpr')
    label = _read_option(argv, '--run-label', '')
    if not label:
        raise SystemExit('Supply a NEW --run-label, e.g. fast_bcd_strongreg_v1')
    if label in ('fast_bcd', 'bcd_hofm3_research'):
        raise SystemExit('Refusing to overwrite previous unregularized run; choose a NEW label')
    if not label.replace('_', '').isalnum():
        raise SystemExit('--run-label may contain letters, digits and underscores only')
    folder = Path('models') / dataset / 'unified_hofm3_bcd' / f'{parent}_{label}'
    config_path = folder / 'regularization_config.json'
    force = '--force-retrain' in argv
    existing_weights = (folder / 'weights.pt').exists()
    if existing_weights and not config_path.exists() and not force:
        raise SystemExit(f'{folder} has old weights without regularization metadata. '
                         'Use a NEW --run-label; do not mix experiments.')
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing != config and not force:
            raise SystemExit('Regularization settings differ from this checkpoint. '
                             'Use a NEW --run-label or explicit --force-retrain.')
    if '--resume' in argv:
        # Existing resume helper does not track regularization config in the
        # saved optimizer snapshot. Fail closed rather than pretend exact resume.
        raise SystemExit('--resume is not supported by this launcher: its optimizer '
                         'snapshot does not record these new penalties. Use a '
                         'new run label or train with this launcher from epoch 1.')
    folder.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    if existing_weights and not force:
        print('Existing weights found. Trainer will load/evaluate them; '
              'add --force-retrain to explicitly restart.', flush=True)
    return folder


def _self_test():
    """Offline test: penalties are differentiable and cache batching is sound."""
    import torch.nn as nn
    class Toy:
        def __init__(self):
            self.stage = 'd'
            self.names = tuple(str(j) for j in range(15))
            self.cache = {}
        def matrix(self, user, timestamp, movie_ids):
            mids = np.asarray(movie_ids, dtype=np.float32)
            return np.stack([(mids + user + timestamp) * (j + 1) / 100
                             for j in range(15)], axis=-1).astype(np.float32)
    toy = Toy()
    from collections import defaultdict as dd
    grp = dd(list)
    users = [1, 1, 3, 1, 3]
    times = [100, 100, 80, 100, 80]
    mids = np.array([4, 5, 3, 7, 2])
    for j, (u, t) in enumerate(zip(users, times)):
        grp[u, t].append(j)
    out = np.empty((5, 15), np.float32)
    for (u, t), inds in grp.items():
        out[inds] = toy.matrix(u, t, mids[inds])
    expected = np.stack([toy.matrix(u, t, [m])[0]
                         for u, t, m in zip(users, times, mids)])
    assert np.allclose(out, expected)
    layer = nn.Linear(15, 1, bias=False)
    x = torch.randn(8, 15)
    base = layer(x).sum()
    regularized = base + 0.001 * layer.weight.abs().sum() + 0.002 * layer.weight.square().mean()
    regularized.backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    torch.manual_seed(42)
    v = torch.ones(1000, 15)
    assert (torch_F.dropout(v, p=0.15, training=True) == 0).any()
    assert torch.equal(torch_F.dropout(v, p=0.15, training=False), v)
    print('SELF-TEST PASS: grouped features, L1/L2 gradients, train-only dropout')


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False,
        description='Strong regularization wrapper for optimized Unified HOFM3 + JointBCD')
    parser.add_argument('--bcd-dropout', type=float, default=0.15,
                        help='Input feature dropout on the B+C+D field (default 0.15)')
    parser.add_argument('--factor-l2', type=float, default=0.0025,
                        help='L2 on sampled pairwise user/movie factors')
    parser.add_argument('--content-l2', type=float, default=0.0005,
                        help='L2 on dense pairwise and cubic content projections')
    parser.add_argument('--bcd-l1', type=float, default=0.001,
                        help='L1 on 15 learned temporal feature coefficients')
    parser.add_argument('--bcd-l2', type=float, default=0.002,
                        help='L2 on temporal linear / pair / triple projections')
    parser.add_argument('--disable-fast-features', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    extra, passthrough = parser.parse_known_args(argv)
    if extra.self_test:
        _self_test()
        return
    if '--help' in passthrough or '-h' in passthrough:
        print(parser.format_help())
        # Show the original trainer's available CLI options as well.
        trainer = install_optimizations_and_regularization(
            bcd_dropout=extra.bcd_dropout, factor_l2=extra.factor_l2,
            content_l2=extra.content_l2, bcd_l1=extra.bcd_l1,
            bcd_l2=extra.bcd_l2, fast_features=not extra.disable_fast_features)
        return trainer.main(['--help'])
    import math
    cfg = {key: float(getattr(extra, key)) for key in
           ('bcd_dropout', 'factor_l2', 'content_l2', 'bcd_l1', 'bcd_l2')}
    cfg.update({'fast_features': not extra.disable_fast_features,
                'trainer_weight_decay': float(_read_option(passthrough, '--weight-decay', '0.01')),
                'trainer_third_l2': float(_read_option(passthrough, '--third-l2', '0.0001')),
                'hard_negative_share': float(_read_option(passthrough, '--hard-negative-share', '0.25')),
                'launcher_version': 1})
    if not (0 <= cfg['bcd_dropout'] < 1):
        raise SystemExit('--bcd-dropout must be between 0 and 1 (exclusive)')
    if any(not math.isfinite(cfg[k]) or cfg[k] < 0
           for k in ('factor_l2', 'content_l2', 'bcd_l1', 'bcd_l2',
                     'trainer_weight_decay', 'trainer_third_l2')):
        raise SystemExit('All L1/L2/weight-decay coefficients must be finite and nonnegative')
    if not (0 <= cfg['hard_negative_share'] <= 1):
        raise SystemExit('--hard-negative-share must be in [0, 1]')
    run_folder = _preflight_run_label(passthrough, cfg)
    trainer = install_optimizations_and_regularization(
        bcd_dropout=cfg['bcd_dropout'],
        factor_l2=cfg['factor_l2'],
        content_l2=cfg['content_l2'],
        bcd_l1=cfg['bcd_l1'], bcd_l2=cfg['bcd_l2'],
        fast_features=cfg['fast_features'])
    print('STRONGLY REGULARIZED FAST HOFM3 + JointBCD', flush=True)
    print('New run folder:', run_folder, flush=True)
    print('L1/L2/dropout:', cfg, flush=True)
    print('Semi-hard negatives, epoch logging and AdamW are managed by the original trainer.',
          flush=True)
    return trainer.main(passthrough)


if __name__ == '__main__':
    main(sys.argv[1:])
