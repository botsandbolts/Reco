"""Frozen score exports for a fresh isolated research run.

Ranker supervision uses three disjoint user cohorts from the ORIGINAL test
period. Base checkpoints may use original validation for model selection, but
never original test labels. No held-out positives are injected into retrieval.
The ranker test cohort remains held out from ranker training and selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

PARENT = 'fm_all_hofm3semi25_bpr'


def load_joint(ev, provider, dataset, order):
    from temporal_fm import unified_hofm3_bcd as unified
    from temporal_fm.preference_history import HistoryIndex
    from temporal_fm.preference_experiment import TemporalPreferenceFeatures
    builder = TemporalPreferenceFeatures(ev, provider, HistoryIndex(ev), 'd')
    parent = Path('models') / dataset / 'temporal_fm' / PARENT / 'metadata.json'
    folder = Path('models') / dataset / 'unified_hofm3_bcd' / f'{PARENT}_fresh_hofm{order}'
    metadata = json.loads((folder / 'metadata.json').read_text())
    expected = {'dataset': dataset, 'sources': ev.meta['sources'],
                'train_cutoff': ev.meta['train_cutoff'], 'parent_model': PARENT,
                'feature_names': builder.names, 'backend': provider.backend}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Joint checkpoint mismatch: {key}')
    model = unified._make_model(ev, provider, json.loads(parent.read_text()), len(builder.names), 'cpu')
    if order == 4:
        from src.hofm4_interaction import FourthOrderJointBCD
        model.fourth_order = FourthOrderJointBCD(model.user.num_embeddings,
            model.movie.num_embeddings, bcd_dim=len(builder.names), rank=24,
            strength=1.0, input_dropout=.20)
        original = model.forward
        model.forward = lambda batch, bcd: original(batch, bcd) + model.fourth_order(batch['user'], batch['item'], bcd)
    # Strict loading prevents silently scoring with untrained quartic factors.
    model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return model, builder


def export(dataset='ml-32m', backend='sentence', threads=2, train_users=700,
           valid_users=200, test_users=300, pool=200):
    from temporal_fm.data import Events
    from temporal_fm.lambdamart import _prepare
    from temporal_fm.ranking import _cohort, _model, score_candidates
    from temporal_fm.advanced_retrieval import AdvancedRetriever
    from temporal_fm.advanced_ltr import _split_users
    from temporal_fm.unified_hofm3_bcd import score_items
    torch.set_num_threads(threads)
    ev = Events(Path('data/temporal') / dataset)
    provider = _prepare(ev, backend)
    provider.backend = backend
    ids, histories, truth, pop, cohort = _cohort(ev, 'test', 4.,
        train_users + valid_users + test_users, 5, 42)
    split_ids = _split_users(ids, train_users, valid_users, test_users, 42)
    fm = _model(ev, provider, 'fm_all_bpr', backend, 'cpu', seed=42)
    retriever = AdvancedRetriever(ev, provider, fm, histories, ids, pop,
        sources=('fm', 'svd', 'bpr', 'pop', 'semantic'), baseline_profile='research',
        seed=42, threads=threads, threshold=4., device='cpu', batch_size=2048)
    model_names = {'temporal_fm': 'fm_all_bpr', 'deepfm': 'deepfm_all_bpr',
                   'semi25': 'fm_all_semi25_bpr', 'hofm3': PARENT}
    models = {key: _model(ev, provider, name, backend, 'cpu', seed=42)
              for key, name in model_names.items()}
    joint = {order: load_joint(ev, provider, dataset, order) for order in (3, 4)}
    score_names = ['bpr', 'svd', 'popularity', 'semantic', *model_names,
                   'hofm3_semi25_jointbcd', 'hofm4_jointbcd']
    out = Path('artifacts/rerank')
    out.mkdir(parents=True, exist_ok=True)
    candidate_cols = ['query_id', 'item_id', 'label', 'total_relevant',
                      *['score_' + name for name in score_names]]
    frozen_weights = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path('models').rglob('weights.pt')}
    # Baseline factors are produced by AdvancedRetriever and fitted only on TRAIN.
    for path in Path('models').rglob('factors.npz'):
        frozen_weights[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for split, users in zip(('train', 'valid', 'test'), split_ids):
        with (out / (split + '.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(candidate_cols)
            informative = 0
            for number, user in enumerate(users, 1):
                quotas = {source: pool for source in retriever.sources}
                row = retriever.query(user, histories[user], truth[user], quotas)
                if row is None:
                    raise ValueError(f'No eligible candidates for {split} user {user}')
                mids = row['movies']
                # Expand the shared pool using every trained neural base model,
                # including unified HOFM3/HOFM4, without inspecting relevance.
                unseen = np.setdiff1d(retriever.catalog, np.array(sorted(histories[user]), dtype=np.int32))
                full_scores = {}
                for name, model in models.items():
                    full_scores[name] = score_candidates(model, ev, provider, user,
                        unseen, retriever.cutoff, device='cpu', batch_size=2048)
                for order, name in ((3, 'hofm3_semi25_jointbcd'), (4, 'hofm4_jointbcd')):
                    model, builder = joint[order]
                    full_scores[name] = score_items(model, ev, provider, builder,
                        user, retriever.cutoff, unseen, device='cpu', chunk_size=1024)
                for values in full_scores.values():
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite base score')
                    mids = np.union1d(mids, unseen[np.argsort(-values, kind='stable')[:pool]])
                loc = np.searchsorted(unseen, mids)
                scores = {name: values[loc] for name, values in full_scores.items()}
                ui = int(ev.user_indices(np.array([user], dtype=np.int32))[0]) - 1
                for name, (uvec, ivec) in retriever.baseline.items():
                    scores[name] = ivec[mids] @ uvec[ui]
                scores['popularity'] = np.log1p(pop[mids])
                scores['semantic'] = retriever.snapshot.title[mids] @ retriever.snapshot.title_affinity[int(user)]
                labels = np.isin(mids, list(truth[user])).astype(int)
                # LightGBM needs positive and negative ranker-training examples.
                # Validation/test retain retrieval misses, contributing zero NDCG.
                if split == 'train' and (not labels.any() or labels.all()):
                    continue
                informative += 1
                for index, movie in enumerate(mids):
                    writer.writerow([f'{split}_{user}', str(int(movie)), int(labels[index]),
                                     len(truth[user]), *[float(scores[name][index]) for name in score_names]])
                if number % 10 == 0:
                    print(f'EXPORT {split}: {number}/{len(users)} users', flush=True)
            if not informative:
                raise ValueError('No usable queries in ' + split)
    manifest = {
        'dataset': {'id': dataset, 'name': 'MovieLens ' + dataset, 'version': dataset},
        'protocol': {'id': 'fresh-global-chrono-user-disjoint-ranker-v1',
            'dataset_id': dataset,
            'split_method': 'Base global train/validation; ORIGINAL test-period labels partitioned by user into ranker train/valid/test',
            'candidate_policy': f'Label-blind union of top {pool} per frozen retriever; no positive injection; supplied-pool metrics',
            'label_rule': 'Original test-period rating >=4; warm items only; training histories excluded',
            'cohort_signature': hashlib.sha256(json.dumps(split_ids).encode()).hexdigest(),
            'cutoff': str(ev.meta['train_cutoff'])},
        'features_from_past_only': True, 'frozen_base_models': True,
        'test_untouched_for_training': True, 'query_id_is_user': True,
        'base_checkpoint_sha256': frozen_weights, 'cohort': cohort,
        'note': 'Final ranker test users are disjoint from its train/valid users. Base-model tuning uses only the earlier original validation period. Stage B-E heads trained but not all included as separate score columns.'}
    (out / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ml-32m')
    parser.add_argument('--backend', default='sentence', choices=['sentence', 'hash'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-users', type=int, default=700)
    parser.add_argument('--valid-users', type=int, default=200)
    parser.add_argument('--test-users', type=int, default=300)
    parser.add_argument('--pool', type=int, default=200)
    args = parser.parse_args()
    if min(args.threads, args.train_users, args.valid_users, args.test_users, args.pool) < 1:
        parser.error('Counts must be positive')
    export(**vars(args))


if __name__ == '__main__':
    main()
