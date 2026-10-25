"""Compare only saved, compatible MovieLens 32M models. Never train a base model.

Example: python -m src.lambdarank_disk --workspace auto --threads 2
Uses an existing temporal data cache and checkpoint metadata. Missing or
incompatible models are reported and excluded; fewer than two valid models
stops explicitly. Only LightGBM rerankers are fitted.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def discover(workspace, dataset='ml-32m'):
    root = Path(workspace)
    result, skipped = [], []
    for family in ('temporal_fm', 'unified_hofm3_bcd'):
        for folder in sorted((root / 'models' / dataset / family).glob('*')):
            if not folder.is_dir() or '_interrupted_' in folder.name:
                continue
            if not (folder / 'weights.pt').is_file() or not (folder / 'metadata.json').is_file():
                skipped.append({'path': str(folder), 'reason': 'missing weights.pt or metadata.json'})
                continue
            try:
                metadata = json.loads((folder / 'metadata.json').read_text())
                if family == 'temporal_fm' and (metadata.get('kind') not in ('fm', 'deepfm') or
                                                metadata.get('objective') != 'bpr'):
                    raise ValueError('unsupported checkpoint kind/objective')
                if family == 'unified_hofm3_bcd' and metadata.get('stage') != 'd':
                    raise ValueError('unsupported unified checkpoint')
                name = re.sub('[^a-z0-9_]', '_', family + '_' + folder.name.lower())
                result.append((name, family, folder, metadata))
            except (ValueError, OSError) as exc:
                skipped.append({'path': str(folder), 'reason': str(exc)})
    return result, skipped


def choose_workspace(project, explicit='auto'):
    if explicit != 'auto':
        return Path(explicit).resolve()
    roots = [Path(project), *Path(project).glob('artifacts/retrain_runs/*/code')]
    available = [root.resolve() for root in roots
                 if (root/'data/temporal/ml-32m/manifest.json').is_file()
                 and (discover(root)[0] or any((root/'models/ml-32m/shared_chrono_baselines').glob('*/factors.npz'))) ]
    if not available:
        raise FileNotFoundError('No on-disk ML32M checkpoints with prepared temporal data found. No retraining was started.')
    if len(available) > 1:
        raise ValueError('Multiple trained workspaces found; specify --workspace: ' + ', '.join(map(str, available)))
    return available[0]


def load_models(ev, provider, workspace, dataset, records):
    import numpy as np
    import torch
    from temporal_fm.ranking import _model, score_candidates
    loaded, skipped, fingerprints = {}, [], {}
    for name, family, folder, metadata in records:
        try:
            if family == 'temporal_fm':
                model = _model(ev, provider, folder.name, provider.backend, 'cpu')
                def predict(user, movies, at, model=model):
                    return score_candidates(model, ev, provider, user, movies, at, device='cpu', batch_size=2048)
            else:
                from temporal_fm import unified_hofm3_bcd as unified
                from temporal_fm.preference_history import HistoryIndex
                from temporal_fm.preference_experiment import TemporalPreferenceFeatures
                builder = TemporalPreferenceFeatures(ev, provider, HistoryIndex(ev), 'd')
                expected = {'dataset': dataset, 'sources': ev.meta['sources'],
                            'train_cutoff': ev.meta['train_cutoff'],
                            'validation_cutoff': ev.meta['val_cutoff'],
                            'backend': provider.backend, 'text_model': provider.model_name,
                            'tag_dims': provider.tags.tag_dims, 'feature_names': builder.names}
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ValueError('incompatible ' + key)
                parent = workspace/'models'/dataset/'temporal_fm'/metadata['parent_model']/'metadata.json'
                model = unified._make_model(ev, provider, json.loads(parent.read_text()), len(builder.names), 'cpu')
                state = torch.load(folder/'weights.pt', map_location='cpu', weights_only=True)
                if any(key.startswith('fourth_order.') for key in state):
                    from src.hofm4_interaction import FourthOrderJointBCD
                    config = json.loads((folder/'regularization_config.json').read_text())
                    if config.get('model_family') != 'hofm4_jointbcd':
                        raise ValueError('missing quartic architecture provenance')
                    model.fourth_order = FourthOrderJointBCD(model.user.num_embeddings,
                        model.movie.num_embeddings, len(builder.names), rank=config['fourth_rank'],
                        strength=config['fourth_strength'], input_dropout=config['fourth_dropout'])
                    original = model.forward
                    # Bind each closure independently; HOFM3 scoring remains unchanged.
                    def forward(batch, bcd, base=original, branch=model.fourth_order):
                        return base(batch, bcd) + branch(batch['user'], batch['item'], bcd)
                    model.forward = forward
                model.load_state_dict(state, strict=True)
                model.eval()
                def predict(user, movies, at, model=model, builder=builder):
                    return unified.score_items(model, ev, provider, builder, user, at, movies,
                                               device='cpu', chunk_size=1024)
            loaded[name] = predict
            from src.lambdarank_rerank import _sha256
            fingerprints[str(folder/'weights.pt')] = _sha256(folder/'weights.pt')
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            skipped.append({'path': str(folder), 'reason': str(exc)})
    # Read cached factors directly. Never call load_or_fit_baselines.
    for folder in sorted((workspace/'models'/dataset/'shared_chrono_baselines').glob('*')):
        try:
            if '_interrupted_' in folder.name:
                continue
            metadata = json.loads((folder/'manifest.json').read_text())
            expected = {'dataset': dataset, 'sources': ev.meta['sources'], 'cutoff': ev.meta['train_cutoff']}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError('incompatible baseline history')
            with np.load(folder/'factors.npz', allow_pickle=False) as cache:
                for kind in ('svd', 'bpr'):
                    uvec, ivec = cache[kind+'_user'].copy(), cache[kind+'_item'].copy()
                    if uvec.shape[0] != len(ev.train_users) or ivec.shape[0] != len(ev.movies):
                        raise ValueError('incompatible factor dimensions')
                    def predict(user, movies, at, uvec=uvec, ivec=ivec):
                        index = int(ev.user_indices(np.array([user], dtype=np.int32))[0]) - 1
                        if index < 0:
                            raise ValueError('Unknown training user')
                        return ivec[movies] @ uvec[index]
                    loaded[kind+'_'+folder.name] = predict
            from src.lambdarank_rerank import _sha256
            fingerprints[str(folder/'factors.npz')] = _sha256(folder/'factors.npz')
        except (ValueError, KeyError, OSError) as exc:
            skipped.append({'path': str(folder), 'reason': str(exc)})
    return loaded, skipped, fingerprints


def run(args):
    import numpy as np
    import torch
    project = Path(__file__).resolve().parents[1]
    workspace = choose_workspace(project, args.workspace)
    output = (Path(args.output).resolve() if args.output else
              workspace/'artifacts/rerank'/('disk_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if output.exists():
        raise FileExistsError('Choose a new output path: ' + str(output))
    # Install shipped modules only if absent; no model/data downloads or training.
    if not (workspace/'temporal_fm/ranking.py').exists():
        sys.path.insert(0, str(project/'scripts'))
        from retrain_all_ml32m import install
        runtime = output.parent / ('inference_code_' + output.name)
        install(runtime)
        sys.path.insert(0, str(runtime))
    else:
        sys.path.insert(0, str(workspace))
    os.chdir(workspace)
    from temporal_fm.data import Events
    from temporal_fm.lambdamart import _prepare
    from temporal_fm.ranking import _cohort
    from temporal_fm.advanced_ltr import _split_users
    torch.set_num_threads(args.threads)
    ev = Events(workspace/'data/temporal/ml-32m')
    provider = _prepare(ev, args.backend); provider.backend = args.backend
    records, skipped = discover(workspace)
    models, incompatible, fingerprints = load_models(ev, provider, workspace, 'ml-32m', records)
    inventory = {'workspace': str(workspace), 'included': sorted(models),
                 'excluded': skipped + incompatible, 'base_model_training': False}
    print(json.dumps(inventory, indent=2), flush=True)
    if not models:
        raise ValueError('No compatible saved models. No training was started.')
    if len(models) < 2:
        raise ValueError('At least two compatible saved models are required for a stacked comparative experiment.')
    if args.audit:
        return inventory
    ids, histories, truth, _, cohort = _cohort(ev, 'test', 4.,
        args.train_users + args.valid_users + args.test_users, 5, args.seed)
    splits = _split_users(ids, args.train_users, args.valid_users, args.test_users, args.seed)
    catalog = np.flatnonzero(ev.movie_has_train).astype(np.int32)
    cutoff = int(ev.meta['train_cutoff']) + 1
    names = sorted(models)
    output.mkdir(parents=True)
    for split, users in zip(('train', 'valid', 'test'), splits):
        count = 0
        with (output/(split+'.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id','item_id','label','total_relevant', *['score_'+name for name in names]])
            for number, user in enumerate(users, 1):
                unseen = np.setdiff1d(catalog, np.array(sorted(histories[user]), dtype=np.int32))
                scores = {name: np.asarray(models[name](user, unseen, cutoff)) for name in names}
                if any(len(values) != len(unseen) or not np.isfinite(values).all() for values in scores.values()):
                    raise ValueError('Invalid base scores')
                locations = np.unique(np.concatenate([np.argsort(-values, kind='stable')[:args.pool]
                                                       for values in scores.values()]))
                mids = unseen[locations]
                labels = np.isin(mids, list(truth[user])).astype(int)
                if split == 'train' and (not labels.any() or labels.all()):
                    continue
                count += 1
                for index, movie in enumerate(mids):
                    writer.writerow([f'{split}_{user}',int(movie),int(labels[index]),len(truth[user]),
                                     *[float(scores[name][locations[index]]) for name in names]])
                print(f'EXPORT {split}: {number}/{len(users)}', flush=True)
        if not count:
            raise ValueError('No usable queries for ' + split)
    manifest = {
        'dataset': {'id':'ml-32m','name':'MovieLens 32M','version':'ml-32m'},
        'protocol': {'id':'disk-only-original-test-user-disjoint-v1','dataset_id':'ml-32m',
            'split_method':'Original test-period users partitioned into ranker train/valid/test; base selection used earlier validation only',
            'candidate_policy':f'Label-blind union of top {args.pool} per saved model; no injected positives; supplied-pool metrics',
            'label_rule':'Original test ratings >=4, warm unseen movies',
            'cohort_signature':hashlib.sha256(json.dumps(splits).encode()).hexdigest(), 'cutoff':str(cutoff)},
        'features_from_past_only':True,'frozen_base_models':True,
        'test_untouched_for_training':True,'query_id_is_user':True,
        'base_checkpoint_sha256':fingerprints,'inventory':inventory,'cohort':cohort,
        'provenance_limit':'Input metadata checks cannot prove prior checkpoints were never tuned on original test labels. Requires original test period to have been unused for base selection.'}
    (output/'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'disk_inventory.json').write_text(json.dumps(inventory, indent=2))
    from src.lambdarank_compare import main
    main(['--train',str(output/'train.csv'),'--valid',str(output/'valid.csv'),
          '--test',str(output/'test.csv'),'--manifest',str(output/'evaluation_manifest.json'),
          '--models',','.join(names),'--threads',str(args.threads),'--output',str(output/'comparison')])
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', default='auto')
    p.add_argument('--backend', default='sentence', choices=['sentence','hash'])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--train-users', type=int, default=700)
    p.add_argument('--valid-users', type=int, default=200)
    p.add_argument('--test-users', type=int, default=300)
    p.add_argument('--pool', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output')
    p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    if min(args.threads,args.train_users,args.valid_users,args.test_users,args.pool) < 1:
        p.error('Counts must be positive')
    run(args)


if __name__ == '__main__':
    main()
