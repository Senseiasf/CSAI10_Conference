"""Export and evaluate the existing splits through the CatBoost training loader."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.scripts.train_ltr import create_catboost_pool, load_and_flatten_jsonl_sources
from app.services.ltr_service import FEATURE_COLUMNS_V2, FEATURE_SCHEMA_V2, LTRService

DATA = ROOT / 'backend/app/data'
MODEL = DATA / 'catboost_travel_ranker_v2.cbm'
METRIC = 'NDCG:top=2;type=Base'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pairs(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def metrics(expected, scores):
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite ranker scores')
    margins = (scores[:, 0] - scores[:, 1]) * np.where(expected[:, 0] == 1, 1, -1)
    correct = int((margins > 0).sum())
    ties = int((margins == 0).sum())
    return dict(pairs=len(expected), correct=correct, ties=ties,
                pairwise_accuracy=(correct + 0.5 * ties) / len(expected),
                ndcg_at_2=float(ndcg_score(expected, scores, k=2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--source-ref', help='Git revision containing the split files and the same checkpoint')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {split: DATA / f'csai_tradeoffs_v2_{split}.jsonl' for split in ('train', 'val', 'test')}
    source_commit = subprocess.check_output(['git', 'rev-parse', args.source_ref or 'HEAD'], cwd=ROOT, text=True).strip()
    if args.source_ref:
        def git_bytes(path):
            return subprocess.check_output(['git', 'show', f'{source_commit}:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT)

        if git_bytes(MODEL) != MODEL.read_bytes():
            raise ValueError('Source revision contains a different model')
        snapshot = output / 'source'
        snapshot.mkdir(exist_ok=True)
        for split, path in paths.items():
            snapshot_path = snapshot / path.name
            snapshot_path.write_bytes(git_bytes(path))
            paths[split] = snapshot_path
    input_hashes = {path.name: sha256(path) for path in [MODEL, *paths.values()]}
    records = {split: read_pairs(path) for split, path in paths.items()}
    for field in ('pair_id', 'scenario_family_id'):
        partitions = [{row[field] for row in rows} for rows in records.values()]
        if sum(map(len, partitions)) != len(set.union(*partitions)):
            raise ValueError(f'Overlap across partitions by {field}')

    model = LTRService.load_model(MODEL, fail_fast=True)
    if LTRService.get_feature_schema() != FEATURE_SCHEMA_V2:
        raise ValueError('Expected a V2 model')
    history = json.loads(model.get_metadata()['training'])['metrics']
    best = history['best_iteration']
    history_score = history['test_metrics_history'][best][0][METRIC]
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_commit=source_commit, input_sha256=input_hashes, columns=FEATURE_COLUMNS_V2, trees=model.tree_count_,
                  model_params=model.get_all_params(), best_iteration_zero_based=best,
                  historical_validation_ndcg=history_score, splits={})
    if args.source_ref:
        report['family_overlap_with_current'] = {}
        for current_split in ('val', 'test'):
            current_rows = read_pairs(DATA / paths[current_split].name)
            for old_split in ('train', 'val'):
                old_families = {row['scenario_family_id'] for row in records[old_split]}
                overlapping = [row for row in current_rows if row['scenario_family_id'] in old_families]
                report['family_overlap_with_current'][f'{old_split}_to_{current_split}'] = {
                    'families': sorted({row['scenario_family_id'] for row in overlapping}),
                    'current_pair_count': len(overlapping),
                }
    for split, path in paths.items():
        by_id = {row['pair_id']: row for row in records[split]}
        df = load_and_flatten_jsonl_sources([path], schema_version=FEATURE_SCHEMA_V2)
        if len(df) != 2 * len(records[split]) or set(df['group_id']) != by_id.keys():
            raise ValueError(f'Pair IDs or counts changed during loading: {split}')
        pool = create_catboost_pool(df, schema_version=FEATURE_SCHEMA_V2)
        expected = df['target'].to_numpy().reshape(-1, 2)
        pair_ids = df['group_id'].to_numpy().reshape(-1, 2)[:, 0]
        original_targets = np.array([[by_id[i]['winner'] == 'A', by_id[i]['winner'] == 'B'] for i in pair_ids])
        if not np.array_equal(expected, original_targets):
            raise ValueError('Labels or A/B order changed during loading')
        rankers = {'price': -df['price'].to_numpy().reshape(-1, 2),
                   'heuristic': np.asarray([LTRService._surrogate_score(row) for row in df.to_dict('records')]).reshape(-1, 2),
                   'catboost': model.predict(pool, thread_count=2).reshape(-1, 2)}
        families = np.array([by_id[i]['scenario_family_id'] for i in pair_ids])
        tradeoffs = np.array([by_id[i]['tradeoff_family'] for i in pair_ids])
        scopes = {'all': np.ones(len(pair_ids), dtype=bool)}
        scopes.update({f'family/{f}': families == f for f in sorted(set(families))})
        scopes.update({f'tradeoff/{t}': tradeoffs == t for t in sorted(set(tradeoffs))})
        report['splits'][split] = {name: {mode: metrics(expected[mask], values[mask])
                                         for mode, values in rankers.items()} for name, mask in scopes.items()}
        rows = df.to_dict('records')
        with (output / f'{split}_processed_pairs.jsonl').open('w', encoding='utf-8') as file:
            for i, pair_id in enumerate(pair_ids):
                source = by_id[pair_id]
                item = dict(pair_id=int(pair_id), family=source['scenario_family_id'],
                            tradeoff=source['tradeoff_family'], winner=source['winner'],
                            tour_a=rows[2*i], tour_b=rows[2*i+1],
                            scores={mode: values[i].tolist() for mode, values in rankers.items()})
                file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        if split == 'val':
            native = model.eval_metrics(pool, [METRIC], thread_count=2)[METRIC][-1]
            report['current_validation_native_ndcg'] = native
            report['validation_history_matches'] = bool(np.isclose(native, history_score, atol=1e-12, rtol=0))
        if split == 'train':
            historical_loss = history['learn_metrics_history'][best]['PairLogit']
            native_loss = model.eval_metrics(pool, ['PairLogit'], thread_count=2)['PairLogit'][-1]
            report['historical_train_pairlogit'] = historical_loss
            report['evaluated_train_pairlogit'] = native_loss
            report['train_history_matches'] = bool(np.isclose(native_loss, historical_loss, atol=1e-12, rtol=0))
        print(split, report['splits'][split]['all'])
    if input_hashes != {path.name: sha256(path) for path in [MODEL, *paths.values()]}:
        raise ValueError('Evaluation inputs changed during the run')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Historical/current native validation NDCG:', history_score, report['current_validation_native_ndcg'])


if __name__ == '__main__':
    main()
