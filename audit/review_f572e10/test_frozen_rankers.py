"""Evaluate the existing V2 checkpoint and heuristic on the same recorded pairs."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.scripts.benchmark_routes import build_routes
from app.scripts.run_csai_benchmark import candidate_pool, scores
from app.services.ltr_service import LTRService

DATA = ROOT / 'backend/app/data'
MODEL = DATA / 'catboost_travel_ranker_v2.cbm'
MODES = ('price', 'heuristic', 'catboost')
VIEWS = ('stored_features', 'catalogue_features')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(a, b, price_a, price_b):
    if not all(map(math.isfinite, (a, b, price_a, price_b))):
        raise ValueError('Non-finite price or score')
    if not math.isclose(a, b, rel_tol=0, abs_tol=1e-6):
        return ('A' if a > b else 'B'), False
    if price_a == price_b:
        return 'TIE', True
    return ('A' if price_a < price_b else 'B'), True


def catalogue_pair(source, routes, cached_pools):
    # The pair corpus supplies one destination and known hard slots per scenario.
    route_a, route_b = (tuple(source[side]['route_key']) for side in ('tour_a', 'tour_b'))
    if route_a[0] != route_b[0]:
        raise ValueError('A same-destination pair was expected')
    slots = source['request_slots']
    request = {**slots, 'days': slots['nights'], 'archetype': source['archetype'],
               'soft_preference': source.get('soft_preference')}
    key = (route_a[0], json.dumps(request, sort_keys=True))
    if key not in cached_pools:
        pool, _ = candidate_pool(routes[route_a[0]], request, strict=True)
        cached_pools[key] = {candidate['route'].key: candidate for candidate in pool}
    pool = cached_pools[key]
    pair = [pool[route_a]['features'], pool[route_b]['features']]
    for features, side in zip(pair, ('tour_a', 'tour_b')):
        if features['price'] != source[side]['price']:
            raise ValueError(f"Catalogue price differs from the judged price for pair {source['pair_id']}")
    return pair, len(pool)


def aggregate(rows):
    return {mode: {'credit': sum(row[mode]['credit'] for row in rows), 'n': len(rows),
                   'score_ties': sum(row[mode]['score_tie'] for row in rows),
                   'unresolved_ties': sum(row[mode]['winner'] == 'TIE' for row in rows)}
            for mode in MODES}


def compare(rows, baseline):
    deltas = [row['catboost']['credit'] - row[baseline]['credit'] for row in rows]
    return dict(C4_better=sum(d > 0 for d in deltas), baseline_better=sum(d < 0 for d in deltas),
                equal=sum(d == 0 for d in deltas), delta_pp=100 * sum(deltas) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    catalogue = DATA / 'destinations.json'
    labels_path = DATA / 'csai_tradeoffs_labeled_v2.json'
    split_paths = {s: DATA / f'csai_tradeoffs_v2_{s}.jsonl' for s in ('train', 'val', 'test')}
    input_paths = [catalogue, labels_path, MODEL, *split_paths.values()]
    input_hashes = {path.name: digest(path) for path in input_paths}
    labels = {row['pair_id']: row for row in read_json(labels_path)['labeled_pairs']}
    routes = build_routes(read_json(catalogue))
    model = LTRService.load_model(MODEL, fail_fast=True)
    if LTRService.get_feature_schema() != 'v2':
        raise ValueError('The selected checkpoint must use feature schema V2')
    cached_pools = {}
    pair_ids = set()
    summaries = {}
    all_rows = []
    for split, path in split_paths.items():
        pairs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows = {view: [] for view in VIEWS}
        features_by_view = {view: [] for view in VIEWS}
        for pair in pairs:
            pair_id = pair['pair_id']
            source = labels[pair_id]
            winner = pair['winner']
            if (pair_id in pair_ids or winner not in ('A', 'B') or winner != source['label']['winner']
                    or pair['scenario_family_id'] != source['scenario_family_id']):
                raise ValueError(f'Duplicate or inconsistent pair {pair_id}')
            pair_ids.add(pair_id)
            runtime, pool_size = catalogue_pair(source, routes, cached_pools)
            stored = [pair['tour_a'], pair['tour_b']]
            prices = [source[side]['price'] for side in ('tour_a', 'tour_b')]
            for view, features in zip(VIEWS, (stored, runtime)):
                features_by_view[view].extend({'features': feature} for feature in features)
                rows[view].append(dict(split=split, view=view, pair_id=pair_id,
                                       family=pair['scenario_family_id'], tradeoff=pair['tradeoff_family'],
                                       expected=winner, pool_size=pool_size, judge_prices=prices,
                                       input_prices=[feature['price'] for feature in features],
                                       price_delta=[feature['price_delta_vs_cheapest'] for feature in features]))

        summaries[split] = {}
        for view, view_rows in rows.items():
            for mode in MODES:
                predicted = scores(features_by_view[view], mode, model)
                for i, row in enumerate(view_rows):
                    score_a, score_b = map(float, predicted[2*i:2*i+2])
                    price_a, price_b = row['input_prices']
                    winner, tie = decide(score_a, score_b, price_a, price_b)
                    if decide(score_b, score_a, price_b, price_a)[0] != {'A': 'B', 'B': 'A', 'TIE': 'TIE'}[winner]:
                        raise AssertionError('Asymmetric tie handling')
                    row[mode] = dict(winner=winner, score_tie=tie, scores=[score_a, score_b],
                                     credit=0.5 if winner == 'TIE' else int(winner == row['expected']))
            groups = defaultdict(list)
            for row in view_rows:
                groups[row['family']].append(row)
            summaries[split][view] = dict(
                total=aggregate(view_rows),
                by_family={name: aggregate(group) for name, group in sorted(groups.items())},
                comparisons={baseline: compare(view_rows, baseline) for baseline in ('price', 'heuristic')},
                label_price_relation=dict(Counter(
                    'equal' if row['judge_prices'][0] == row['judge_prices'][1]
                    else 'cheaper' if row['expected'] == ('A' if row['judge_prices'][0] < row['judge_prices'][1] else 'B')
                    else 'dearer' for row in view_rows)),
            )
            all_rows.extend(view_rows)

    if pair_ids != labels.keys() or input_hashes != {path.name: digest(path) for path in input_paths}:
        raise ValueError('Incomplete split coverage or modified inputs')
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  script_sha256=digest(Path(__file__)), input_sha256=input_hashes,
                  model_trees=model.tree_count_, partitions=summaries)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'pairs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in all_rows), encoding='utf-8')
    for split, views in summaries.items():
        for view, summary in views.items():
            print(split, view, {mode: (row['credit'], row['n']) for mode, row in summary['total'].items()})


if __name__ == '__main__':
    main()
