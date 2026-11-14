"""Frozen-data audit: diagnostic assertions, no training and no repository-data writes."""
import collections, copy, hashlib, importlib.metadata, json, math, platform, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scipy.stats import binomtest
from catboost import CatBoostRanker
from app.scripts.benchmark_csai_v2 import evaluate_models_on_test
from app.scripts.benchmark_routes import build_routes, assess, select_return
from app.scripts.run_csai_benchmark import candidate_pool, offer_is_success
from app.scripts.split_csai_tradeoffs import split_labeled_dataset
from app.services.ltr_service import LTRService
from app.services.travel_preferences import merge_preferences, matches_preference

OUT = ROOT / 'experiment_outputs/review_f572e10'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'backend/app/data'
def read(name): return json.loads((DATA/name).read_text(encoding='utf8'))
def dump(name, data): (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
splits = {s: [json.loads(l) for l in (DATA/f'csai_tradeoffs_v2_{s}.jsonl').read_text(encoding='utf8').splitlines()] for s in ('train','val','test')}
labeled = read('csai_tradeoffs_labeled_v2.json')
by_id = {p['pair_id']:p for p in labeled['labeled_pairs']}
models = []
for name in ('catboost_travel_ranker.cbm','catboost_travel_ranker_v2.cbm','catboost_travel_ranker_v2_ablation.cbm'):
    m = CatBoostRanker(); m.load_model(DATA/name); models.append(m)

# Only replace price-derived features with the price actually presented to the judge.
# This is sensitivity analysis of FROZEN checkpoints, not a valid corrected-training experiment.
corrected = copy.deepcopy(splits['test'])
for p in corrected:
    source = by_id[p['pair_id']]
    ref = min(source['tour_a']['price'],source['tour_b']['price'])
    for side in ('tour_a','tour_b'):
        f = p[side]; price = source[side]['price']; per_day = price/max(1,f['days'])
        f.update(price=price, log_price=math.log1p(price), log_price_per_day=math.log1p(per_day),
                 stars_per_10k=round(f['hotel_stars']/max(.1,per_day/10000),4),
                 price_delta_vs_cheapest=round(max(0,price-ref),2),
                 relative_surcharge_pct=round(max(0,price-ref)/max(1,ref),4))
result = evaluate_models_on_test(corrected,*models)
dump('price_correction_sensitivity.json',result)

original = json.loads((OUT/'pairs/benchmark_summary.json').read_text(encoding='utf8'))
def comparison(rows, left, right):
    lookup = {(r['pair_id'],r['config']):r['correct'] for r in rows}
    deltas = [(p['pair_id'],lookup[p['pair_id'],left]-lookup[p['pair_id'],right]) for p in splits['test']]
    wins = sum(d==1 for _,d in deltas); losses = sum(d==-1 for _,d in deltas)
    families = collections.defaultdict(list)
    for pid,d in deltas:families[by_id[pid]['scenario_family_id']].append(d)
    return {'wins_only':wins,'losses_only':losses,
            'naive_pair_binomial_p':binomtest(wins,wins+losses).pvalue if wins+losses else 1,
            'family_mean_difference':{f:sum(v)/len(v) for f,v in families.items()},
            'note':'Pairs share scenario families; naive pair p-value is not cluster-adjusted.'}

stats={'original_vs_price':comparison(original['detailed_predictions'],'C4_V2_Full_Context','Price_Baseline'),
       'price_corrected_vs_price':comparison(result['detailed_predictions'],'C4_V2_Full_Context','Price_Baseline'),
       'corrected_price_summary':result['summary'],
       'label_payload_keys':list(labeled),
       'label_fields':sorted({k for p in by_id.values() for k in p['label']})}
# Grouping and representation, including exact cross-split vector overlap.
stats['splits']={}
for name,rows in splits.items():
    stats['splits'][name]={'families':dict(collections.Counter(p['scenario_family_id'] for p in rows)),
                          'tradeoff_families':dict(collections.Counter(p['tradeoff_family'] for p in rows)),
                          'archetypes':dict(collections.Counter(p['archetype'] for p in rows))}
stats['overlap']={}
for a,b in [('train','val'),('train','test'),('val','test')]:
    route_sets=[{tuple(by_id[p['pair_id']][s]['route_key']) for p in splits[t] for s in ('tour_a','tour_b')} for t in (a,b)]
    feat_sets=[{json.dumps(p[s],sort_keys=True) for p in splits[t] for s in ('tour_a','tour_b')} for t in (a,b)]
    stats['overlap'][a+'_'+b]={'routes':len(route_sets[0]&route_sets[1]),'exact_feature_vectors':len(feat_sets[0]&feat_sets[1])}

checks=[]
def record(name, expected, actual, context=None):
    checks.append({'name':name,'passed':expected==actual,'expected':expected,'actual':actual,'context':context})
record('feature price matches judge card price for every card',0,sum(p[s]['price']!=p[s]['features_v2']['price'] for p in by_id.values() for s in ('tour_a','tour_b')))
meta=read('csai_tradeoffs_v2_split_meta.json')
record('all split SHA256 match manifest',True,all(hashlib.sha256((DATA/f'csai_tradeoffs_v2_{s}.jsonl').read_bytes()).hexdigest()==meta[s+'_sha256'] for s in splits))

# Improving time preference at fixed price must not lower a heuristic score.
base=copy.deepcopy(splits['test'][0]['tour_a'])
base.update(matches_time_pref=0,matches_direct_pref=0,relative_surcharge_pct=1.0)
improved={**base,'matches_time_pref':1}
record('heuristic monotonicity in matching time preference at fixed cost',True,LTRService._surrogate_score(improved)>=LTRService._surrogate_score(base),
       {'before':LTRService._surrogate_score(base),'after':LTRService._surrogate_score(improved)})

queries = {n:read(n) for n in ('csai_200_benchmark_queries.json','csai_top3_holdout.json')}
routes=build_routes(read('destinations.json'))
LTRService.load_model(DATA/'catboost_travel_ranker_v2.cbm',fail_fast=True)
preference_examples=[]; reference_examples=[]
for q in queries['csai_top3_holdout.json']:
    req=merge_preferences(q,q['query'])
    pool,_=candidate_pool([r for d in q['expected_destinations'] for r in routes[d]],req,True)
    pref=req.get('soft_preference')
    if pref and pref['max_extra_rub'] is None and pool:
        cheap=min(pool,key=lambda c:c['features']['price'])
        matches=[c for c in pool if matches_preference(c,pref['attribute'],cheap)]
        nonmatches=[c for c in pool if not matches_preference(c,pref['attribute'],cheap)]
        if matches and nonmatches:
            c=nonmatches[0]
            preference_examples.append({'query_id':q['id'],'preference':pref,'nonmatching_route':c['route'].key,
                'preferred_alternative':matches[0]['route'].key,'counted_success':offer_is_success(c,req,pool)})
    if pool:
        c=pool[0]; gold={**req,'budget':1}; predicted={**req,'budget':100000000}
        reference_examples.append({'query_id':q['id'],'gold_success':offer_is_success(c,gold,pool),
           'merged_prediction_success':offer_is_success(c,{**gold,**predicted},pool)})
record('qualitative preference evaluated when feasible preferred alternative exists',False,any(p['counted_success'] for p in preference_examples),preference_examples[:3])
record('predicted budget cannot relax evaluation reference',False,any(not p['gold_success'] and p['merged_prediction_success'] for p in reference_examples),reference_examples[:1])
stats['qualitative_preference_examples']=preference_examples
stats['query_sets']={n:{'n':len(qs),'unique_ids':len({q['id'] for q in qs}),'unique_texts':len({q['query'] for q in qs}),
   'preferences':dict(collections.Counter(str(merge_preferences(q,q['query']).get('soft_preference')) for q in qs))} for n,qs in queries.items()}
a,b=queries.values()
stats['query_text_overlap']=len({q['query'] for q in a}&{q['query'] for q in b})

# Different documented ratio arguments must change split sizes on this sufficiently large dataset.
size_results=[]
for tag,ratio in [('small',.1),('large',.4)]:
    folder=OUT/('split_probe_'+tag);folder.mkdir(exist_ok=True)
    res=split_labeled_dataset(DATA/'csai_tradeoffs_labeled_v2.json',folder,test_ratio=ratio,val_ratio=.1)
    size_results.append([res['train_pairs'],res['val_pairs'],res['test_pairs']])
record('test_ratio controls split sizes',True,size_results[0]!=size_results[1],size_results)
stats['invariants']=checks
dump('invariant_audit.json',stats)
print(json.dumps({'checks':checks,'price_corrected_summary':result['summary'],'original_vs_price':stats['original_vs_price']},ensure_ascii=False,indent=2))

# Remaining catalogue and representation probes; separate from normative assertions.
flat={r.key:r for rs in routes.values() for r in rs}
extra={'unique_pair_query_texts':len({p['query_text'] for p in by_id.values()}),
       'label_counts':dict(collections.Counter(p['label']['winner'] for p in by_id.values())),
       'judge_price_vs_catalogue_mismatches':sum(
           p[s]['price'] != assess(flat[tuple(p[s]['route_key'])],
               {**p['request_slots'],'days':p['request_slots']['nights']},
               select_return(flat[tuple(p[s]['route_key'])]))[2]
           for p in by_id.values() for s in ('tour_a','tour_b'))}
for name,qs in queries.items():
    solvable=[]
    for q in qs:
        if any(not a and not b for d in q['expected_destinations'] for r in routes[d]
               for a,b,_ in [assess(r,q,select_return(r,q.get('prefer_direct_flight')))]):
            solvable.append(q['id'])
    extra[name]={'n':len(qs),'hard_feasible_expected':len(solvable),
                 'unsolvable_expected':[q['id'] for q in qs if q['id'] not in solvable]}
f=next(iter(by_id.values()))['tour_a']['features_v2']
tie_pair=dict(pair_id=1,winner='A',archetype='family',tradeoff_family='probe',destination_id='probe',tour_a=f,tour_b=f)
extra['equal_candidate_tie_predictions']=evaluate_models_on_test([tie_pair],models[0],None,None)['detailed_predictions']
route=next(iter(flat.values()))
profile={'archetype':'family','soft_preference':{'attribute':'breakfast','max_extra_rub':0}}
kw=dict(hotel=route.hotel,destination_id=route.destination,profile=profile,nights=7,
        logistics={'flight':route.flight},breakdown={'mandatory':{'total_mandatory_rub':100000}},reference_price=50000)
a=LTRService.extract_candidate_features_v2(**kw)
kw['profile']={**profile,'soft_preference':{'attribute':'breakfast','max_extra_rub':100000}}
b=LTRService.extract_candidate_features_v2(**kw)
kw['profile']={**profile,'soft_preference':{'attribute':'stars','max_extra_rub':100000}}
c=LTRService.extract_candidate_features_v2(**kw)
extra.update(cap_changes_features=a!=b,breakfast_vs_stars_changes_features=b!=c)
dump('additional_probes.json',extra)
files = ['destinations.json','csai_200_benchmark_queries.json','csai_top3_holdout.json',
         'csai_tradeoffs_labeled_v2.json','csai_tradeoffs_v2_split_meta.json',
         *[f'csai_tradeoffs_v2_{s}.jsonl' for s in splits],
         'catboost_travel_ranker.cbm','catboost_travel_ranker_v2.cbm','catboost_travel_ranker_v2_ablation.cbm']
dump('audit_manifest.json', {
    'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'python':sys.version,'platform':platform.platform(),
    'packages':{p:importlib.metadata.version(p) for p in ['catboost','numpy','pandas','scipy','sentence-transformers','pytest']},
    'data_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in files},
    'mode':'frozen checkpoints; read-only diagnostics; no live Neo4j or PostgreSQL verification',
})
