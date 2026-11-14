"""Read-only diagnostic experiments for review/f572e10. No training or label changes."""
import asyncio, collections, copy, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.scripts.run_csai_benchmark import candidate_pool, rank, retrieve, offer_is_success
from app.scripts.benchmark_routes import build_routes, assess, select_return
from app.services.ltr_service import LTRService
from app.services.travel_preferences import merge_preferences, matches_preference, prioritize_preference, select_offers
from app.services.vector_service import generate_text_embedding

OUT=ROOT/'experiment_outputs/review_f572e10'
OUT.mkdir(parents=True, exist_ok=True)
DATA=ROOT/'backend/app/data'
def read(name): return json.loads((DATA/name).read_text(encoding='utf8'))
def dump(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

async def main():
 catalog=read('destinations.json')
 routes=build_routes(catalog)
 route_index={r.key:r for rs in routes.values() for r in rs}
 labeled=read('csai_tradeoffs_labeled_v2.json')['labeled_pairs']
 invalid=[]; price_mismatch=[]; missing=[]
 for pair in labeled:
  req={**pair['request_slots'],'days':pair['request_slots']['nights']}
  for side in ['tour_a','tour_b']:
   card=pair[side]
   route=route_index.get(tuple(card['route_key']))
   if route is None: missing.append([pair['pair_id'],side]);continue
   log,con,price=assess(route,req,select_return(route))
   if log or con: invalid.append([pair['pair_id'],side,sorted(log|con)])
   if card['features_v2']['price']!=price:price_mismatch.append([pair['pair_id'],side,card['features_v2']['price'],price])
 sets={s:[json.loads(l) for l in (DATA/f'csai_tradeoffs_v2_{s}.jsonl').read_text(encoding='utf8').splitlines()] for s in ['train','val','test']}
 pair_map={p['pair_id']:p for p in labeled}
 metadata=read('csai_tradeoffs_v2_split_meta.json')
 stats={'catalogue_revalidation':{'cards':len(labeled)*2,'missing_routes':missing,'inadmissible_cards':invalid,'feature_price_mismatches':price_mismatch},
 'splits':{s:{'pairs':len(rs),'families':len({p['scenario_family_id'] for p in rs}),'hash_matches_manifest':hashlib.sha256((DATA/f'csai_tradeoffs_v2_{s}.jsonl').read_bytes()).hexdigest()==metadata[f'{s}_sha256']} for s,rs in sets.items()}}
 dump('data_audit.json',stats)
 print('DATA', {k:len(v) if isinstance(v,list) else v for k,v in stats['catalogue_revalidation'].items()}, flush=True)
 model=LTRService.load_model(DATA/'catboost_travel_ranker_v2.cbm',fail_fast=True)
 embeddings={d['id']:generate_text_embedding(d['vibe_description']) for d in catalog['destinations']}
 # Comparator evaluates preferences against all feasible retrieved reference destinations;
 # includes qualitative preferences. This is a diagnostic alternative to the current protocol.
 def intended_success(c,q,pool):
  r=c['route']
  if r.destination not in q['expected_destinations']:return False
  log,con,_=assess(r,q,c['back'])
  if log or con:return False
  pref=q.get('soft_preference')
  if not pref:return True
  acceptable=[p for p in pool if p['route'].destination in q['expected_destinations']]
  if not acceptable:return True
  cheap=min(acceptable,key=lambda p:(p['features']['price'],p['route'].key))
  cap=pref['max_extra_rub']
  def meets(p):return matches_preference(p,pref['attribute'],cheap) and (cap is None or p['features']['price']<=cheap['features']['price']+cap)
  return meets(c) if any(meets(p) for p in acceptable) else True
 allrows=[]
 for filename in ['csai_200_benchmark_queries.json','csai_top3_holdout.json']:
  queries=read(filename)
  for qi,q in enumerate(queries):
   request=merge_preferences(q,q['query'])
   ids=await retrieve(q['query'],catalog['destinations'],embeddings,None,True)
   paths=[r for dest in ids for r in routes[dest]]
   pool,_=candidate_pool(paths,request,True)
   ref={**q,**request}
   for mode in ['price','heuristic','catboost']:
    raw=rank(pool,mode,model)
    ordered=prioritize_preference(raw,request.get('soft_preference'))
    top=select_offers(ordered,diversify=True)
    current=lambda c:offer_is_success(c,ref,pool)
    intended=lambda c:intended_success(c,ref,pool)
    allrows.append({'dataset':filename,'id':q['id'],'mode':mode,'pool_count':len(pool),'hit2':bool(set(ids)&set(q['expected_destinations'])),
      'pref':request.get('soft_preference'),'pure1':bool(raw and current(raw[0])),'policy1':bool(top and current(top[0])),
      'score3':any(current(c) for c in ordered[:3]),'diverse3':any(current(c) for c in top),
      'alternative1':bool(top and intended(top[0])),'alternative3':any(intended(c) for c in top),
      'top_changed':bool(raw and raw[0] is not ordered[0]),
      'top_route':list(top[0]['route'].key) if top else None})
   if qi%25==0:print(filename,qi,flush=True)
 dump('catalogue_query_results.json',allrows)
 summary=[]
 for dataset in {r['dataset'] for r in allrows}:
  for mode in ['price','heuristic','catboost']:
   sub=[r for r in allrows if r['dataset']==dataset and r['mode']==mode]
   summary.append({'dataset':dataset,'mode':mode,'n':len(sub),**{key:sum(r[key] for r in sub) for key in ['pure1','policy1','score3','diverse3','alternative1','alternative3','top_changed']}})
 dump('catalogue_summary.json',summary)
 print(json.dumps(summary,indent=2),flush=True)
asyncio.run(main())
