"""Saved-data closure and own-surface receipt checks; stdlib only."""
from pathlib import Path
import json,gzip,hashlib,collections
HERE=Path(__file__).resolve().parent;CAP=Path('/private/tmp/unfold-s9a-thirtysecond/current-pages/capture');REPO=HERE.parents[6];PAGES=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
summary=json.loads((HERE/'summary.json').read_bytes());capture=json.loads((CAP/'result.json').read_bytes());lane=json.loads((CAP.parent/'result.json').read_bytes());assert lane['passed'] and lane['pins_equal'] and lane['before']==lane['after'] and lane['artifacts_before']==lane['artifacts_after'];assert capture['captured_count']==39 and capture['failed_count']==0
native=collections.Counter();status=collections.Counter();styles=True;receipts=[];cards=[];soft=[];legacy=[];banner=[];mutations=[];htmltot=collections.Counter();pages=[];verified=0
for row in summary['models']:
 slug=row['slug'];d=json.loads((HERE/(slug+'.json')).read_bytes());folder=CAP/slug
 assert not d['native_deltas'] and d['native_before']==d['native_after']
 assert all(not p['findings'] and not p['class3_records'] for p in d['phases'].values())
 assert not d['html']['findings'] and not d['html']['missing_unknown_paths'] and not d['html']['unexpected_unknown_paths'] and d['html']['four_styles_present_and_distinct']
 assert not d['live_capture_census']['findings']
 for p in d['inputs']:assert sha(folder/p['path'])==p['sha256'];verified+=1
 entry=next(x for x in capture['models'] if x['slug']==slug)
 for name,h in entry['artifact_sha256'].items():assert sha(folder/name)==h
 page=PAGES/entry['published_page'];raw=gzip.decompress((folder/'page.html.gz').read_bytes());assert page.read_bytes()==raw;pages.append({'slug':slug,'page':str(page),'sha256':sha(page)})
 for key in ('facts','qualified','without_proof'):native[key]+=d['native_after'][key]
 status.update(d['native_after']['status'])
 for key in ('html_bytes','gzip_bytes','serialized_element_count','inline_detail_cards','deferred_template_expanded_element_count','deferred_template_expanded_detail_cards','banner_scalar_markers'):htmltot[key]+=d['html'][key]
 context=json.loads(gzip.decompress((folder/'render-context-local.json.gz').read_bytes()))['context'];br=[]
 for event in context['events']:
  for receipt in event['receipts']:
   r={'slug':slug,'view':event['view'],'block_path':event['block_path'],'event_node_ids':event['node_ids']['members'],'receipt':receipt}
   if receipt['surface']=='opgraph':assert set(receipt['node_ids'])<=set(event['node_ids']['members']),r
   elif receipt['surface']=='card':
    assert receipt['node_ids']==[receipt['structural_target']] and event['block_path'][-1:]==receipt['node_ids'],r
    cards.append(r)
   elif receipt['surface']=='html':
    assert set(receipt['node_ids'])<=set(event['node_ids']['members']),r
    if event['view']=='stats_banner':
     assert receipt['fact_id']=='model.hidden_size',r
     br.append(r)
   else:raise AssertionError(r)
   if receipt['fact_id']=='decoder.attention.logit_softcap':soft.append(r)
   receipts.append(r)
 if br:banner.append({'slug':slug,'count':len(br)})
 elif d['html']['banner_scalar_markers']:legacy.append(slug)
 if not all(d['raw_before_after_render_equal'].values()):mutations.append(slug)
assert len(receipts)==197 and len(cards)==6 and len(soft)==2 and len(banner)==24 and len(legacy)==14
write(HERE/'receipt-surfaces.json',{'all_receipts':receipts,'six_card_rows':cards,'softcap_rows':soft,'card_contract':'Card identities are the actual enclosing projector block, not graph node identities. receipts.py defines node_ids within the named surface; declared_ops.py emits surface=card for block rid. No opgraph subset rule is applied across surfaces.'})
old=Path('verification/receipts/S9-A-schema-candidate-fff99e22/new-resume/reviews/current30-saved-audit-r2/owner-debt-addendum.json');debt=json.loads(old.read_bytes());write(HERE/'carried-debt.json',{'prior_addendum':str(old),'prior_sha256':sha(old),'legacy_banner_models':legacy,'markers_not_receipts':True,'prior_debt_record':debt,'same_pre_post_render_mutation_models':mutations,'nested_limit':'147 separately audited nested deltas encode105 consumption obligations on80 child targets in15 models. This root-native audit establishes no child proof change or child ledger validity; actual child ledger observer is separate.'})
write(HERE/'published-pages.json',pages)
write(HERE/'verdict.json',{'status':'PASS_FINITE_SAVED_FLEET_SCOPE','models_inspected':39,'captured_files_rehashed':verified,'native_totals':dict(native),'native_statuses':dict(status),'native_value_status_kind_proof_delta_keys':0,'unknown_and_html_totals':summary['totals'],'html_detail_totals':dict(htmltot),'four_styles_distinct_all39':True,'all197_receipts_valid_on_declared_surface':True,'receipt_surface_counts':dict(collections.Counter(r['receipt']['surface'] for r in receipts)),'card_receipts':len(cards),'fact_backed_banner_receipts':sum(x['count'] for x in banner),'legacy_scalar_markers_without_receipts':len(legacy),'actual_unresolved_softcap_fleet_cases':sum(x['actual_unresolved_softcap'] for x in summary['models']),'missing_softcap_branch_scope':'Fleet exercises zero missing-operand cases; separate six-case saved diagnostic establishes actual missing HTML limits.','scope':'Standard-library saved-data audit only; no model/tests/browser/product imports, portable proof replay, occurrence proof, output normalization or blessing. All published page bytes compared exactly. Root-native facts do not stand in for nested child ledgers.','source_manifest_sha256':summary['source_manifest_sha256'],'capture_result_sha256':sha(CAP/'result.json'),'lane_result_sha256':sha(CAP.parent/'result.json')})
write(HERE/'manifest.json',{'files':[{'path':p.relative_to(HERE).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file()]})
print(json.dumps({'verdict_sha256':sha(HERE/'verdict.json'),'manifest_sha256':sha(HERE/'manifest.json'),'native':dict(native),'html':dict(htmltot),'published_pages':len(pages),'legacy_models':legacy,'mutation_models':mutations},indent=2))
