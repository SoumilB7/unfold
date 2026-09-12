"""Read-only checks of existing saved artifacts. No project/model/test imports."""
from pathlib import Path
from collections import Counter
import copy,gzip,hashlib,html,json,re
ROOT=Path('/private/tmp/unfold-s8-typed-phase-7168440')
def read(p):
 b=Path(p).read_bytes();return json.loads(gzip.decompress(b) if str(p).endswith('.gz') else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def walk(x):
 if isinstance(x,dict):
  if 'id' in x:yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
paths=sorted(ROOT.glob('*/result.json'));assert len(paths)==11
rows=[]
for path in paths:
 case=path.parent;r=read(path);assert r['status']=='PASS' and not any(r['checks'].values())
 base=Path(r['base_case']);fc=Path(r['fact_case']);pd=Path(r['summary_proof_dir']);pins=read(case/'input-pins.json')
 assert r['commit']=='7168440c81a3e79b1e220437e2f9fa77e25bc296'
 for area,folder in [('base',base),('facts',fc),('proofs',pd)]:
  assert all(sha(folder/n)==v for n,v in pins[area].items()),(case.name,area)
 assert read(case/'source-before.json')==read(case/'source-after.json')
 pr=read(pd/'result.json'); proofs=read(pd/'proofs.json'); facts=read(fc/'facts.json')
 assert all(pr[n+'_sha256']==pins['facts'][n+'.json'] for n in ('input','facts','inventory'))
 assert pr['three_fact_values_exact'] and pr['no_model_or_execution']
 shape=proofs['root.denoiser.constructed_parameter_shapes']['value']; stages=proofs['root.denoiser.constructed_stage_relations']['value']; population=proofs['root.denoiser.constructed_modules']['value']
 assert all(v['value']==facts[k]['value'] for k,v in proofs.items())
 assert set(shape['by_module'])==set(population)
 stagepaths=sum((stages[k] for k in ('producer_stages','intermediate_stages','consumer_stages')),[])
 assert set(stagepaths)<=set(population)
 raw=read(case/'ir.json'); old=read(base/'ir.json'); delta=read(case/'ir-deltas.json')
 summary=raw['construction_summary']; assert summary==read(pd/'summary.json')==pr['summary']
 assert (summary['parameter_count'],summary['parameterized_module_count'],summary['stage_count'])==(shape['total'],shape['parameterized_modules'],len(stagepaths))
 assert all(type(summary[k]) is int for k in ('parameter_count','parameterized_module_count','stage_count'))
 archive=read(pd/'archive-content-check.json');assert archive['proof_result_sha256']==sha(pd/'result.json')
 archived=set()
 for item in archive['archived_source_bytes_verified']:
  p=Path(item['artifact']);assert sha(p)==item['stored_sha256']
  assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==item['content_sha256'];archived.add(item['content_sha256'])
 assert all(n['sha256'] in archived for n in read(pd/'indexed-sources.json') if n['archival_required'])
 reverse=copy.deepcopy(raw);reverse.pop('construction_summary');reverse.pop('component_entry',None)
 if delta['removed_legacy_component_scope'] is not None:
  reverse['extras']['render']['component_scope']=delta['removed_legacy_component_scope']
  reverse['extras']['render']['component_input_ids']=delta['removed_legacy_component_input_ids']
 byid={}
 for item in delta['block_fields']:byid.setdefault(item['id'],[]).append(item)
 for b in walk(reverse):
  for d in byid.get(b['id'],[]):
   if d['field']=='detail.fact_display_lines':
    b['detail'].pop('fact_display_lines')
    if not b['detail']:b.pop('detail')
   else:b[d['field']]=d['before']
 assert reverse==old,(case.name,'complete reverse IR')
 cap=read(case/'render-input.json');oldcap=read(base/'render-input.json')
 assert cap['parameters']==oldcap['parameters'] and cap['mount_id']==oldcap['mount_id'] and cap['warnings']==oldcap['warnings']
 page=(case/'page.html').read_text();assert sha(case/'page.html')==r['html_sha256']
 expected=(base/'page.html').read_text()
 for d in delta['block_fields']:
  if d['field']!='title':continue
  prefix='data-card-id="'+d['id']+'" data-card-size="compact"><div class="uf-card-title">'
  oldtext=prefix+html.escape(d['before'])+'</div>';newtext=prefix+html.escape(d['after'])+'</div>'
  assert oldtext in expected;expected=expected.replace(oldtext,newtext)
 if raw.get('component_entry'):
  for cls,oldtext,key in [('uf-section-label','SAMPLING LOOP','title'),('uf-section-sub','Denoiser applied iteratively · click it to open its architecture','subtitle')]:
   src='<span class="'+cls+'">'+oldtext+'</span>';dst='<span class="'+cls+'">'+html.escape(raw['component_entry'][key])+'</span>'
   assert expected.count(src)==1;expected=expected.replace(src,dst)
 assert expected==page,(case.name,'actual HTML')
 oldevents=read(base/'render-events.json.gz');events=read(case/'render-events.json.gz');ed=read(case/'event-deltas.json')
 oc=Counter(map(canon,oldevents));nc=Counter(map(canon,events));assert not oc-nc
 assert Counter(map(canon,ed['added']))==nc-oc and not ed['removed']
 blocks={b['id']:b for b in walk(raw)}
 headers=list(re.finditer(r'<div\b[^>]*data-card-id="([^"]+)"[^>]*>',page));cards={}
 for i,h in enumerate(headers):cards.setdefault(h[1],[]).append(page[h.start():headers[i+1].start() if i+1<len(headers) else len(page)])
 counts=Counter()
 for event in ed['added']:
  bid=event['block_path'][-1];b=blocks[bid];keys=set(event['facts_projected']);counts[event['view']]+=1
  assert keys<=set(b['source_fact_keys']) and keys and bid in cards
  if event['view']=='card_fact_lines':
   assert set(event['node_ids'])=={bid}
   for key in keys:
    lines=b['detail']['fact_display_lines'][key];assert lines
    assert any(all('<span class="uf-fact">'+html.escape(line,quote=False)+'</span>' in card for line in lines) for card in cards[bid])
  else:
   assert event['view']=='runtime_ffn_fact' and keys=={'root.denoiser.ffn_mechanisms'}
   ids={c['id'] for c in b['children'] if c.get('role')=='operation'};assert ids and ids==set(event['node_ids'])
   def nodeids(card):
    return {m[1] for g in re.findall(r'<g\b[^>]*>',card) if re.search(r'class="[^"]*\buf-node\b',g) for m in [re.search(r'data-id="([^"]+)"',g)] if m}
   assert any(ids<=nodeids(card) for card in cards[bid]),(case.name,bid)
 assert r['svg_deltas']==[]
 rows.append({'case':case.name,'pass':True,'html_sha256':r['html_sha256'],'parameter_count':summary['parameter_count'],'stage_count':summary['stage_count'],'added_receipts':dict(counts),'archived_sources_checked':len(archived),'component_entry':bool(raw.get('component_entry')),'result_sha256':sha(path)})
assert len({r['html_sha256'] for r in rows if r['case'] in ('sdxl-ordinary','sdxl-rewrite','sdxl-unchanged')})==1
print(json.dumps({'cases':rows,'all_pass':True,'no_model_or_pytest':True},indent=2,sort_keys=True))
