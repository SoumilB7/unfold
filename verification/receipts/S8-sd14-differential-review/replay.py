from pathlib import Path
import json,re,hashlib,ast,collections,math,gzip
R=Path('/private/tmp/unfold-s8-render-phase-c38-v2');P=R/'sd-v1-4';O=Path(__file__).resolve().parent
report=json.loads((P/'differential-report.json').read_text());pin=json.loads((R/'phase-pin.json').read_text());assert pin['production_commit']=='c38b008af8153302d85bbaf2a8e3593841d87e1a'
pins={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [R/'phase-pin.json',R/'scripts/report_s8_demonstration.py',P/'differential-report.json',P/'html.diff']+[(P/n/f)for n in ('ordinary','legacy')for f in ('ir.json','facts.json','qualified-facts.json','page.html','observation.json','result.json')]}
cases={n:{f:json.loads((P/n/(f+'.json')).read_text()) for f in ('ir','facts','qualified-facts','observation','result')}for n in ('ordinary','legacy')};pages={n:(P/n/'page.html').read_text()for n in cases}
for n,c in cases.items():assert hashlib.sha256(pages[n].encode()).hexdigest()==c['result']['html_sha256']
module=ast.parse((R/'scripts/report_s8_demonstration.py').read_text());ns={};exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='changes'],type_ignores=[]),'pinned-changes','exec'),ns)
raw=list(ns['changes'](cases['legacy']['ir'],cases['ordinary']['ir']));stripped=[{k:v for k,v in r.items()if k in ('path','operation','before','after')}for r in report['ir_deltas']];assert raw==stripped

def walk(x):
 if isinstance(x,dict):
  if 'id'in x and any(k in x for k in ('kind','view','children')):yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
blocks={n:{b['id']:b for b in walk(c['ir']['extras']['render'])}for n,c in cases.items()}
def vh(svg):
 s=re.sub(r'\b(id|data-id|data-card-id|data-target|href|xlink:href|aria-labelledby)="[^"]*"','',svg);return hashlib.md5(re.sub(r'url\(#[^)]*\)','url()',s).encode()).hexdigest()
svgrows={}
for n,page in pages.items():
 markers=list(re.finditer(r'data-card-id="([^"]+)"',page));intervals=[(m.start(),m.group(1))for m in markers];rows=[];cursor=0
 for i,m in enumerate(re.finditer(r'<svg\b.*?</svg>',page,re.S)):
  while cursor+1<len(intervals) and intervals[cursor+1][0]<m.start():cursor+=1
  cid=intervals[cursor][1]if intervals and intervals[cursor][0]<m.start()else '<architecture>'
  svg=m.group();rows.append({'ordinal':i,'card_id':cid,'visual_hash':vh(svg),'arrows':len(re.findall('marker-end=',svg)),'node_ids':re.findall(r'class="[^"]*uf-node[^\"]*" data-id="([^"]+)"',svg)})
 assert [r['visual_hash']for r in rows]==cases[n]['observation']['svg_visual_hashes'];svgrows[n]=rows
oldc=collections.Counter(r['visual_hash']for r in svgrows['legacy']);newc=collections.Counter(r['visual_hash']for r in svgrows['ordinary']);assert dict(oldc-newc)==report['svg_deltas']['removed'] and dict(newc-oldc)==report['svg_deltas']['added']
f=cases['ordinary']['facts'];q=cases['ordinary']['qualified-facts'];shape=f['root.denoiser.constructed_parameter_shapes']['value'];inv=json.loads((P/'ordinary/inventory.json').read_text());modules={m['path']:m for m in inv['modules']};direct={m['path']+'.'+p['name']:p['shape']for m in inv['modules']for p in m['parameters']};assert {k:v['shape']for k,v in shape['parameters'].items()}==direct;assert sum(math.prod(s)for s in direct.values())==shape['by_module']['']==859520964
canonical=[b for b in blocks['ordinary'].values()if not b.get('target') and 'source_instance_path'in b];shape_cards=0
for b in canonical:
 path=b['source_instance_path'];assert path in modules
 for fact in b.get('facts',[]):
  if fact.endswith(' parameters in subtree'):
   assert fact==f"{shape['by_module'][path]:,} parameters in subtree";shape_cards+=1
assert all(k in q and q[k].get('proof') for k in f if k.startswith('root.denoiser.'))
causes={
 'render':'Constructed occurrence/stage containment, qualified local mechanisms, and conditional source-port projection replace the old aggregate UNet drawing. Must assess individual flow/label removals; ancestor fact citations alone are insufficient.',
 'shapes':'Exact constructed parameter inventory replaces legacy aggregate banner/width metadata. All saved parameter shapes and canonical subtree quantity chips were cross-checked.',
 'proofs':'New qualified reader facts are recorded in the existing ledger; proof metadata addition alone does not approve a product claim.',
 'audit':'The new instance/reader path changes config-access obligations and visible unresolved findings; these findings are retained debt, not an accepted waiver.',
 'legacy_geometry':'Old hand-authored geometry fields are removed or replaced with canonical constructed/source route records. Per-field preservation/limitation mapping remains required.',
 'limited_metadata':'Legacy hidden_size1280/tie_word_embeddingsTrue become unknown, reflecting absent qualified global-width/tied-embedding claims for this variable-width UNet.',
 'source_address':'Absolute root source path becomes module@content-hash address; source content identity is retained, not promoted to mechanism.',
 'other':'No bounded evidence-level cause assigned; remains a blocking review finding.'}
annotated=[]
for index,row in enumerate(report['ir_deltas']):
 p=row['path']
 if p.startswith('/extras/render/'):group='render'
 elif p.startswith('/extras/unet/parameter_shapes'):group='shapes'
 elif p.startswith('/extras/fact_provenance/'):group='proofs'
 elif p.startswith(('/extras/config_','/extras/ship_findings','/warnings')):group='audit'
 elif p.startswith(('/extras/unet/','/extras/diffusion')):group='legacy_geometry'
 elif p in ('/hidden_size','/tie_word_embeddings'):group='limited_metadata'
 elif p.startswith('/extras/source_provenance/'):group='source_address'
 else:group='other'
 annotated.append({'ordinal':index,'original_delta':row,'review_group':group,'group_cause':causes[group],'review_disposition':'cause_group_only_not_output_approval','blessed':False})
summary={'checkpoint':pin['production_commit'],'raw_delta_count':len(raw),'report_counts':report['counts'],'groups':dict(collections.Counter(r['review_group']for r in annotated)),'svg_counts':{n:len(r)for n,r in svgrows.items()},'svg_removed':sum((oldc-newc).values()),'svg_added':sum((newc-oldc).values()),'unchanged_svg_multiplicity':sum((oldc&newc).values()),'actual_cards':{n:len(c['observation']['page']['cards'])for n,c in cases.items()},'constructed_modules':len(modules),'shape_total':859520964,'canonical_quantity_chips_checked':shape_cards,'all_raw_rows_preserved':True,'artifact_pins':pins,'status':'review_findings_remain; no blanket re-proof acceptance'}
(O/'per-output-review.json.gz').write_bytes(gzip.compress((json.dumps(annotated,indent=2,sort_keys=True)+'\n').encode(),mtime=0));(O/'svg-inventory.json').write_text(json.dumps(svgrows,indent=2,sort_keys=True)+'\n');(O/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');assert pins=={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in pins};print(json.dumps({k:v for k,v in summary.items()if k!='artifact_pins'},indent=2))
