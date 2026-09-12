from pathlib import Path
import sys,json,gzip,dataclasses,hashlib,time,traceback,html,gc
sys.path.insert(0,str(Path.cwd()))
from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.diagram import Diagram
root=Path.cwd();out=Path('/private/tmp/unfold-s9a-twentyfourth/census');out.mkdir(exist_ok=False)
pages_root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design')
old_path=Path('/private/tmp/unfold-s9a-independent-discovery/current-unstamped-census.json')
old=json.loads(old_path.read_text());old_by_slug={Path(r['file']).name.removesuffix('.json.gz'):r for r in old['models']}
models=json.loads((root/'verification/s7/matrix.json').read_text())['models'];priority=['llama-7b','gemma-2-9b','bloom','musicgen-small','t5-small']
models=sorted(models,key=lambda r:(priority.index(r['slug']) if r['slug'] in priority else len(priority),r['slug']))
results=[];families={}
for target in models:
 slug=target['slug'];dest=out/slug;dest.mkdir();start=time.monotonic();stage='parse'
 try:
  payload=json.loads((root/target['input']).read_text());cfg=payload.get('config',payload)
  baseline=json.loads(gzip.decompress((root/'verification/s7/models'/f'{slug}.json.gz').read_bytes()))
  context=ParseContext.build(cfg);bundle=context.source_bundle
  arch=dict(bundle.component_architectures);arch['root']=baseline['inventory_provenance']['resolved_class']['qualname']
  context.source_bundle=dataclasses.replace(bundle,component_architectures=arch)
  ir=config_to_ir(cfg,parse_context=context);facts=context.facts.typed_records();qualified={}
  for key,fact in facts.items():
   if fact.claim_evidence is not None:
    validate_fact_claim(fact,fact.claim_evidence);qualified[key]=fact.claim_kind
  ledger=context.facts.to_dict();(dest/'facts.json').write_text(json.dumps(ledger,indent=2)+'\n')
  with gzip.open(dest/'ir.json.gz','wt') as f:json.dump(ir.to_dict(),f,sort_keys=True)
  counts=old_by_slug[slug]['by_fact_key'];remaining={k:n for k,n in counts.items() if k not in qualified}
  stage='render';diagram=Diagram(ir);page=diagram.to_html(standalone=True)
  with gzip.open(dest/'page.html.gz','wt') as f:f.write(page)
  render=ir.extras.get('render') or {};family='unet' if render.get('denoiser_view')=='unet_constructed' else render.get('family','transformer')
  if not family.replace('_','').replace('-','').isalnum():raise ValueError('invalid render family address')
  parent=pages_root/f'S9-{family}';parent.mkdir(exist_ok=True);page_dir=parent/'schema-twentyfourth';page_dir.mkdir(exist_ok=True)
  (page_dir/f'{slug}.html').write_text(page);families.setdefault(family,[]).append((slug,target['model']))
  listing=''.join(f'<li><a href="{html.escape(s)}.html">{html.escape(n)}</a></li>' for s,n in families[family])
  (page_dir/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>S9-A {html.escape(family)} witnesses</title><style>body{{font:16px/1.6 system-ui;max-width:950px;margin:48px auto;padding:0 24px}}</style><h1>S9-A {html.escape(family)} actual pages</h1><p>Twentyfourth isolated diagnostic candidate. No output is blessed. Browser visual review is pending. No design choice is proposed; questions begin at Q9 when needed.</p><ul>{listing}</ul></html>')
  entry=parent/'index.html';link='<p><a href="schema-twentyfourth/index.html">S9-A twentyfourth-candidate actual pages (unblessed diagnostic)</a></p>'
  if not entry.exists():entry.write_text((page_dir/'index.html').read_text().replace('href="','href="schema-twentyfourth/'))
  elif 'schema-twentyfourth/index.html' not in entry.read_text():
   text=entry.read_text();entry.write_text(text.replace('</body>',link+'</body>') if '</body>' in text else text+link)
  row={'slug':slug,'status':'PASS','seconds':time.monotonic()-start,'fact_count':len(facts),'qualified_fact_count':len(qualified),'previous_unstamped_findings':sum(counts.values()),'previous_citations_now_qualified':sum(n for k,n in counts.items() if k in qualified),'remaining_prior_citations':remaining,'qualified_kinds':qualified,'declarations':{key:{'claim_kind':fact.claim_kind,'claim_readers':list(fact.claim_readers),'status':fact.status} for key,fact in facts.items()},'page_bytes':len(page.encode()),'page_sha256':hashlib.sha256(page.encode()).hexdigest(),'page_family':family}
 except Exception as exc:
  (dest/'failure.txt').write_text(traceback.format_exc());row={'slug':slug,'status':'FAIL','stage':stage,'seconds':time.monotonic()-start,'error':str(exc)}
 results.append(row);(out/'result.json').write_text(json.dumps({'scope':'diagnostic fact census against fixed prior cited-key counts; not a regenerated reconciliation matrix','prior_census_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'models':results},indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ('qualified_kinds','remaining_prior_citations','declarations')}),flush=True);gc.collect()
if any(r['status']=='FAIL' for r in results):raise SystemExit(1)
