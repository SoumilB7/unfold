from pathlib import Path
import dataclasses,gzip,hashlib,json,runpy,sys
root=Path.cwd();sys.path.insert(0,str(root))
from model_unfolder import config_to_ir
from model_unfolder.diagram import Diagram
from model_unfolder.evidence.context import ParseContext
out=Path('/private/tmp/unfold-s9a-thirtieth/softcap-diagnostic/capture');out.mkdir(parents=True,exist_ok=False)
fixture=root/'tests/test_attention_softcap.py';namespace=runpy.run_path(str(fixture))
fixture_hash=hashlib.sha256(fixture.read_bytes()).hexdigest()
def serial(v):
 if dataclasses.is_dataclass(v):return {f.name:serial(getattr(v,f.name)) for f in dataclasses.fields(v)}
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(serial(x) for x in v)
 return v
def save(p,v):p.write_text(json.dumps(serial(v),indent=2,sort_keys=True)+'\n')
rows=[]
for case in ['injected_missing','plain_missing','explicit_17','explicit_none','no_mechanism','gemma_default']:
 d=out/case;d.mkdir();cfg=json.loads((root/'tests/sable_test_corpus'/('gemma-2-2b-it.json' if case=='gemma_default' else 'llama-7b.json')).read_text())['config']
 if case=='gemma_default':
  cfg.pop('attn_logit_softcapping',None);ctx=ParseContext.build(cfg)
 else:
  _,bundle,_,_=namespace['_pipeline'](d,protocol='weights = weights + 0' if case=='no_mechanism' else namespace['_EXACT'])
  cfg.update(hidden=128,layers=2);cfg.pop('attention_cap',None)
  if case=='explicit_17':cfg['attention_cap']=17.0
  if case=='explicit_none':cfg['attention_cap']=None
  ctx=ParseContext(source_bundle=bundle,declared_decoderness='decoder_only_wrapper',class_defaults={'attention_cap':17.0} if case in {'injected_missing','no_mechanism'} else {})
 save(d/'input.json',cfg);save(d/'context-defaults-before.json',ctx.class_defaults)
 ir=config_to_ir(cfg,parse_context=ctx);save(d/'ir-before.json',ir.to_dict());save(d/'native-facts.json',ctx.facts.to_dict())
 binding=ctx.prepared_documents['root'];doc=binding.prepared
 save(d/'prepared-document.json',{'checkpoint':doc.checkpoint,'class_overlay':doc.class_overlay,'failure':doc.failure,'provenance':doc.provenance})
 readers=[]
 for key,value in ctx.reader_results.items():
  if 'softcap' in str(key):readers.append({'key':key,'status':value.status,'value':value.value,'failures':value.failures,'provenance':value.provenance})
 save(d/'actual-softcap-readers.json',readers)
 events=[e for e in ctx.config_access.events if 'softcap' in str(getattr(e,'fact_key','')) or 'attention_cap' in str(getattr(e,'config_path','')) or 'softcapp' in str(getattr(e,'config_path',''))]
 save(d/'events.json',events)
 diagram=Diagram(ir);page=diagram.to_html(standalone=True);(d/'page.html.gz').write_bytes(gzip.compress(page.encode(),mtime=0));save(d/'ir-after.json',diagram.ir.to_dict())
 save(d/'render-contexts.json',{str(k):list(v.events) for k,v in diagram._render_contexts.items()})
 native=ctx.facts.typed_records().get('decoder.attention.logit_softcap')
 row={'case':case,'layers':len(ir.layers),'caps':[layer.attention.logit_softcap for layer in ir.layers],'native_softcap_present':native is not None,'native_softcap':None if native is None else {'value':native.value,'status':native.status,'claim_kind':native.claim_kind,'has_proof':native.claim_evidence is not None,'unknown_reason':native.unknown_reason.to_dict() if native.unknown_reason else None},'consumed_softcap_events':sum(getattr(e,'intent','')=='consumed' and 'softcap' in str(getattr(e,'fact_key','')) for e in events),'reader_statuses':[(str(k),v.status) for k,v in ctx.reader_results.items() if 'softcap' in str(k)],'page_sha256':hashlib.sha256(page.encode()).hexdigest(),'page_bytes':len(page.encode())}
 rows.append(row);save(d/'result.json',row);print(json.dumps(row),flush=True)
assert hashlib.sha256(fixture.read_bytes()).hexdigest()==fixture_hash
save(out/'result.json',{'scope':'Actual original producer observation; no normalization, acceptance or repair','fixture_sha256':fixture_hash,'cases':rows})
