from pathlib import Path
import dataclasses,json,sys
root=Path.cwd();sys.path.insert(0,str(root))
from model_unfolder import config_to_ir
from model_unfolder.diagram import Diagram
from model_unfolder.evidence.context import ParseContext
out=Path('/private/tmp/unfold-s9a-thirtyfourth/bloom-default-diagnostic/capture');out.mkdir(parents=True,exist_ok=False)
def plain(value):
 if dataclasses.is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in dataclasses.fields(value)}
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [plain(v) for v in value]
 if isinstance(value,(set,frozenset)):return sorted(plain(v) for v in value)
 if value is None or type(value) in (str,bool,int,float):return value
 return {'local_type':type(value).__name__}
def save(p,v):p.write_text(json.dumps(plain(v),sort_keys=True,indent=2)+'\n')
rows=[]
for case in ['ordinary','missing_root_count']:
 d=out/case;d.mkdir();cfg=json.loads((root/'tests/sable_test_corpus/bloom.json').read_text())['config']
 if case=='missing_root_count':assert cfg.pop('n_layer')==70
 save(d/'input.json',cfg);ctx=ParseContext.build(cfg);traces=[]
 def profile(frame,event,arg):
  if event!='return' or frame.f_code.co_name!='_scoped' or frame.f_locals.get('field')!='num_hidden_layers' or Path(frame.f_code.co_filename)!=root/'model_unfolder/adapters/transformer/parser.py':return
  parent=frame
  while parent and '_text_path' not in parent.f_locals:parent=parent.f_back
  loc=parent.f_locals if parent else {}
  traces.append({'field':frame.f_locals['field'],'text_path':loc.get('_text_path'),'shape_completion_defaults':loc.get('_shape_completion_defaults'),'fact_class_defaults':loc.get('_fact_class_defaults'),'selected_text_config':loc.get('text_cfg'),'resolution':{name:plain(getattr(arg,name,None)) for name in ['value','state','source_kind','provenance','present','ambiguous','selected_path','canonical','path']},'has_default_premise':getattr(arg,'default_premise',None) is not None})
 previous=sys.getprofile();error=None;ir=None
 try:
  sys.setprofile(profile);ir=config_to_ir(cfg,parse_context=ctx)
 except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
 finally:sys.setprofile(previous)
 save(d/'count-resolution-traces.json',traces)
 save(d/'prepared-documents.json',{k:{'owner':v.owner,'document_path':v.document_path,'checkpoint':v.prepared.checkpoint,'overlay':v.prepared.class_overlay,'provenance':v.prepared.provenance,'failure':v.prepared.failure} for k,v in ctx.prepared_documents.items()})
 save(d/'context-defaults.json',ctx.class_defaults);save(d/'events.json',[e for e in ctx.config_access.events if 'num_hidden_layers' in str(getattr(e,'canonical','')) or 'num_layers' in str(getattr(e,'fact_key',''))]);save(d/'native-facts.json',ctx.facts.to_dict())
 if ir is not None:
  save(d/'ir.json',ir.to_dict())
  (d/'page.html').write_text(Diagram(ir).to_html(standalone=True))
 row={'case':case,'layers':None if ir is None else len(ir.layers),'error':error,'count_traces':len(traces)};rows.append(row);save(d/'result.json',row);print(json.dumps(row),flush=True)
assert rows[0]['layers']==70 and rows[0]['error'] is None
assert rows[1]['layers'] > 0 and rows[1]['error'] is None
f=json.loads((out/'missing_root_count/native-facts.json').read_text())['decoder.attention.position_schedule']
assert f['status']=='class_default' and len(f['value'])==rows[1]['layers']
save(out/'result.json',{'scope':'LOCAL_ONLY actual original source/default-resolution diagnostic; no bypass or approval','cases':rows})
