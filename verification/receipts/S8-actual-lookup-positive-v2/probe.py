from pathlib import Path
import sys,json,gzip,dataclasses,hashlib,time,subprocess,traceback
repo=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');sys.path.insert(0,str(repo));out=Path('/private/tmp/unfold-s8-actual-lookup-probe-v2');out.mkdir(exist_ok=False)
critical=['runtime_inventory','unet_call_binding','unet_lookup_closure','unet_wrapper_binding','unet_cell_connections','unet_iteration_binding','local_port_routes','runtime_source','program_index','reconciliation','context','component_owner','import_source']
files=sorted(list((repo/'physics').glob('*.py'))+[repo/'model_unfolder/evidence'/f'{name}.py' for name in critical])
def pin():return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def write(name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,default=dataclasses.asdict)+'\n').encode();p=out/name;p.write_bytes(gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw)
before=pin();write('source-before.json',before);write('checkout.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),'scope':'Actual SDXL meta construction plus requested lookup/source closure only; no render, forward run, final acceptance or blessings.'})
start=time.monotonic()
try:
 from model_unfolder.evidence.context import ParseContext
 from model_unfolder.evidence.component_owner import resolve_component_root
 from model_unfolder.evidence.document import prepare_document
 from model_unfolder.evidence.runtime_inventory import build_resolved_instance,request_from_resolved_source
 from model_unfolder.evidence.unet_call_binding import extend_lookup_sources, read_root_invocations
 from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure
 from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
 from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
 from model_unfolder.evidence.reconciliation import reconcile
 from model_unfolder.evidence.program_index import SymbolId
 input_path=repo/'verification/receipts/S8-final-run-preparation/inputs/sdxl.json';raw=input_path.read_bytes();config=json.loads(raw);write('input.json',config);write('input-pin.json',{'path':str(input_path),'sha256':hashlib.sha256(raw).hexdigest()})
 context=ParseContext.build(config);index=context.program_index();bundle=context.source_bundle;root=resolve_component_root(index,bundle,'root');document=prepare_document(config,merge=False)
 request=request_from_resolved_source(document,bundle,root,index=index);write('construction-request.json',request.to_dict());print(json.dumps({'event':'constructing','requests':[x.attribute for x in request.attribute_lookups]}),flush=True)
 built=build_resolved_instance(document,bundle,root,index=index);write('inventory-result.json.gz',built.to_dict());write('construction-timing.json',{'elapsed_seconds':round(time.monotonic()-start,3),'status':built.status});print(json.dumps({'event':'constructed','status':built.status,'modules':len(built.inventory.modules) if built.inventory else None}),flush=True)
 if built.status!='ok':raise RuntimeError(str(built.failure))
 inventory=built.inventory;index=extend_lookup_sources(index,bundle,inventory);write('indexed-sources.json',[{'path':x.source_id.canonical_path,'sha256':x.source_id.content_fingerprint} for x in index.source_nodes]);table=reconcile(model='sdxl-actual-lookup-probe',inventory=inventory,observations=(),config_document=document,program_index=index);bindings=RuntimeSourceBindings(table,inventory,index)
 closures=[]
 for witness in bindings._modules[''].attribute_bindings:
  result=read_lookup_closure(bindings,witness);closures.append({'attribute':witness.attribute,'observation_kind':witness.kind,'closure_kind':result.kind,'reason':result.reason,'child_path':result.child_path,'storage_name':result.storage_name,'conditions':dataclasses.asdict(result)['conditions'],'spans':dataclasses.asdict(result)['spans'],'actual_function':dataclasses.asdict(witness.function) if witness.function else None})
 write('lookup-closures.json',closures)
 observed=next((x for x in bindings._modules[''].attribute_bindings if x.attribute=='forward'),None);expected=SymbolId(root.graph.root.symbol.source,root.graph.root.symbol.qualified_name+'.forward');wrapper=read_wrapper_binding(index,observed,expected);write('wrapper-binding.json',dataclasses.asdict(wrapper))
 method=index.callable_by_symbol(expected);invocations, invocation_spans, investigations=read_root_invocations(bindings,method,wrapper);write('root-invocations.json',{'invocations':invocations,'spans':invocation_spans,'investigations':[{'method':item.symbol.qualified_name,'closure':result} for item,result in investigations]});print(json.dumps({'event':'invocations','count':len(invocations),'kinds':{kind:sum(row['kind']==kind for row in invocations.values()) for kind in sorted({row['kind'] for row in invocations.values()})}}),flush=True)
 write('summary.json',{'status':'review_required','blessed':False,'constructed':len(inventory.modules),'lookups':len(closures),'closure_kinds':{k:sum(x['closure_kind']==k for x in closures) for k in sorted({x['closure_kind'] for x in closures})},'wrapper_kind':wrapper.kind,'wrapper_reason':wrapper.reason,'elapsed_seconds':round(time.monotonic()-start,3)})
 print(json.dumps({'event':'closures','rows':[{k:x[k] for k in ('attribute','observation_kind','closure_kind','reason')} for x in closures],'wrapper_kind':wrapper.kind,'wrapper_reason':wrapper.reason}),flush=True)
except BaseException as exc:
 write('failure.json',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':round(time.monotonic()-start,3)});raise
finally:
 after=pin();write('source-after.json',after);write('source-comparison.json',{'unchanged':before==after,'changed':[k for k in before if before[k]!=after[k]]})
