from pathlib import Path
import sys,json,gzip,dataclasses,time
repo=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');sys.path.insert(0,str(repo))
from physics.instance_inventory import InventoryResult
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.unet_call_binding import extend_lookup_sources
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
from model_unfolder.evidence.unet_iteration_binding import read_iteration_binding
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.program_index import SymbolId
p=Path('/private/tmp/unfold-s8-actual-lookup-probe-v1');config=json.loads((p/'input.json').read_text());inventory=InventoryResult.from_dict(json.loads(gzip.decompress((p/'inventory-result.json.gz').read_bytes()))).inventory
context=ParseContext.build(config);index=context.program_index();bundle=context.source_bundle;root=resolve_component_root(index,bundle,'root');index=extend_lookup_sources(index,bundle,inventory);document=prepare_document(config,merge=False);table=reconcile(model='saved-actual-sdxl',inventory=inventory,observations=(),config_document=document,program_index=index);bindings=RuntimeSourceBindings(table,inventory,index);method=index.callable_by_symbol(SymbolId(root.graph.root.symbol.source,root.graph.root.symbol.qualified_name+'.forward'));witnesses={x.attribute:x for x in bindings._modules[''].attribute_bindings};wrapper=read_wrapper_binding(index,witnesses['forward'],method.symbol);rows=[]
for loop in index.loops_in(method.symbol):
 for name in ['down_blocks','up_blocks']:
  if name not in (loop.iterable.source_segment or ''):continue
  result=read_iteration_binding(index,method,loop,read_lookup_closure(bindings,witnesses[name]),wrapper.function,parent_stable=True)
  rows.append({'field':name,'line':loop.span.line,'kind':result.kind,'reason':result.reason,'slot_count':len(result.slots),'target_count':len(result.targets),'element_name':result.element_name,'slots':[dataclasses.asdict(x) for x in result.slots]})
r=repo/'verification/receipts/S8-iteration-binding';r.mkdir(exist_ok=True);(r/'actual-source-replay.json').write_text(json.dumps({'scope':'Saved actual SDXL inventory and current exact source syntax; parent_stable=True is supplied as a probe premise, not independently established here. No model/forward/render run.','rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
