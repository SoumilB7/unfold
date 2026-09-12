from pathlib import Path
from types import SimpleNamespace
import sys,json,gzip,hashlib,dataclasses
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure,close_parent_helpers
from model_unfolder.evidence.unet_call_binding import read_root_invocations
from physics.instance_inventory import InventoryResult
receipt=ROOT/'verification/receipts/S8-actual-lookup-positive-v2'
record=InventoryResult.from_dict(json.loads(gzip.decompress((receipt/'inventory-result.json.gz').read_bytes())))
paths=json.loads((receipt/'indexed-sources.json').read_text())
for row in paths:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
index=build_program_index(SourceBundle(source='path',component_files={'root':tuple(x['path'] for x in paths)}))
bindings=SimpleNamespace(index=index,_modules={x.path:x for x in record.inventory.modules})
root=bindings._modules[''];lookups=[read_lookup_closure(bindings,x) for x in root.attribute_bindings]
forward=next(x for x in index.callables if x.symbol.qualified_name=='UNet2DConditionModel.forward' and x.symbol.source.content_fingerprint=='052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26')
witness=next(x for x in root.attribute_bindings if x.attribute=='forward')
wrapper=read_wrapper_binding(index,witness,forward.symbol)
parent=close_parent_helpers(bindings,forward,lookups,function_witness=wrapper.function)
targets,spans,investigations=read_root_invocations(bindings,forward,wrapper)
def ref(s):return f'sha256:{s.source.content_fingerprint}:{s.line}:{s.col}:{s.end_line}:{s.end_col}'
rows=[]
for k,row in targets.items():
 contradictions=[c for c in row['conditions'] if c.get('branch') is False and any(g.kind=='if' and g.test and ref(g.test.span)==c.get('source') for g in row['call'].guard)]
 rows.append({'source':k,'kind':row['kind'],'attribute':row.get('attribute'),'targets':row.get('targets'),'method':row['call'].enclosing_callable.qualified_name,'line':row['call'].span.line,'condition_count':len(row['conditions']),'same_guard_contradictions':contradictions})
out={'wrapper_kind':wrapper.kind,'root_closed_helpers':[{'line':c.span.line,'attribute':c.callee.name} for c in parent.closed_helpers],'root_open_helpers':[(s.line,r) for s,r in parent.unresolved],'lookup_kinds':{x.witness.attribute:x.kind for x in lookups},'targets':rows,'same_guard_contradiction_count':sum(bool(x['same_guard_contradictions']) for x in rows),'inventory_sha256':hashlib.sha256((receipt/'inventory-result.json.gz').read_bytes()).hexdigest(),'limit':'Static replay of saved evidence, not a fresh model run or acceptance of all conditions.'}
(OUT/'saved-helper-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in {'targets','lookup_kinds'}},indent=2))
