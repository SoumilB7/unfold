"""Container projection controls using existing fixture definitions, no pytest."""
import ast
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path.cwd()))
path=Path('tests/test_s7_reconciliation.py'); tree=ast.parse(path.read_text())
keep=[]
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom)):
  if isinstance(node,ast.Import) and any(a.name=='pytest' for a in node.names):continue
  keep.append(node)
 elif isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in {'FP','CONFIG','CONFIG_HASH','CLASS'} for t in node.targets):keep.append(node)
 elif isinstance(node,ast.FunctionDef) and node.name in {'_inventory','_raw_fact','_product_index','_product_ir','_static'}:keep.append(node)
ns={};exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),ns)
fact=ns['_raw_fact']();ir=ns['_product_ir'](head=False);inventory=ns['_inventory']();index=ns['_product_index']()
project=ns['projection_claims_from_product']
def row():
 return next(row for row in project(index=index,inventory=inventory,static_claims=(),ir=ir,facts={fact.ledger_key():fact},render_events=()) if row.instance_path=='blocks')
fallback=row();assert fallback.axis.kind=='non_architectural'
ir.extras['render']['model_blocks'].append({'id':'container_card','kind':'opaque','label':'Container','source_instance_path':'blocks','source_fact_keys':[fact.ledger_key()]})
drawn=row();assert drawn.axis.kind=='rendered'
assert drawn.axis.fact_findings==(ns['ProjectionFactFinding'](fact.ledger_key()),)
assert drawn.axis.unqualified_fact_keys==(fact.ledger_key(),)
modules=list(inventory.modules)
modules[2]=dataclasses.replace(modules[2],children=('inner',))
cls=ns['ResolvedClass']('torch.nn.modules.container','ModuleList')
modules.append(ns['ModuleNode']('blocks.0.inner',cls,cls.module,(cls,),(),(),{},()))
group_inventory=dataclasses.replace(inventory,modules=tuple(modules))
static=ns['_static']();nested=dataclasses.replace(static,path_pattern=('blocks','*','inner'))
event=ns['RenderEvent']('block',('block',),'root','','','',None,frozenset(),frozenset({'inner'}),facts_projected=frozenset({fact.ledger_key()}))
grouped=next(row for row in project(index=index,inventory=group_inventory,static_claims=(static,nested),ir=ns['_product_ir'](),facts={fact.ledger_key():fact},render_events=(event,)) if row.instance_path=='blocks.0.inner')
assert grouped.axis.kind=='grouped'
assert grouped.axis.parent=='blocks.0'
assert grouped.axis.fact_findings==(ns['ProjectionFactFinding'](fact.ledger_key()),)
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'fallback':dataclasses.asdict(fallback.axis),'drawn':dataclasses.asdict(drawn.axis),'grouped':dataclasses.asdict(grouped.axis)},indent=2))
