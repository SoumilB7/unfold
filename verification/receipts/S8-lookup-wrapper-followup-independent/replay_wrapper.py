from pathlib import Path
import sys,importlib.util,tempfile,json,hashlib,dataclasses
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from physics.attribute_bindings import AttributeBindingWitness,_function_witness,capture_attribute_lookup_types
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
CASES={
 'contradictory_delegation':'''if enabled:
    before(owner)
    return captured(owner, value)''',
 'compatible_else_delegation':'''if enabled:
    before(owner)
else:
    return captured(owner, value)''',
 'nested_else_exposure':'''if outer:
    if enabled:
        pass
    else:
        before(owner)
return captured(owner, value)''',
 'handler_target':'''try:
    missing()
except Exception as owner:
    return captured(owner, value)''',
 'finally_return':'''try:
    return captured(owner, value)
finally:
    return 9''',
 'finally_effect':'''try:
    return captured(owner, value)
finally:
    before(owner)''',
 'conditional_finally_positive':'''if enabled:
    before(owner)
try:
    return captured(owner, value)
finally:
    if enabled:
        before(owner)''',
}
results={}
for name,body in CASES.items():
 source='events=[]\nenabled=False\nouter=False\ndef before(owner):\n    events.append("parent_exposed")\nclass Cell:\n    def forward(owner, value):\n        events.append("delegated")\n        return value\ndef wrap(captured):\n    def wrapper(owner, value):\n'+''.join('        '+line+'\n' for line in body.splitlines())+'    return wrapper\n'
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'fixture.py';path.write_text(source)
  spec=importlib.util.spec_from_file_location('wrapper_'+name,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  wrapper=module.wrap(module.Cell.forward)
  function,reason=_function_witness(wrapper,capture_attribute_lookup_types());assert function,reason
  witness=AttributeBindingWitness('','forward','plain_bound_method',function=function,lookup_kind='object_getattribute')
  index=build_program_index(SourceBundle(source='path',component_files={'root':(str(path),)}))
  target=next(x.symbol for x in index.callables if x.symbol.qualified_name=='Cell.forward')
  symbol=next(x.symbol for x in index.callables if x.symbol.qualified_name=='wrap.wrapper')
  result=read_wrapper_binding(index,witness,target)
  calls=[{'callee':c.callee.name,'guards':[{'kind':g.kind,'test':g.test.source_segment if g.test else None,'span_line':g.span.line} for g in c.guard]} for c in index.calls_in(symbol)]
  module.events.clear()
  try:value=wrapper(module.Cell(),7);actual={'value':value,'events':list(module.events)}
  except Exception as exc:actual={'exception':type(exc).__name__,'events':list(module.events)}
  results[name]={'source':source,'kind':result.kind,'reason':result.reason,'conditions':result.conditions,'calls':calls,'actual_false_branches':actual,'tries':[{'handler_targets':[h.bound_name for h in t.handlers],'body':t.body_span.line,'finally':t.finally_span.line if t.finally_span else None} for t in index.try_observations_in(symbol)],'unsupported':[x.construct_kind for x in index.unsupported_execution_in(symbol)]}
FILES=('program_index.py','unet_wrapper_binding.py','unet_call_binding.py')
results['hashes']={p:hashlib.sha256((ROOT/'model_unfolder/evidence'/p).read_bytes()).hexdigest() for p in FILES}
(OUT/'wrapper-results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps({n:{'kind':r['kind'],'conditions':r['conditions'],'actual':r['actual_false_branches']} for n,r in results.items() if n!='hashes'},indent=2))
