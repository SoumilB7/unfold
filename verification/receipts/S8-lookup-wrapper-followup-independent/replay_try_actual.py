from pathlib import Path
from dataclasses import replace,asdict
import sys,tempfile,json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index,portable_source_index_fingerprint
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
from physics.instance_inventory import InventoryResult
SOURCE='''def wrapper(owner, value):
    try:
        delegated(owner, value)
    except Failure as owner:
        handled(owner)
    except:
        fallback()
    else:
        succeeded()
    finally:
        owner.child = replacement
        return 9
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'fixture.py';p.write_text(SOURCE);bundle=SourceBundle(source='path',component_files={'root':(str(p),)})
 index=build_program_index(bundle);row=index.try_observations[0];symbol=row.enclosing_callable
 assert row.handlers[0].bound_name=='owner' and row.handlers[1].bound_name is None
 assert row.handlers[0].exception.kind=='name' and row.handlers[0].exception.name=='Failure'
 assert len(index.try_observations_in(symbol))==1
 assert any(r.construct_kind=='try' and r.span==row.span for r in index.unsupported_execution_in(symbol))
 assert [row.body_span.line,row.handlers[0].body_span.line,row.handlers[1].body_span.line,row.else_span.line,row.finally_span.line]==[3,5,7,9,11]
 rejected={}
 poisons={'order':{'handlers':tuple(reversed(row.handlers))},'cross_source':{'body_span':replace(row.body_span,source=replace(row.span.source,content_fingerprint='f'*64))},'handler_erasure':{'handlers':()}}
 for name,values in poisons.items():
  try: replace(row,**values)
  except (ValueError,TypeError):rejected[name]=True
  else:rejected[name]=False
 assert all(rejected.values())
 cached=build_program_index(bundle);assert cached.try_observations==index.try_observations
 before=portable_source_index_fingerprint(index);p.write_text(SOURCE.replace('as owner:','as error:'));changed=build_program_index(bundle)
 assert portable_source_index_fingerprint(changed)!=before and changed.try_observations[0].handlers[0].bound_name=='error'
 p.write_text('def f(owner):\n try:\n  call(owner)\n except* Failure as owner:\n  recover(owner)\n');star=build_program_index(bundle)
 assert not star.try_observations and star.unsupported_execution
 producer={'status':'PASS','poisons_rejected':rejected,'clause_spans':asdict(row),'except_star_unknown':True,'cache_and_changed_source':True}
receipt=ROOT/'verification/receipts/S8-actual-lookup-positive'
record=InventoryResult.from_dict(json.loads(gzip.decompress((receipt/'inventory-result.json.gz').read_bytes())))
paths=json.loads((receipt/'indexed-sources.json').read_text())
for row in paths: assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
index=build_program_index(SourceBundle(source='path',component_files={'root':tuple(x['path'] for x in paths)}))
root=next(x for x in record.inventory.modules if x.path=='')
witness=next(x for x in root.attribute_bindings if x.attribute=='forward')
target=next(x.symbol for x in index.callables if x.symbol.qualified_name=='UNet2DConditionModel.forward' and x.symbol.source.content_fingerprint=='052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26')
result=read_wrapper_binding(index,witness,target)
assert result.kind=='conditional_wrapper' and len(result.conditions)==2 and all(x['branch'] is False for x in result.conditions)
(OUT/'try-actual-results.json').write_text(json.dumps({'producer':producer,'saved_actual_sdxl_wrapper':asdict(result),'saved_inventory_sha256':hashlib.sha256((receipt/'inventory-result.json.gz').read_bytes()).hexdigest(),'current_source_bytes_match_all_saved_sources':True},indent=2,sort_keys=True)+'\n')
print('PASS: neutral Try records, unknown retention, handler/order/source poisons, cache changes; saved actual SDXL wrapper stays conditional with two unknown source conditions')
