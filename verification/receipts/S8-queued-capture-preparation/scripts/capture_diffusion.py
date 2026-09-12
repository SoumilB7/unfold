"""One exact fixture parse; preserve actual constructor success or typed failure."""
from pathlib import Path
import argparse,dataclasses,json
from unittest.mock import patch
from capture_common import begin,finish,failure,write,sha,canonical,load_module,save_diagram

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--expect-construction',choices=('ok','failed'),required=True);a=p.parse_args()
root,out,before,start=begin(a)
try:
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import config_to_ir
    from model_unfolder.diagram import Diagram
    from model_unfolder.evidence import runtime_inventory
    from model_unfolder.adapters.diffusor import unet_cutover
    helpers=load_module('s8_capture_demonstration_helpers',root/'scripts/demonstrate_s8_unet.py')
    input_pin=sha(a.input);config=json.loads(Path(a.input).read_text());write(out/'input.json',config);write(out/'input-pin.json',{'path':a.input,'sha256':sha(a.input)})
    requests=[];results=[];request_function=runtime_inventory.request_from_resolved_source;builder=unet_cutover.build_resolved_instance
    def captured_request(*args,**kwargs):
        request=request_function(*args,**kwargs);requests.append(request);write(out/f'request-{len(requests)-1}.json',request.to_dict());return request
    def captured_builder(*args,**kwargs):
        result=builder(*args,**kwargs);results.append(result);write(out/f'construction-result-{len(results)-1}.json',result.to_dict());return result
    context=ParseContext.build(config)
    with patch.object(runtime_inventory,'request_from_resolved_source',captured_request),patch.object(unet_cutover,'build_resolved_instance',captured_builder):
        ir=config_to_ir(config,parse_context=context)
    diagram=Diagram(ir);diagram._mount_id='uf-s8-queued-capture'
    page,checks=save_diagram(out,diagram,helpers)
    write(out/'facts.json',{key:dataclasses.asdict(fact.to_record()) for key,fact in context.facts.typed.items()})
    helpers._qualified_facts(context,out)
    inventories=[]
    for number,result in enumerate(results):
        if result.inventory is not None:
            write(out/f'inventory-{number}.json',dataclasses.asdict(result.inventory));inventories.append(result.inventory)
    write(out/'reader-results.json',[{'key':key,'has_value':result.has_value,'failures':canonical(result.failures)} for key,result in context.reader_results.items() if key[0].startswith('root.denoiser.unet.')])
    statuses=[result.status for result in results]
    result={'status':'CAPTURED_REVIEW_REQUIRED','constructor_statuses':statuses,'expected_constructor_status':a.expect_construction,'request_count':len(requests),'inventory_count':len(inventories),'checks':checks,'html_sha256':sha(out/'page.html'),'semantic_limits':'Occurrence/shape/source-port evidence must be reviewed; no temporal alpha formula or runtime frame count inferred.','no_model_forward_or_weights':True}
    assert sha(a.input)==input_pin
    finish(root,out,before,start,result)
    assert len(results)==1,'expected one actual root construction result, not an inferred absent-unet disposition'
    if a.expect_construction=='ok':assert results[0].status=='ok',canonical(results[0].failure)
    else:
        assert results[0].status!='ok' and results[0].failure is not None
        assert any('UNet investigation_missing:' in str(w) for w in ir.warnings),'typed constructor failure must stay visible'
    assert all(not finding for finding in checks.values()),checks
except BaseException as error:
    failure(out,error);raise
