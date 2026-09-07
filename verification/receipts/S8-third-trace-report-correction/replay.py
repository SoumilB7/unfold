import importlib.util,json,tempfile,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("reporter",root/"scripts/report_s8_demonstration.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
case=m.read_case(Path("/private/tmp/unfold-s8-final-c38b008/sdxl/ordinary"))
all_blocks=list(m.blocks(case["ir"]["extras"]["render"]))
cards=case["observation"]["page"]["cards"]
positive=m._conditioning_trace(case,all_blocks,cards)
assert not positive["chain_gaps"] and positive["actual_svg_arrows"]==3
assert positive["numbers_on_actual_cards"][0]["value"]==2050560
assert len(positive["established_connection_or_function"]["conditions"])==7
before=list(cards["denoiser"]["node_ids"])
cards["denoiser"]["node_ids"]=[x for x in before if x!="instance_time_embedding"]
negative=m._conditioning_trace(case,all_blocks,cards)
assert "stage overview" in negative["chain_gaps"]
cards["denoiser"]["node_ids"]=before
boundary=next(b for b in all_blocks if b.get("id")==positive["block"]["id"]+"__call")
conditions=boundary["detail"]["invocation_conditions"]
boundary["detail"]["invocation_conditions"]=conditions[:-1]
assert "exact conditional target link" in m._conditioning_trace(case,all_blocks,cards)["chain_gaps"]
print(json.dumps({"positive_chain_gaps":[],"arrows":3,"parameters":2050560,"conditions":7,"missing_visible_bookend_rejected":True,"dropped_condition_rejected":True},indent=2))
