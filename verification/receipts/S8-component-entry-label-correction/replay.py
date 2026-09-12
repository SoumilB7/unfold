"""Renderer-only label fit replay from an actual saved ModelIR; no model or pytest."""
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
from model_unfolder.renderers.html.views_diffusion import _build_loop_view,_stub_info
from model_unfolder.renderers.html.graph_engine import render_graph
out=Path(__file__).resolve().parent
source=Path("model_unfolder/renderers/html/views_diffusion.py")
before=source.read_bytes()
input_path=Path(sys.argv[1])
ir=json.loads(input_path.read_text());inputs=ir["extras"]["render"]["component_input_ids"];captured=[]
def capture(graph,*args,**kwargs):
    captured.extend(graph.nodes)
    return render_graph(graph,*args,**kwargs)
with patch("model_unfolder.renderers.html.graph_engine.render_graph",capture):
    svg=_build_loop_view(ir,_stub_info(),"uf-s8-demonstration")
labels={x["id"]:x["label"] for x in ir["extras"]["render"]["loop_blocks"]}
for node in captured:
    if node.id in inputs:
        assert "".join(node.label)==labels[node.id]
        assert max(map(len,node.label))*(node.font+3)*.6+30<=node.width()
        assert node.height()>=16+len(node.label)*(node.font+7)
assert source.read_bytes()==before
(out/"after.svg").write_text(svg)
(out/"results.json").write_text(json.dumps({"input":str(input_path),"ir_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),"renderer_sha256":hashlib.sha256(before).hexdigest(),"exact_labels_fitted":len(inputs),"source_unchanged":True,"scope":"saved actual IR renderer-only label fit; full page phase separate"},indent=2)+"\n")
print("PASS",len(inputs),"exact labels fitted")
