"""Two source operand rails, without asserting dispatch or mutation effects."""
from pathlib import Path
import json
import runpy
import sys
from xml.etree import ElementTree

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.adapters.diffusor.unet_projection import _port_route_block
from model_unfolder.renderers.html.block_views.unet import build_runtime_port_route
from model_unfolder.renderers.html.views_diffusion import _stub_info
from model_unfolder.preview import svg_to_png

geometry = runpy.run_path(str(ROOT / 'tests/test_s8_call_port_layout.py'))
out = Path(__file__).parent
results = []
for kind in ('source_operation', 'inplace_operation'):
    route = {'kind': kind, 'operator': '+', 'operands': [
        {'kind': 'formal', 'formal': 'emb'}, {'kind': 'formal', 'formal': 'aug_emb'}]}
    block = _port_route_block(route, 'operator_fixture', 'root.denoiser.primary_state_ports')
    svg = build_runtime_port_route({}, _stub_info(), 'source_operator_fixture', block)
    rectangles, segments, circles = geometry['_geometry'](svg)
    assert len(rectangles) == 3 and not circles
    assert not any(geometry['_enters_rectangle'](segment, rect)
                   for rect in rectangles.values() for segment in segments)
    assert block['detail']['argument_ids'] == ['operator_fixture__operand_0', 'operator_fixture__operand_1']
    assert set(rectangles) == {*block['detail']['argument_ids'], block['detail']['boundary_id']}
    assert block['children'][-1]['kind'] == 'unknown' and block['children'][-1]['resolved'] is False
    assert 'Operation result' in svg and ('Source +=' if kind == 'inplace_operation' else 'Source +') in svg
    # Each argument has its own outgoing rail and both destination arrowheads
    # end at the operator, while a third arrow leaves for the result port.
    root = ElementTree.fromstring(svg)
    arrows = [node for node in root.iter() if node.get('marker-end')]
    assert len(arrows) == 3
    for argument in block['detail']['argument_ids']:
        rect = rectangles[argument]
        top_center = ((rect[0]+rect[2])/2, rect[1])
        assert any(abs(point[0]-top_center[0]) < .01 and abs(point[1]-top_center[1]) < .01
                   for segment in segments for point in segment)
    (out / (kind + '.svg')).write_text(svg)
    svg_to_png(svg, str(out / (kind + '.png')), scale=1, highlight_clickable=True)
    (out / (kind + '.json')).write_text(json.dumps(block, indent=2)+'\n')
    results.append({'kind': kind, 'clickable_cards': len(rectangles), 'incoming_operand_rails': 2,
                    'result_arrow': 1, 'card_crossings': 0, 'dispatch_and_mutation': 'unresolved'})
print(json.dumps({'scope': 'Synthetic source-port renderer/projection control, no model or acceptance', 'results': results}, indent=2))
