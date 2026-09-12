"""Render retained canonical graphs; no model parse or baseline writes."""
from pathlib import Path
import hashlib
import json
import re
import types
import xml.etree.ElementTree as ET

from model_unfolder.opgraph import ffn_region
from model_unfolder.renderers.html import graph_engine
from model_unfolder.renderers.html.graph import Edge, Graph, Group, Lane, Node, Parallel, SideInput
from model_unfolder.renderers.html.op_render import region_to_graph

OUT = Path(__file__).parent
CAPTURE = Path('/private/tmp/unfold-s8-bloom-current-736fdab/capture/graphs-and-regions.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def graph_from_dict(data):
    return Graph(**{**data, 'nodes': [Node(**x) for x in data['nodes']],
                    'edges': [Edge(**x) for x in data['edges']],
                    'groups': [Group(**x) for x in data['groups']],
                    'side_inputs': [SideInput(**x) for x in data['side_inputs']],
                    'parallels': [Parallel(**{**x, 'lanes': [Lane(**y) for y in x['lanes']]})
                                  for x in data['parallels']]})


def inspect_attention(svg):
    root = ET.fromstring(svg)
    ns = '{http://www.w3.org/2000/svg}'
    nodes = {g.get('data-id'): g for g in root.iter(ns + 'g') if g.get('data-id')}
    source = nodes['kv_cache'].find(ns + 'rect')
    target = nodes['attn_apply_v'].find(ns + 'circle')
    source_top = float(source.get('y'))
    target_y = float(target.get('cy'))
    paths = [tuple(map(float, re.findall(r'-?\d+(?:\.\d+)?', p.get('d'))))
             for p in root.iter(ns + 'path') if p.get('marker-end')]
    bypass = next(p for p in paths if p[1] == source_top - 28 and p[-1] == target_y)
    obstacles = []
    for node_id, g in nodes.items():
        rect = g.find(ns + 'rect')
        if rect is None or node_id == 'kv_cache':
            continue
        x, y, width, height = (float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
        if y < bypass[1] and y + height > target_y:
            obstacles.append({'node_id': node_id, 'left': x, 'right': x + width,
                              'intersects_rail': x <= bypass[0] <= x + width})
    stems = [line for line in root.iter(ns + 'line')
             if float(line.get('x1')) == float(line.get('x2')) == 0
             and float(line.get('y1')) == source_top
             and float(line.get('y2')) == source_top - 16]
    return {'rail_x': bypass[0], 'stem_count': len(stems), 'obstacles': obstacles,
            'arrow_paths': len(paths), 'viewBox': root.get('viewBox')}


def main():
    before_path = OUT / 'graph_engine.before.py.txt'
    before = types.ModuleType('model_unfolder.renderers.html._geometry_before')
    before.__package__ = 'model_unfolder.renderers.html'
    exec(compile(before_path.read_text(), str(before_path), 'exec'), before.__dict__)
    records = json.loads(CAPTURE.read_text())
    (OUT / 'canonical-graphs.json').write_bytes(CAPTURE.read_bytes())
    results = []
    graphs = [('bloom-attention', graph_from_dict(records[0]['graph'])),
              ('bloom-ffn', graph_from_dict(records[1]['graph'])),
              ('fused-gated-ffn', region_to_graph(ffn_region({
                  'kind': 'dense', 'gated': True, 'activation': 'gelu',
                  'projection_mode': 'fused_gate_up'}, None), clickable=True))]
    for name, graph in graphs:
        pair = []
        for phase, engine in [('before', before), ('after', graph_engine)]:
            svg = engine.render_graph(graph, {}, 'geometry-review', name, name)
            (OUT / f'{name}.{phase}.svg').write_text(svg)
            pair.append(svg)
        results.append({'graph': name, 'before_sha256': sha(pair[0].encode()),
                        'after_sha256': sha(pair[1].encode()), 'byte_identical': pair[0] == pair[1]})
    assert all(r['byte_identical'] for r in results[1:]), results
    geometry = {phase: inspect_attention((OUT / f'bloom-attention.{phase}.svg').read_text())
                for phase in ('before', 'after')}
    assert geometry['before']['stem_count'] == 0
    assert geometry['after']['stem_count'] == 1
    assert any(r['intersects_rail'] for r in geometry['before']['obstacles'])
    assert not any(r['intersects_rail'] for r in geometry['after']['obstacles'])
    assert geometry['before']['arrow_paths'] == geometry['after']['arrow_paths']
    assert geometry['before']['viewBox'] == geometry['after']['viewBox']
    source_path = Path(graph_engine.__file__)
    (OUT / 'graph_engine.after.py.txt').write_bytes(source_path.read_bytes())
    (OUT / 'replay.json').write_text(json.dumps({
        'command': 'PYTHONPATH=. python3 verification/receipts/S8-empty-lane-geometry/reproduce.py',
        'model_runs': 0, 'baselines_written': False,
        'before_source_sha256': sha(before_path.read_bytes()),
        'after_source_sha256': sha(source_path.read_bytes()),
        'capture_sha256': sha(CAPTURE.read_bytes()), 'results': results,
        'actual_geometry': geometry}, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
