"""Renderer-only saved-data probe; no model construction or pytest lane."""
import copy
import gzip
import json
from pathlib import Path
import runpy
import sys
from xml.etree import ElementTree

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.renderers.html.block_views.unet import build_unet_constructed_view
from model_unfolder.renderers.html.views_diffusion import _stub_info

fixture = json.loads(gzip.decompress(Path(__file__).with_name('fixture.json.gz').read_bytes()))
geometry = runpy.run_path(str(ROOT / 'tests/test_s8_call_port_layout.py'))


def inspect(data):
    svg = build_unet_constructed_view({'extras': {'unet': data['unet']}}, _stub_info(), 'primary_geometry', data['denoiser'])
    rects, segments, _ = geometry['_geometry'](svg)
    crossings = [(name, segment) for name, rect in rects.items() for segment in segments
                 if geometry['_enters_rectangle'](segment, rect)]
    routes = [(node.get('data-source'), node.get('data-target')) for node in ElementTree.fromstring(svg).iter()
              if node.get('data-route-kind') == 'primary_state_port']
    expected = [(before['id'], after['id']) for before, after in zip(data['unet']['primary_regions'], data['unet']['primary_regions'][1:])
                if after['receives_previous_state']]
    assert routes == expected
    assert not crossings
    assert set(data['unet']['stage_block_ids'].values()) <= set(rects)
    assert set(data['unet']['other_block_ids']) <= set(rects)
    assert all(row['id'] in rects for row in data['unet']['primary_regions'])
    return {'cards': len(rects), 'primary_links': len(routes), 'crossings': len(crossings)}, svg


positive, _ = inspect(fixture)
poison = copy.deepcopy(fixture)
poison['unet']['primary_regions'][4]['receives_previous_state'] = False
limited, svg = inspect(poison)
assert limited['primary_links'] == positive['primary_links'] - 1
assert limited['cards'] == positive['cards']
assert 'Input link unresolved' in svg
print(json.dumps({'scope': 'Saved IR plus source-record region fixture; no production run or semantic acceptance',
                  'positive': positive, 'unproven_link_omitted_without_losing_cards': limited}, indent=2))
