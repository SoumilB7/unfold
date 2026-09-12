"""Stdlib-only pins, delegating unchanged source/address law to held actual37."""
from pathlib import Path
import hashlib
import gzip
import importlib.util
import json

HERE = Path(__file__).resolve().parent
SOURCE_PIN = 'ea7b4cde20d1dbaa2322175cc54a7fdc56bf10d1cedeb5bfb501ce10fe298c99'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_bytes())

def pins(source_pin):
    if source_pin != SOURCE_PIN:
        raise ValueError('requires explicitly supplied exact frozen37 source pin')
    manifest = load(HERE / 'manifest.json')
    required = {'observe.py', 'run-observation.py', 'observation_contract.py', 'plan.json', 'obligations.json', 'obligation-transitions.json', 'README.md'}
    if set(manifest['files']) != required:
        raise ValueError('proposal dependency membership differs')
    own = {}
    for name, digest in manifest['files'].items():
        own[name] = sha(HERE / name)
        if own[name] != digest:
            raise ValueError('held proposal changed: ' + name)
    own['manifest.json'] = sha(HERE / 'manifest.json')
    plan = load(HERE / 'plan.json')
    original = Path(plan['original_capture'])
    if sha(original / 'manifest.json') != plan['original_capture_manifest_sha256']:
        raise ValueError('original actual37 capture manifest changed')
    original_manifest = load(original / 'manifest.json')
    if sha(original / 'capture_contract.py') != original_manifest['files']['capture_contract.py']:
        raise ValueError('original guard changed before import')
    spec = importlib.util.spec_from_file_location('held_actual37_source_contract', original / 'capture_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_pins = module.validate_execution_inputs(original / 'plan.json', source_pin)
    saved = {}
    for path, digest in plan['saved_inputs'].items():
        saved[path] = sha(Path(path))
        if saved[path] != digest:
            raise ValueError('saved actual37/audit input changed: ' + path)
    completion_path = Path(plan['completion_result'])
    if completion_path != Path('/private/tmp/unfold-s9a-thirtyseventh/current-pages/result.json'):
        raise ValueError('current37 page completion address differs')
    completion = load(completion_path)
    if (not completion['passed'] or not completion['pins_equal']
            or completion['source_manifest_sha256'] != SOURCE_PIN
            or completion['before'] != completion['after']
            or completion['before'] != plan['source_fingerprint']
            or completion['artifacts_before'] != completion['artifacts_after']
            or completion['artifacts_before'] != '5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e'):
        raise ValueError('requires actual completed37 pages with matching source/artifact brackets')
    rows = load(HERE / 'obligations.json')
    if len(rows) != 105 or len({(r['slug'], json.dumps(r['semantic_address'], sort_keys=True)) for r in rows}) != 105:
        raise ValueError('expected exact 105 unique obligations')
    for row in rows:
        expected_address = {k: row['obligation'][k] for k in ('source', 'target', 'mechanism')}
        if row['semantic_address'] != expected_address:
            raise ValueError('semantic obligation address differs from exact current37 record')
        actual_ir = json.loads(gzip.decompress(
            (Path(plan['actual37_capture']) / row['slug'] / 'ir-before-render.json.gz').read_bytes()))
        actual_obligations = actual_ir['extras']['config_access']['projection_obligations']
        matches = [(i, o) for i, o in enumerate(actual_obligations)
                   if {k: o[k] for k in ('source', 'target', 'mechanism')} == row['semantic_address']]
        if len(matches) != 1 or matches[0] != (row['index'], row['obligation']):
            raise ValueError('current37 saved obligation missing/changed/duplicated')
    if len({(r['slug'], r['qualified_key']) for r in rows}) != 80:
        raise ValueError('expected exact 80 native targets')
    if sorted({r['slug'] for r in rows}) != plan['targets'] or len(plan['targets']) != 15:
        raise ValueError('expected exact 15 checkpoint inputs')
    return {'original36': original_pins, 'proposal': own, 'saved_inputs': saved}
