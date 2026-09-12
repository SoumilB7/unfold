"""Stdlib-only execution guards; no default, placeholder or inherited source pin."""
from pathlib import Path
import hashlib
import json
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) or value == '0' * 64:
        raise ValueError(label + ' requires an explicit exact SHA256; placeholders cannot execute')
    return value


def validate_execution_inputs(plan_path, source_manifest_sha256):
    source_pin = _digest(source_manifest_sha256, 'frozen32 source manifest')
    here = Path(__file__).resolve().parent
    if Path(plan_path).resolve() != here / 'plan.json':
        raise ValueError('capture requires its own held plan')
    manifest_path = here / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    required = {'census.py', 'run-census.py', 'portable_source.py', 'capture_contract.py', 'plan.json', 'contract.json'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('held capture manifest has missing or extra executable dependencies')
    for name, expected in manifest['files'].items():
        if sha(here / name) != _digest(expected, 'held script/plan'):
            raise ValueError('capture differs from held source: ' + name)
    plan = json.loads((here / 'plan.json').read_bytes())
    if plan.get('generation') != 'thirtysecond' or plan.get('execution_status') != 'TEMPLATE_REQUIRES_EXPLICIT_FROZEN_MANIFEST_PIN':
        raise ValueError('capture generation differs from held template')
    if 'source_manifest_sha256' in plan:
        raise ValueError('source pin must be supplied explicitly, never inherited from the template')
    tree = Path(plan['tree'])
    if tree.name != 'verify-s9-a-thirtysecond' or tree.resolve() != tree or not tree.is_dir():
        raise ValueError('capture requires the actual immutable32 worktree')
    source_manifest = Path(plan['source_manifest'])
    if source_manifest.resolve() != source_manifest or not source_manifest.is_file():
        raise ValueError('exact frozen32 source manifest is absent')
    if sha(source_manifest) != source_pin:
        raise ValueError('explicit frozen32 source manifest SHA256 differs')
    source_rows = json.loads(source_manifest.read_bytes())
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError('frozen source manifest must enumerate actual files')
    sources = {}
    for row in source_rows:
        relative = row['path']
        path = tree / relative
        if not isinstance(relative, str) or path.resolve().relative_to(tree).as_posix() != relative:
            raise ValueError('source manifest address is not canonical under the frozen tree')
        if relative in sources:
            raise ValueError('duplicate source manifest row')
        expected = row['sha256']
        if expected is None:
            # A deletion is an exact negative source pin. Even a broken
            # symlink violates absence; never skip or hash away this row.
            if path.exists() or path.is_symlink():
                raise ValueError('deleted frozen source exists: ' + relative)
            sources[relative] = None
        else:
            actual = sha(path)
            if actual != _digest(expected, 'source row'):
                raise ValueError('frozen source differs from its explicit manifest: ' + relative)
            sources[relative] = actual
    targets = plan['targets']
    if len(targets) != 39 or len({target['slug'] for target in targets}) != 39:
        raise ValueError('capture requires exact39 unique target addresses')
    inputs = {}
    for target in targets:
        for field, hash_field in [('input', 'input_sha256'), ('inventory', 'inventory_sha256')]:
            relative = target[field]
            path = tree / relative
            if path.resolve().relative_to(tree).as_posix() != relative:
                raise ValueError('noncanonical target address')
            actual = sha(path)
            if actual != _digest(target[hash_field], 'target address input'):
                raise ValueError('reviewed address input changed: ' + relative)
            inputs[relative] = actual
    matrix_path = tree / 'verification/s7/matrix.json'
    inputs['verification/s7/matrix.json'] = sha(matrix_path)
    if inputs['verification/s7/matrix.json'] != plan['active_matrix_sha256']:
        raise ValueError('active S7 address matrix changed')
    matrix = json.loads(matrix_path.read_bytes())
    if [row['slug'] for row in matrix['models']] != [row['slug'] for row in targets]:
        raise ValueError('active address membership/order differs from held targets')
    # Bracket all importable production sources as well as the changed-file
    # freeze manifest. This census imports no product code.
    production = {path.relative_to(tree).as_posix(): sha(path)
        for prefix in ('model_unfolder', 'physics')
        for path in sorted((tree / prefix).rglob('*.py'))}
    production['scripts/verify_commit.py'] = sha(tree / 'scripts/verify_commit.py')
    return {'source_manifest_sha256': source_pin, 'source': sources,
            'all_production_sources': production, 'inputs': inputs,
            'scripts': {name: sha(here / name) for name in sorted(required | {'manifest.json'})}}
