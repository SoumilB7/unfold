"""Actual saved SDXL inventory replay: operation counts, no model or cache budget."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_pins(repo):
    return {str(path.relative_to(repo)): digest(path)
            for name in ('model_unfolder', 'physics')
            for path in sorted((repo / name).rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in {'.pyc', '.pyo'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, origin, out = args.repo.resolve(), args.origin.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = [origin / name for name in ('input.json', 'build-request.json', 'inventory-result.json', 'ir.json',
                                        'selected-external-before.json')]
    before = source_pins(repo)
    initial_inputs = {str(path): digest(path) for path in inputs}
    tool_before = digest(Path(__file__))
    selected = json.loads((origin / 'selected-external-before.json').read_bytes())
    external_before = {path: digest(Path(path)) for path in selected}
    record = {'status': 'FAIL', 'source_before': before, 'input_before': initial_inputs,
              'external_before': external_before, 'tool_before': tool_before,
              'source_pin_scope': 'all regular model_unfolder/physics package files excluding bytecode caches; fresh membership before/finally',
              'limits': 'Actual source-bundle replay using one exact historical inventory; no construction, persistent cache hit or latency budget claim. Only derived hash computations counted; observer overhead is included in diagnostic timings.'}
    descriptor_originals = []
    computations = {}
    retained = {}
    original_inventory = inventory_module = None
    calls = []
    try:
        assert selected == external_before
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        from model_unfolder import unfold
        from model_unfolder.evidence import program_index
        from model_unfolder.everchanging import clear_load_cache
        from physics import instance_inventory as inventory_module
        assert Path(program_index.__file__).resolve().is_relative_to(repo)
        assert Path(inventory_module.__file__).resolve().is_relative_to(repo)
        request_record = json.loads((origin / 'build-request.json').read_bytes())
        inventory_record = json.loads((origin / 'inventory-result.json').read_bytes())
        inventory_result = inventory_module.InventoryResult.from_dict(inventory_record)
        assert inventory_result.status == 'ok'
        assert json.loads(json.dumps(inventory_result.to_dict())) == inventory_record
        original_inventory = inventory_module.inventory_in_subprocess
        def replay(request):
            actual = json.loads(json.dumps(request.to_dict()))
            calls.append(actual)
            write(out / 'actual-request.json', actual)
            assert len(calls) == 1 and actual == request_record
            return inventory_result
        inventory_module.inventory_in_subprocess = replay
        for cls in (program_index.SourceId, program_index.SourceSpan, program_index.ExprNode):
            descriptor = cls.__dict__['_structural_hash']
            original = descriptor.func
            descriptor_originals.append((descriptor, original))
            computations[cls.__name__] = 0
            retained[cls.__name__] = {}
            def counted(value, _name=cls.__name__, _original=original):
                computations[_name] += 1
                retained[_name][id(value)] = value
                return _original(value)
            descriptor.func = counted
        program_index.clear_program_index_source_cache()
        clear_load_cache()
        document = json.loads((origin / 'input.json').read_bytes())
        started, cpu = time.perf_counter(), time.process_time()
        diagram = unfold(document['config'])
        record['replay_wall_seconds'] = time.perf_counter() - started
        record['replay_process_cpu_seconds'] = time.process_time() - cpu
        assert len(calls) == 1
        write(out / 'ir.json', diagram.ir.to_dict())
        assert (out / 'ir.json').read_bytes() == (origin / 'ir.json').read_bytes()
        assert json.loads(json.dumps(inventory_result.to_dict())) == inventory_record
        prior = dict(computations)
        for name, objects in retained.items():
            assert len(objects) > 0 and computations[name] == len(objects)
            for value in objects.values():
                exact = hash(tuple(getattr(value, field.name) for field in fields(value)
                                   if (field.compare if field.hash is None else field.hash)))
                assert all(hash(value) == exact for _ in range(3))
        assert computations == prior
        record['actual_inventory_calls_replayed'] = len(calls)
        record['exact_ir_bytes'] = True
        record['status'] = 'PASS'
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        for descriptor, original in descriptor_originals:
            descriptor.func = original
        if original_inventory is not None:
            inventory_module.inventory_in_subprocess = original_inventory
        record['counts'] = {name: {'field_tuple_computations': count,
                                  'distinct_strongly_retained_objects': len(retained[name])}
                            for name, count in computations.items()}
        record['source_after'] = source_pins(repo)
        record['input_after'] = {str(path): digest(path) for path in inputs}
        record['external_after'] = {path: digest(Path(path)) for path in selected}
        record['tool_after'] = digest(Path(__file__))
        if (record['source_after'] != before or record['input_after'] != initial_inputs
                or record['external_after'] != external_before or record['tool_after'] != tool_before):
            record['status'] = 'FAIL'
        write(out / 'result.json', record)
    print(json.dumps({'status': record['status'], 'counts': record['counts']}), flush=True)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
