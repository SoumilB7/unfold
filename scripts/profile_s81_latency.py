#!/usr/bin/env python3
"""Serial, fresh-process UNet cold/warm latency gate with retained artifacts.

No model work occurs for --check. The default command executes twelve processes;
run it only in the repository's exclusive measurement lane. Cold means an empty
result cache, not cold OS filesystem pages. See verification/latency.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

TOOL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL_ROOT))
from test_support.latency_contract import (  # noqa: E402
    TARGETS, canonical_bytes, check_receipt, expected_samples, linked_entries, require, sha, validate_budget,
)

SOURCE_ROOTS = ('model_unfolder', 'physics')
SOURCE_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.js', '.css', '.html'}
TOOL_PATHS = ('scripts/profile_s81_latency.py', 'test_support/latency_contract.py')
FIXED_ENV = {'PYTHONHASHSEED': '0', 'HF_HUB_OFFLINE': '1',
             'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1',
             'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}


def host_telemetry():
    """Boundary context only: never normalize timing or drop a sample."""
    row = {}
    for name, read in (('cpu_count', os.cpu_count),
                       ('load_average', lambda: list(os.getloadavg()))):
        try:
            row[name] = read()
        except (OSError, AttributeError) as exc:
            row[name] = None
            row[name + '_unavailable'] = f'{type(exc).__name__}: {exc}'
    return row


def write_raw(root, relative, raw):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {'path': relative, 'bytes': len(raw), 'sha256': sha(raw)}


def write_json(root, relative, value):
    return write_raw(root, relative, canonical_bytes(value) + b'\n')


def file_pins(paths):
    return {str(path): sha(path.read_bytes()) for path in sorted(set(paths))}


def pins(repo, budget):
    # Membership is enumerated again in finally. No evidence imports/validation
    # occur here; this hashing is outside the library budget and can warm files.
    sources = {str(path.relative_to(repo)): sha(path.read_bytes())
               for name in SOURCE_ROOTS for path in sorted((repo / name).rglob('*'))
               if path.is_file() and path.suffix in SOURCE_SUFFIXES}
    return {'sources': sources,
            'inputs': {relative: sha((repo / relative).read_bytes()) for relative in TARGETS.values()},
            'tools': file_pins([TOOL_ROOT / relative for relative in TOOL_PATHS]),
            'budget': {str(budget): sha(budget.read_bytes())}}


def cache_state(path):
    return {str(item.relative_to(path)): sha(item.read_bytes())
            for item in sorted(path.rglob('*')) if item.is_file()}


def config_document(raw):
    document = json.loads(raw)
    if isinstance(document, dict) and isinstance(document.get('config'), dict):
        return document['config']
    require(isinstance(document, dict), 'config must be an object')
    return document


def check_local_imports(repo):
    for name, module in tuple(sys.modules.items()):
        if name == 'physics' or name.startswith('physics.') or name == 'model_unfolder' or name.startswith('model_unfolder.'):
            path = getattr(module, '__file__', None)
            if path:
                require(Path(path).resolve().is_relative_to(repo), f'foreign measured module: {name}: {path}')


@contextmanager
def observe_request_chain(runtime, input_sha256):
    """Observe the existing resolver once; return its exact request unchanged."""
    original = runtime.request_from_resolved_source
    rows = []

    def observed(document, bundle, root, **kwargs):
        request = original(document, bundle, root, **kwargs)
        symbol = root.graph.root.symbol
        row = {'input_sha256': input_sha256,
               'prepared_checkpoint': document.checkpoint,
               'resolved_source': {'path': symbol.source.canonical_path,
                                   'sha256': symbol.source.content_fingerprint,
                                   'component': symbol.source.component_key,
                                   'qualified_name': symbol.qualified_name},
               'request': request.to_dict()}
        rows.append(json.loads(canonical_bytes(row)))
        return request

    runtime.request_from_resolved_source = observed
    try:
        yield rows
    finally:
        unchanged = runtime.request_from_resolved_source is observed
        runtime.request_from_resolved_source = original
        require(unchanged, 'request observer was replaced during unfold')


def child(args):
    repo, output, budget = args.repo.resolve(), args.output.resolve(), args.budgets.resolve()
    relative = f'samples/{args.target}/{args.pair}/{args.mode}'
    sample_dir = output / relative
    sample_dir.mkdir(parents=True, exist_ok=False)
    cache_relative = f'cache/{args.target}/{args.pair}'
    cache = output / cache_relative
    record = {'schema_version': 1, 'status': 'FAIL', 'target': args.target,
              'pair': args.pair, 'mode': args.mode, 'config_relative': TARGETS[args.target],
              'cache_path': cache_relative, 'host_before': host_telemetry(), 'timings': {}, 'diagnostics': [], 'entries': {}, 'request_observations': []}
    before = None
    runtime_paths = []
    try:
        require(Path.cwd().resolve() == repo, 'child cwd must be measured checkout')
        require(os.environ.get('UNFOLD_EVIDENCE_CACHE_DIR') == str(cache), 'explicit owned pair cache')
        before = pins(repo, budget)
        record['pins_before'] = write_json(output, f'{relative}/pins-before.json', before)
        record['cache_before'] = write_json(output, f'{relative}/cache-before.json', cache_state(cache))
        if args.mode == 'cold':
            require(cache.is_dir() and not any(cache.iterdir()), 'cold cache is not empty')
        raw_input = (repo / TARGETS[args.target]).read_bytes()
        record['input'] = write_raw(output, f'{relative}/input.json', raw_input)
        config = config_document(raw_input)
        # Match S2's explicit import floor. Do not import readers/cache modules
        # separately to warm them before the actual public library call.
        sys.path.insert(0, str(repo))
        imported = {}
        import_start = time.perf_counter()
        for name in ('torch', 'transformers', 'diffusers', 'model_unfolder'):
            started = time.perf_counter()
            module = importlib.import_module(name)
            imported[name] = {'seconds': time.perf_counter() - started,
                              'version': getattr(module, '__version__', None),
                              'file': getattr(module, '__file__', None)}
        record['timings']['imports_seconds'] = time.perf_counter() - import_start
        record['imports'] = imported
        check_local_imports(repo)
        runtime_paths = [Path(sys.executable).resolve()]
        runtime_paths += [Path(row['file']).resolve() for row in imported.values() if row['file']]
        record['runtime_before'] = write_json(output, f'{relative}/runtime-before.json', file_pins(runtime_paths))
        record['runtime_pin_scope'] = 'interpreter and explicit library entry modules; full worker dependency seals retained in cache entries'
        # The diagnostics import is INSIDE this conservative budget. Any lazy
        # imports plus cache identity/capture/validation stay inside, too.
        started = time.perf_counter()
        try:
            from physics.result_cache import cache_diagnostics
            from model_unfolder.evidence import runtime_inventory
            with cache_diagnostics() as diagnostics, observe_request_chain(
                    runtime_inventory, sha(raw_input)) as requests:
                try:
                    call_started = time.perf_counter()
                    try:
                        diagram = sys.modules['model_unfolder'].unfold(config)
                    finally:
                        record['timings']['unfold_seconds'] = time.perf_counter() - call_started
                finally:
                    record['diagnostics'] = list(diagnostics)
                    record['request_observations'] = list(requests)
        finally:
            record['timings']['budget_seconds'] = time.perf_counter() - started
        check_local_imports(repo)
        record['import_origins_checked'] = True
        # Preserve the pre-render IR; known renderer materialization is outside
        # the library budget. Public HTML generation gets its own timer.
        record['ir'] = write_json(output, f'{relative}/ir.json', diagram.ir.to_dict())
        from model_unfolder.params import estimate_params
        record['params'] = write_json(output, f'{relative}/params.json', estimate_params(diagram.ir))
        started = time.perf_counter()
        html = diagram.to_html(standalone=True)
        record['timings']['html_seconds'] = time.perf_counter() - started
        record['html'] = write_raw(output, f'{relative}/page.html', html.encode('utf-8'))
        record['post_render_ir'] = write_json(output, f'{relative}/post-render-ir.json', diagram.ir.to_dict())
        record['status'] = 'PASS'
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        # Retain observer/cache/source state even when unfold or HTML failed.
        try:
            record['requests'] = write_json(output, f'{relative}/requests.json',
                                            record.pop('request_observations'))
            for row in record['diagnostics']:
                entry_sha = row.get('entry_sha256')
                if entry_sha:
                    require(len(entry_sha) == 64 and all(c in '0123456789abcdef' for c in entry_sha), 'entry address')
                    raw = (cache / 'entries' / f'{entry_sha}.json').read_bytes()
                    record['entries'][entry_sha] = write_raw(output, f'{relative}/entries/{entry_sha}.json', raw)
            record['cache_after'] = write_json(output, f'{relative}/cache-after.json', cache_state(cache))
            if record['status'] == 'PASS':
                linked_entries(output, record)
                summary = diagram.ir.construction_summary
                require(summary is not None and summary.parameter_count > 0 and summary.stage_count > 0,
                        'successful shape-backed UNet construction required')
        except BaseException as exc:
            record['status'] = 'FAIL'
            record['archive_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            # An unreadable cache entry must never skip final source/runtime
            # fingerprints. Their failure remains separate from archive failure.
            try:
                after = pins(repo, budget)
                record['pins_after'] = write_json(output, f'{relative}/pins-after.json', after)
                if runtime_paths:
                    record['runtime_after'] = write_json(output, f'{relative}/runtime-after.json', file_pins(runtime_paths))
                    require(record['runtime_before']['sha256'] == record['runtime_after']['sha256'], 'initial runtime files changed')
                require(before is not None and before == after, 'source/input/tool changed during child')
            except BaseException as exc:
                record['status'] = 'FAIL'
                record['finally_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        record['host_after'] = host_telemetry()
        write_json(output, f'{relative}/result.json', record)
    print(f"{args.target} pair={args.pair} {args.mode}: {record['status']} {record['timings']}", flush=True)
    return 0 if record['status'] == 'PASS' else 1


def campaign(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    budget = (args.budgets or repo / 'verification/latency_budgets.json').resolve()
    require(not output.is_relative_to(repo), 'output/cache must be outside measured checkout')
    require(not output.exists(), 'output already exists; never overwrite prior outcomes')
    output.mkdir(parents=True)
    record = {'schema_version': 1, 'status': 'FAIL', 'repo': str(repo), 'launches': [],
              'cold_scope': 'empty enabled result cache; OS files may be warm from initial hashing',
              'timer_boundary': 'S2 explicit library imports outside; lazy imports, request observer and all cache work inside; HTML separate',
              'python': sys.version, 'platform': platform.platform(), 'environment_overrides': FIXED_ENV,
              'host_before': host_telemetry()}
    before = None
    try:
        before = pins(repo, budget)
        record['pins_before'] = write_json(output, 'pins-before.json', before)
        record['budget'] = write_raw(output, 'latency_budgets.json', budget.read_bytes())
        validate_budget(json.loads(budget.read_bytes()))
        record['commit'] = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                         check=True, capture_output=True, text=True).stdout.strip()
        record['executed_tools'] = {str(TOOL_ROOT / relative): write_raw(
            output, f'tools/{relative}', (TOOL_ROOT / relative).read_bytes()) for relative in TOOL_PATHS}
        for target, pair, mode in expected_samples():
            cache = output / f'cache/{target}/{pair}'
            if mode == 'cold':
                cache.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(Path(__file__).resolve()), '--child', '--repo', str(repo),
                       '--output', str(output), '--budgets', str(budget), '--target', target,
                       '--pair', str(pair), '--mode', mode]
            environment = dict(os.environ, **FIXED_ENV, UNFOLD_EVIDENCE_CACHE_DIR=str(cache))
            launch = {'target': target, 'pair': pair, 'mode': mode, 'command': command,
                      'cwd': str(repo), 'cache_directory': str(cache)}
            log = output / f'{target}-{pair}-{mode}.log'
            started = time.monotonic()
            launch['start_monotonic'] = started
            record['launches'].append(launch)
            write_json(output, 'campaign.json', record)
            completed = None
            try:
                with log.open('wb') as stream:
                    completed = subprocess.run(command, cwd=repo, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT, check=False)
            finally:
                launch['end_monotonic'] = time.monotonic()
                launch['process_seconds'] = launch['end_monotonic'] - started
                launch['returncode'] = completed.returncode if completed is not None else None
                if log.exists():
                    launch['log'] = {'path': str(log.relative_to(output)), 'bytes': log.stat().st_size,
                                     'sha256': sha(log.read_bytes())}
                write_json(output, 'campaign.json', record)
            result_path = f'samples/{target}/{pair}/{mode}/result.json'
            raw_result = (output / result_path).read_bytes()
            launch['result'] = {'path': result_path, 'bytes': len(raw_result), 'sha256': sha(raw_result)}
            write_json(output, 'campaign.json', record)
            print(f"{target}/{pair}/{mode}: exit={completed.returncode} process={launch['process_seconds']:.3f}s", flush=True)
            # Fail-stop retains this outcome. A partial campaign cannot satisfy
            # the exact twelve-sample checker and never invents warm samples.
            require(completed.returncode == 0, 'fresh child failed; remaining samples not run')
        record['status'] = 'COMPLETE_PENDING_CHECK'
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        try:
            after = pins(repo, budget)
            record['pins_after'] = write_json(output, 'pins-after.json', after)
            require(before is not None and before == after, 'campaign source/input/tool changed')
        except BaseException as exc:
            record['status'] = 'FAIL'
            record['finally_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        record['host_after'] = host_telemetry()
        write_json(output, 'campaign.json', record)
    return check(output)


def check(output):
    try:
        result = check_receipt(output)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        result = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
    write_json(output, 'check.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--target', choices=tuple(TARGETS), help=argparse.SUPPRESS)
    parser.add_argument('--pair', type=int, choices=range(3), help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('cold', 'warm'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check:
        return check(args.output.resolve())
    if args.child:
        require(args.target is not None and args.pair is not None and args.mode is not None
                and args.budgets is not None, 'complete internal sample address required')
        return child(args)
    return campaign(args)


if __name__ == '__main__':
    raise SystemExit(main())
