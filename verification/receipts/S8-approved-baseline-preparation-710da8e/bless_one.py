"""Run the unchanged guarded bless writer once against an approved staged witness."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
import sys
import time
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_pins(root):
    return {str(p.relative_to(root)): sha(p) for d in ('model_unfolder', 'physics', 'scripts', 'test_support') for p in sorted((root / d).rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkout', 'approval', 'gallery-manifest', 'slug', 'report', 'verdict', 'pixel-review', 'corpus', 'external-pins', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, corpus, out = Path(args.checkout), Path(args.corpus), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    approval = json.loads(Path(args.approval).read_text())
    manifest = json.loads(Path(args.gallery_manifest).read_text())
    row = next(x for x in manifest['cases'] if x['slug'] == args.slug)
    assert args.slug in approval['guarded_gallery_rebless_witnesses']
    assert corpus.resolve() == Path(approval['stage']) / 'corpus'
    fixture_path = corpus / (args.slug + '.json')
    original = json.loads(fixture_path.read_text())
    assert sha(fixture_path) == row['input_sha256'][row['prior_fixture']]
    assert Path.cwd().resolve() == root.resolve(), 'Isolated inventory worker requires checkout cwd'
    inputs = {str(Path(p).resolve()): sha(p) for p in (args.approval, args.gallery_manifest, args.report, args.verdict, args.pixel_review, args.external_pins, __file__)}
    raw_report = json.loads(Path(args.report).read_text())
    for p in raw_report['gallery']:
        inputs[str(Path(p).resolve())] = sha(p)
    gallery_manifest = Path(raw_report['gallery'][0]).parent / 'MANIFEST.txt'
    inputs[str(gallery_manifest)] = sha(gallery_manifest)
    external_doc = json.loads(Path(args.external_pins).read_text())
    external = {**external_doc['installed_source_sha256'], **external_doc['cached_task_config_sha256']}
    before = source_pins(root)
    assert {k: before[k] for k in manifest['production_source_sha256']} == manifest['production_source_sha256']
    assert all(sha(p) == value for p, value in external.items())
    write(out / 'source-before.json', before)
    write(out / 'input-pins.json', inputs)
    write(out / 'external-before.json', external)
    write(out / 'fixture-before.json', original)
    started = time.monotonic()
    try:
        sys.path.insert(0, str(root))
        module = importlib.import_module('model_unfolder.sable')
        assert Path(module.__file__).resolve().is_relative_to(root)
        fields = dict(raw_report)
        fields['checks'] = [module.SableCheck(**x) for x in fields['checks']]
        report = module.SableReport(**fields)
        assert json.loads(json.dumps(dataclasses.asdict(report))) == raw_report
        actual_sable = module.sable
        reproductions = []

        def observe_actual_sable(*positional, **keywords):
            value = actual_sable(*positional, **keywords)
            reproductions.append(value)
            write(out / f'actual-offline-report-{len(reproductions)}.json', dataclasses.asdict(value))
            return value

        module.sable = observe_actual_sable
        try:
            result = module.bless(report, original['config'], source=original['source'], corpus_dir=corpus, review_verdict=args.verdict, implementer='executor')
        finally:
            module.sable = actual_sable
        assert len(reproductions) == 1
        assert Path(result) == fixture_path
        updated = json.loads(fixture_path.read_text())
        assert all(updated[k] == original[k] for k in ('config', 'source', 'model'))
        for png in raw_report['gallery']:
            assert sha(png) == sha(corpus / 'galleries' / args.slug / Path(png).name)
        assert sha(gallery_manifest) == sha(corpus / 'galleries' / args.slug / 'MANIFEST.txt')
        shutil.copy2(args.pixel_review, corpus / 'galleries' / args.slug / 'her_eyes_review.md')
        write(out / 'fixture-after.json', updated)
        write(out / 'result.json', {'status': 'GUARDED_STAGED_BLESS_PASS', 'slug': args.slug, 'fixture': str(fixture_path), 'fixture_sha256': sha(fixture_path), 'mandatory_actual_offline_reproductions': len(reproductions), 'observer_returns_original_report': True, 'gallery_pngs': len(raw_report['gallery']), 'pixel_review_sha256': sha(args.pixel_review), 'elapsed_seconds': round(time.monotonic() - started, 3), 'live_baseline_mutated': False})
    except BaseException as error:
        write(out / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        after = source_pins(root)
        current_inputs = {p: sha(p) for p in inputs}
        current_external = {p: sha(p) for p in external}
        write(out / 'source-after-finally.json', after)
        write(out / 'external-after-finally.json', current_external)
        write(out / 'pin-check-finally.json', {'source_equal': before == after, 'inputs_equal': inputs == current_inputs, 'external_equal': external == current_external})
        assert before == after and inputs == current_inputs and external == current_external


if __name__ == '__main__':
    main()
