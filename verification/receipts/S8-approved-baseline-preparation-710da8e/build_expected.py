"""Generate the live preservation manifest with the unchanged canonical writer."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def tree(root, suffixes=None):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and (suffixes is None or p.suffix in suffixes)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkout', 'approval', 'corpus', 'external-pins', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, corpus, out = Path(args.checkout), Path(args.corpus), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    approval = json.loads(Path(args.approval).read_text())
    assert corpus.resolve() == Path(approval['stage']) / 'corpus'
    assert sorted(p.stem for p in corpus.glob('*.json')) == approval['approved_preservation_witnesses']
    for slug in approval['guarded_gallery_rebless_witnesses']:
        assert (corpus / 'galleries' / slug / 'her_eyes_review.md').stat().st_size > 0
        fixture = json.loads((corpus / (slug + '.json')).read_text())
        assert fixture['review_verdict']['decision'] == 'ACCEPT'
        assert fixture['review_verdict']['implementer'] == 'executor'
    assert Path.cwd().resolve() == root.resolve()
    source = {d + '/' + k: v for d in ('model_unfolder', 'physics', 'scripts', 'test_support') for k, v in tree(root / d, ('.py', '.yaml', '.yml')).items()}
    inputs = tree(corpus)
    tool_inputs = {str(Path(p).resolve()): sha(Path(p)) for p in (args.approval, args.external_pins, __file__)}
    external_doc = json.loads(Path(args.external_pins).read_text())
    external = {**external_doc['installed_source_sha256'], **external_doc['cached_task_config_sha256']}
    assert all(sha(Path(p)) == h for p, h in external.items())
    write(out / 'source-before.json', source)
    write(out / 'corpus-before.json', inputs)
    write(out / 'tool-input-pins.json', tool_inputs)
    write(out / 'external-before.json', external)
    start = time.monotonic()
    try:
        sys.path.insert(0, str(root))
        from test_support import preservation
        assert Path(preservation.__file__).resolve().is_relative_to(root)
        target = out / 'preservation_expected_manifest.json'
        print('Starting unchanged build_expected_manifest for all29 canonical witnesses', flush=True)
        preservation.build_expected_manifest(corpus, target)
        manifest = json.loads(target.read_text())
        assert manifest['witness_count'] == 29
        assert sorted(manifest['witnesses']) == approval['approved_preservation_witnesses']
        write(out / 'result.json', {'status': 'ACTUAL_CANONICAL_EXPECTED_MANIFEST_GENERATED', 'witnesses': 29, 'path': str(target), 'sha256': sha(target), 'elapsed_seconds': round(time.monotonic() - start, 3), 'historical_baseline_regenerated': False, 'live_baseline_mutated': False})
        print('Actual canonical manifest generation complete', flush=True)
    except BaseException as error:
        write(out / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        after = {d + '/' + k: v for d in ('model_unfolder', 'physics', 'scripts', 'test_support') for k, v in tree(root / d, ('.py', '.yaml', '.yml')).items()}
        after_external = {p: sha(Path(p)) for p in external}
        write(out / 'source-after-finally.json', after)
        write(out / 'external-after-finally.json', after_external)
        equality = {'source_equal': source == after, 'corpus_gallery_equal': inputs == tree(corpus), 'external_equal': external == after_external, 'tool_inputs_equal': tool_inputs == {p: sha(Path(p)) for p in tool_inputs}}
        write(out / 'pin-check-finally.json', equality)
        assert all(equality.values())


if __name__ == '__main__':
    main()
