"""Root-owned serial lane. Requires explicit frozen37 pin; no active outputs."""
from pathlib import Path
import argparse
import dataclasses
import importlib.util
import json
import os
import sys
from observation_contract import HERE, load, pins


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    before = pins(args.source_manifest_sha256)
    plan = load(HERE / 'plan.json')
    original_plan = load(Path(plan['original_capture']) / 'plan.json')
    tree = Path(original_plan['tree'])
    out = Path(plan['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'pins-before.json', before)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      UNFOLD_EVIDENCE_CACHE_DIR=str(out / 'cache'))
    spec = importlib.util.spec_from_file_location('child_observation36_verify', tree / 'scripts/verify_commit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._run_lane(module.Lane('actual37-child-native-ledgers', (
        sys.executable, str(HERE / 'observe.py'), '--source-manifest-sha256', args.source_manifest_sha256)), tree, out)
    after = pins(args.source_manifest_sha256)
    save(out / 'pins-after.json', after)
    packet = out / 'capture/result.json'
    packet_passed = packet.is_file() and load(packet)['passed']
    record = {**dataclasses.asdict(result), 'log_path': str(result.log_path),
              'source_manifest_sha256': args.source_manifest_sha256,
              'pins_equal': before == after, 'packet_passed': packet_passed,
              'passed': result.passed and before == after and packet_passed,
              'scope': 'Native child-ledger observation only; missing proof stays missing; no output approval.'}
    save(out / 'result.json', record)
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
