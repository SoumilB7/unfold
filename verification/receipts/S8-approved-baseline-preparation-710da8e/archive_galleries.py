"""Archive exact selected gallery streams with collision-free lossless paths."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    approval = json.loads(Path(args.approval).read_text())
    rows = []
    for slug in approval['guarded_gallery_rebless_witnesses']:
        for p in sorted((source / slug).rglob('*')):
            if not p.is_file():
                continue
            relative = p.relative_to(source).as_posix()
            data = p.read_bytes()
            target = output / 'files' / (relative + '.gzip')
            target.parent.mkdir(parents=True, exist_ok=True)
            stored = gzip.compress(data, mtime=0)
            target.write_bytes(stored)
            assert gzip.decompress(target.read_bytes()) == data == p.read_bytes()
            rows.append({'original_path': str(p.resolve()), 'logical_path': relative, 'stored_path': target.relative_to(output).as_posix(), 'original_sha256': digest(data), 'stored_sha256': digest(stored), 'original_bytes': len(data)})
    assert len({r['stored_path'] for r in rows}) == len(rows)
    (output / 'archive-map.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    print(len(rows), 'unique lossless gallery streams archived')


if __name__ == '__main__':
    main()
