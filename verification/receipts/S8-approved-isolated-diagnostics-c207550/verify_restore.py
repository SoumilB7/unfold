"""Verify every archived byte; optionally restore into a new inspection directory."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new, nonexistent destination")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    entries = json.loads((root / "artifact-map.json").read_text())["entries"]
    digest = lambda data: hashlib.sha256(data).hexdigest()
    logical = [entry["logical_path"] for entry in entries]
    stored = [entry["stored_path"] for entry in entries]
    assert len(set(logical)) == len(set(stored)) == len(entries)
    actual = {p.relative_to(root).as_posix() for p in (root / "streams").rglob("*") if p.is_file()}
    assert actual == set(stored)
    if args.out is not None:
        assert not args.out.exists(), "Refuse to overwrite an existing directory"
    for entry in entries:
        for key in ("logical_path", "stored_path"):
            path = PurePosixPath(entry[key])
            assert not path.is_absolute() and ".." not in path.parts
        assert entry["stored_path"] == "streams/" + entry["logical_path"] + ".gzip"
        packed = (root / entry["stored_path"]).read_bytes()
        assert len(packed) == entry["stored_bytes"]
        assert digest(packed) == entry["stored_sha256"]
        raw = gzip.decompress(packed)
        assert len(raw) == entry["bytes"] and digest(raw) == entry["sha256"]
    if args.out is not None:
        args.out.mkdir(parents=True)
        for entry in entries:
            target = args.out / entry["logical_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.decompress((root / entry["stored_path"]).read_bytes()))
            assert digest(target.read_bytes()) == entry["sha256"]
    print(json.dumps({"status": "PASS", "streams": len(entries), "restored_to": str(args.out) if args.out else None}))


if __name__ == "__main__":
    main()
