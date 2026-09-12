"""Check exact historical bytes after archival gzip packaging; no models/tests."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
manifest = json.loads(Path(__file__).with_name("path-map.json").read_text())
for row in manifest["rows"]:
    archive = (root / row["archive_path"]).read_bytes()
    assert hashlib.sha256(archive).hexdigest() == row["archive_sha256"]
    raw = gzip.decompress(archive)
    assert hashlib.sha256(raw).hexdigest() == row["original_sha256"]
    original = subprocess.check_output(
        ["git", "show", manifest["historical_commit"] + ":" + row["original_path"]],
        cwd=root,
    )
    assert raw == original
print(f"Verified {len(manifest['rows'])} exact historical archives")
