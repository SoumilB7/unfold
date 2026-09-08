# Lossless archival whitespace packaging

The exact committed static check at ca15e58 passed Python lint, then correctly failed `git diff --check` because 13 preserved pytest logs/raw diff artifacts contain their original trailing whitespace. Its command/result and losslessly compressed output are retained here. This was archival data, not production code whitespace.

Each entry in `path-map.json` maps the original logical path and SHA256 to a gzip archive and SHA256. Decompressing every archive reproduces the exact bytes committed at ca15e58. Original receipt manifests retain their historical logical paths and hashes and are resolved through this map. No output text was stripped or rewritten; no gate exemption was added. The original files also remain available in ca15e58 history.

This packaging change does not alter the frozen ca15e58 diagnostic model inputs or production source. A later committed static check must pass before another whole coordinator run.
