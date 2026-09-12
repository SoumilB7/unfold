"""Pinned exact portable source-multiset law; no project imports."""
import hashlib

def portable_rows(index):
    # Exact _compute_portable_source_index_fingerprint law at both fff99e22
    # and frozen26, exposed as rows. Never drop failures or deduplicate sources.
    sources = [node.source_id for node in index.source_nodes]
    sources.extend(failure.source for failure in index.parse_failures)
    if not sources:
        raise ValueError("portable source-index fingerprint needs a source census")
    paths = tuple(sorted(set(source.canonical_path for source in sources)))

    def parts(path):
        values = tuple(part for part in path.replace("\\", "/").split("/")
                       if part and not part.endswith(":"))
        if not values:
            raise ValueError("a source-index entry has no portable path parts")
        return values

    path_parts = {path: parts(path) for path in paths}

    def logical_locator(path):
        value = path_parts[path]
        for width in range(1, len(value) + 1):
            suffix = value[-width:]
            if sum(other[-width:] == suffix for other in path_parts.values()
                   if len(other) >= width) == 1:
                return "/".join(suffix)
        return "@source-root/" + "/".join(value)

    def record(source):
        return {
            "component": source.component_key or "",
            "external": source.external,
            "external_provenance": source.external_provenance,
            "locator": logical_locator(source.canonical_path),
            "content_sha256": source.content_fingerprint,
        }

    def encoded(row):
        return "\x1f".join((row["component"], "1" if row["external"] else "0",
                             row["external_provenance"], row["locator"], row["content_sha256"]))

    rows = sorted((record(source) for source in sources), key=encoded)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(encoded(row).encode("utf-8", "surrogatepass"))
        digest.update(b"\x1e")
    failures = [{"source": record(failure.source), "kind": failure.kind,
                 "detail": failure.detail} for failure in index.parse_failures]
    return rows, failures, digest.hexdigest()
