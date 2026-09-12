"""Compare saved actual closures, IR and HTML without running product code."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import difflib
import gzip
import hashlib
import json
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_compressed(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def delta_category(path):
    # These are address categories, not explanations or approvals. In
    # particular, a fact value/status change is not discarded as metadata.
    if path[:2] == ["extras", "fact_provenance"]:
        return "fact_value_or_status_requires_review" if len(path) < 4 or path[3] in {"value", "status"} else "fact_authority_metadata_requires_review"
    if path[:2] in (["extras", "config_access"], ["extras", "config_audit"], ["extras", "config_consumed"]):
        return "config_accounting_metadata_requires_review"
    if path[:2] == ["extras", "source_provenance"]:
        return "source_evidence_metadata_requires_review"
    if any(isinstance(part, str) and part.startswith("presentation_") for part in path) or "unknown_reason" in path:
        return "presentation_or_unknown_metadata_requires_review"
    if path[:1] in (["notes"], ["warnings"]):
        return "diagnostic_text_requires_review"
    return "structural_or_unclassified_requires_review"


def path_deltas(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            if key in before and key in after:
                yield from path_deltas(before[key], after[key], (*path, key))
            else:
                yield {"path": [*path, key], "before_present": key in before,
                       "after_present": key in after, "before": before.get(key), "after": after.get(key)}
    elif isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            if index < len(before) and index < len(after):
                yield from path_deltas(before[index], after[index], (*path, index))
            else:
                yield {"path": [*path, index], "before_present": index < len(before),
                       "after_present": index < len(after),
                       "before": before[index] if index < len(before) else None,
                       "after": after[index] if index < len(after) else None}
    elif type(before) is not type(after) or before != after:
        yield {"path": list(path), "before_present": True, "after_present": True,
               "before": before, "after": after}


def annotated_deltas(before, after):
    rows = list(path_deltas(before, after))
    for row in rows:
        row.update(category=delta_category(row["path"]), cause_status="NOT_YET_PROVEN",
                   owner_approval="NOT_RECORDED_BY_THIS_COMPARISON")
    return rows


def row_key(row):
    return (row["component"], row["external"], row["external_provenance"], row["locator"], row["content_sha256"])


def source_row(key, count):
    return dict(zip(("component", "external", "external_provenance", "locator", "content_sha256"), key), count=count)


def check_closure(capture, target, phase):
    expected = target[phase]["source_index_fingerprint"]
    if capture["status"] != "REPRODUCED" or any(capture[field] != expected for field in (
            "expected_matrix_seal", "actual_portable_seal", "recomputed_from_rows")):
        raise ValueError(f"{target['slug']} {phase}: corresponding actual matrix seal was not reproduced")
    rows = capture["source_multiset"]
    if len(rows) != capture["source_node_count"] + capture["parse_failure_count"]:
        raise ValueError("source multiset dropped healthy or failed source occurrences")
    encoded = sorted("\x1f".join((row["component"], "1" if row["external"] else "0",
                                  row["external_provenance"], row["locator"], row["content_sha256"])) for row in rows)
    actual = hashlib.sha256()
    for row in encoded:
        actual.update(row.encode("utf-8", "surrogatepass"))
        actual.update(b"\x1e")
    if actual.hexdigest() != expected:
        raise ValueError("saved complete multiset no longer reproduces its matrix seal")


def closure_diff(before, after):
    a, b = [Counter(row_key(row) for row in capture["source_multiset"]) for capture in (before, after)]
    owners = []
    groups = []
    for counter in (a, b):
        current = defaultdict(Counter)
        for (component, external, provenance, locator, content), count in counter.items():
            current[(locator, content)][(component, external, provenance)] += count
        groups.append(current)
    for identity in sorted(groups[0].keys() & groups[1].keys()):
        if groups[0][identity] != groups[1][identity]:
            owners.append({"locator": identity[0], "content_sha256": identity[1],
                           "before": [{"component": key[0], "external": key[1], "external_provenance": key[2], "count": count}
                                      for key, count in sorted(groups[0][identity].items())],
                           "after": [{"component": key[0], "external": key[1], "external_provenance": key[2], "count": count}
                                     for key, count in sorted(groups[1][identity].items())]})
    return {
        "before_seal": before["actual_portable_seal"], "after_seal": after["actual_portable_seal"],
        "before_source_count": sum(a.values()), "after_source_count": sum(b.values()),
        "removed": [source_row(key, count) for key, count in sorted((a - b).items())],
        "added": [source_row(key, count) for key, count in sorted((b - a).items())],
        "ownership_or_external_provenance_distributions_changed": owners,
        "before_parse_failures": before["parse_failures"], "after_parse_failures": after["parse_failures"],
        "limit": "Exact multiset differences, not architectural proof. A shortest unique locator can change when closure membership changes; additions/removals do not by themselves prove filesystem additions or renames. Ownership distributions compare only identical locator+content pairs.",
    }


SVG = re.compile(rb"<svg\b[^>]*>.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
STATS = re.compile(rb'<div class="uf-stat-key">(.*?)</div><div class="uf-stat-val">(.*?)</div>', re.DOTALL)


def html_summary(data):
    svgs = SVG.findall(data)
    return {"bytes": len(data), "sha256": digest(data), "inline_svg_count": len(svgs),
            "ordered_inline_svg_sha256": [digest(svg) for svg in svgs],
            "raw_banner_fields": [{"label_html": key.decode(), "value_html": value.decode()} for key, value in STATS.findall(data)]}


def structure_summary(ir, html_data):
    layers = ir.get("layers", [])
    extras = ir.get("extras", {})
    render = extras.get("render") or {}
    parameter_surfaces = []

    def parameters(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"parameter_shapes", "parameter_count", "total_params", "active_params", "params"}:
                    parameter_surfaces.append({"path": [*path, key], "value": child})
                else:
                    parameters(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                parameters(child, (*path, index))

    parameters(ir)
    return {
        "root_dimensions": {key: ir[key] for key in ("architecture", "hidden_size", "vocab_size", "max_position_embeddings", "tie_word_embeddings") if key in ir},
        "layer_count": len(layers),
        "layer_summaries": [{key: layer[key] for key in ("index", "count", "norm_kind", "norm_placement", "parallel_norm_count", "residual_topology", "blocks", "attention", "ffn") if key in layer} for layer in layers],
        "cross_layer_edges": ir.get("cross_layer_edges"),
        "render_family": render.get("family"), "denoiser_view": render.get("denoiser_view"),
        "unet_stage_surfaces": {key: extras.get("unet", {})[key] for key in ("stage_block_ids", "stage_relations", "skip_routes", "context_routes") if key in extras.get("unet", {})},
        "recorded_parameter_surfaces": parameter_surfaces,
        "raw_parameter_banner_fields": [item for item in html_summary(html_data)["raw_banner_fields"] if "PARAM" in item["label_html"].upper()],
        "parameter_limit": "Only recorded IR surfaces and literal HTML banner values. No parameter engine executed; rounded or estimated banners are not exact totals. Full IR diff remains authoritative over this summary.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--current-pages", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if len(plan["targets"]) != 39 or len({row["slug"] for row in plan["targets"]}) != 39:
        raise ValueError("the reviewed comparison requires all39 target identities")
    inputs = {}

    def pinned(path):
        inputs[str(path)] = digest(path.read_bytes())
        return load(path)

    for phase in ("before", "after"):
        result = pinned(args.replay_root / phase / "capture/result.json")
        if result["plan_sha256"] != digest(plan_bytes) or {r["slug"] for r in result["models"]} != {r["slug"] for r in plan["targets"]}:
            raise ValueError("closure phase is incomplete or uses another target plan")
    page_result = pinned(args.current_pages / "result.json")
    pages = {row["slug"]: row for row in page_result["models"]}
    if set(pages) != {row["slug"] for row in plan["targets"]} or any(row["status"] != "CAPTURED" for row in pages.values()):
        raise ValueError("current actual page capture is incomplete")
    closures_by_slug = {}
    for target in plan["targets"]:
        captures = {}
        for phase in ("before", "after"):
            capture = pinned(args.replay_root / phase / "capture" / target["slug"] / "closure.json")
            check_closure(capture, target, phase)
            captures[phase] = capture
        closures_by_slug[target["slug"]] = captures
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for target in plan["targets"]:
        slug = target["slug"]
        old = args.replay_root / "before/capture" / slug
        new_closure = args.replay_root / "after/capture" / slug
        current = args.current_pages / slug
        before_closure, after_closure = closures_by_slug[slug]["before"], closures_by_slug[slug]["after"]
        dest = out / slug
        dest.mkdir()
        closures = closure_diff(before_closure, after_closure)
        write(dest / "closure-deltas.json", closures)
        counts = {}
        for phase in ("before-render", "after-render"):
            before_ir = pinned(old / f"ir-{phase}.json.gz")
            after_ir = pinned(current / f"ir-{phase}.json.gz")
            recorded_key = "ir_before_sha256" if phase == "before-render" else "ir_after_sha256"
            if digest(gzip.decompress((old / f"ir-{phase}.json.gz").read_bytes())) != before_closure["baseline_output"][recorded_key]:
                raise ValueError("saved baseline IR differs from its actual capture receipt")
            rows = annotated_deltas(before_ir, after_ir)
            write_compressed(dest / f"ir-{phase}-deltas.json.gz", rows)
            counts[phase] = {"total": len(rows), "categories": dict(Counter(row["category"] for row in rows))}
        before_html_path, after_html_path = old / "page.html.gz", current / "page.html.gz"
        inputs[str(before_html_path)] = digest(before_html_path.read_bytes())
        inputs[str(after_html_path)] = digest(after_html_path.read_bytes())
        before_html = gzip.decompress(before_html_path.read_bytes())
        after_html = gzip.decompress(after_html_path.read_bytes())
        if digest(before_html) != before_closure["baseline_output"]["page_sha256"] or digest(after_html) != pages[slug]["page_sha256"]:
            raise ValueError("saved HTML differs from its actual capture receipt")
        a, b = html_summary(before_html), html_summary(after_html)
        html_delta = {"before": a, "after": b, "byte_equal": before_html == after_html,
                      "ordered_inline_svg_bytes_equal": SVG.findall(before_html) == SVG.findall(after_html),
                      "limit": "Raw inline SVG byte extraction is not DOM/browser verification and cannot establish chips-only output changes."}
        write(dest / "html-summary.json", html_delta)
        diff = "".join(difflib.unified_diff(before_html.decode().splitlines(keepends=True), after_html.decode().splitlines(keepends=True), fromfile=f"clean-fff99e22/{slug}.html", tofile=f"frozen26/{slug}.html"))
        (dest / "html-full.patch.gz").write_bytes(gzip.compress(diff.encode(), mtime=0))
        a_structure = structure_summary(pinned(old / "ir-before-render.json.gz"), before_html)
        b_structure = structure_summary(pinned(current / "ir-before-render.json.gz"), after_html)
        summary_equal = not list(path_deltas(a_structure, b_structure))
        write_compressed(dest / "structural-and-parameter-summaries.json.gz", {
            "before": a_structure, "after": b_structure,
            "summary_equal": summary_equal,
            "scope": "Review aid only; full path diffs enumerate all IR deltas without ignoring metadata or changing list order."})
        row = {"slug": slug, "status": "ENUMERATED_NOT_ACCEPTED", "closure_delta_rows": {"removed": len(closures["removed"]), "added": len(closures["added"])},
               "ir_deltas": counts, "html_byte_equal": html_delta["byte_equal"],
               "ordered_inline_svg_bytes_equal": html_delta["ordered_inline_svg_bytes_equal"],
               "structural_summary_equal": summary_equal,
               "chips_only": None, "cause_status": "ALL_CAUSES_REQUIRE_EVIDENCE_REVIEW"}
        results.append(row)
        write(out / "result.json", {"scope": "Exact saved-artifact enumeration; no normalized fields, architectural acceptance or blessing.", "models": results})
        print(json.dumps(row), flush=True)
    changed = [path for path, fingerprint in inputs.items() if digest(Path(path).read_bytes()) != fingerprint]
    if args.plan.read_bytes() != plan_bytes or changed:
        raise ValueError("saved inputs changed during comparison")
    write(out / "input-sha256.json", inputs)


if __name__ == "__main__":
    main()
