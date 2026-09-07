#!/usr/bin/env python3
"""Review-only comparisons for the one-time S8 experiment; never bless outputs.

All differences remain in the report. Narrow metadata normalization is listed
explicitly; an unmatched difference becomes a blocking finding, never a waiver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

EVIDENCE_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}:\d+:\d+:\d+:\d+")
SEMANTIC_FACTS = (
    "constructed_modules", "constructed_parameter_shapes", "ffn_mechanisms",
    "runtime_primitives", "cell_connections", "cell_arithmetic",
    "context_connections", "stage_join_connections", "spatial_mechanisms",
    "constructed_stage_relations",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class PageEvidence(HTMLParser):
    """Capture baked card prose and SVG membership, excluding script/style data."""
    VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})

    def __init__(self, page):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.cards = {}
        self.text = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cid = attrs.get("data-card-id") if tag == "div" else None
        if cid:
            self.cards.setdefault(cid, {"text": [], "facts": [], "svg_count": 0, "node_ids": []})
        if tag == "svg":
            for _, active, _ in self.stack:
                if active:
                    self.cards[active]["svg_count"] += 1
        if "uf-node" in attrs.get("class", "").split() and attrs.get("data-id"):
            for _, active, _ in self.stack:
                if active:
                    self.cards[active]["node_ids"].append(attrs["data-id"])
        if tag not in self.VOID:
            self.stack.append((tag, cid, attrs.get("class", "")))

    def handle_startendtag(self, tag, attrs):
        if tag not in self.VOID:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, value):
        if any(tag in {"script", "style"} for tag, _, _ in self.stack):
            return
        value = " ".join(value.split())
        if not value:
            return
        self.text.append(value)
        for _, cid, _ in self.stack:
            if cid:
                self.cards[cid]["text"].append(value)
                if any("uf-fact" in classes.split() for _, _, classes in self.stack):
                    self.cards[cid]["facts"].append(value)

    def record(self):
        return {"text": "\n".join(self.text), "cards": {
            key: {**value, "text": "\n".join(value["text"])}
            for key, value in self.cards.items()}}


def blocks(value):
    if isinstance(value, dict):
        if "id" in value and any(key in value for key in ("kind", "children", "view", "source_instance_path")):
            yield value
        for child in value.values():
            yield from blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from blocks(child)


def semantic_value(value):
    """Only source-span identity strings are normalized; values/routes survive."""
    if isinstance(value, dict):
        return {key: semantic_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [semantic_value(child) for child in value]
    if isinstance(value, str):
        return EVIDENCE_ADDRESS.sub("<source-span-address>", value)
    return value


def observation(ir, facts, page):
    from model_unfolder.preview import svg_views, _visual_hash
    selected = {key: semantic_value(facts["root.denoiser." + key]["value"])
                for key in SEMANTIC_FACTS if "root.denoiser." + key in facts}
    render = ir.get("extras", {}).get("render", {})
    structure = [{key: row[key] for key in (
        "id", "kind", "label", "view", "source_instance_path", "source_fact_keys") if key in row}
        for row in blocks(render)]
    svgs = svg_views(page)
    return {"semantic_facts": selected, "block_structure": structure,
            "svg_visual_hashes": [_visual_hash(svg) for _, svg in svgs],
            "svg_count": len(svgs), "distinct_svg_count": len({_visual_hash(svg) for _, svg in svgs}),
            "normalization": ["Only sha256:<hash>:line:col:end_line:end_col evidence-address strings inside selected fact values are replaced; paths, literals, source formals, mechanisms, operations and connections remain exact."],
            "signatures": {"semantic_facts": digest(selected), "block_structure": digest(structure)},
            "page": PageEvidence(page).record()}


def changes(before, after, path=""):
    if type(before) is not type(after):
        yield {"path": path or "/", "before": before, "after": after}
    elif isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            p = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before:
                yield {"path": p, "operation": "added", "after": after[key]}
            elif key not in after:
                yield {"path": p, "operation": "removed", "before": before[key]}
            else:
                yield from changes(before[key], after[key], p)
    elif isinstance(before, list):
        for i in range(max(len(before), len(after))):
            p = path + "/" + str(i)
            if i >= len(before):
                yield {"path": p, "operation": "added", "after": after[i]}
            elif i >= len(after):
                yield {"path": p, "operation": "removed", "before": before[i]}
            else:
                yield from changes(before[i], after[i], p)
    elif before != after:
        yield {"path": path or "/", "before": before, "after": after}


def read_case(path):
    result = {name: json.loads((path / (name + ".json")).read_text())
              for name in ("result", "facts", "ir", "controls", "observation", "qualified-facts")}
    result["path"] = str(path)
    return result


def condition_checks(ordinary, current):
    condition = current["result"]["condition"]
    a, b = ordinary["observation"], current["observation"]
    checks = []

    def check(name, passed, evidence):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    check("actual HTML generated", bool(current["result"].get("html_sha256")), current["path"] + "/page.html")
    check("graph wiring", not current["result"].get("wiring_problems"), current["result"].get("wiring_problems"))
    projected = current["result"].get("projected_fact_keys", [])
    qualified = current["qualified-facts"]
    gaps = [key for key in projected if key not in qualified or not qualified[key].get("proof")]
    check("every projected family fact carries a proof", bool(projected) and not gaps, gaps)
    check("same frozen implementation as ordinary", current["result"].get("implementation_source_sha256") == ordinary["result"].get("implementation_source_sha256")
          and current["result"].get("implementation_unchanged_during_run") is True,
          current["result"].get("implementation_source_sha256"))
    if condition in {"sparse", "misleading", "rewrite", "unchanged"}:
        check("mechanism, population and shape facts preserved", a["semantic_facts"] == b["semantic_facts"], {
            "ordinary": a["signatures"]["semantic_facts"], "condition": b["signatures"]["semantic_facts"]})
        check("canonical block structure preserved", a["block_structure"] == b["block_structure"], {
            "ordinary": a["signatures"]["block_structure"], "condition": b["signatures"]["block_structure"]})
        check("actual SVG diagrams preserved", a["svg_visual_hashes"] == b["svg_visual_hashes"], {
            "ordinary_count": a["svg_count"], "condition_count": b["svg_count"]})
    if condition == "sparse":
        defaults = current["facts"].get("root.denoiser.declared_constructor_defaults", {}).get("value", {})
        missing = [key for key in current["controls"] if key not in defaults or defaults[key].get("provenance") != "class_default"]
        check("every omitted equal default labelled as a class default fact", bool(current["controls"]) and not missing, missing)
        card = b["page"]["cards"].get("denoiser", {})
        expected = {key: f"Declared class default · {key}: {json.dumps(row['omitted'])} (checkpoint omitted)"
                    for key, row in current["controls"].items()}
        missing_text = [key for key, chip in expected.items() if chip not in card.get("facts", [])]
        check("each omitted key/value has class-default provenance on the actual denoiser card", not missing_text,
              {"card": "denoiser", "expected_chips": expected, "missing": missing_text})
    if condition == "misleading":
        page = b["page"]["text"]
        windows = [page[max(0, m.start()-200):m.end()+300] for m in re.finditer("hidden_act", page)]
        check("misleading field flagged in HTML prose", any(re.search(r"unused|unconsumed|unclaimed|unsupported|not consumed|uninterpreted|unread|no consumer", window, re.I) for window in windows), windows)
    if condition in {"rewrite", "unchanged", "changed", "missing"}:
        check("static reader and builder share exact scratch root bytes", current["result"].get("same_source") is True, {
            key: current["result"].get(key) for key in ("static_source_sha256", "runtime_source_sha256", "scratch_source_sha256")})
    if condition == "unchanged":
        check("unchanged scratch is byte-identical HTML", ordinary["result"]["html_sha256"] == current["result"]["html_sha256"], {
            "ordinary": ordinary["result"]["html_sha256"], "unchanged": current["result"]["html_sha256"]})
    if condition == "changed":
        check("real computation change changes facts", a["semantic_facts"] != b["semantic_facts"], str(Path(current["path"]) / "source.diff"))
        check("real computation change changes actual drawing", a["svg_visual_hashes"] != b["svg_visual_hashes"], {"ordinary": a["svg_count"], "changed": b["svg_count"]})
        old_params, new_params = ordinary["result"].get("parameters"), current["result"].get("parameters")
        check("shape-derived parameter number changes", old_params is not None and new_params is not None and old_params != new_params, {"before": old_params, "after": new_params})
        old_ffn, new_ffn = a["semantic_facts"].get("ffn_mechanisms", {}), b["semantic_facts"].get("ffn_mechanisms", {})
        added_ffn = sorted(set(new_ffn) - set(old_ffn))
        removed_ffn = sorted(set(old_ffn) - set(new_ffn))
        old_shapes = a["semantic_facts"].get("constructed_parameter_shapes", {})
        new_shapes = b["semantic_facts"].get("constructed_parameter_shapes", {})
        current_blocks = {row["source_instance_path"]: row for row in blocks(current["ir"].get("extras", {}).get("render", {})) if row.get("source_instance_path")}
        growth = []
        for path in added_ffn:
            block = current_blocks.get(path, {})
            card = b["page"]["cards"].get(block.get("id"), {})
            params = new_shapes.get("by_module", {}).get(path)
            expected_chip = f"{params:,} parameters in subtree" if params is not None else None
            growth.append({"occurrence": path, "mechanism": new_ffn[path], "block_id": block.get("id"),
                           "drill_svg_count": card.get("svg_count", 0), "shape_parameters": params,
                           "expected_card_chip": expected_chip, "card_matches_shapes": expected_chip in card.get("facts", []),
                           "new_parameter_paths": [key for key in new_shapes.get("parameters", {}) if key.startswith(path + ".") and key not in old_shapes.get("parameters", {})]})
        check("specific added computation is proven, drilled and counted on its own card",
              bool(growth) and not removed_ffn and all(row["drill_svg_count"] and row["card_matches_shapes"] and row["new_parameter_paths"] for row in growth),
              {"added_ffn_occurrences": growth, "removed_ffn_occurrences": removed_ffn})
    if condition == "missing":
        ordinary_ffn = a["semantic_facts"].get("ffn_mechanisms", {})
        current_ffn = b["semantic_facts"].get("ffn_mechanisms", {})
        check("removed evidence limits the FFN proof", bool(ordinary_ffn) and len(current_ffn) < len(ordinary_ffn), {"ordinary": len(ordinary_ffn), "missing": len(current_ffn)})
        lost = sorted(set(ordinary_ffn) - set(current_ffn))
        current_blocks = {row["source_instance_path"]: row for row in blocks(current["ir"].get("extras", {}).get("render", {})) if row.get("source_instance_path")}
        limitations = []
        for path in lost:
            block = current_blocks.get(path, {})
            card = b["page"]["cards"].get(block.get("id"), {})
            lines = [line for line in card.get("text", "").splitlines()
                     if re.search(r"mechanism|source|computation|reader|ffn", line, re.I)
                     and re.search(r"unresolved|investigation_missing|missing|unavailable|not established|not proven", line, re.I)]
            limitations.append({"occurrence": path, "lost_fact": "root.denoiser.ffn_mechanisms", "card_id": block.get("id"), "visible_limitation": lines})
        check("each affected occurrence exposes its missing mechanism evidence on its own card",
              bool(limitations) and all(row["visible_limitation"] for row in limitations), limitations)
    return checks


def report(root):
    ordinary = read_case(root / "ordinary")
    rows = {}
    for name in ("ordinary", "sparse", "misleading", "rewrite", "unchanged", "changed", "missing"):
        path = root / name
        if (path / "failure.json").exists():
            rows[name] = {"status": "FAIL", "checks": [{"check": "experiment completed", "status": "FAIL",
                           "evidence": json.loads((path / "failure.json").read_text())}]}
            continue
        if not (path / "result.json").exists():
            rows[name] = {"status": "NOT_RUN", "checks": []}
            continue
        current = read_case(path)
        checks = condition_checks(ordinary, current)
        rows[name] = {"status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL", "checks": checks}
        delta = list(changes(ordinary["ir"], current["ir"]))
        for row in delta:
            row["disposition"] = "blocking_finding"
            row["reason"] = "Output delta awaits evidence-level review; this report never self-approves it."
        dump(path / "output-deltas.json", delta)
        rows[name]["output_delta_count"] = len(delta)
    result = {"status": "review_required", "blessed": False, "conditions": rows,
              "blocking_checks": sum(row["status"] == "FAIL" for case in rows.values() for row in case["checks"]),
              "not_run": [key for key, value in rows.items() if value["status"] == "NOT_RUN"]}
    dump(root / "demonstration-report.json", result)
    return result


def _at(value, pointer):
    for part in pointer.strip("/").split("/") if pointer else ():
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _nearest_fact_keys(ir, pointer):
    parts = pointer.split("/")
    for end in range(len(parts), 0, -1):
        try:
            value = _at(ir, "/".join(parts[:end]))
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get("source_fact_keys"):
            return value["source_fact_keys"]
    return []


def differential(root):
    """Enumerate every IR/HTML delta, with reviewable proof candidates or debt."""
    before, after = read_case(root / "legacy"), read_case(root / "ordinary")
    rows = list(changes(before["ir"], after["ir"]))
    proofs = after["qualified-facts"]
    for row in rows:
        keys = _nearest_fact_keys(after["ir"], row["path"])
        qualified = [key for key in keys if key in proofs and proofs[key].get("proof")]
        row.update(disposition="blocking_finding", cause="Output change has not yet been matched to a complete evidence-level re-proof.",
                   evidence_fact_keys=qualified, approval="pending_owner_review")
        if qualified and len(qualified) == len(keys):
            row.update(disposition="named_reproof_candidate", cause="Canonical block projection now cites these qualified reader facts; owner must verify that the cited claim kinds cover this exact changed output.")
        if row["path"].startswith("/extras/unet/parameter_shapes"):
            key = "root.denoiser.constructed_parameter_shapes"
            if proofs.get(key, {}).get("proof"):
                row.update(disposition="named_reproof_candidate", cause="Exact constructed parameter shapes replace the legacy banner/count derivation.", evidence_fact_keys=[key])
    old_svgs = Counter(before["observation"]["svg_visual_hashes"])
    new_svgs = Counter(after["observation"]["svg_visual_hashes"])
    result = {"status": "review_required", "blessed": False, "witness": root.name,
              "legacy_html_sha256": before["result"]["html_sha256"],
              "new_html_sha256": after["result"]["html_sha256"],
              "ir_deltas": rows, "counts": dict(Counter(row["disposition"] for row in rows)),
              "svg_deltas": {"removed": dict(old_svgs-new_svgs), "added": dict(new_svgs-old_svgs)},
              "limitations": ["Named re-proof candidates are not accepted re-proofs or approval to re-bless.",
                              "Each IR delta is retained. HTML byte diff and normalized SVG changes are also written; no output is blessed."]}
    dump(root / "differential-report.json", result)
    import difflib
    old_page = (root / "legacy" / "page.html").read_text().splitlines(keepends=True)
    new_page = (root / "ordinary" / "page.html").read_text().splitlines(keepends=True)
    (root / "html.diff").write_text("".join(difflib.unified_diff(old_page, new_page, fromfile="legacy/page.html", tofile="ordinary/page.html")))
    return result


def claim_traces(root):
    """Index three witness claims into the actual baked cards, without blessing."""
    case = read_case(root / "ordinary")
    facts, qualified = case["facts"], case["qualified-facts"]
    all_blocks = list(blocks(case["ir"].get("extras", {}).get("render", {})))
    by_occurrence = {row["source_instance_path"]: row for row in all_blocks if row.get("source_instance_path")}
    cards = case["observation"]["page"]["cards"]
    traces = []
    for short, description in (("ffn_mechanisms", "Selected FFN computation and projection shapes"),
                               ("cell_connections", "Exact cell-member call connections"),
                               ("spatial_mechanisms", "Selected spatial operation and constructed shapes")):
        key = "root.denoiser." + short
        values = facts.get(key, {}).get("value", {})
        occurrence = next((path for path in sorted(values) if path in by_occurrence), None)
        if occurrence is None:
            traces.append({"claim": description, "status": "BLOCKING", "reason": "No qualified occurrence and actual canonical block pair."})
            continue
        block = by_occurrence[occurrence]
        stage = next((row for path, row in by_occurrence.items()
                      if occurrence.startswith(path + ".") and path.count(".") == 1), None)
        card = cards.get(block["id"], {})
        proof = qualified.get(key, {}).get("proof")
        shape_key = "root.denoiser.constructed_parameter_shapes"
        shapes = facts.get(shape_key, {}).get("value", {})
        expected_numbers = []
        subtree = shapes.get("by_module", {}).get(occurrence)
        if subtree is not None:
            expected_numbers.append({"card_id": block["id"], "chip": f"{subtree:,} parameters in subtree",
                                     "fact": shape_key, "value_path": ["by_module", occurrence], "value": subtree})
        mechanism = values[occurrence]
        connected_members = set()
        if short == "ffn_mechanisms":
            for field in ("input_projection", "output_projection"):
                member = mechanism.get(field)
                weight = shapes.get("parameters", {}).get(str(member) + ".weight")
                if weight:
                    expected_numbers.append({"card_id": block["id"], "chip": field.replace("_", " ") + ": " + " × ".join(map(str, weight["shape"])),
                                             "fact": shape_key, "value_path": ["parameters", str(member) + ".weight", "shape"], "value": weight["shape"]})
                if member:
                    connected_members.add(member)
        if short == "cell_connections":
            for edge in mechanism.get("connections", []):
                for endpoint in ("source", "target"):
                    call = mechanism.get("calls", {}).get(edge.get(endpoint), {})
                    if call.get("member"):
                        connected_members.add(call["member"])
        for member in sorted(connected_members):
            member_block = by_occurrence.get(member, {})
            for parameter, value in shapes.get("parameters", {}).items():
                if parameter.rpartition(".")[0] == member:
                    expected_numbers.append({"card_id": member_block.get("id"),
                                             "chip": parameter.rpartition(".")[2] + ": " + " × ".join(map(str, value["shape"])),
                                             "fact": shape_key, "value_path": ["parameters", parameter, "shape"], "value": value["shape"]})
        for number in expected_numbers:
            number["present_on_its_actual_card"] = number["chip"] in cards.get(number["card_id"], {}).get("facts", [])
        stage_visible = bool(stage and stage["id"] in cards.get("denoiser", {}).get("node_ids", []))
        connection = ({"kind": "returned FFN computation", "input_projection": mechanism.get("input_projection"),
                       "activation": mechanism.get("activation"), "gated": mechanism.get("gated"),
                       "output_projection": mechanism.get("output_projection")}
                      if short == "ffn_mechanisms" else mechanism)
        traces.append({"claim": description, "status": "REVIEW_REQUIRED", "occurrence": occurrence,
                       "implementation_evidence": proof, "established_connection_or_function": connection,
                       "canonical_fact": {"key": key, "claim_kind": qualified.get(key, {}).get("claim_kind"), "value": values[occurrence]},
                       "overview_stage": {"id": stage["id"], "label": stage.get("label"), "drawn_in_actual_denoiser_svg": stage_visible} if stage else None,
                       "block": {name: block.get(name) for name in ("id", "kind", "view", "source_fact_keys")},
                       "actual_card": card, "shape_evidence": qualified.get(shape_key, {}).get("proof"),
                       "numbers_on_actual_cards": expected_numbers,
                       "chain_gaps": [name for name, present in (("qualified proof", bool(proof)), ("stage overview", stage_visible),
                           ("actual card", bool(card)), ("actual drill SVG", bool(card.get("svg_count"))),
                           ("shape-backed numbers on exact cards", bool(expected_numbers) and all(row["present_on_its_actual_card"] for row in expected_numbers))) if not present],
                       "limitation": "An artifact linkage is not a semantic re-proof. Review the typed proof's exact claims and cited source before acceptance."})
    dump(root / "claim-traces.json", traces)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--differential", action="store_true", help="Also compare the explicit legacy experiment against ordinary")
    parser.add_argument("--claim-traces", action="store_true", help="Index three claim-to-card evidence chains")
    args = parser.parse_args()
    print(json.dumps(report(args.root), sort_keys=True))
    if args.differential:
        result = differential(args.root)
        print(json.dumps({"differential": result["counts"]}, sort_keys=True))
    if args.claim_traces:
        print(json.dumps({"claim_trace_count": len(claim_traces(args.root))}))


if __name__ == "__main__":
    main()
