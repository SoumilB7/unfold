"""UNet detail views over canonical source-backed facts and constructed cards.

The production overview connects source-port boundary summaries only where
previous-state transport is proven. Constructed stages remain visible inside
containment frames; their layout does not assert individual execution order.
Skip accumulation and context links retain their separately proven endpoints.
Unbound bookends remain explicit containment cards, and boundary drills expose
local source-port details and unresolved call/guard meaning.

The legacy layout interpreter was deleted in S10-1.
"""
from __future__ import annotations

from ..graph_engine import render_graph
from ..stack_view import fit_svg, point
from ..svg import _ids, _path, _svg_tag, _svg_text
from ..theme import C, FONT_HEAD, FONT_MONO


def build_unet_constructed_view(ir, info, mount_id, block):
    """Bounded source-port boundaries, stage containment, and proven side routes."""
    unet = ir["extras"]["unet"]
    relation, ids = unet["stage_relations"], unet["stage_block_ids"]
    arrow_id, shadow_id = _ids(mount_id, "unet_constructed")
    parts, regions, placed, frames, wires = [], [], {}, [], []
    cards = {child["id"]: child for child in block.get("children", ())}

    def place(block_id, x, y, width=230, label=None, resolved=None):
        child = cards[block_id]
        geometry = _box(parts, x, y, width, 58, child["label"] if label is None else label,
                        shadow_id, node_id=block_id,
                        resolved=child.get("resolved", True) if resolved is None else resolved)
        regions.append(geometry)
        placed[block_id] = geometry
        return geometry

    def wire(route, kind, path):
        wires.append(_svg_tag("g", {"data-route-kind": kind,
                                   "data-source": route["source"], "data-target": route["target"]},
                              _path(path, arrow_id)))

    def containment(x, y, width, height, label):
        frames.append(_svg_tag("rect", {"x": x, "y": y, "width": width, "height": height,
                       "rx": 14, "fill": "none", "stroke": C["border"], "stroke-width": 1,
                       "data-region-kind": "containment"}))
        labels = label if isinstance(label, (tuple, list)) else (label,)
        for number, text in enumerate(labels):
            frames.append(_svg_text(x + width / 2, y + height - 12 - (len(labels)-number-1)*16, text,
                          {"text-anchor": "middle", "font-family": FONT_MONO, "font-size": 11,
                           "fill": C["text"]}))
        regions.extend((point(x, y), point(x + width, y + height)))

    producer, consumer = relation["producer_stages"], relation["consumer_stages"]
    primary = unet.get("primary_regions", ())
    left_x, right_x = 170, 820
    if primary:
        down_ids = {ids[path] for path in producer}
        up_ids = {ids[path] for path in consumer}
        down_positions = [number for number, row in enumerate(primary)
                          if down_ids.intersection(row["stage_block_ids"])]
        up_positions = [number for number, row in enumerate(primary)
                        if up_ids.intersection(row["stage_block_ids"])]
        down_end = max(down_positions, default=-1) + 1
        up_start = min(up_positions, default=len(primary))
        # Source order chooses the route. Left/right columns are only layout;
        # no order or execution edge is asserted between the contained stages.
        left_rows, bridge_rows, right_rows = primary[:down_end], primary[down_end:up_start], primary[up_start:]
        right_x = left_x + max(650, 560 * max(0, len(bridge_rows) - 1))
        center_x = (left_x + right_x) / 2
        region_heights = {}

        def region(row, x, y):
            child = cards[row["id"]]
            heading = {"if": "Conditional boundary", "for": "Repeat boundary",
                       "while": "Repeat boundary"}.get(row["kind"], "Operation boundary")
            detail = str(child["label"]).partition(": ")[2] or "Source ports · drill for details"
            if row.get("primary_target_id"):
                heading, detail = child["label"], "Conditional invocation"
            geometry = place(row["id"], x, y, 280, [heading, detail], resolved=False)
            if row.get("primary_target_id"):
                placed[row["primary_target_id"]] = geometry
            stage_ids = list(row["stage_block_ids"])
            if up_ids.intersection(stage_ids):
                stage_ids.reverse()
            invocation_ids = [target for target in row.get("extra_target_ids", ()) if target not in placed]
            visible_ids = list(dict.fromkeys((*stage_ids, *invocation_ids)))
            for number, stage_id in enumerate(stage_ids):
                place(stage_id, x, y + 90 + number * 86)
            height = 58 if not stage_ids else 110 + len(stage_ids) * 86
            bound = set(row.get("binding_target_ids", ()))
            if stage_ids:
                containment(x - 155, y - 15, 310, height + 30,
                            ("Conditional slot identity links" if row["kind"] in {"for", "while"}
                             else "Conditional call identity links") if set(stage_ids) <= bound else
                            "Constructed stages · open targets retained")
            if invocation_ids:
                extra_top = y + height + 55 if stage_ids else y + 75
                for number, target_id in enumerate(invocation_ids):
                    place(target_id, x, extra_top + 20 + number * 86)
                extra_height = 50 + len(invocation_ids) * 86
                containment(x - 155, extra_top, 310, extra_height,
                            ("Conditioning / argument call targets", "No loop execution implied")
                            if stage_ids else "Conditional call identity links")
                height = extra_top - y + extra_height
            if visible_ids:
                # These are target-identity relations, not tensor arrows or
                # execution order between slots. Actual input/result ports
                # remain in the selected source-bound call drill.
                for target_id in visible_ids:
                    if target_id not in bound:
                        continue
                    target = placed[target_id]
                    rail = x - 171
                    path = f"M {geometry['left']} {geometry['cy']} L {rail} {geometry['cy']} L {rail} {target['cy']} L {target['left']} {target['cy']}"
                    wires.append(_svg_tag("path", {"d": path, "fill": "none", "stroke": C["muted"],
                        "stroke-width": 1.5,
                        "data-route-kind": "conditional_call_target", "data-source": row["id"],
                        "data-target": target_id}))
                    regions.append(point(rail, target["cy"]))
            if not row["receives_previous_state"]:
                parts.append(_svg_text(x, y - 24, "Input link unresolved",
                             {"text-anchor": "middle", "font-family": FONT_MONO,
                              "font-size": 11, "fill": C["text"]}))
                regions.append(point(x, y - 35))
            region_heights[row["id"]] = height
            return height

        floor = 40
        for x, rows in ((left_x, left_rows), (right_x, list(reversed(right_rows)))):
            cursor = 40
            for row in rows:
                cursor += region(row, x, cursor) + 52
            floor = max(floor, cursor)
        bridge_y = floor + 35
        bottom = bridge_y
        for number, row in enumerate(bridge_rows):
            x = center_x if len(bridge_rows) == 1 else center_x + number * (right_x-center_x) / (len(bridge_rows)-1)
            bottom = max(bottom, bridge_y + region(row, x, bridge_y))
        for previous, current in zip(primary, primary[1:]):
            if not current["receives_previous_state"]:
                continue
            source, target = placed[previous["id"]], placed[current["id"]]
            if source["cx"] == target["cx"] and source["cy"] < target["cy"] and region_heights[previous["id"]] == 58:
                path = f"M {source['cx']} {source['bottom']} L {target['cx']} {target['top'] - 5}"
            elif source["cx"] == target["cx"] and source["cy"] > target["cy"] and region_heights[current["id"]] == 58:
                path = f"M {source['cx']} {source['top']} L {target['cx']} {target['bottom'] + 5}"
            elif source["cx"] == target["cx"] and source["cy"] > target["cy"]:
                # Leave through the upper port before taking a separate outer
                # rail. Sharing the incoming side port would look like a
                # bidirectional branch at this source boundary.
                rail, elbow = source["cx"] + 215, source["top"] - 22
                path = f"M {source['cx']} {source['top']} L {source['cx']} {elbow} L {rail} {elbow} L {rail} {target['cy']} L {target['right'] + 5} {target['cy']}"
                regions.extend((point(rail, elbow), point(rail, target["cy"])))
            elif source["cy"] == target["cy"]:
                path = f"M {source['right']} {source['cy']} L {target['left'] - 5} {target['cy']}"
            else:
                left = source["cx"] == left_x
                rail = left_x - 190 if left else right_x + 190
                start = source["left"] if left else source["right"]
                end = target["left"] - 5 if left else target["right"] + 5
                path = f"M {start} {source['cy']} L {rail} {source['cy']} L {rail} {target['cy']} L {end} {target['cy']}"
                regions.extend((point(rail, source["cy"]), point(rail, target["cy"])))
            wire({"source": previous["id"], "target": current["id"]}, "primary_state_port", path)
        # A newly proven stage without a matching region must still be visible.
        unplaced = [block_id for block_id in ids.values() if block_id not in placed]
        for number, block_id in enumerate(unplaced):
            place(block_id, left_x + number % 3 * (right_x-left_x)/2, bottom + 75 + number//3 * 95)
        if unplaced:
            bottom += 75 + ((len(unplaced)+2)//3)*95
        stage_tops = [placed[block_id]["top"] for block_id in ids.values()]
        place("unet_skip_bank", center_x, max(230, min(stage_tops, default=300) + 65), 250)
        bottom += 90
    else:
        center_x = (left_x + right_x) / 2
        for column, paths in enumerate((producer, list(reversed(consumer)))):
            for row, path in enumerate(paths):
                place(ids[path], (left_x, right_x)[column], 40 + row * 125)
        floor = 40 + max(len(producer), len(consumer)) * 125
        for row, path in enumerate(relation["intermediate_stages"]):
            place(ids[path], center_x, floor + row * 95, 270)
        place("unet_skip_bank", center_x, 75)
        bottom = floor + len(relation["intermediate_stages"]) * 95 + 60
    context_ids = list(dict.fromkeys(route["source"] for route in unet.get("context_routes", ())))
    for number, source_id in enumerate(context_ids):
        place(source_id, center_x, bottom + number * 95, 270)
    bottom += len(context_ids) * 95
    other_ids = [block_id for block_id in unet["other_block_ids"] if block_id not in placed]
    for number, block_id in enumerate(other_ids):
        place(block_id, left_x + number % 3 * (right_x-left_x)/2, bottom + 50 + number//3 * 95)
    if other_ids:
        containment(left_x-145, bottom+25, right_x-left_x+290, ((len(other_ids)+2)//3)*95+35,
                    ("Constructed call targets · conditional connections in source-port drills"
                     if "unbound_bookend_paths" in unet and not unet["unbound_bookend_paths"] else
                     "Other constructed modules · some call targets remain unbound"))
    for route in unet.get("skip_routes", ()):
        source, target = placed[route["source"]], placed[route["target"]]
        rail = (source["right"] + target["left"]) / 2
        wire(route, "skip_accumulation", f"M {source['right']} {source['cy']} L {rail} {source['cy']} "
             f"L {rail} {target['cy']} L {target['left'] - 5} {target['cy']}")
    for number, route in enumerate(unet.get("context_routes", ())):
        source, target = placed[route["source"]], placed[route["target"]]
        left = target["cx"] < center_x
        rail = left_x - 250 - number*5 if left else right_x + 250 + number*5
        start = source["left"] if left else source["right"]
        end = target["left"] - 5 if left else target["right"] + 5
        wire(route, "external_context", f"M {start} {source['cy']} L {rail} {source['cy']} "
             f"L {rail} {target['cy']} L {end} {target['cy']}")
        regions.append(point(rail, source["cy"]))
    return fit_svg(arrow_id, shadow_id, frames + wires + parts, regions,
                   "Source port boundaries and constructed stages; containment is not execution order",
                   min_width=720, pad=44)


def build_constructed_children_view(ir, info, mount_id, block):
    """A containment view has no implied sequential arrows."""
    from ..render_context import current_render_context
    from xml.etree import ElementTree

    arrow_id, shadow_id = _ids(mount_id, "constructed_children")
    parts, regions = [], []
    for number, child in enumerate(block.get("children", ())):
        regions.append(_box(parts, 145 + number % 3 * 270,
                            35 + number // 3 * 100, 235, 64,
                            child["label"], shadow_id,
                            node_id=child["id"]))
    svg = fit_svg(arrow_id, shadow_id, parts, regions,
                  "Constructed children; containment only", min_width=720, pad=40)
    context = current_render_context()
    if context is None or not context.block_stack:
        return svg
    try:
        returned = ElementTree.fromstring(svg)
    except ElementTree.ParseError:
        return svg
    visible = {}
    for element in returned.iter():
        if (element.tag.rsplit("}", 1)[-1] == "g"
                and "uf-node" in element.get("class", "").split()
                and element.get("data-id") and len(element)):
            visible[element.get("data-id")] = tuple(
                "".join(text.itertext()) for text in element.iter()
                if text.tag.rsplit("}", 1)[-1] == "text")
    key = "root.denoiser.runtime_primitives"
    row = context.fact_rows.get(key, {})
    if row.get("status") != "code_proven":
        return svg
    values = row.get("value", {})
    actual_children = context.block_stack[-1].get("children", ())
    address_fields = ("id", "source_instance_path", "source_component", "source_owner", "label")
    for child in block.get("children", ()):
        actual = next((candidate for candidate in actual_children
                       if all(candidate.get(field) == child.get(field) for field in address_fields)
                       and set(candidate.get("source_fact_keys", ()))
                       == set(child.get("source_fact_keys", ()))), None)
        if (actual is None or actual.get("source_component") != "root"
                or key not in actual.get("source_fact_keys", ())):
            continue
        value = values.get(actual.get("source_instance_path"), {})
        label = actual.get("label")
        if (not isinstance(label, str) or not label
                or value.get("kind") != actual.get("kind") or value.get("label") != label
                or visible.get(actual.get("id")) != (label,)):
            continue
        with context.block(actual):
            context.note_facts_projected("constructed_primitive_label", (key,),
                                         node_ids=(actual["id"],))
    return svg


def build_runtime_ffn_view(ir, info, mount_id, block):
    """Reuse the canonical FFN graph and keep its actual children inspectable."""
    from .feed_forward import build_ffn_view
    from ..render_context import current_render_context
    from xml.etree import ElementTree

    context = current_render_context()
    start = len(context.events) if context is not None else 0
    block_path = tuple(str(item.get("id") or item.get("view") or "?")
                       for item in context.block_stack) if context is not None else ()
    ffn_svg = build_ffn_view(ir, info, mount_id, block)
    key = "root.denoiser.ffn_mechanisms"
    if (context is not None and ffn_svg and block_path
            and block_path[-1] == block.get("id")
            and key in block.get("source_fact_keys", ())):
        try:
            returned = ElementTree.fromstring(ffn_svg)
            visible_nodes = frozenset(
                element.get("data-id") for element in returned.iter()
                if element.tag.rsplit("}", 1)[-1] == "g"
                and "uf-node" in element.get("class", "").split()
                and element.get("data-id") and len(element))
        except ElementTree.ParseError:
            visible_nodes = frozenset()
        operation_ids = frozenset(child["id"] for child in block.get("children", ())
                                  if child.get("role") == "operation")
        graphs = [event for event in context.events[start:]
                  if event.view == "ffn" and event.block_path == block_path
                  and operation_ids and operation_ids <= event.node_ids
                  and operation_ids <= visible_nodes]
        if graphs:
            context.note_facts_projected(
                "runtime_ffn_fact", (key,),
                node_ids=operation_ids)
    constructed = [child for child in block.get("children", ())
                   if "source_instance_path" in child]
    return (ffn_svg
            + build_constructed_children_view(ir, info, mount_id,
                                              {"children": constructed}))


def build_runtime_stage_connections(ir, info, mount_id, block):
    """Project declared connections through the shared graph/wiring engine."""
    from ..graph import Graph, Node, SideInput
    children = {child["id"]: child for child in block.get("children", ())}
    used, rendered = set(), []
    for number, route in enumerate(block["detail"]["join_routes"]):
        operands = route["operands"]
        if not operands:
            continue
        join_id = route["join"]
        nodes = [Node(operand, children[operand]["kind"], children[operand]["label"],
                      resolved=children[operand].get("resolved", True)) for operand in operands]
        nodes.extend((Node(join_id, "concat"),
                      Node(route["target"], "opaque", "Repeated child calls")))
        graph = Graph(nodes, [operands[0], join_id, route["target"]],
                      side_inputs=[SideInput(operand, join_id,
                                             "right" if offset % 2 == 0 else "left")
                                   for offset, operand in enumerate(operands[1:])])
        rendered.append(render_graph(
            graph, info, mount_id, "runtime_stage_connections",
            "Source-proven concat; conditional input routes remain explicit",
            facts_projected=frozenset(block.get("source_fact_keys", ()))))
        used.update((*operands, join_id, route["target"]))
    remaining = [child for child in children.values() if child["id"] not in used]
    if remaining:
        rendered.append(build_constructed_children_view(ir, info, mount_id,
                                                        {"children": remaining}))
    return "".join(rendered)


def build_runtime_port_route(ir, info, mount_id, block):
    """Arguments enter one call boundary; result ports assert no inner algebra."""
    from ..graph import Graph, Node, Parallel, Lane
    children = block.get("children", ())
    kind = block["detail"]["port_route_kind"]
    if kind == "conditional":
        return build_constructed_children_view(ir, info, mount_id, block)
    if not children:
        return ""
    nodes = []
    for child in children:
        label = child["label"]
        layout = {}
        if "argument_port" in child.get("detail", {}):
            from textwrap import wrap
            headings = label if isinstance(label, list) else [label]
            if child["kind"] == "unknown" and headings == ["Input unresolved"]:
                headings = ["Unresolved"]
            label = [line for heading in (*headings, "Port " + str(child["detail"]["argument_port"]))
                     for line in wrap(str(heading), width=14, break_on_hyphens=False)]
            layout = {"w": 164, "font": 12}
        nodes.append(Node(child["id"], child["kind"], label,
                          resolved=child.get("resolved", True), **layout))
    end = block["id"] + "__result_port"
    operator_boundary = kind in {"source_operation", "inplace_operation"}
    label = ("Operation result" if operator_boundary else "Selected value" if kind == "selection"
             else "Returned slot " + str(block["detail"].get("result_slot", [])))
    nodes.append(Node(end, "port", label, static=True))
    arguments = block["detail"].get("argument_ids", [child["id"] for child in children])
    boundary = block["detail"].get("boundary_id", end)
    if kind == "call_result" or operator_boundary:
        flow = ([boundary] if boundary != end else []) + [end]
        parallels = [Parallel(None, boundary, [Lane([argument]) for argument in arguments])] if arguments else []
    else:
        flow = list(arguments) + [end]
        parallels = []
    return render_graph(Graph(nodes, flow, parallels=parallels),
                        info, mount_id, "runtime_port_route",
                        "Source operator ports only; dispatch and mutation unresolved" if operator_boundary
                        else "Call boundary ports only; internal dependency unresolved" if kind == "call_result"
                        else "Source-proven selection from the input value",
                        facts_projected=frozenset(block.get("source_fact_keys", ())))


def build_runtime_cell_connections(ir, info, mount_id, block):
    """Only the edges supplied by the connection fact become arrows."""
    from ..graph import Graph, Node, SideInput
    calls = block["detail"]["connection_calls"]
    edges = list(dict.fromkeys((row["source"], row["target"])
                               for row in block["detail"]["connections"]))
    remaining, fragments = set(edges), []
    while remaining:
        source, target = next(edge for edge in edges if edge in remaining)
        chain = [source, target]
        remaining.remove((source, target))
        while True:
            following = [edge for edge in edges if edge in remaining and edge[0] == chain[-1]]
            if len(following) != 1 or following[0][1] in chain:
                break
            edge = following[0]
            remaining.remove(edge)
            chain.append(edge[1])
        fragments.append(chain)
    rendered = []
    for number, fragment in enumerate(fragments):
        nodes = []
        for key in fragment:
            row = calls[key]
            kind = "conv" if row["kind"] in {"conv1d", "conv2d", "conv3d"} else row["kind"]
            nodes.append(Node(row["id"], kind, row["label"], target=row["target"],
                              sub="conditional call" if row["guard"] == "conditional" else None))
        rendered.append(render_graph(
            Graph(nodes, [node.id for node in nodes]), info, f"{mount_id}_{number}",
            "runtime_cell_connections", "Source-proven local call connections",
            facts_projected=frozenset(block.get("source_fact_keys", ()))))
    arithmetic = block["detail"].get("return_arithmetic")
    if arithmetic:
        operands = arithmetic["operands"]
        children = {child["id"]: child for child in block.get("children", ())}
        merge_id = block["id"] + "__return_add"
        nodes = [Node(operand, "unknown", children[operand]["label"], resolved=False)
                 for operand in operands]
        nodes.append(Node(merge_id, "residual_add"))
        flow = [operands[0], merge_id]
        if arithmetic["scale"] == "divide":
            scale_id = block["id"] + "__return_scale"
            nodes.append(Node(scale_id, "opaque", "Divide by scale",
                              sub="value unresolved"))
            flow.append(scale_id)
        rendered.append(render_graph(
            Graph(nodes, flow, side_inputs=[SideInput(operands[1], merge_id)]),
            info, f"{mount_id}_return", "runtime_cell_return",
            "Proven return arithmetic; complete operand routes remain under investigation",
            facts_projected=frozenset(block.get("source_fact_keys", ()))))
    children = {child["id"]: child for child in block.get("children", ())}
    for number, route in enumerate(block["detail"].get("conditioning_arithmetic", ())):
        operands = route["operands"]
        nodes = [Node(operand, children[operand]["kind"], children[operand]["label"],
                      resolved=children[operand].get("resolved", True)) for operand in operands]
        nodes.append(Node(route["merge"], "residual_add"))
        rendered.append(render_graph(
            Graph(nodes, [operands[0], route["merge"]],
                  side_inputs=[SideInput(operands[1], route["merge"])]),
            info, f"{mount_id}_conditioning_{number}", "runtime_cell_conditioning",
            "Conditional operand addition; guard unresolved" if route["conditional"] else "Source-proven operand addition",
            facts_projected=frozenset(block.get("source_fact_keys", ()))))
    contained = [child for child in block.get("children", ()) if "source_instance_path" in child]
    rendered.append(build_constructed_children_view(ir, info, mount_id, {"children": contained}))
    return "".join(rendered)


def build_runtime_context_connection(ir, info, mount_id, block):
    from ..graph import Graph, Node
    detail = block["detail"]
    target = block["id"] + "__context_argument"
    graph = Graph([Node(detail["source"], "source", detail["source_label"]),
                   Node(target, "port", detail["target_formal"], static=True)],
                  [detail["source"], target])
    return (render_graph(graph, info, mount_id, "runtime_context_connection",
                         "Context input proven; query role remains under investigation",
                         facts_projected=frozenset(block.get("source_fact_keys", ())))
            + build_constructed_children_view(ir, info, mount_id, {
                "children": [child for child in block.get("children", ())
                             if "source_instance_path" in child]}))


def _box(parts, cx, y, w, h, main, shadow_id, *, resolved=True, node_id=None) -> dict:
    """A solid, clickable U-net node (name only).  No light input/output accent —
    conv-in/out are real conv ops, drawn solid like every other stage; their
    dims live on the card.  ``node_id`` wraps it as a click target coupled to its
    card."""
    fill = C["block"] if resolved else C["badge_bg"]
    main_color = C["text_block"] if resolved else C["text"]
    stroke = C["block_alt"] if resolved else C["border"]
    stroke_width = 0.6 if resolved else 1.0
    x = cx - w / 2
    children = [
        _svg_tag("rect", {
            "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
            "fill": fill, "stroke": stroke, "stroke-width": stroke_width,
            "filter": f"url(#{shadow_id})"}),
    ]
    # A list label stacks: a heading line plus a smaller second line (e.g. the
    # "2× CLIP → 2,048" origin under "Encoded text").
    lines = [s for s in main if s] if isinstance(main, (list, tuple)) else [main]
    if len(lines) == 1:
        children.append(_svg_text(cx, y + h / 2, lines[0],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_HEAD, "font-size": 17,
             "pointer-events": "none"}))
    else:
        children.append(_svg_text(cx, y + h / 2 - 9, lines[0],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_HEAD, "font-size": 16,
             "pointer-events": "none"}))
        children.append(_svg_text(cx, y + h / 2 + 11, lines[1],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_MONO, "font-size": 12,
             "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": cx, "cy": y + h / 2, "w": w, "h": h}
