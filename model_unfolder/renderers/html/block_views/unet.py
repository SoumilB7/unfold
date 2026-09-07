"""UNet detail views over canonical source-backed facts and constructed cards.

The production overview connects source-port boundary summaries only where
previous-state transport is proven. Constructed stages remain visible inside
containment frames; their layout does not assert individual execution order.
Skip accumulation and context links retain their separately proven endpoints.
Unbound bookends remain explicit containment cards, and boundary drills expose
local source-port details and unresolved call/guard meaning.

The later legacy UNet view builders remain for the explicit old-path
comparison. They are not the production overview's topology authority.
"""
from __future__ import annotations

from ....block_schema import DIFFUSION_PART_KINDS
from ..graph_engine import render_graph
from ..stack_view import fit_svg, point
from ..svg import _ids, _path, _svg_tag, _svg_text, _v_seg
from ..theme import C, FONT_HEAD, FONT_MONO
from ..tower import tower_graph


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
                       "stroke-dasharray": "5 4", "data-region-kind": "containment"}))
        frames.append(_svg_text(x + width / 2, y + height - 12, label,
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

        def region(row, x, y):
            child = cards[row["id"]]
            heading = {"if": "Conditional boundary", "for": "Repeat boundary",
                       "while": "Repeat boundary"}.get(row["kind"], "Operation boundary")
            detail = str(child["label"]).partition(": ")[2] or "Source ports · drill for details"
            if row.get("primary_target_id"):
                heading, detail = child["label"], "Conditional source-bound invocation"
            geometry = place(row["id"], x, y, 280, [heading, detail], resolved=False)
            if row.get("primary_target_id"):
                placed[row["primary_target_id"]] = geometry
            stage_ids = list(row["stage_block_ids"])
            if up_ids.intersection(stage_ids):
                stage_ids.reverse()
            invocation_ids = [target for target in row.get("extra_target_ids", ()) if target not in placed]
            visible_ids = list(dict.fromkeys((*stage_ids, *invocation_ids)))
            for number, stage_id in enumerate(visible_ids):
                place(stage_id, x, y + 90 + number * 86)
            height = 58 if not visible_ids else 110 + len(visible_ids) * 86
            if visible_ids:
                bound = set(row.get("binding_target_ids", ()))
                containment(x - 155, y - 15, 310, height + 30,
                            "Conditional call targets · dashed links" if set(visible_ids) <= bound else
                            "Constructed modules · open targets retained")
                # These are target-identity relations, not tensor arrows or
                # execution order between slots. Actual input/result ports
                # remain in the selected source-bound call drill.
                for target_id in visible_ids:
                    if target_id not in bound:
                        continue
                    target = placed[target_id]
                    rail = x - 171
                    path = f"M {geometry['left']} {geometry['cy']} L {rail} {geometry['cy']} L {rail} {target['cy']} L {target['left']} {target['cy']}"
                    wires.append(_svg_tag("path", {"d": path, "fill": "none", "stroke": C["border"],
                        "stroke-width": 1.5, "stroke-dasharray": "5 3",
                        "data-route-kind": "conditional_call_target", "data-source": row["id"],
                        "data-target": target_id}))
                    regions.append(point(rail, target["cy"]))
            if not row["receives_previous_state"]:
                parts.append(_svg_text(x, y - 24, "Input link unresolved",
                             {"text-anchor": "middle", "font-family": FONT_MONO,
                              "font-size": 11, "fill": C["text"]}))
                regions.append(point(x, y - 35))
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
            if source["cx"] == target["cx"] and source["cy"] < target["cy"] and not previous["stage_block_ids"]:
                path = f"M {source['cx']} {source['bottom']} L {target['cx']} {target['top'] - 5}"
            elif source["cx"] == target["cx"] and source["cy"] > target["cy"] and not current["stage_block_ids"]:
                path = f"M {source['cx']} {source['top']} L {target['cx']} {target['bottom'] + 5}"
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
    arrow_id, shadow_id = _ids(mount_id, "constructed_children")
    parts, regions = [], []
    for number, child in enumerate(block.get("children", ())):
        regions.append(_box(parts, 145 + number % 3 * 270,
                            35 + number // 3 * 100, 235, 64,
                            child["label"], shadow_id,
                            node_id=child["id"]))
    return fit_svg(arrow_id, shadow_id, parts, regions,
                   "Constructed children; containment only", min_width=720, pad=40)


def build_runtime_ffn_view(ir, info, mount_id, block):
    """Reuse the canonical FFN graph and keep its actual children inspectable."""
    from .feed_forward import build_ffn_view
    constructed = [child for child in block.get("children", ())
                   if "source_instance_path" in child]
    return (build_ffn_view(ir, info, mount_id, block)
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
            + build_constructed_children_view(ir, info, mount_id, block))


def _text_source_label(ir: dict):
    """Label for the 'encoded text' source — makes the two-CLIP origin visible.

    The text the U-net cross-attends is a SINGLE tensor: the text encoders' token
    features concatenated along the feature axis (SDXL: 768 + 1,280 = 2,048).  So
    one box is correct, but the label shows it is the concatenation of the N
    encoders, answering 'where did the second CLIP go?'."""
    extras = ir.get("extras") or {}
    diff = extras.get("diffusion") or {}
    encs = diff.get("text_encoders") or []
    cond = diff.get("conditioning") or {}
    modality = cond.get("kv_modality")
    # F1: when the declared cross-attention K/V is NOT text (image_proj / hint /
    # unknown), the source label is the resolved modality's own — never "Encoded
    # text" for a component the pipeline conditions on differently.
    if modality and modality != "text":
        return cond.get("kv_label") or "External conditioning"
    # The cross-attention width is a dim — it belongs on a card/chip, never on the
    # block label (design policy). The label shows only the structural origin: the
    # concatenation of N text encoders ("where did the second CLIP go?").
    if len(encs) >= 2:
        fam = "CLIP" if all("CLIP" in str(e) for e in encs) else "encoders"
        return ["Encoded text", f"{len(encs)}× {fam} (concat)"]
    return cond.get("kv_label") or "Encoded text"


def build_unet_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    u = (ir.get("extras") or {}).get("unet") or {}
    down = u.get("down") or []
    up = u.get("up") or []
    mid = u.get("mid") or {}
    n = len(down)

    arrow_id, shadow_id = _ids(mount_id, "unet")
    parts: list[str] = []
    regions: list[dict] = []
    LX, RX = 188.0, 588.0
    sw, sh, row_gap, y0 = 200.0, 58.0, 120.0, 92.0
    MERGE_DROP = 26.0                 # the concat connector sits this far below an up stage
    DOWN_DROP = sh / 2 + MERGE_DROP   # each down stage sits at its skip's entry level, so
                                      # the skip runs STRAIGHT into the connector (no elbow)

    def stage(cx: float, y: float, st: dict, default_kind: str) -> dict:
        # Stage NAME only on the box; channels / ResNet / attention counts are
        # fact chips on the stage's card (numbers never ride the diagram block).
        resolved = st.get("diffusion_part_kind") in DIFFUSION_PART_KINDS
        return _box(parts, cx, y, sw, sh, _stage_title(st, default_kind), shadow_id,
                    resolved=resolved, node_id=st.get("id"))

    conv_in = _box(parts, LX, 22, sw, 44, "Conv in", shadow_id, node_id="unet_conv_in")
    conv_out = _box(parts, RX, 22, sw, 44, "Conv out", shadow_id, node_id="unet_conv_out")
    regions += [conv_in, conv_out]

    # Right column = up stages; left column row i = the matching down stage, sat
    # DOWN_DROP lower so its right-edge centre is level with the up stage's concat
    # connector — that makes every skip a straight horizontal arrow.
    down_g, up_g = [], []
    for i in range(n):
        y = y0 + i * row_gap
        up_g.append(stage(RX, y, up[n - 1 - i], "up_stage"))
        down_g.append(stage(LX, y + DOWN_DROP, down[i], "down_stage"))
    regions += down_g + up_g

    # F2: the mid (bottleneck) box is drawn ONLY when the denoiser class
    # constructs one — a Kandinsky3-shape conv-U (conv_in -> down -> up -> conv_out,
    # no bottleneck) draws NO mid, and the deepest down stage connects straight
    # across to the deepest up stage.
    has_mid = bool(mid)
    mid_cx = (LX + RX) / 2
    if has_mid:
        mid_y = down_g[n - 1]["bottom"] + 26
        mid_g = _box(parts, mid_cx, mid_y, sw + 44, sh, _stage_title(mid, "mid_stage"), shadow_id,
                     resolved=mid.get("diffusion_part_kind") in DIFFUSION_PART_KINDS,
                     node_id="unet_mid")
        regions.append(mid_g)
        cross_y = mid_g["cy"]
    else:
        mid_g = None
        cross_y = down_g[n - 1]["bottom"] + 26

    # --- DOWN path: conv_in -> down stages -> mid (the resolution change per
    # stage lives on the cards + the denoiser description, not on the diagram). ---
    parts.append(_v_seg(LX, conv_in["bottom"], down_g[0]["top"] - 6, arrow_id))
    for i in range(n - 1):
        parts.append(_v_seg(LX, down_g[i]["bottom"], down_g[i + 1]["top"] - 6, arrow_id))
    if has_mid:
        parts.append(_path(f"M {LX} {down_g[n-1]['bottom']} L {LX} {cross_y} L {mid_g['left'] - 6} {cross_y}", arrow_id))

    # --- UP path: at EACH up stage, the skip from the matching down stage and the
    # features coming up from below MERGE at a concat connector (‖) just below the
    # stage, then go on up into it.  UNet skips concatenate along the channel axis
    # — two arrows in, one out; solid lines, ⊕ reserved for addition. ---
    if has_mid:
        parts.append(_path(  # mid -> below the bottom up stage (into its concat)
            f"M {mid_g['right']} {cross_y} L {RX} {cross_y} "
            f"L {RX} {up_g[n-1]['bottom'] + MERGE_DROP + 11}", arrow_id))
    else:
        # no bottleneck: deepest down stage flows straight across to the deepest
        # up stage's concat entry (the U's floor is a plain crossover).
        parts.append(_path(
            f"M {LX} {down_g[n-1]['bottom']} L {LX} {cross_y} L {RX} {cross_y} "
            f"L {RX} {up_g[n-1]['bottom'] + MERGE_DROP + 11}", arrow_id))
    for i in range(n):
        b = up_g[i]
        my = b["bottom"] + MERGE_DROP              # == down_g[i] centre → straight skip
        if i < n - 1:                              # features coming up from below (straight)
            parts.append(_v_seg(RX, up_g[i + 1]["top"], my + 11, arrow_id))
        parts.append(_svg_tag("line", {           # the skip, straight in from the left
            "x1": down_g[i]["right"], "y1": my, "x2": RX - 11, "y2": my,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
        _concat_node(parts, RX, my)
        parts.append(_v_seg(RX, my - 11, b["bottom"] + 4, arrow_id))   # ‖ -> up stage (straight)
    parts.append(_v_seg(RX, up_g[0]["top"], conv_out["bottom"] + 6, arrow_id))

    # --- TEXT CONDITIONING: the encoded prompt (the text encoders' K/V) enters
    # EVERY cross-attention stage. Drawn as a source at the bottom that broadcasts
    # laterally into each CrossAttn stage from its outer edge — latent flows
    # vertically through the U, conditioning enters from the side. ---
    _draw_text_conditioning(parts, regions, shadow_id, arrow_id, u,
                            down, up, mid, down_g, up_g, mid_g, LX, RX, sw,
                            text_label=_text_source_label(ir))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} U-net denoiser", min_width=720, pad=44)


def _draw_text_conditioning(parts, regions, shadow_id, arrow_id, u, down, up,
                            mid, down_g, up_g, mid_g, LX, RX, sw,
                            text_label="Encoded text") -> None:
    """Show the encoded text feeding the cross-attention stages.

    Only the stages whose block type carries cross-attention receive it (down[i]
    / up[n-1-i] with ``attn``, and the mid block).  A source box at the bottom
    fans out: straight up into the bottleneck, and out to a left/right lateral bus
    that taps each cross-attn stage on its OUTER edge.  Skipped entirely for an
    unconditional U-net (no ``cross_attention_dim`` / no attention stages)."""
    cad = u.get("cross_attention_dim")
    n = len(down)
    down_attn = [g for i, g in enumerate(down_g) if down[i].get("attn")]
    up_attn = [g for i, g in enumerate(up_g) if up[n - 1 - i].get("attn")]
    mid_attn = bool(mid.get("attn")) and mid_g is not None
    if not cad or not (down_attn or up_attn or mid_attn):
        return

    mid_cx = (LX + RX) / 2
    tb_h = 56.0 if isinstance(text_label, (list, tuple)) else 48.0
    # Anchor the conditioning source below the bottleneck; with no mid block the
    # floor of the U is the deepest down/up row.
    floor = (mid_g["bottom"] if mid_g is not None
             else max(down_g[n - 1]["bottom"], up_g[n - 1]["bottom"]))
    tb = _box(parts, mid_cx, floor + 44, 252.0, tb_h, text_label,
              shadow_id, node_id="unet_text_cond")
    regions.append(tb)
    cy = tb["cy"]
    x_L = (LX - sw / 2) - 34          # lane left of the down column
    x_R = (RX + sw / 2) + 34          # lane right of the up column

    def tap(x_from: float, into_left_edge: bool, g: dict) -> None:
        edge = (g["left"] - 4) if into_left_edge else (g["right"] + 4)
        parts.append(_svg_tag("line", {
            "x1": x_from, "y1": g["cy"], "x2": edge, "y2": g["cy"],
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))

    if mid_attn:                                   # straight up into the bottleneck
        parts.append(_svg_tag("line", {
            "x1": mid_cx, "y1": tb["top"], "x2": mid_cx, "y2": mid_g["bottom"] + 5,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    if down_attn:                                  # left bus + taps into down stages
        top_y = min(g["cy"] for g in down_attn)
        parts.append(_svg_tag("path", {
            "d": f"M {tb['left']} {cy} L {x_L} {cy} L {x_L} {top_y}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round"}))
        for g in down_attn:
            tap(x_L, True, g)
        regions.append(point(x_L - 6, cy))
    if up_attn:                                    # right bus + taps into up stages
        top_y = min(g["cy"] for g in up_attn)
        parts.append(_svg_tag("path", {
            "d": f"M {tb['right']} {cy} L {x_R} {cy} L {x_R} {top_y}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round"}))
        for g in up_attn:
            tap(x_R, False, g)
        regions.append(point(x_R + 6, cy))


def build_unet_stage_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """A resolution stage: in → [ResNet block (+ Transformer block when the stage
    has cross-attention)] × layers_per_block → optional 2× resample → out.  The
    ResNet block and Transformer block are clickable and DRILL FURTHER (into the
    residual cell, and into self→cross→FF × depth respectively).

    Mid block special case (direction=None): UNetMidBlock2DCrossAttn forward is
    ResNet₀ → Transformer → ResNet₁ — a sandwich, not a paired repeat.  Drawn as
    three sequential ``pre`` blocks, no ``× N`` frame."""
    d = block.get("detail") or {}
    ch = d.get("channels")
    resnets = int(d.get("resnets") or 1)
    direction = d.get("direction")
    op = {"w": 224, "h": 48}
    # Node ids are scoped by the stage id so each stage drills into ITS OWN
    # resnet / transformer cards (matching _unet_stage_children), not the first
    # stage's deduped card.
    sid = block.get("id") or "unet_stage"
    # F2: only a source-proven Transformer2D stage draws the "Transformer block";
    # SimpleCrossAttn / code-defined / unresolved stages draw a plain attention
    # cell (its card id is {sid}__crossattn), NEVER a fabricated Transformer2D.
    akind = d.get("attn_kind")
    tf2d = akind == "transformer2d"
    attn_id = f"{sid}__transformer" if tf2d else f"{sid}__crossattn"
    attn_label = ("Transformer block" if tf2d
                  else "Attention" if akind == "code_defined"
                  else "Cross-attention")

    if direction is None and d.get("attn"):
        # Mid block: ResNet₀ → [attn] → ResNet₁ (UNetMidBlock2D*.forward)
        # Shown as a sequential ``pre`` chain — no repeat frame, no ×N badge.
        spec = {
            "source": {"id": "unet_stage_in", "label": f"in ({ch:,} ch)" if ch else "in"},
            "pre": [
                {"id": f"{sid}__resnet_pre", "kind": "norm", "label": "ResNet block", **op},
                {"id": attn_id, "kind": "attention", "label": attn_label, **op},
                {"id": f"{sid}__resnet_post", "kind": "norm", "label": "ResNet block", **op},
            ],
            "output": {"id": "unet_stage_out"},
            "side_inputs": [{
                "node": {"id": "unet_stage_text", "kind": "embedding",
                         "label": _text_source_label(ir), "w": 210},
                "target": attn_id,
            }],
        }
        graph = tower_graph(spec)
        return render_graph(graph, info, mount_id, f"unetstage_{sid}",
                            f"{ir.get('name', 'model')} {block.get('title') or 'U-net mid stage'}",
                            min_width=560)

    # Block NAME only — the per-block facts (transformer depth, self/cross/FF,
    # stride-2 conv) are chips/prose on each block's card, never a sub-caption on
    # the diagram block.
    cell = [{"id": f"{sid}__resnet", "kind": "norm", "label": "ResNet block", **op}]
    if d.get("attn"):
        cell.append({"id": attn_id, "kind": "attention", "label": attn_label, **op})
    post = ([{"id": f"{sid}__{'down' if direction == 'down' else 'up'}sample",
              "kind": "embedding",
              "label": "Downsample" if direction == "down" else "Upsample",
              "w": 224, "h": 46}]
            if d.get("sample") else [])

    spec = {
        "source": {"id": "unet_stage_in", "label": f"in ({ch:,} ch)" if ch else "in"},
        "cell": cell,
        "repeat": resnets,
        "post": post,
        "output": {"id": "unet_stage_out"},
    }
    if d.get("attn"):
        # The conditioning enters the attention cell (its cross-attention) — show
        # it feeding in from the side, the same conditioning seen one level up.
        spec["side_inputs"] = [{
            "node": {"id": "unet_stage_text", "kind": "embedding",
                     "label": _text_source_label(ir), "w": 210},
            "target": attn_id,
        }]
    graph = tower_graph(spec)
    return render_graph(graph, info, mount_id, f"unetstage_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} {block.get('title') or 'U-net stage'}",
                        min_width=560)


def build_unet_resnet_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One ResNet block — the actual ResnetBlock2D.forward() cell:
    in → GroupNorm+activation → Conv 3×3 → ⊕ timestep emb →
    GroupNorm+activation → Conv 3×3 → ⊕ → out.

    The residual bypass (shortcut) goes around the ENTIRE cell from the raw block
    input — not from norm1.  The timestep embedding is injected between conv1 and
    norm2 (the UNet's conditioning mechanism, as distinct from DiT/AdaLN).
    The activation label is projected from the canonical child card; this renderer
    never supplies a SiLU/GELU mechanism default."""
    d = block.get("detail") or {}
    ch = d.get("channels")
    temporal = bool(d.get("temporal"))
    child_by_id = {
        child.get("id"): child
        for child in (block.get("children") or [])
        if isinstance(child, dict)
    }
    norm_activation = str(
        (child_by_id.get("unet_op_norm1") or {}).get("title")
        or "GroupNorm + Activation"
    )
    op = {"w": 216, "h": 44}
    # A ResNet block is ONE residual cell, not a repeated stack (the per-stage
    # repeat = layers_per_block is shown one level up, on the stage). So its ops
    # are ``pre`` (a plain chain + residual loop), never a "× N" repeat-frame.
    # residual_from "unet_res_in" (the block's input port): the shortcut bypasses
    # norm1 + activation + conv1 + temb_inject + norm2 + activation + conv2.
    # F3: a spatio-temporal block appends the temporal 1-D-conv branch + AlphaBlender.
    pre = [
        {"id": "unet_op_norm1", "kind": "norm",
         "label": norm_activation, **op},
        {"id": "unet_op_conv1", "kind": "embedding", "label": "Conv 3×3", **op},
        {"id": "unet_op_temb", "kind": "residual_add", "label": "⊕ Timestep emb"},
        {"id": "unet_op_norm2", "kind": "norm",
         "label": norm_activation, **op},
        {"id": "unet_op_conv2", "kind": "embedding", "label": "Conv 3×3", **op},
        {"id": "unet_op_residual", "kind": "residual_add", "residual_from": "unet_res_in"},
    ]
    if temporal:
        pre += [
            {"id": "unet_op_temporal", "kind": "embedding", "label": "Temporal conv",
             "sub": "1-D over frames", **op},
            {"id": "unet_op_alphablend", "kind": "residual_add", "label": "α AlphaBlender"},
        ]
    graph = tower_graph({
        "source": {"id": "unet_res_in", "label": f"in ({ch:,} ch)" if ch else "in"},
        "pre": pre,
        "output": {"id": "unet_res_out"},
        # The projected timestep embedding enters beside the injection node.
        "side_inputs": [{
            "node": {"id": "unet_res_temb", "kind": "source",
                     "label": "Timestep emb", "w": 148},
            "target": "unet_op_temb",
        }],
    })
    return render_graph(graph, info, mount_id, f"unetres_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} ResNet block", min_width=540)


def build_unet_transformer_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One Transformer2D block: in → [Self-attention → Cross-attention (text) →
    Feed-forward] × transformer_layers → out.  Each sub-block opens the canonical
    attention / feed-forward view (the same opener a transformer layer uses)."""
    d = block.get("detail") or {}
    t = int(d.get("transformers") or 1)
    temporal = bool(d.get("temporal"))
    # match the scoped sub-block ids built in _unet_transformer_subblocks
    sid = d.get("prefix") or block.get("id") or "unet"
    op = {"w": 232, "h": 46}
    cell = [
        {"id": f"{sid}__selfattn", "kind": "attention", "label": "Self-attention", **op},
        {"id": f"{sid}__crossattn", "kind": "attention", "label": "Cross-attention (text)", **op},
        {"id": f"{sid}__ff", "kind": "ffn", "label": "Feed-forward", **op},
    ]
    if temporal:
        # F3: TransformerSpatioTemporalModel — a temporal transformer + AlphaBlender.
        cell += [
            {"id": f"{sid}__temporal_tf", "kind": "attention", "label": "Temporal transformer",
             "sub": "attention over frames", **op},
            {"id": f"{sid}__temporal_blend", "kind": "residual_add", "label": "α AlphaBlender"},
        ]
    graph = tower_graph({
        "source": {"id": "unet_tf_in", "label": "in (latent tokens)"},
        "cell": cell,
        "repeat": t,
        "output": {"id": "unet_tf_out"},
        # The encoded text enters beside the cross-attention sub-block — it supplies
        # that sublayer's K/V (self-attention and FF stay on the latent).
        "side_inputs": [{
            "node": {"id": "unet_tf_text", "kind": "embedding",
                     "label": _text_source_label(ir), "w": 210},
            "target": f"{sid}__crossattn",
        }],
    })
    return render_graph(graph, info, mount_id, f"unettf_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} Transformer block", min_width=540)


def build_encoded_text_concat_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """How the text encoders combine into the cross-attention K/V.

    Each encoder produces a token sequence at its own width; SDXL concatenates the
    two CLIP penultimate hidden states along the feature axis (768 + 1,280 =
    2,048) — the single tensor the U-net cross-attends.  Drawn as N encoder boxes
    feeding one ``‖`` concat connector (the SAME op as the U-net skips) → K/V.  A
    single-encoder model (SD1.5) draws a straight pass-through (no concat).  A
    declared ``encoder_hid_dim`` bridge (Kolors: ChatGLM 4,096-d → 2,048-d)
    draws its projection box between the lane(s) and the K/V — a width change
    is an OP, never a bare arrow."""
    d = block.get("detail") or {}
    encoders = d.get("encoders") or []
    cad = d.get("cross_attention_dim")
    proj = d.get("projection") if isinstance(d.get("projection"), dict) else None
    arrow_id, shadow_id = _ids(mount_id, "txtconcat")
    parts: list[str] = []
    regions: list[dict] = []

    bw, bh, gap = 168.0, 58.0, 52.0
    n = max(len(encoders), 1)
    y_enc, y_concat, y_out = 196.0, 96.0, 36.0
    if proj:                       # projection row: everything below shifts down
        y_proj = 100.0
        # A lone encoder tucks in under the projection; a concat lane keeps the
        # full extra row (‖ sits where the projection's feed arrives).
        y_enc += 96.0 if len(encoders) >= 2 else 26.0
        y_concat += 96.0
    total = n * bw + (n - 1) * gap
    x0 = -total / 2 + bw / 2

    enc_g: list[dict] = []
    for i in range(n):
        e = encoders[i] if i < len(encoders) else {"name": "Text encoder"}
        cxn = x0 + i * (bw + gap)
        hid = e.get("hidden")
        name = str(e.get("name") or "Text encoder")
        label = [name, f"{hid:,}-d"] if hid else name
        g = _box(parts, cxn, y_enc, bw, bh, label, shadow_id)
        enc_g.append(g)
        regions.append(g)

    # Whatever the lane(s) resolve to feeds either the projection (when the
    # config declares one) or the K/V directly.  _box is TOP-anchored, so the
    # box spans y_proj..y_proj+44; +10 keeps the arrowhead clear of its bottom
    # edge AND the clickable highlight ring.
    y_next = (y_proj + 44.0 + 10.0) if proj else y_out
    if len(encoders) >= 2:
        # the ‖ concat connector is clickable — opens a card explaining the op
        _concat_node(parts, 0.0, y_concat, node_id="text_concat_op")
        regions.append(point(0.0, y_concat))
        for g in enc_g:                              # each encoder → the ‖ connector
            if abs(g["cx"]) < 1:
                parts.append(_v_seg(0.0, g["top"], y_concat + 11, arrow_id))
            else:
                side = -11 if g["cx"] < 0 else 11
                parts.append(_path(
                    f"M {g['cx']} {g['top']} L {g['cx']} {y_concat} L {side} {y_concat}",
                    arrow_id))
        parts.append(_v_seg(0.0, y_concat - 11, y_next, arrow_id))
    else:                                            # single encoder: straight up
        parts.append(_v_seg(0.0, enc_g[0]["top"], y_next, arrow_id))

    if proj:
        # encoder_hid_proj: text_proj IS one nn.Linear; other declared types are
        # code-defined modules — labelled as the bare op, dims live on the card.
        pg = _box(parts, 0.0, y_proj, bw, 44.0,
                  "Linear" if proj.get("type") == "text_proj" else "Projection",
                  shadow_id, node_id="text_proj_op")
        regions.append(pg)
        parts.append(_v_seg(0.0, pg["top"], y_out, arrow_id))

    out_label = f"K/V ({cad:,})" if cad else "cross-attention K/V"
    parts.append(_svg_text(0.0, y_out - 10, out_label, {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO,
        "font-size": 12}))
    regions.append(point(0.0, y_out - 18))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} encoded text → K/V", min_width=520, pad=44)


def _stage_title(st: dict, default_kind: str) -> str:
    if st.get("custom_label"):
        return str(st["custom_label"])
    kind = st.get("diffusion_part_kind") or default_kind
    return {
        "down_stage": "Down stage",
        "mid_stage": "Mid stage",
        "up_stage": "Up stage",
    }.get(kind, "Stage")


def _concat_node(parts: list[str], cx: float, cy: float, node_id: str | None = None) -> None:
    """A concatenation connector (‖) where lanes join along the feature axis.  Two
    parallel bars — deliberately NOT a '+': concatenation, and ⊕ is reserved
    across the package for addition.  ``node_id`` makes it a clickable drill
    target coupled to a card explaining the op."""
    children = [_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": 11, "fill": C["block"],
        "stroke": C["block_alt"], "stroke-width": 0.6})]
    for dx in (-3.0, 3.0):
        children.append(_svg_tag("line", {
            "x1": cx + dx, "y1": cy - 5, "x2": cx + dx, "y2": cy + 5,
            "stroke": C["text_block"], "stroke-width": 2,
            "stroke-linecap": "round", "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)


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
