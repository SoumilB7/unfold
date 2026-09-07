"""Independent argument arrows must not pass through other argument cards."""
import re
from xml.etree import ElementTree

from model_unfolder.renderers.html.graph import Graph, Lane, Node, Parallel, SideInput
from model_unfolder.renderers.html.graph_engine import render_graph
from model_unfolder.renderers.html.views_diffusion import _stub_info


def _geometry(svg):
    rectangles, segments, circles = {}, [], []

    def path_points(d):
        tokens = re.findall(r"[MLQ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
        points, cursor, i = [], (0.0, 0.0), 0
        while i < len(tokens):
            command = tokens[i]
            i += 1
            if command in {"M", "L"}:
                cursor = (float(tokens[i]), float(tokens[i+1]))
                i += 2
                points.append(cursor)
            else:
                assert command == "Q", "the geometry probe must understand every rendered path command"
                control = (float(tokens[i]), float(tokens[i+1]))
                end = (float(tokens[i+2]), float(tokens[i+3]))
                i += 4
                start = cursor
                for step in range(1, 17):
                    t = step / 16
                    points.append(tuple((1-t)**2*start[axis] + 2*(1-t)*t*control[axis] + t*t*end[axis] for axis in (0, 1)))
                cursor = end
        return points

    def walk(node, offset=(0.0, 0.0), owner=None):
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "defs":
            return
        if node.get("transform"):
            transform = re.fullmatch(r"translate\(\s*([-+\d.]+)[ ,]+([-+\d.]+)\s*\)", node.get("transform"))
            assert transform, "the geometry probe must understand every coordinate transform"
            offset = tuple(offset[i] + float(transform.group(i+1)) for i in (0, 1))
        if "uf-node" in node.get("class", "").split():
            owner = node.get("data-id")
        if tag == "rect" and owner:
            x, y = float(node.get("x")) + offset[0], float(node.get("y")) + offset[1]
            rectangles[owner] = (x, y, x + float(node.get("width")), y + float(node.get("height")))
        if tag == "circle":
            circles.append(node.attrib)
        if not owner and tag in {"line", "path"}:
            if tag == "line":
                points = [(float(node.get("x1")), float(node.get("y1"))),
                          (float(node.get("x2")), float(node.get("y2")))]
            else:
                points = path_points(node.get("d"))
            points = [(x+offset[0], y+offset[1]) for x, y in points]
            segments.extend(zip(points, points[1:]))
        for child in node:
            walk(child, offset, owner)

    walk(ElementTree.fromstring(svg))
    return rectangles, segments, circles


def _enters_rectangle(segment, rectangle):
    """Intersect the segment with a slightly inset card interior."""
    start, end = segment
    lower, upper = 0.0, 1.0
    for axis in (0, 1):
        minimum, maximum = rectangle[axis] + .1, rectangle[axis+2] - .1
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-9:
            if not minimum < start[axis] < maximum:
                return False
            continue
        enter, leave = sorted(((minimum-start[axis])/delta, (maximum-start[axis])/delta))
        lower, upper = max(lower, enter), min(upper, leave)
        if lower >= upper:
            return False
    return lower < upper


def _fixture(independent):
    arguments = [f"argument_{number}" for number in range(7)]
    nodes = [Node(key, "unknown", ["Input unresolved", f"Argument {number}"],
                  h=60 + 8*(number % 3)) for number, key in enumerate(arguments)]
    nodes += [Node("call", "unknown", "Unresolved call"),
              Node("result", "port", "Returned slot [1]", static=True)]
    if independent:
        graph = Graph(nodes, ["call", "result"], parallels=[Parallel(None, "call", [Lane([key]) for key in arguments])])
    else:
        graph = Graph(nodes, [arguments[0], "call", "result"], side_inputs=[
            SideInput(key, "call", "left" if number % 2 else "right")
            for number, key in enumerate(arguments[1:])])
    return render_graph(graph, _stub_info(), "independent_inputs", "call_ports", "Argument boundary"), arguments


def test_independent_incoming_lanes_have_no_card_crossings_or_fake_split():
    svg, arguments = _fixture(True)
    rectangles, segments, circles = _geometry(svg)
    assert set(rectangles) == {*arguments, "call"}
    assert not circles, "independent arguments must not acquire a common split dot"
    assert not [(owner, segment) for owner, rectangle in rectangles.items()
                for segment in segments if _enters_rectangle(segment, rectangle)]
    for number in range(7):
        assert f"Argument {number}" in svg
    assert "Returned slot [1]" in svg


def test_geometry_probe_detects_the_original_side_input_crossing():
    svg, _ = _fixture(False)
    rectangles, segments, _ = _geometry(svg)
    assert any(_enters_rectangle(segment, rectangle)
               for rectangle in rectangles.values() for segment in segments)


def test_call_port_view_preserves_seven_named_inputs_in_compact_lanes():
    from model_unfolder.renderers.html.block_views.unet import build_runtime_port_route
    ports = ["0", "1", "2", "s1", "s2", "b1", "b2"]
    arguments = [{"id": "arg_" + port, "kind": "unknown", "label": "Input unresolved",
                  "detail": {"argument_port": port}} for port in ports]
    block = {"id": "route", "children": [*arguments, {"id": "call", "kind": "unknown", "label": "Unresolved call"}],
             "detail": {"port_route_kind": "call_result", "argument_ids": [row["id"] for row in arguments],
                        "boundary_id": "call", "result_slot": [1]}}
    svg = build_runtime_port_route({}, _stub_info(), "compact_inputs", block)
    rectangles, segments, circles = _geometry(svg)
    assert set(rectangles) == {*(row["id"] for row in arguments), "call"}
    assert not circles
    assert not any(_enters_rectangle(segment, rectangle)
                   for rectangle in rectangles.values() for segment in segments)
    assert float(ElementTree.fromstring(svg).get("viewBox").split()[2]) < 1600
    for port in ports:
        assert f"Port {port}" in svg
