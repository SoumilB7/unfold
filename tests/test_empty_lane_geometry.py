"""Direct parallel edges retain their meaning while clearing intervening boxes."""
import re
import xml.etree.ElementTree as ET

from model_unfolder.renderers.html.graph import Lane, Node, Parallel
from model_unfolder.renderers.html.graph_engine import _draw_parallel, _geom


def _draw(lanes, *, extra=None):
    geometry = {
        "source": _geom(0, 650, 180, 50),
        "near": _geom(0, 440, 220, 50),
        "wide": _geom(0, 260, 520, 50),
        "far": _geom(0, 100, 200, 50),
        # A wide box outside the bypass interval must not move its rail.
        "outside": _geom(0, 750, 1800, 50),
    }
    geometry.update(extra or {})
    nodes = {key: Node(key, "linear") for key in geometry}
    parts, regions = [], []
    _draw_parallel(parts, regions, {}, "shadow", "arrow",
                   Parallel("source", "near", lanes), nodes, geometry, 0)
    return ET.fromstring("<svg>" + "".join(parts) + "</svg>"), regions, geometry


def _arrow_paths(svg):
    return [tuple(map(float, re.findall(r"-?\d+(?:\.\d+)?", p.attrib["d"])))
            for p in svg.iter("path") if "marker-end" in p.attrib]


def test_empty_bypass_clears_widest_intermediate_box_on_both_sides():
    svg, regions, geometry = _draw([
        Lane([], dst=["far"]), Lane([]),
        Lane([], src="source", dst=["far"]),
    ])
    paths = _arrow_paths(svg)
    bypasses = [p for p in paths if p[-1] < geometry["wide"]["top"]]
    assert len(bypasses) == 2
    assert {p[0] for p in bypasses} == {-296, 296}
    for p in bypasses:
        # Every rectangle intersecting the actual vertical rail is cleared.
        for key in ("near", "wide"):
            g = geometry[key]
            assert p[0] < g["left"] or p[0] > g["right"]
        assert any(r["left"] <= p[0] <= r["right"] and
                   r["top"] <= p[-1] <= r["bottom"] for r in regions)


def test_explicit_same_source_has_one_connected_stem_and_both_edges():
    svg, _, geometry = _draw([
        Lane([], src="source"), Lane([], src="source", dst=["far"]),
    ])
    stems = [line for line in svg.iter("line")
             if float(line.attrib["x1"]) == float(line.attrib["x2"]) == 0
             and float(line.attrib["y1"]) == geometry["source"]["top"]
             and float(line.attrib["y2"]) == geometry["source"]["top"] - 16]
    assert len(stems) == 1
    assert len(_arrow_paths(svg)) == 2


def test_lower_destination_is_an_obstacle_for_the_same_lanes_higher_edge():
    svg, _, _ = _draw([Lane([], dst=["near", "far"])],
                      extra={"near": _geom(0, 440, 700, 50)})
    paths = _arrow_paths(svg)
    assert len(paths) == 2
    assert {p[0] for p in paths} == {386}


def test_adjacent_empty_lane_keeps_original_position():
    svg, _, _ = _draw([Lane([]), Lane([])])
    assert {p[0] for p in _arrow_paths(svg)} == {-78, 78}


def test_unknown_nonempty_lane_is_not_invented_as_a_direct_edge():
    svg, _, _ = _draw([Lane(["missing-operation"])])
    assert _arrow_paths(svg) == []
