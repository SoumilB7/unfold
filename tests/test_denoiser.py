"""U3-D1 — the denoiser temporal-axis reader (evidence/denoiser.py).

The first production reader on the ProgramIndex + exact-occurrence + ReaderResult
path.  Controls (Section 5): video positive / image negative; class/field/local
rename metamorphism; sibling-component non-contamination; missing forward;
unsupported expression; unrelated helper marker; and a real video/image corpus
parity control.  A missing forward or unsupported expression is a typed failure,
never False.
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.denoiser import denoiser_temporal_axis
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.reader_result import ReaderValueUnavailable


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(component_files, architectures):
    flat = []
    for group in component_files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(
        source="local", files=tuple(flat),
        component_files={k: tuple(v) for k, v in component_files.items()},
        component_architectures=dict(architectures))


def _read(tmp_path, files, architectures, component="root"):
    bundle = _bundle(files, architectures)
    index = pi.build_program_index(bundle)
    resolution = resolve_component_root(index, bundle, component)
    assert resolution.status == "resolved", resolution.status
    return denoiser_temporal_axis(index, resolution.occurrence)


# --------------------------------------------------------------------------- #
# Positive video / negative image
# --------------------------------------------------------------------------- #

def test_video_forward_with_num_frames_param_is_true(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, num_frames):
                return hidden_states
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.status == "resolved" and res.require_value() is True
    assert res.provenance and res.provenance[0].kind == "source"
    assert res.provenance[0].spans


def test_parameter_result_cites_the_exact_identifier_span(tmp_path):
    path = _write(tmp_path, "m.py", """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, num_frames):
                return hidden_states
    """)
    bundle = _bundle({"root": (path,)}, {"root": "Denoiser"})
    index = pi.build_program_index(bundle)
    resolution = resolve_component_root(index, bundle, "root")
    result = denoiser_temporal_axis(index, resolution.occurrence)
    fwd = index.callable_by_symbol(next(
        item.symbol for item in index.callables
        if item.symbol.qualified_name == "Denoiser.forward"))
    identifier = next(item for item in index.identifiers_in(fwd.symbol)
                      if item.name == "num_frames"
                      and item.context == "parameter")
    assert result.provenance[0].spans == (identifier.span,)
    assert identifier.span.end_col > identifier.span.col


def test_video_forward_with_unpacked_num_frames_is_true(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states):
                batch, num_frames, channels = hidden_states.shape
                return hidden_states.reshape(batch, num_frames, channels)
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.status == "resolved" and res.require_value() is True


def test_image_forward_without_num_frames_is_false(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, timestep):
                return self.proj(hidden_states)
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.status == "resolved" and res.require_value() is False


@pytest.mark.parametrize("body", [
    "for num_frames in range(1):\n            pass\n        return hidden_states",
    "num_frames\n        return hidden_states",
    "values = [x for num_frames in hidden_states]\n        return values",
])
def test_binding_and_bare_identifier_positions_are_observed(tmp_path, body):
    src = ("class Denoiser:\n"
           "    def __init__(self, config): pass\n"
           "    def forward(self, hidden_states):\n"
           f"        {body}\n")
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)},
                {"root": "Denoiser"})
    assert res.status == "resolved" and res.require_value() is True


# --------------------------------------------------------------------------- #
# Rename metamorphism — the result tracks the marker, not identity
# --------------------------------------------------------------------------- #

def test_class_and_field_rename_preserves_true(tmp_path):
    a = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, num_frames):
                return hidden_states
    """
    b = """
        class TotallyRenamedModel:
            def __init__(self, cfg): pass
            def forward(self, latents, num_frames):
                return latents
    """
    ra = _read(tmp_path, {"root": (_write(tmp_path, "a.py", a),)}, {"root": "Denoiser"})
    rb = _read(tmp_path, {"root": (_write(tmp_path, "b.py", b),)}, {"root": "TotallyRenamedModel"})
    assert ra.require_value() is True and rb.require_value() is True


def test_rename_that_drops_marker_flips_to_false(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, timestep):
                return hidden_states
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.require_value() is False


# --------------------------------------------------------------------------- #
# Sibling-component non-contamination
# --------------------------------------------------------------------------- #

def test_sibling_component_num_frames_does_not_influence_root(tmp_path):
    root_src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states, timestep):
                return hidden_states
    """
    vision_src = """
        class VisionTower:
            def __init__(self, config): pass
            def forward(self, hidden_states, num_frames):
                return hidden_states
    """
    files = {"root": (_write(tmp_path, "root.py", root_src),),
             "vision": (_write(tmp_path, "vision.py", vision_src),)}
    res = _read(tmp_path, files, {"root": "Denoiser", "vision": "VisionTower"}, "root")
    assert res.require_value() is False   # the vision num_frames is never consulted


# --------------------------------------------------------------------------- #
# Unrelated helper / local marker does not affect the exact forward
# --------------------------------------------------------------------------- #

def test_num_frames_in_a_helper_method_does_not_make_forward_temporal(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def _prepare(self, num_frames):
                return num_frames
            def forward(self, hidden_states):
                return self.proj(hidden_states)
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.require_value() is False


# --------------------------------------------------------------------------- #
# Missing forward and unsupported expression are typed failures, never False
# --------------------------------------------------------------------------- #

def test_missing_forward_is_failed_not_false(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config):
                self.proj = None
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.status == "failed"
    assert res.failures and res.failures[0].kind == "incomplete_graph"
    with pytest.raises(ReaderValueUnavailable):
        res.require_value()


def test_unsupported_expression_without_marker_is_failed_not_false(tmp_path):
    # a walrus in the forward normalizes to an unsupported ExprNode; with no
    # temporal name observed, absence is unprovable -> failed, never False.
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states):
                if (ready := self.check(hidden_states)):
                    return ready
                return hidden_states
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)}, {"root": "Denoiser"})
    assert res.status == "failed"
    assert res.failures and res.failures[0].kind == "unsupported_syntax"
    with pytest.raises(ReaderValueUnavailable):
        res.require_value()


def test_bare_unsupported_expression_is_failed_not_false(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states):
                (ready := hidden_states)
                return hidden_states
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)},
                {"root": "Denoiser"})
    assert res.status == "failed"
    assert res.failures[0].kind == "unsupported_syntax"


def test_nested_callable_marker_is_incomplete_not_false(tmp_path):
    src = """
        class Denoiser:
            def __init__(self, config): pass
            def forward(self, hidden_states):
                def helper(num_frames):
                    return num_frames
                return hidden_states
    """
    res = _read(tmp_path, {"root": (_write(tmp_path, "m.py", src),)},
                {"root": "Denoiser"})
    assert res.status == "failed"
    assert res.failures[0].kind == "unsupported_syntax"


def test_child_occurrence_cannot_be_stamped_with_root_forward(tmp_path):
    path = _write(tmp_path, "m.py", """
        class Child:
            def __init__(self, config): pass
            def forward(self, num_frames): return num_frames
        class Denoiser:
            def __init__(self, config):
                self.child = Child(config)
            def forward(self, hidden_states):
                return hidden_states
    """)
    bundle = _bundle({"root": (path,)}, {"root": "Denoiser"})
    index = pi.build_program_index(bundle)
    resolution = resolve_component_root(index, bundle, "root")
    assert resolution.graph.root.children
    child = resolution.graph.root.children[0].occurrence
    result = denoiser_temporal_axis(index, child)
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


# --------------------------------------------------------------------------- #
# Resolution guards: ambiguous/failed roots never reach the reader as False
# --------------------------------------------------------------------------- #

def test_two_same_spelled_roots_are_ambiguous_before_the_reader(tmp_path):
    a = _write(tmp_path, "a.py", "class Denoiser:\n    def __init__(self, config): pass\n"
                                 "    def forward(self, x): return x\n")
    b = _write(tmp_path, "b.py", "class Denoiser:\n    def __init__(self, config): pass\n"
                                 "    def forward(self, x): return x\n")
    bundle = _bundle({"root": (a, b)}, {"root": "Denoiser"})
    resolution = resolve_component_root(pi.build_program_index(bundle), bundle, "root")
    assert resolution.status == "ambiguous"   # reader is invoked only when address_resolved


def test_syntax_failed_root_is_failed_before_the_reader(tmp_path):
    broken = _write(tmp_path, "broken.py", "class Denoiser(:\n    def forward(self):\n")
    bundle = _bundle({"root": (broken,)}, {"root": "Denoiser"})
    resolution = resolve_component_root(pi.build_program_index(bundle), bundle, "root")
    assert resolution.status == "failed"      # reader never runs -> never a False from source


# --------------------------------------------------------------------------- #
# Real video/image corpus parity control
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("witness,expected", [
    ("cogvideox-5b", True),
    ("hunyuanvideo", True),
    ("ltx-video", True),
    ("mochi-1-preview", True),
    ("wan2-2-t2v-a14b-diffusers", True),
    ("flux-2-dev", False),
    ("fluxtransformer2dmodel", False),
    ("auraflow-v0-3", False),
    ("lumina-image-2-0", False),
    ("pixart-sigma-xl-2-1024-ms", False),
    ("sana-1600m-1024px-diffusers", False),
])
def test_real_corpus_video_and_image_controls(witness, expected):
    import json
    import pathlib
    import model_unfolder as mu
    from model_unfolder.parser import _coerce
    from model_unfolder.evidence.context import ParseContext

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    resolution = resolve_component_root(index, context.source_bundle, "root")
    assert resolution.status == "resolved"
    result = denoiser_temporal_axis(index, resolution.occurrence)
    assert result.status == "resolved" and result.require_value() is expected
