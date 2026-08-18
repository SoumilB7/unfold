"""U10-A exact diffusion-root topology boundary.

The poisons attack the intended seams: identity/config cannot select shape,
container addresses cannot imply execution, and a U-shape requires an exact
producer -> accumulator -> derived bypass -> later consumer route.
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import (
    DiffusionRootTopology,
    RepeatedRootStage,
    read_diffusion_root_topology,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(files, architecture="Root"):
    return SourceBundle(
        source="test", files=tuple(files), architecture=architecture,
        component_files={"root": tuple(files)},
        component_architectures={"root": architecture})


def _read(tmp_path, source, *, architecture="Root", extra_files=()):
    path = _write(tmp_path, "model.py", source)
    bundle = _bundle((path, *extra_files), architecture)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    return read_diffusion_root_topology(index, root), root, index


REPEATED = """
    from torch.nn import ModuleList
    class Cell:
        def forward(self, value): return value
    class Root:
        def __init__(self, config):
            self.items = ModuleList([Cell() for _ in range(config.depth)])
        def forward(self, value):
            for item in self.items:
                value = item(value)
            return value
"""


U_SHAPED = """
    from torch.nn import ModuleList
    class Cell:
        def forward(self, value, side=None): return value, (value,)
    class Root:
        def __init__(self, config):
            self.alpha = ModuleList([Cell() for _ in range(config.depth)])
            self.omega = ModuleList([Cell() for _ in range(config.depth)])
        def forward(self, value):
            saved = (value,)
            for first in self.alpha:
                value, branch = first(value)
                saved += branch
            for second in self.omega:
                side = saved[-1:]
                value = second(value, side)
            return value
"""


def test_exact_repeated_container_invocation_proves_repeated_stack(tmp_path):
    result, _root, _index = _read(tmp_path, REPEATED)
    assert result.status == "incomplete"
    assert result.require_value().kind == "repeated_stack"
    assert len(result.require_value().stages) == 1
    assert result.provenance[0].kind == "source"


def test_exact_bypass_route_proves_u_shape(tmp_path):
    result, _root, _index = _read(tmp_path, U_SHAPED)
    assert result.status == "incomplete"
    topology = result.require_value()
    assert topology.kind == "u_shaped"
    assert len(topology.stages) == 2
    assert topology.skip_route is not None
    assert topology.skip_route.bypass_output == "branch"
    assert topology.skip_route.derived_value == "side"


def test_class_named_transformer_with_u_shape_stays_u_shaped(tmp_path):
    source = U_SHAPED.replace("class Root:", "class Transformer:")
    result, _root, _index = _read(
        tmp_path, source, architecture="Transformer")
    assert result.require_value().kind == "u_shaped"


def test_class_named_unet_with_one_stack_stays_repeated_stack(tmp_path):
    source = REPEATED.replace("class Root:", "class UNet:")
    result, _root, _index = _read(tmp_path, source, architecture="UNet")
    assert result.require_value().kind == "repeated_stack"


def test_down_up_spellings_without_bypass_do_not_create_u_shape(tmp_path):
    source = REPEATED.replace("items", "down_blocks").replace(
        "class Root:", "class UNet:")
    result, _root, _index = _read(tmp_path, source, architecture="UNet")
    assert result.require_value().kind == "repeated_stack"


def test_complete_class_field_and_local_rename_preserves_shape(tmp_path):
    renamed = (U_SHAPED.replace("Root", "Obscure")
               .replace("Cell", "Unit")
               .replace("alpha", "left_store")
               .replace("omega", "right_store")
               .replace("first", "one")
               .replace("second", "two")
               .replace("saved", "memory")
               .replace("branch", "detour")
               .replace("side", "later_input")
               .replace("value", "tensor"))
    result, _root, _index = _read(
        tmp_path, renamed, architecture="Obscure")
    assert result.require_value().kind == "u_shaped"


def test_same_declared_config_path_cannot_change_source_topology(tmp_path):
    # Both snippets read config.depth.  The source execution route alone decides.
    repeated, _, _ = _read(tmp_path / "a", REPEATED)
    u_shaped, _, _ = _read(tmp_path / "b", U_SHAPED)
    assert repeated.require_value().kind == "repeated_stack"
    assert u_shaped.require_value().kind == "u_shaped"


def test_config_stage_lists_without_source_loops_prove_nothing(tmp_path):
    source = """
        class Root:
            def __init__(self, config):
                self.claim = config.down_block_types
            def forward(self, value):
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_container_construction_without_invocation_proves_nothing(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value):
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "failed"


def test_two_loops_without_bypass_are_a_repeated_stack_not_a_u_shape(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                self.alpha = ModuleList([Cell() for _ in range(config.depth)])
                self.omega = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value):
                for first in self.alpha:
                    value = first(value)
                for second in self.omega:
                    value = second(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"


def test_bypass_not_consumed_by_later_stage_does_not_create_u_shape(tmp_path):
    source = U_SHAPED.replace(
        "value = second(value, side)", "value = second(value)")
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"


def test_unconditional_accumulator_overwrite_kills_the_u_route(tmp_path):
    source = U_SHAPED.replace(
        "            for second in self.omega:",
        "            saved = ()\n            for second in self.omega:")
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"


def test_unconditional_carried_value_overwrite_kills_the_u_route(tmp_path):
    source = U_SHAPED.replace(
        "            for second in self.omega:",
        "            value = object()\n            for second in self.omega:")
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"


def test_same_element_class_constructed_twice_is_not_two_runtime_occurrences(
        tmp_path):
    source = REPEATED.replace(
        "[Cell() for _ in range(config.depth)]", "[Cell(), Cell()]")
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"
    assert len(result.require_value().stages) == 1


def test_helper_constructed_elements_do_not_need_a_name_based_fallback(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        def build(): return Cell()
        class Root:
            def __init__(self, config):
                self.items = ModuleList([build() for _ in range(config.depth)])
            def forward(self, value):
                for item in self.items:
                    value = item(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.require_value().kind == "repeated_stack"


def test_guarded_rival_container_constructors_are_preserved_not_picked(tmp_path):
    source = """
        from torch.nn import ModuleList
        class A:
            def forward(self, value): return value
        class B:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                if config.flag:
                    self.items = ModuleList([A()])
                else:
                    self.items = ModuleList([B()])
            def forward(self, value):
                for item in self.items:
                    value = item(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    stage = result.require_value().stages[0]
    assert result.require_value().kind == "repeated_stack"
    assert len(stage.container_records) == 1
    assert len(stage.container_records[0].records) == 2


def test_u_shape_plus_independent_repeated_stack_is_typed_ambiguity(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, side=None): return value, (value,)
        class Root:
            def __init__(self, config):
                self.alpha = ModuleList([Cell() for _ in range(config.depth)])
                self.omega = ModuleList([Cell() for _ in range(config.depth)])
                self.other = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value):
                saved = (value,)
                for first in self.alpha:
                    value, branch = first(value)
                    saved += branch
                for second in self.omega:
                    side = saved[-1:]
                    value = second(value, side)
                for extra in self.other:
                    value = extra(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 3


def test_imported_aliases_preserve_addresses_without_selecting_mechanism(
        tmp_path):
    helper = _write(tmp_path, "parts.py", """
        class Neutral:
            def forward(self, value): return value
    """)
    source = """
        from torch.nn import ModuleList
        from parts import Neutral as Unit
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Unit() for _ in range(config.depth)])
            def forward(self, value):
                for item in self.items:
                    value = item(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source, extra_files=(helper,))
    assert result.require_value().kind == "repeated_stack"


def test_broken_sibling_source_blocks_the_hidden_rival(tmp_path):
    broken = _write(tmp_path, "broken.py", "class Root(:\n")
    result, root, _index = _read(tmp_path, REPEATED, extra_files=(broken,))
    assert root.status == "failed"
    assert result.status == "failed"
    assert result.failures[0].kind == "parse_failure"


def test_missing_declared_root_is_typed_failure_not_default(tmp_path):
    result, root, _index = _read(
        tmp_path, "class Other:\n    pass\n", architecture="Missing")
    assert root.status == "absent"
    assert result.status == "failed"
    assert result.failures[0].kind == "missing_source"


def test_forged_u_shape_without_skip_route_is_rejected(tmp_path):
    result, _root, _index = _read(tmp_path, REPEATED)
    topology = result.require_value()
    with pytest.raises(ValueError, match="requires two stages and a skip route"):
        DiffusionRootTopology("u_shaped", topology.owner, topology.stages)


def test_forged_stage_with_wrong_container_field_is_rejected(tmp_path):
    result, _root, _index = _read(tmp_path, REPEATED)
    stage = result.require_value().stages[0]
    with pytest.raises(ValueError, match="exact owner and field"):
        RepeatedRootStage(
            stage.owner, "other", stage.container_records, stage.loop,
            stage.element_target, stage.calls)


@pytest.mark.parametrize("witness,expected,stages", [
    ("auraflow-v0-3", "repeated_stack", 2),
    ("cogvideox-5b", "repeated_stack", 1),
    ("flux-2-dev", "repeated_stack", 2),
    ("fluxtransformer2dmodel", "repeated_stack", 2),
    ("hunyuanvideo", "repeated_stack", 2),
    ("ltx-video", "repeated_stack", 1),
    ("lumina-image-2-0", "repeated_stack", 3),
    ("mochi-1-preview", "repeated_stack", 1),
    ("pixart-sigma-xl-2-1024-ms", "repeated_stack", 1),
    ("prxpixel-t2i", "repeated_stack", 1),
    ("qwen-image", "repeated_stack", 1),
    ("sana-1600m-1024px-diffusers", "repeated_stack", 1),
    ("stable-diffusion-3-5-large", "repeated_stack", 1),
    ("stable-diffusion-xl-base-1-0", "u_shaped", 2),
    ("wan2-2-t2v-a14b-diffusers", "repeated_stack", 1),
])
def test_real_diffusion_corpus_root_topology_matrix(
        witness, expected, stages):
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    result = read_diffusion_root_topology(index, root)
    assert result.status == "incomplete"
    assert result.require_value().kind == expected
    assert len(result.require_value().stages) == stages


def test_shadow_publication_is_call_local_and_does_not_author_ir(tmp_path):
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_root_topology,
    )
    from model_unfolder.evidence.context import ParseContext

    path = _write(tmp_path, "shadow.py", REPEATED)
    context = ParseContext(source_bundle=_bundle((path,)))
    first = _shadow_diffusion_root_topology(context)
    second = _shadow_diffusion_root_topology(context)
    assert first is second
    assert context.reader_results[("root.denoiser.topology", ())] is first
    assert first.require_value().kind == "repeated_stack"
