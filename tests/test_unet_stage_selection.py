"""U11-F1 occurrence-exact U-Net stage selection controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_selection import (
    read_unet_stage_selection,
)
from model_unfolder.evidence.unet_stage_operands import (
    read_unet_selected_stage_operands,
)
from model_unfolder.evidence.unet_stage_constructor_operands import (
    read_unet_selected_stage_constructor_operands,
)


SOURCE = """
from torch.nn import ModuleList
from diffusers.configuration_utils import register_to_config

class First:
    def forward(self, value, side=None): return value, (value,)
class Second:
    def forward(self, value, side=None): return value, (value,)

def choose(token):
    token = token[2:] if token.startswith("X:") else token
    if token == "first":
        return First()
    if token == "second":
        return Second()
    raise ValueError(token)

class Root:
    @register_to_config
    def __init__(self, alpha_types=("first",), omega_types=("second",)):
        self.alpha = ModuleList([])
        self.omega = ModuleList([])
        for i, token in enumerate(alpha_types):
            item = choose(token)
            self.alpha.append(item)
        for i, token in enumerate(omega_types):
            item = choose(token)
            self.omega.append(item)

    def forward(self, value):
        saved = (value,)
        for left in self.alpha:
            value, branch = left(value)
            saved += branch
        for right in self.omega:
            value = right(value, saved[-1:])
        return value
"""


def _bundle(tmp_path, source=SOURCE, *, architecture="Root"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return SourceBundle(
        source="test", architecture=architecture,
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture})


def _evidence(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    topology = read_diffusion_root_topology(index, root)
    assert topology.has_value and topology.value.kind == "u_shaped"
    construction = read_unet_stage_construction(
        index, bundle, root, topology.value)
    assert construction.has_value
    return root, construction.value


def _read(tmp_path, document, source=SOURCE):
    root, construction = _evidence(_bundle(tmp_path, source))
    binding = DocumentBinding(
        "root", (), prepare_document(document, merge=False))
    return read_unet_stage_selection(construction, root, binding)


def _rows(result):
    return tuple((item.source.template.topology_stage.field,
                  item.position, item.selector_value,
                  item.candidate.symbol.qualified_name)
                 for item in result.require_value().occurrences)


def test_exact_checkpoint_lists_select_exact_factory_returns(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first", "second"],
        "omega_types": ["X:second", "first"],
    })
    assert result.status == "resolved"
    assert _rows(result) == (
        ("alpha", 0, "first", "First"),
        ("alpha", 1, "second", "Second"),
        ("omega", 0, "X:second", "Second"),
        ("omega", 1, "first", "First"),
    )
    assert {path for origin in result.provenance
            for path in origin.config_paths} == {
                ("alpha_types",), ("omega_types",)}


def test_same_class_twice_remains_two_occurrences(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first", "first"],
        "omega_types": ["second"],
    })
    first = result.require_value().stages[0].selected
    assert len(first) == 2
    assert first[0].candidate == first[1].candidate
    assert first[0].position != first[1].position


def test_short_and_long_lists_are_not_mirrored_or_filled(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first"],
        "omega_types": ["second", "first", "second"],
    })
    assert result.status == "resolved"
    assert [(stage.source.template.topology_stage.field,
             len(stage.source.selector_values))
            for stage in result.value.stages] == [("alpha", 1), ("omega", 3)]
    assert len(result.value.occurrences) == 4


def test_missing_sibling_selector_cannot_borrow_the_present_list(tmp_path):
    result = _read(tmp_path, {"alpha_types": ["first"]})
    assert result.status == "incomplete"
    assert [item.source.template.topology_stage.field
            for item in result.value.stages] == ["alpha"]
    assert [item.template.topology_stage.field
            for item in result.value.unresolved_templates] == ["omega"]


def test_registered_class_default_does_not_instantiate_checkpoint_occurrences(
        tmp_path):
    result = _read(tmp_path, {})
    assert result.status == "incomplete"
    assert result.value.occurrences == ()
    assert len(result.value.unresolved_templates) == 2


def test_unresolved_registration_source_is_retained_not_bypassed(tmp_path):
    source = SOURCE.replace(
        "from diffusers.configuration_utils import register_to_config\n",
        "def register_to_config(value): return value\n")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]}, source)
    assert result.status == "incomplete"
    assert result.value.registration_result.status == "failed"
    assert result.value.registration_result.failures[0].kind \
        == "unresolved_import"
    assert result.value.occurrences == ()


def test_selector_alias_or_transform_is_not_guessed(tmp_path):
    source = SOURCE.replace(
        "for i, token in enumerate(alpha_types):",
        "selected_types = alpha_types\n"
        "        for i, token in enumerate(selected_types):")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]}, source)
    assert result.status == "incomplete"
    assert result.value.unresolved_templates[0].issues[0].kind \
        == "selector_route_unresolved"


def test_nonsequence_selector_is_typed_incomplete(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": "first", "omega_types": ["second"]})
    assert result.status == "incomplete"
    assert result.value.occurrences[0].candidate.symbol.qualified_name == "Second"
    assert result.value.unresolved_templates[0].issues[0].kind \
        == "selector_not_sequence"


def test_multiple_direct_templates_are_all_preserved_as_unresolved(tmp_path):
    source = SOURCE.replace(
        "self.alpha = ModuleList([])",
        "self.alpha = ModuleList([First(), Second()])").replace(
        "        for i, token in enumerate(alpha_types):\n"
        "            item = choose(token)\n"
        "            self.alpha.append(item)\n", "")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]}, source)
    assert result.status == "incomplete"
    direct = [item for item in result.value.unresolved_templates
              if item.template is not None
              and item.template.topology_stage.field == "alpha"]
    assert len(direct) == 2
    assert all(item.issues[0].kind == "selector_route_unresolved"
               for item in direct)


def test_unknown_token_does_not_choose_a_familiar_fallback(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["unknown"], "omega_types": ["second"]})
    assert result.status == "incomplete"
    unresolved = result.value.stages[0].unresolved[0]
    assert unresolved.live_candidates == ()
    assert unresolved.issues[-1].kind == "no_live_candidate"


def test_dynamic_factory_guard_stays_unresolved(tmp_path):
    source = SOURCE.replace(
        'if token == "first":', 'if decide(token):')
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]}, source)
    assert result.status == "incomplete"
    assert result.value.stages[0].unresolved[0].issues[0].kind \
        == "candidate_guard_unresolved"


def test_nested_unrelated_loop_cannot_steal_the_selector_binding(tmp_path):
    source = SOURCE.replace(
        "item = choose(token)\n            self.alpha.append(item)",
        "for decoy in runtime_values:\n"
        "                item = choose(token)\n"
        "                self.alpha.append(item)")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]}, source)
    assert result.status == "incomplete"
    unresolved = result.value.unresolved_templates[0]
    assert unresolved.template.topology_stage.field == "alpha"
    assert unresolved.issues[0].kind == "selector_route_unresolved"


def test_two_live_factory_returns_are_preserved_as_rivals(tmp_path):
    source = SOURCE.replace(
        'if token == "second":', 'if token == "first":')
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["first"]}, source)
    assert result.status == "incomplete"
    for stage in result.value.stages:
        row = stage.unresolved[0]
        assert len(row.live_candidates) == 2
        assert any(item.kind == "rival_live_candidates" for item in row.issues)


def test_complete_rename_preserves_the_structural_join(tmp_path):
    source = (SOURCE.replace("Root", "Opaque")
              .replace("First", "One")
              .replace("Second", "Two")
              .replace("alpha_types", "north_kinds")
              .replace("omega_types", "south_kinds")
              .replace("alpha", "north")
              .replace("omega", "south")
              .replace("token", "selector")
              .replace("choose", "construct"))
    bundle = _bundle(tmp_path, source, architecture="Opaque")
    root, construction = _evidence(bundle)
    result = read_unet_stage_selection(
        construction, root, DocumentBinding(
            "root", (), prepare_document({
                "north_kinds": ["first"],
                "south_kinds": ["second"],
            }, merge=False)))
    assert result.status == "resolved"
    assert {(row.source.template.topology_stage.field,
             row.candidate.symbol.qualified_name)
            for row in result.value.occurrences} == {
                ("north", "One"), ("south", "Two")}


def test_selected_occurrence_rejects_candidate_outside_template_census(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"]})
    first = result.value.occurrences[0]
    foreign = _read(tmp_path / "foreign", {
        "alpha_types": ["first"], "omega_types": ["second"]}) \
        .value.occurrences[0].candidate
    with pytest.raises(ValueError, match="complete template census"):
        replace(first, candidate=foreign)


def test_root_and_construction_from_different_indices_are_rejected(tmp_path):
    left_root, _left = _evidence(_bundle(tmp_path / "left"))
    _right_root, right = _evidence(_bundle(tmp_path / "right"))
    binding = DocumentBinding("root", (), prepare_document({
        "alpha_types": ["first"], "omega_types": ["second"]}, merge=False))
    result = read_unet_stage_selection(right, left_root, binding)
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_real_sdxl_instantiates_six_exact_stage_occurrences():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    root, construction = _evidence(bundle)
    corpus = json.loads(Path(
        "tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json"
    ).read_text(encoding="utf-8"))["config"]
    result = read_unet_stage_selection(
        construction, root,
        DocumentBinding("root", (), prepare_document(corpus, merge=False)))
    assert result.status == "resolved"
    assert _rows(result) == (
        ("down_blocks", 0, "DownBlock2D", "DownBlock2D"),
        ("down_blocks", 1, "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        ("down_blocks", 2, "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        ("up_blocks", 0, "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        ("up_blocks", 1, "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        ("up_blocks", 2, "UpBlock2D", "UpBlock2D"),
    )
    assert result.value.unresolved_occurrences == ()
    assert result.value.unresolved_templates == ()

    operands = read_unet_selected_stage_operands(
        result.require_value())
    assert operands.status == "incomplete"
    down_channels = {
        (item.selected.position, item.formal.name): item.value
        for item in operands.value.operands
        if item.selected.source.template.topology_stage.field == "down_blocks"
        and item.formal.name in {"in_channels", "out_channels"}
    }
    assert down_channels == {
        (0, "in_channels"): 320,
        (0, "out_channels"): 320,
        (1, "out_channels"): 640,
        (2, "out_channels"): 1280,
    }
    loop_carried = tuple(
        item for item in operands.value.issues
        if item.selected.source.template.topology_stage.field == "down_blocks"
        and item.formal is not None and item.formal.name == "in_channels")
    assert [(item.selected.position, item.kind) for item in loop_carried] == [
        (1, "local_lineage_unresolved"),
        (2, "local_lineage_unresolved"),
    ]
    per_stage = {
        (item.selected.source.template.topology_stage.field,
         item.selected.position, item.formal.name): item.value
        for item in operands.value.operands
        if item.formal.name in {
            "num_layers", "add_downsample", "add_upsample",
            "cross_attention_dim",
        }
    }
    assert tuple(per_stage[("down_blocks", position, "num_layers")]
                 for position in range(3)) == (2, 2, 2)
    assert tuple(per_stage[("up_blocks", position, "num_layers")]
                 for position in range(3)) == (3, 3, 3)
    assert tuple(per_stage[("down_blocks", position, "add_downsample")]
                 for position in range(3)) == (True, True, False)
    assert tuple(per_stage[("up_blocks", position, "add_upsample")]
                 for position in range(3)) == (True, True, False)
    assert tuple(per_stage[(field, position, "cross_attention_dim")]
                 for field in ("down_blocks", "up_blocks")
                 for position in range(3)) == (2048,) * 6

    stage_operands = read_unet_selected_stage_constructor_operands(
        operands.require_value())
    assert stage_operands.status == "incomplete"
    assert len(stage_operands.require_value().operands) == 103
    assert len(stage_operands.require_value().issues) == 10
    stage_values = {
        (item.selected.source.template.topology_stage.field,
         item.selected.position, item.formal.name): item.value
        for item in stage_operands.value.operands
    }
    assert tuple(stage_values[("down_blocks", position, "num_layers")]
                 for position in range(3)) == (2, 2, 2)
    assert tuple(stage_values[("up_blocks", position, "num_layers")]
                 for position in range(3)) == (3, 3, 3)
    assert tuple(stage_values[("down_blocks", position, "add_downsample")]
                 for position in range(3)) == (True, True, False)
    assert tuple(stage_values[("up_blocks", position, "add_upsample")]
                 for position in range(3)) == (True, True, False)
    assert tuple(stage_values[(field, position, "cross_attention_dim")]
                 for field, position in (
                     ("down_blocks", 1), ("down_blocks", 2),
                     ("up_blocks", 0), ("up_blocks", 1))) == (2048,) * 4
    assert ("down_blocks", 0, "cross_attention_dim") not in stage_values
    assert ("up_blocks", 2, "cross_attention_dim") not in stage_values
