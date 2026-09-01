"""U11-F4 exact root self-helper preprocessing controls."""
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
from model_unfolder.evidence.unet_root_preprocess import (
    RootPreprocessIssue,
    read_unet_root_preprocessing,
)
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_selection import (
    read_unet_stage_selection,
)


SOURCE = """
from torch.nn import ModuleList
from diffusers.configuration_utils import register_to_config

class Stage:
    def forward(self, value, side=None):
        return value, (value,)

def build(token):
    if token == "stage":
        return Stage()
    raise ValueError(token)

class Root:
    @register_to_config
    def __init__(self, down_types=("stage",), up_types=("stage",), mode="plain"):
        self.mode = mode
        self.down = ModuleList([])
        self.up = ModuleList([])
        for token in down_types:
            item = build(token)
            self.down.append(item)
        for token in up_types:
            item = build(token)
            self.up.append(item)

    def project(self, *values):
        return values[0]

    def preprocess(self, context, auxiliary):
        if self.mode == "projected":
            context = self.project(context)
        elif self.mode == "mixed":
            context = self.project(context, auxiliary)
        elif self.mode == "replace":
            context = self.project(auxiliary)
        return context

    def forward(self, value, context, auxiliary):
        context = self.preprocess(context, auxiliary)
        saved = (value,)
        for stage in self.down:
            value, branch = stage(value, context)
            saved += branch
        for stage in self.up:
            value = stage(value, saved[-1:], context)
        return value
"""


def _bundle(tmp_path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})


def _read(tmp_path, mode="plain", source=SOURCE):
    bundle = _bundle(tmp_path, source)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    document = {
        "down_types": ["stage"],
        "up_types": ["stage"],
        "mode": mode,
    }
    selection = read_unet_stage_selection(
        construction, root,
        DocumentBinding("root", (), prepare_document(document, merge=False)))
    assert selection.status == "resolved"
    return read_unet_root_preprocessing(selection.require_value(), root)


def _route(result):
    value = result.require_value()
    matches = tuple(item for item in value.routes
                    if item.call.callee.name == "preprocess")
    assert len(matches) == 1, value.issues
    return matches[0]


@pytest.mark.parametrize(("mode", "sources", "relation"), (
    ("plain", ("context",), "single_source"),
    ("projected", ("context",), "single_source"),
    ("mixed", ("auxiliary", "context"), "mixed_sources"),
    ("replace", ("auxiliary",), "single_source"),
))
def test_selected_constructor_guard_controls_exact_helper_lineage(
        tmp_path, mode, sources, relation):
    route = _route(_read(tmp_path, mode))
    assert tuple(item.name for item in route.caller_sources) == sources
    assert route.relation == relation
    assert route.config_paths == (("mode",),)
    assert route.owner == route.transport.owner_occurrence


def test_helper_arguments_do_not_become_sources_unless_the_return_reaches_them(
        tmp_path):
    source = SOURCE.replace(
        "        return context\n\n    def forward",
        "        return auxiliary\n\n    def forward")
    route = _route(_read(tmp_path, "plain", source))
    assert tuple(item.name for item in route.caller_sources) == ("auxiliary",)


def test_constructor_branch_decision_reaches_the_instance_guard_provenance(
        tmp_path):
    source = SOURCE.replace(
        "        self.mode = mode\n",
        "        if mode == \"plain\":\n"
        "            self.marker = None\n"
        "        else:\n"
        "            self.marker = 1\n")
    start = source.index("    def preprocess")
    end = source.index("\n    def forward", start)
    source = (source[:start]
              + "    def preprocess(self, context, auxiliary):\n"
                "        if self.marker is not None:\n"
                "            context = self.project(auxiliary)\n"
                "        return context\n"
              + source[end:])
    route = _route(_read(tmp_path, "plain", source))
    assert tuple(item.name for item in route.caller_sources) == ("context",)
    assert route.config_paths == (("mode",),)


def test_unknown_instance_guard_stays_typed_unresolved(tmp_path):
    source = SOURCE.replace(
        'if self.mode == "projected":', 'if decide(self.mode):')
    result = _read(tmp_path, "plain", source)
    assert result.status == "incomplete"
    assert any(item.kind == "source_lineage_unresolved"
               for item in result.require_value().issues)


def test_selected_root_call_guard_is_part_of_the_route_proof(tmp_path):
    source = SOURCE.replace(
        "        context = self.preprocess(context, auxiliary)\n",
        "        if self.mode != \"off\":\n"
        "            context = self.preprocess(context, auxiliary)\n")
    route = _route(_read(tmp_path, "plain", source))
    assert route.config_paths == (("mode",),)


def test_false_root_call_guard_does_not_author_a_route(tmp_path):
    source = SOURCE.replace(
        "        context = self.preprocess(context, auxiliary)\n",
        "        if self.mode != \"off\":\n"
        "            context = self.preprocess(context, auxiliary)\n")
    result = _read(tmp_path, "off", source)
    assert result.status == "incomplete"
    assert not any(item.call.callee.name == "preprocess"
                   for item in result.require_value().routes)


def test_active_early_exit_blocks_the_later_return_route(tmp_path):
    source = SOURCE.replace(
        "    def preprocess(self, context, auxiliary):\n",
        "    def preprocess(self, context, auxiliary):\n"
        "        if self.mode == \"replace\":\n"
        "            raise ValueError(self.mode)\n")
    result = _read(tmp_path, "replace", source)
    assert result.status == "incomplete"
    assert any(item.kind == "control_flow_unresolved"
               for item in result.require_value().issues)
    assert not any(item.call.callee.name == "preprocess"
                   for item in result.require_value().routes)


def test_inactive_early_exit_does_not_poison_selected_route(tmp_path):
    source = SOURCE.replace(
        "    def preprocess(self, context, auxiliary):\n",
        "    def preprocess(self, context, auxiliary):\n"
        "        if self.mode == \"replace\":\n"
        "            raise ValueError(self.mode)\n")
    route = _route(_read(tmp_path, "plain", source))
    assert tuple(item.name for item in route.caller_sources) == ("context",)


def test_unsupported_helper_execution_never_becomes_a_positive_route(tmp_path):
    source = SOURCE.replace(
        "    def preprocess(self, context, auxiliary):\n",
        "    def preprocess(self, context, auxiliary):\n"
        "        try:\n"
        "            context = context\n"
        "        finally:\n"
        "            context = context\n")
    result = _read(tmp_path, "plain", source)
    assert result.status == "incomplete"
    assert any(item.kind == "execution_unresolved"
               for item in result.require_value().issues)


def test_complete_renaming_preserves_the_relation(tmp_path):
    source = (SOURCE
              .replace("Root", "RenamedRoot")
              .replace("context", "memory")
              .replace("auxiliary", "conditioning")
              .replace("preprocess", "prepare")
              .replace("mode", "choice"))
    bundle = _bundle(tmp_path, source)
    bundle = replace(
        bundle, architecture="RenamedRoot",
        component_architectures={"root": "RenamedRoot"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    document = {
        "down_types": ["stage"], "up_types": ["stage"], "choice": "replace"}
    selection = read_unet_stage_selection(
        construction, root,
        DocumentBinding("root", (), prepare_document(document, merge=False)))
    result = read_unet_root_preprocessing(selection.require_value(), root)
    route = next(item for item in result.require_value().routes
                 if item.call.callee.name == "prepare")
    assert tuple(item.name for item in route.caller_sources) == ("conditioning",)
    assert route.config_paths == (("choice",),)


def test_route_and_outer_inventory_recompute_exactly(tmp_path):
    result = _read(tmp_path, "replace")
    value = result.require_value()
    route = _route(result)
    with pytest.raises(ValueError):
        replace(route, relation="mixed_sources")
    with pytest.raises(ValueError):
        replace(route, config_paths=(("invented",),))
    with pytest.raises(ValueError):
        replace(value, routes=())
    with pytest.raises(ValueError):
        RootPreprocessIssue("invented", "not in the closed vocabulary")


def test_foreign_root_or_index_cannot_cross_the_boundary(tmp_path):
    first = _read(tmp_path / "a", "plain")
    second = _read(tmp_path / "b", "plain")
    with pytest.raises(ValueError):
        replace(first.require_value(), root=second.require_value().root)


def test_real_sdxl_preprocesses_encoder_state_from_its_exact_input():
    diffusers = pytest.importorskip("diffusers")
    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    config = json.loads(Path(
        "tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json"
    ).read_text(encoding="utf-8"))["config"]
    selection = read_unet_stage_selection(
        construction, root,
        DocumentBinding("root", (), prepare_document(config, merge=False)))
    assert selection.status == "resolved"
    result = read_unet_root_preprocessing(selection.require_value(), root)
    routes = tuple(item for item in result.require_value().routes
                   if item.call.callee.name == "process_encoder_hidden_states")
    compact_issues = tuple((item.kind, item.detail,
                            item.span.line if item.span else None)
                           for item in result.require_value().issues)
    assert len(routes) == 1, compact_issues
    route = routes[0]
    assert route.relation == "single_source"
    assert tuple(item.name for item in route.caller_sources) == (
        "encoder_hidden_states",)
    assert ("encoder_hid_dim_type",) in route.config_paths
