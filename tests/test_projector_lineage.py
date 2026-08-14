"""U9-C exact fusion-operand producer lineage controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.projector_lineage import projector_lineage_result
from model_unfolder.evidence.projector import projector_result_for_context


def _result(tmp_path, source):
    path = tmp_path / "modeling_lineage.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    return projector_lineage_result(build_program_index(bundle), bundle)


def test_terminal_affine_child_reaching_fusion_is_selected_without_name_markers(tmp_path):
    result = _result(tmp_path, """
        import torch
        from torch.nn import Linear
        class Arbitrary:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self): self.any_spelling = Arbitrary()
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.any_spelling(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert len(result.value.candidates) == 1
    candidate = result.value.candidates[0]
    assert candidate.field == "any_spelling"
    assert [op.kind for op in candidate.chain.operations] == ["linear"]


def test_width_or_raw_feature_presence_does_not_create_a_projector(tmp_path):
    result = _result(tmp_path, """
        class Root:
            def forward(self, inputs_embeds, image_features, mask):
                width_mismatch = inputs_embeds.shape[-1] != image_features.shape[-1]
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "absent"


def test_constructed_affine_decoy_not_on_fusion_lineage_is_excluded(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.decoy = Linear(4, 4)
            def forward(self, inputs_embeds, image_features, mask):
                ignored = self.decoy(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "absent"


def test_exact_self_method_return_attribute_is_followed(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Output: pass
        class Arbitrary:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self): self.child = Arbitrary()
            def build_features(self, image_features):
                result = Output()
                result.pooler_output = self.child(image_features)
                return result
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.build_features(image_features).pooler_output
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "child"


def test_local_name_keeps_the_requested_self_method_result_attribute(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Output: pass
        class Root:
            def __init__(self, factory):
                self.tower = factory()
                self.bridge = Linear(4, 4)
            def build_features(self, image_features):
                result = self.tower(image_features)
                result.pooler_output = self.bridge(result.last_hidden_state)
                return result
            def forward(self, inputs_embeds, image_features, mask):
                output = self.build_features(image_features)
                image_features = output.pooler_output
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_self_method_parameter_binds_to_the_exact_call_not_all_calls(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear, Sequential, GELU
        class Root:
            def __init__(self):
                self.decoy = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
                self.bridge = Linear(4, 4)
            def identity(self, value):
                return value
            def forward(self, inputs_embeds, image_features, mask):
                ignored = self.identity(self.decoy(image_features))
                image_features = self.identity(self.bridge(image_features))
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_constructed_child_return_attribute_excludes_sibling_output(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear, Sequential, GELU
        class Output:
            def __init__(self, last_hidden_state, pooler_output): pass
        class Child:
            def __init__(self):
                self.decoy = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
                self.bridge = Linear(4, 4)
            def forward(self, x):
                return Output(
                    last_hidden_state=self.decoy(x),
                    pooler_output=self.bridge(x))
        class Root:
            def __init__(self): self.child = Child()
            def forward(self, inputs_embeds, image_features, mask):
                output = self.child(image_features)
                image_features = output.pooler_output
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_two_terminal_mechanisms_do_not_silently_pick_one(tmp_path):
    result = _result(tmp_path, """
        import torch
        from torch.nn import Linear, Sequential, GELU
        class A:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class B:
            def __init__(self): self.out = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self):
                self.a = A()
                self.b = B()
            def forward(self, inputs_embeds, image_features, video_features, mask):
                image_features = self.a(image_features)
                video_features = self.b(video_features)
                x = inputs_embeds.masked_scatter(mask, image_features)
                return x.masked_scatter(mask, video_features)
    """)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_repeated_tower_is_not_mislabeled_as_its_terminal_merger(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear, ModuleList
        class Block:
            def forward(self, x): return x
        class Merger:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class Tower:
            def __init__(self):
                self.blocks = ModuleList([Block()])
                self.merger = Merger()
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                x = self.merger(x)
                return x
        class Root:
            def __init__(self): self.tower = Tower()
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.tower(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    candidate = result.value.candidates[0]
    assert candidate.field == "merger"
    assert candidate.chain.owner_symbol.qualified_name == "Merger"


def test_parser_and_conformance_share_one_call_local_result(tmp_path):
    path = tmp_path / "modeling_cache.py"
    path.write_text(textwrap.dedent("""
        from torch.nn import Linear
        class Root:
            def __init__(self): self.any = Linear(4, 4)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.any(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    context = ParseContext(bundle)
    assert projector_result_for_context(context) is projector_result_for_context(context)


def test_exact_active_component_owner_can_supply_the_fusion_producer(tmp_path):
    child = tmp_path / "modeling_child.py"
    child.write_text(textwrap.dedent("""
        from torch.nn import Linear
        class Child:
            def __init__(self, config): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
    """), encoding="utf-8")
    root = tmp_path / "modeling_root.py"
    root.write_text(textwrap.dedent("""
        from modeling_child import Child
        class Root:
            def __init__(self, config): self.tower = Child(config.vision_config)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.tower(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(root), str(child)), architecture="Root",
        component_files={
            "root": (str(root),), "vision_config": (str(child),)},
        component_architectures={
            "root": "Root", "vision_config": "Child"})
    result = projector_lineage_result(build_program_index(bundle), bundle)
    assert result.status == "resolved"
    candidate = result.value.candidates[0]
    assert candidate.field == "tower"
    assert candidate.chain.owner_symbol.qualified_name == "Child"
    assert candidate.chain.owner_symbol.source.component_key == "vision_config"
    assert candidate.owner_graph.node_for(candidate.chain.owner_occurrence) \
        is not None


def test_config_value_selects_only_a_code_authored_constructor_branch(tmp_path):
    path = tmp_path / "modeling_guarded.py"
    path.write_text(textwrap.dedent("""
        from torch.nn import Linear
        class Bridge:
            def __init__(self, config): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self, config):
                self.bridge = Bridge(config.child) if config.child is not None else None
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.bridge(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    values = {("child",): {"width": 4}}
    result = projector_lineage_result(
        index, bundle,
        config_selector=lambda exact: (
            exact in values, values.get(exact), "config_declared"))
    assert result.status == "resolved"
    assert result.value.candidates[0].chain.owner_symbol.qualified_name \
        == "Bridge"
    assert result.value.candidates[0].config_dependencies \
        == ((('child',), 'config_declared'),)
    assert result.provenance[0].kind == "code_and_config"
    assert result.provenance[0].config_paths == (("child",),)

    inactive = projector_lineage_result(
        index, bundle,
        config_selector=lambda exact: (
            exact == ("child",), None, "config_declared"))
    assert inactive.status == "absent"


def test_guarded_builtin_projector_is_positive_on_its_exact_runtime_path(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.bridge = Linear(4, 4)
            def forward(self, x, cross_attention_states=None, pixels=None):
                if pixels is not None:
                    cross_attention_states = self.bridge(pixels).reshape(-1, 4)
                return consume(x, cross_attention_states=cross_attention_states)
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.candidates[0].chain.operations] \
        == ["linear"]


def test_latest_definition_wins_within_one_guarded_path(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self, factory):
                self.tower = factory()
                self.bridge = Linear(4, 4)
            def forward(self, x, cross_attention_states=None, pixels=None):
                if pixels is not None:
                    cross_attention_states = self.tower(pixels).last_hidden_state
                    cross_attention_states = self.bridge(cross_attention_states)
                return consume(x, cross_attention_states=cross_attention_states)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_terminal_affine_boundary_does_not_require_classifying_its_upstream_tower(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Bridge:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self, factory):
                self.tower = factory()
                self.bridge = Bridge()
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.bridge(self.tower(image_features))
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_shape_metadata_is_not_a_competing_tensor_producer(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self, factory):
                self.tower = factory()
                self.bridge = Linear(4, 4)
                self.width = 4
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.tower(image_features)
                image_features = self.bridge(image_features).reshape(
                    -1, image_features.shape[-2], self.width)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_subscript_index_is_not_a_competing_tensor_producer(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self, factory):
                self.bridge = Linear(4, 4)
                self.mask_source = factory()
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.bridge(image_features)
                selection = self.mask_source(mask)
                image_features = image_features[selection]
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_tensor_device_metadata_is_not_a_competing_producer(tmp_path):
    result = _result(tmp_path, """
        from torch.nn import Linear, Sequential, GELU
        class Root:
            def __init__(self):
                self.bridge = Linear(4, 4)
                self.decoy = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
            def forward(self, inputs_embeds, image_features, mask):
                inputs_embeds = self.decoy(inputs_embeds)
                image_features = self.bridge(image_features).to(
                    inputs_embeds.device, inputs_embeds.dtype)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"


def test_split_sizes_are_metadata_not_a_competing_producer(tmp_path):
    result = _result(tmp_path, """
        import torch
        from torch.nn import Linear, Sequential, GELU
        class Root:
            def __init__(self):
                self.bridge = Linear(4, 4)
                self.decoy = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.bridge(image_features)
                split_sizes = self.decoy(inputs_embeds).shape[-1]
                image_features = torch.split(image_features, split_sizes)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert result.status == "resolved"
    assert result.value.candidates[0].field == "bridge"
