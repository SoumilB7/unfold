"""U9-C exact projector width operand controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.projector_lineage import projector_lineage_result
from model_unfolder.evidence.projector_width import projector_width_evidence


def _widths(tmp_path, source):
    path = tmp_path / "modeling_width.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    lineage = projector_lineage_result(index, bundle)
    assert lineage.status == "resolved"
    assert len(lineage.value.candidates) == 1
    return projector_width_evidence(
        index, root.graph, lineage.value.candidates[0])


def _nested_widths(tmp_path, *, attribute_map='{"hidden_size": "d_model"}'):
    model = tmp_path / "modeling_nested_width.py"
    config = tmp_path / "configuration_nested_width.py"
    model.write_text(textwrap.dedent("""
        from torch.nn import Linear
        from transformers.modeling_utils import PreTrainedModel
        from .configuration_nested_width import ChildConfig

        class Child(PreTrainedModel):
            def __init__(self, config: ChildConfig):
                super().__init__(config)

        class Root:
            def __init__(self, config):
                self.text_encoder = Child(config.text_encoder)
                self.proj = Linear(
                    self.text_encoder.config.hidden_size,
                    config.output_width,
                )
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.proj(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """), encoding="utf-8")
    config.write_text(textwrap.dedent(f"""
        class ChildConfig:
            attribute_map = {attribute_map}
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(model),), architecture="Root",
        component_files={
            "root": (str(model),), "text_encoder": (str(model),)},
        supporting_files={"text_encoder": (str(config),)},
        component_architectures={
            "root": "Root", "text_encoder": "Child"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    lineage = projector_lineage_result(index, bundle)
    assert root.status == lineage.status == "resolved", lineage.failures
    assert len(lineage.value.candidates) == 1
    return projector_width_evidence(
        index, root.graph, lineage.value.candidates[0],
        nested_config_addresses=lineage.value.nested_config_addresses)


def test_exact_child_config_prefix_is_carried_to_linear_operands(tmp_path):
    widths = _widths(tmp_path, """
        from torch.nn import Linear
        class Child:
            def __init__(self, config):
                self.out = Linear(config.input_width, config.output_width)
            def forward(self, x): return self.out(x)
        class Root:
            def __init__(self, config): self.any = Child(config.vision_config)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.any(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert widths.input.source == "config_bound"
    assert widths.input.path == ("vision_config", "input_width")
    assert widths.output.path == ("vision_config", "output_width")


def test_constructor_parameter_chain_preserves_paths_and_derivation(tmp_path):
    widths = _widths(tmp_path, """
        from torch.nn import Linear
        class Merge:
            def __init__(self, hidden, context, repeat):
                self.out = Linear(context * repeat, hidden)
            def forward(self, x): return self.out(x)
        class Tower:
            def __init__(self, config):
                self.merge = Merge(config.hidden, config.context, config.repeat)
            def forward(self, x): return self.merge(x)
        class Root:
            def __init__(self, config): self.tower = Tower(config.vision_config)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.tower(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert widths.input.source == "derived"
    assert widths.output.source == "config_bound"
    assert widths.output.path == ("vision_config", "hidden")


def test_literal_widths_are_code_bound(tmp_path):
    widths = _widths(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 8)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.out(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert (widths.input.source, widths.input.value) == ("code_bound", 4)
    assert (widths.output.source, widths.output.value) == ("code_bound", 8)


def test_dynamic_operand_stays_unavailable(tmp_path):
    widths = _widths(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self, config): self.out = Linear(resolve(config), config.width)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.out(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert widths.input.source == "unavailable"
    assert widths.output.path == ("width",)


def test_sequential_elements_keep_the_exact_first_and_last_widths(tmp_path):
    widths = _widths(tmp_path, """
        from torch.nn import GELU, Linear, Sequential
        class Merge:
            def __init__(self, hidden, context, repeat):
                self.inner = context * repeat
                self.mlp = Sequential(
                    Linear(self.inner, self.inner),
                    GELU(),
                    Linear(self.inner, hidden),
                )
            def forward(self, x): return self.mlp(x)
        class Root:
            def __init__(self, config):
                self.merge = Merge(
                    config.hidden, config.context, config.repeat)
            def forward(self, inputs_embeds, image_features, mask):
                image_features = self.merge(image_features)
                return inputs_embeds.masked_scatter(mask, image_features)
    """)
    assert widths.input.source == "derived"
    assert widths.output.source == "config_bound"
    assert widths.output.path == ("hidden",)


def test_nested_component_attribute_map_translates_code_name_to_checkpoint_path(
        tmp_path):
    widths = _nested_widths(tmp_path)
    assert widths.input.source == "config_bound"
    assert widths.input.path == ("text_encoder", "d_model")
    assert widths.output.path == ("output_width",)


def test_nonliteral_nested_attribute_map_cannot_publish_a_guessed_path(tmp_path):
    widths = _nested_widths(tmp_path, attribute_map="build_aliases()")
    assert widths.input.source == "unavailable"
    assert widths.output.path == ("output_width",)
