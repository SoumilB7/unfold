"""Conditional slot targets require both iterator and container stability."""
from dataclasses import replace
import textwrap

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_iteration_binding import read_iteration_binding
from model_unfolder.evidence.unet_lookup_closure import LookupClosure
from physics.attribute_bindings import (
    AttributeBindingWitness, ModuleListIterationWitness, ModuleListSlot, PythonFunctionWitness,
)


def fixture(tmp_path, body):
    path = tmp_path / "iteration_source.py"
    source = "class Cell:\n    def forward(self, sample):\n" + textwrap.indent(textwrap.dedent(body).strip(), "        ") + "\n"
    path.write_text(source)
    index = build_program_index(SourceBundle(source="path", files=(str(path),), component_files={"root": (str(path),)}))
    method = next(x for x in index.callables if x.symbol.qualified_name == "Cell.forward")
    function = PythonFunctionWitness("iteration_source", "Cell.forward", path.name,
        method.symbol.source.content_fingerprint, method.span.line, "a" * 64, ("enumerate", "len"))
    primitive = PythonFunctionWitness("torch.nn.modules.container", "ModuleList.__iter__", "container.py", "b" * 64, 1, "c" * 64)
    iteration = ModuleListIterationWitness(primitive, (
        ModuleListSlot("0", "stages.0"), ModuleListSlot("1", "stages.1"), ModuleListSlot("2", None)),
        length=replace(primitive, qualname="ModuleList.__len__"))
    witness = AttributeBindingWitness("", "stages", "registered_child",
        lookup_kind="object_getattribute_module_getattr", lookup_getattr=primitive,
        child_path="stages", iteration=iteration)
    closure = LookupClosure("registered_child", witness, child_path="stages")
    loop = next(x for x in index.loops_in(method.symbol) if x.target and any(
        y.name == "block" for y in (x.target, *x.target.children) if y is not None))
    return index, method, loop, closure, function


def read(args, *, stable=True):
    return read_iteration_binding(*args, parent_stable=stable)


def test_direct_and_enumerated_slots_are_conditional_and_keep_none(tmp_path):
    for header in ("for block in self.stages:", "for i, block in enumerate(self.stages):"):
        args = fixture(tmp_path, header + "\n    if block is not None:\n        sample = block(sample)\nreturn sample")
        result = read(args)
        assert result.kind == "iteration", result.reason
        assert [(x.index, x.name, x.occurrence_path, x.kind) for x in result.slots] == [
            (0, "0", "stages.0", "constructed"), (1, "1", "stages.1", "constructed"), (2, "2", None, "none")]
        assert [x.occurrence_path for x in result.targets] == ["stages.0", "stages.1"]
        assert all(x.call.guard for x in result.targets)


def test_canonical_length_needs_its_own_witness(tmp_path):
    args = fixture(tmp_path, """
        for i, block in enumerate(self.stages):
            last = i == len(self.stages) - 1
            sample = block(sample)
        return sample
    """)
    assert read(args).kind == "iteration"
    index, method, loop, closure, function = args
    missing = replace(closure.witness.iteration, length=None, length_reason="changed")
    poisoned = replace(closure, witness=replace(closure.witness, iteration=missing))
    assert read((index, method, loop, poisoned, function)).kind == "unresolved"


def test_container_alias_escape_writes_and_dynamic_lookup_are_refused(tmp_path):
    prefixes = [
        "alias = self.stages\nalias.append(replacement)",
        "helper(self.stages)",
        "self.stages.append(replacement)",
        "self.stages[0] = replacement",
        "self.stages = replacement",
        "holder = [self.stages]\nhelper(holder)",
        "alias = self\nalias.stages.append(replacement)",
        "alias = getattr(self, 'stages')\nhelper(alias)",
        "self = replacement",
        "yield token",
    ]
    for prefix in prefixes:
        args = fixture(tmp_path, prefix + "\nfor block in self.stages:\n    sample = block(sample)\nreturn sample")
        assert read(args).kind == "unresolved", prefix
    args = fixture(tmp_path, "for block in self.stages:\n    self.stages.append(replacement)\n    sample = block(sample)")
    assert read(args).kind == "unresolved"


def test_local_rebinding_nested_targets_and_after_loop_calls_are_refused(tmp_path):
    cases = [
        "for block in self.stages:\n    block = replacement\n    sample = block(sample)",
        "for block in self.stages:\n    for block in others:\n        pass\n    sample = block(sample)",
        "for block in self.stages:\n    pass\nreturn block(sample)",
        "for block in self.stages:\n    with manager() as block:\n        pass\n    sample = block(sample)",
        "for block in self.stages:\n    del block\n    sample = block(sample)",
        "for block in self.stages:\n    import other as block\n    sample = block(sample)",
        "for block in self.stages:\n    def block(value):\n        return value\n    sample = block(sample)",
        "for block in self.stages:\n    class block:\n        pass\n    sample = block(sample)",
    ]
    for body in cases:
        assert read(fixture(tmp_path, body)).kind == "unresolved", body


def test_missing_parent_iterator_and_shadowed_enumerate_are_refused(tmp_path):
    args = fixture(tmp_path, "for i, block in enumerate(self.stages):\n    sample = block(sample)")
    assert read(args, stable=False).reason == "parent_member_stability_not_proven"
    index, method, loop, closure, function = args
    absent = replace(closure, witness=replace(closure.witness, iteration=None, iteration_reason="changed iterator"))
    assert read((index, method, loop, absent, function)).kind == "unresolved"
    shadowed = replace(function, canonical_builtins=("len",))
    assert read((index, method, loop, closure, shadowed)).reason == "enumerate_adapter_not_canonical"
    source_shadow = fixture(tmp_path, "enumerate = other\nfor i, block in enumerate(self.stages):\n    sample = block(sample)")
    assert read(source_shadow).kind == "unresolved"


def test_empty_container_and_conditional_lookup_are_not_execution_claims(tmp_path):
    index, method, loop, closure, function = fixture(tmp_path, "for block in self.stages:\n    sample = block(sample)")
    empty = replace(closure.witness.iteration, slots=())
    conditional = replace(closure, kind="conditional_registered_child", conditions=("retained source branch",),
                          witness=replace(closure.witness, iteration=empty))
    result = read((index, method, loop, conditional, function))
    assert result.kind == "iteration" and not result.slots and not result.targets
    assert "retained source branch" in result.conditions


def test_boolean_guards_preserve_conditional_targets_without_hiding_effects(tmp_path):
    args = fixture(tmp_path, "for block in self.stages:\n    if enabled and block is not None:\n        sample = block(sample)")
    result = read(args)
    assert result.kind == "iteration", result.reason
    assert all(target.call.guard for target in result.targets)
    for prefix in (
        "enabled and helper(self.stages)",
        "enabled and (alias := self.stages)",
        "if self.stages and enabled:\n    pass",
    ):
        args = fixture(tmp_path, prefix + "\nfor block in self.stages:\n    sample = block(sample)")
        assert read(args).kind == "unresolved", prefix
    args = fixture(tmp_path, "for block in self.stages:\n    if enabled and (block := replacement):\n        sample = block(sample)")
    assert read(args).kind == "unresolved"
