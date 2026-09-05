"""Poisons for the model-blind S7 relation-probe planner."""
from __future__ import annotations

import dataclasses
import hashlib
import ast
import inspect
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.relation_probe import (
    RelationProbeResolution,
    resolve_relation_probes,
)
import model_unfolder.evidence.relation_probe as relation_probe_module
from physics.execution_observation import (
    ExecutionObservation,
    ExecutionRecipe,
    ModuleCall,
    ObservationResult,
    TensorArgument,
)
from physics.instance_inventory import (
    Failure,
    InstanceInventory,
    ModuleNode,
    PackageVersion,
    Provenance,
    ResolvedClass,
    SourceFile,
)


TORCH_LIST = ResolvedClass("torch.nn.modules.container", "ModuleList")
TORCH_SEQUENCE = ResolvedClass("torch.nn.modules.container", "Sequential")


def test_model_unfolder_relation_probe_does_not_import_physics():
    tree = ast.parse(inspect.getsource(relation_probe_module))
    imports = []
    for row in ast.walk(tree):
        if isinstance(row, ast.Import):
            imports.extend(alias.name for alias in row.names)
        elif isinstance(row, ast.ImportFrom) and row.module:
            imports.append(row.module)
    assert not any(name == "physics" or name.startswith("physics.")
                   for name in imports)


def _node(path: str, cls: ResolvedClass, children=()) -> ModuleNode:
    return ModuleNode(path, cls, cls.module, (cls,), tuple(children), (), {}, ())


def _fixture(tmp_path: Path, *, field: str = "spine", cell: str = "Cell",
             second: bool = False, decoy: bool = False,
             custom_container: bool = False):
    source = f"""
        class Root:
            def forward(self, state):
                return state

        class {cell}:
            def forward(self, state):
                left = state * 2
                right = state + 1
                return left + right

        class Plain:
            def forward(self, state):
                return state
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="path", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"},
    )
    index = build_program_index(bundle)
    fingerprint = index.source_nodes[0].source_id.content_fingerprint
    root_cls = ResolvedClass("fixture", "Root")
    cell_cls = ResolvedClass("fixture", cell)
    plain_cls = ResolvedClass("fixture", "Plain")
    container_cls = (ResolvedClass("fixture", "ModuleList")
                     if custom_container else TORCH_LIST)
    modules = [
        _node("", root_cls, (field,) + (("other",) if second else ())
              + (("decoy",) if decoy else ())),
        _node(field, container_cls, ("0", "1")),
        _node(f"{field}.0", cell_cls),
        _node(f"{field}.1", cell_cls),
    ]
    paths = ["", f"{field}.0", f"{field}.1"]
    classes = [root_cls, cell_cls, cell_cls]
    if second:
        modules.extend([
            _node("other", TORCH_SEQUENCE, ("first", "second")),
            _node("other.first", cell_cls), _node("other.second", cell_cls),
        ])
        paths.extend(("other.first", "other.second"))
        classes.extend((cell_cls, cell_cls))
    if decoy:
        modules.extend([
            _node("decoy", TORCH_LIST, ("0", "1", "2", "3")),
            *(_node(f"decoy.{number}", plain_cls) for number in range(4)),
        ])
        paths.extend(f"decoy.{number}" for number in range(4))
        classes.extend(plain_cls for _ in range(4))
    config_hash = hashlib.sha256(b"{}").hexdigest()
    provenance = Provenance(
        (PackageVersion("fixture", "1"),),
        (SourceFile("fixture", path.name, fingerprint),),
        config_hash, root_cls, "fixture.Root", "fixture.Root(config)", {},
        {"python": "3", "platform": "test", "hash_seed": "0",
         "network": "off", "hf_hub_offline": "1",
         "transformers_offline": "1", "diffusers_offline": "1"},
    )
    inventory = InstanceInventory(1, provenance, tuple(modules), (), ())
    recipe = ExecutionRecipe(
        "signature", "callable_signature", "eval", "disabled", "unspecified",
        False, "float32", {"fixture": "1"},
        tensor_arguments=(TensorArgument("state", (1, 2, 8), "float32"),),
        flags={"source": "resolved_callable_signature"},
    )
    calls = tuple(ModuleCall(number, name, cls)
                  for number, (name, cls) in enumerate(zip(paths, classes)))
    observation = ExecutionObservation(
        1, provenance, recipe, calls, (), (), "observed")
    result = ObservationResult(
        "ok", recipe, observation, provenance)
    return index, inventory, result


def test_runtime_probe_selection_is_mechanism_neutral(tmp_path):
    index, inventory, execution = _fixture(tmp_path, decoy=True)
    result = resolve_relation_probes(index, inventory, execution)
    assert result.status == "resolved"
    assert [row.stack_path for row in result.plans] == ["decoy", "spine"]
    assert not result.issues


def test_full_class_field_and_external_label_rename_preserves_rule(tmp_path):
    first = _fixture(tmp_path / "first", field="spine", cell="Cell")
    second = _fixture(tmp_path / "second", field="renamed", cell="RenamedCell")
    left = resolve_relation_probes(*first)
    right = resolve_relation_probes(*second)
    assert len(left.plans) == len(right.plans) == 1
    assert left.plans[0].stack_path == "spine"
    assert right.plans[0].stack_path == "renamed"


def test_two_independent_capabilities_produce_two_plans_never_a_pick(tmp_path):
    result = resolve_relation_probes(*_fixture(tmp_path, second=True))
    assert result.status == "resolved"
    assert [row.stack_path for row in result.plans] == ["other", "spine"]


@pytest.mark.parametrize(
    ("mutation", "kind"),
    (("partial", "partial_invocation"),
     ("repeat", "repeated_invocation"),
     ("reverse", "execution_order_conflict")),
)
def test_call_census_failures_stay_typed(tmp_path, mutation, kind):
    index, inventory, result = _fixture(tmp_path)
    calls = list(result.observation.module_calls)
    if mutation == "partial":
        calls = [row for row in calls if row.path != "spine.1"]
    elif mutation == "repeat":
        calls.append(dataclasses.replace(calls[-1], index=len(calls)))
    else:
        calls[1], calls[2] = (
            dataclasses.replace(calls[2], index=1),
            dataclasses.replace(calls[1], index=2),
        )
    calls = tuple(dataclasses.replace(row, index=number)
                  for number, row in enumerate(calls))
    observation = dataclasses.replace(result.observation, module_calls=calls)
    changed = dataclasses.replace(result, observation=observation)
    resolved = resolve_relation_probes(index, inventory, changed)
    assert not resolved.plans
    assert resolved.status == "unresolved"
    assert any(row.kind == kind for row in resolved.issues)


def test_custom_class_merely_named_modulelist_is_not_a_protocol(tmp_path):
    result = resolve_relation_probes(*_fixture(tmp_path, custom_container=True))
    assert result.status == "absent"
    assert not result.plans and not result.issues


def test_foreign_inventory_and_trace_fail_closed(tmp_path):
    index, inventory, result = _fixture(tmp_path / "one")
    _other_index, other_inventory, _other_result = _fixture(tmp_path / "two")
    other_inventory = dataclasses.replace(
        other_inventory,
        provenance=dataclasses.replace(
            other_inventory.provenance, config_sha256="b" * 64),
    )
    resolved = resolve_relation_probes(index, other_inventory, result)
    assert resolved.status == "failed"
    assert resolved.failure_kind == "provenance_mismatch"


def test_probe_planning_never_infers_meaning_from_member_class(tmp_path):
    index, inventory, result = _fixture(tmp_path)
    plain = ResolvedClass("fixture", "Plain")
    inventory = dataclasses.replace(
        inventory,
        modules=tuple(
            dataclasses.replace(row, class_ref=plain, origin_module=plain.module,
                                mro_entries=(plain,))
            if row.path == "spine.1" else row
            for row in inventory.modules
        ),
    )
    observation = dataclasses.replace(
        result.observation,
        module_calls=tuple(
            dataclasses.replace(row, class_ref=plain)
            if row.path == "spine.1" else row
            for row in result.observation.module_calls
        ),
    )
    result = dataclasses.replace(result, observation=observation)
    resolved = resolve_relation_probes(index, inventory, result)
    # Planning is purely an execution/address capability.  The later observer
    # must prove an exact source-bound operation; neither class is interpreted.
    assert len(resolved.plans) == 1
    assert resolved.status == "resolved"
    assert not resolved.issues


def test_failed_base_execution_is_a_typed_global_failure(tmp_path):
    index, inventory, result = _fixture(tmp_path)
    failed = ObservationResult(
        "failed", recipe=result.recipe,
        failure=Failure("ExecutionFailed", "execute", "fixture failure"))
    resolved = resolve_relation_probes(index, inventory, failed)
    assert resolved.status == "failed"
    assert resolved.failure_kind == "base_execution_failed"


def test_target_must_contain_the_candidate(tmp_path):
    index, inventory, result = _fixture(tmp_path)
    recipe = dataclasses.replace(result.recipe, target_path="elsewhere")
    observation = dataclasses.replace(result.observation, recipe=recipe)
    result = dataclasses.replace(
        result, recipe=recipe, observation=observation)
    resolved = resolve_relation_probes(index, inventory, result)
    assert not resolved.plans
    assert resolved.status == "unresolved"
    assert any(row.kind == "target_not_ancestor" for row in resolved.issues)


def test_derived_recipe_changes_only_audit_identity_and_flags(tmp_path):
    result = resolve_relation_probes(*_fixture(tmp_path))
    plan = result.plans[0]
    derived = plan.execution_recipe()
    assert derived.recipe_id != plan.base_recipe.recipe_id
    assert derived.flags != plan.base_recipe.flags
    assert dataclasses.replace(
        derived, recipe_id=plan.base_recipe.recipe_id,
        flags=plan.base_recipe.flags) == plan.base_recipe
    assert "gemma" not in derived.recipe_id.lower()
    assert "deepseek" not in derived.recipe_id.lower()


def test_plan_closure_prevents_proof_from_one_observation_authorizing_another(
        tmp_path):
    result = resolve_relation_probes(*_fixture(tmp_path))
    plan = result.plans[0]
    changed = dataclasses.replace(
        plan.member_calls[0],
        class_ref=ResolvedClass("fixture", "Other"))
    with pytest.raises(ValueError, match="exact observation"):
        dataclasses.replace(
            plan, member_calls=(changed, *plan.member_calls[1:]))

    foreign_observation = dataclasses.replace(
        plan.observation,
        module_calls=(
            dataclasses.replace(
                plan.observation.module_calls[0],
                class_ref=ResolvedClass("fixture", "ForeignRoot"),
            ),
            *plan.observation.module_calls[1:],
        ),
    )
    with pytest.raises(ValueError, match="exact base observation"):
        dataclasses.replace(plan, observation=foreign_observation)


def test_result_closure_never_calls_absent_a_semantic_negative():
    result = RelationProbeResolution("absent")
    assert result.to_dict()["semantic_negative"] is False
    with pytest.raises(ValueError):
        RelationProbeResolution("resolved")
    with pytest.raises(ValueError):
        RelationProbeResolution("failed", failure_kind="unknown",
                                failure_detail="x")
    with pytest.raises(ValueError):
        RelationProbeResolution("unresolved")
