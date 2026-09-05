"""Evidence-qualified relation-probe planning.

This module replaces neither a relation reader nor the runtime observer.  It
answers one narrower question: which *exact constructed ordered containers*
may lawfully receive the existing recurrent-relation probe?

Selection is deliberately independent of model, family, class and field
vocabulary.  A container qualifies only when all of these positive facts hold:

* its runtime type is exactly one of the closed torch ordered-container types;
* every direct member was called exactly once, in stored-child order, by one
  already-successful, recipe-qualified execution observation; and
* the probe itself remains mechanism-neutral.  Exact source meaning is captured
  later by the executed operation observer; a plan cannot manufacture it from
  a class, field, or generic dependency summary.

No largest-container, familiar-name or configuration convention is used.  If
several containers qualify, all of them become plans.  Every examined container
that cannot qualify remains visible as a typed issue; lack of a supported plan
never means that the model has no relation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .program_index import ProgramIndex


_ORDERED_CONTAINER_TYPES = frozenset({
    ("torch.nn.modules.container", "ModuleList"),
    ("torch.nn.modules.container", "Sequential"),
})
_ISSUE_KINDS = frozenset({
    "target_not_ancestor",
    "container_protocol_invalid",
    "partial_invocation",
    "repeated_invocation",
    "execution_order_conflict",
    "call_class_mismatch",
})
_FAILURE_KINDS = frozenset({
    "base_execution_failed",
    "provenance_mismatch",
})


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _observation_hash(observation: Any) -> str:
    return _canonical_hash(dataclasses.asdict(observation))


def _child_path(container: str, child: str) -> str:
    return f"{container}.{child}".lstrip(".")


def _is_at_or_below(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(root + ".")


def _external_type(value: Any, module: str, qualname: str) -> bool:
    kind = type(value)
    return kind.__module__ == module and kind.__qualname__ == qualname


def _is_resolved_class(value: Any) -> bool:
    return _external_type(
        value, "physics.instance_inventory", "ResolvedClass")


def _is_module_node(value: Any) -> bool:
    return _external_type(value, "physics.instance_inventory", "ModuleNode")


def _is_module_call(value: Any) -> bool:
    return _external_type(
        value, "physics.execution_observation", "ModuleCall")


def _is_execution_recipe(value: Any) -> bool:
    return _external_type(
        value, "physics.execution_observation", "ExecutionRecipe")


def _is_execution_observation(value: Any) -> bool:
    return _external_type(
        value, "physics.execution_observation", "ExecutionObservation")


def _class_key(value: Any) -> tuple[str, str]:
    return value.module, value.qualname


@dataclass(frozen=True)
class RelationProbeIssue:
    """A visible reason an exact ordered container could not be probed.

    This is unresolved probe capability, not evidence that a relation is absent.
    ``member_paths`` and ``calls`` preserve the exact construction/execution side
    of the disagreement instead of reducing it to prose.
    """

    container: Any
    kind: str
    detail: str
    member_paths: tuple[str, ...]
    calls: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not _is_module_node(self.container):
            raise TypeError("a probe issue retains its exact inventory node")
        if _class_key(self.container.class_ref) not in _ORDERED_CONTAINER_TYPES:
            raise ValueError("a probe issue belongs to a supported container type")
        if self.kind not in _ISSUE_KINDS or not self.detail:
            raise ValueError("a probe issue needs a closed kind and detail")
        expected_paths = tuple(
            _child_path(self.container.path, child)
            for child in self.container.children
        )
        if self.member_paths != expected_paths:
            raise ValueError("a probe issue covers the exact direct-member census")
        if len(set(self.member_paths)) != len(self.member_paths):
            raise ValueError("probe-issue members must be occurrence-exact")
        if any(not _is_module_call(row) for row in self.calls):
            raise TypeError("probe-issue calls must be typed observations")
        member_set = set(self.member_paths)
        if any(row.path not in member_set for row in self.calls):
            raise ValueError("probe-issue calls belong to the cited members")

    @property
    def stack_path(self) -> str:
        return self.container.path


@dataclass(frozen=True)
class RelationProbePlan:
    """One exact, positively qualified recurrent-relation observation plan."""

    container: Any
    member_paths: tuple[str, ...]
    member_calls: tuple[Any, ...]
    base_recipe: Any
    observation: Any
    observation_sha256: str
    inventory_config_sha256: str
    index_fingerprint: str

    def __post_init__(self) -> None:
        if not _is_module_node(self.container):
            raise TypeError("a relation plan retains its exact inventory node")
        if _class_key(self.container.class_ref) not in _ORDERED_CONTAINER_TYPES:
            raise ValueError("a relation plan needs a closed ordered-container type")
        expected_paths = tuple(
            _child_path(self.container.path, child)
            for child in self.container.children
        )
        if len(expected_paths) < 2 or self.member_paths != expected_paths:
            raise ValueError("a relation plan covers every direct member in storage order")
        if (len(self.member_calls) != len(self.member_paths)
                or tuple(row.path for row in self.member_calls) != self.member_paths):
            raise ValueError("a relation plan has one ordered call per direct member")
        if len({row.index for row in self.member_calls}) != len(self.member_calls):
            raise ValueError("member calls must be occurrence-exact")
        if tuple(row.index for row in self.member_calls) != tuple(sorted(
                row.index for row in self.member_calls)):
            raise ValueError("member calls follow container storage order")
        if any(not _is_module_call(row) for row in self.member_calls):
            raise TypeError("member calls must be typed execution observations")
        for value in (
                self.observation_sha256,
                self.inventory_config_sha256,
                self.index_fingerprint,
        ):
            if (len(value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in value)):
                raise ValueError("relation plan fingerprints must be SHA-256")
        if not _is_execution_recipe(self.base_recipe):
            raise TypeError("a relation plan derives from one typed base recipe")
        if not _is_execution_observation(self.observation):
            raise TypeError("a relation plan retains its exact base observation")
        if self.observation.recipe != self.base_recipe:
            raise ValueError("the plan recipe must be the observed base recipe")
        if _observation_hash(self.observation) != self.observation_sha256:
            raise ValueError("the plan must cite its exact base observation")
        if (self.observation.provenance.config_sha256
                != self.inventory_config_sha256):
            raise ValueError("the plan observation and inventory config must agree")
        observed_calls = tuple(
            row for member in self.member_paths
            for row in self.observation.module_calls if row.path == member
        )
        if observed_calls != self.member_calls:
            raise ValueError("the plan calls must come from its exact observation")
        if not _is_at_or_below(self.container.path, self.base_recipe.target_path):
            raise ValueError("the observed target must contain the exact stack")

    @property
    def stack_path(self) -> str:
        return self.container.path

    def execution_recipe(self) -> Any:
        """Derive the probe recipe without changing any execution input.

        Only the audit identity/flags change.  Tensor and literal arguments,
        dtype, framework versions and target address remain those already proven
        executable by the base signature recipe.
        """
        address = _canonical_hash({
            "base_recipe": self.base_recipe.recipe_id,
            "stack_path": self.stack_path,
            "observation": self.observation_sha256,
        })[:16]
        flags = dict(self.base_recipe.flags)
        flags.update({
            "source": "resolved_signature_and_ordered_container_capability",
            "base_recipe_id": self.base_recipe.recipe_id,
            "relation_stack_path": self.stack_path,
            "base_observation_sha256": self.observation_sha256,
        })
        return dataclasses.replace(
            self.base_recipe,
            recipe_id=f"relation-{address}",
            flags=flags,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "stack_path": self.stack_path,
            "container": dataclasses.asdict(self.container),
            "container_class": dataclasses.asdict(self.container.class_ref),
            "member_paths": list(self.member_paths),
            "member_calls": [dataclasses.asdict(row) for row in self.member_calls],
            "base_recipe": self.base_recipe.to_dict(),
            "recipe": self.execution_recipe().to_dict(),
            "observation_sha256": self.observation_sha256,
            "inventory_config_sha256": self.inventory_config_sha256,
            "index_fingerprint": self.index_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt"] = RelationProbePlanReceipt.from_payload(
            payload).to_dict()
        return payload


@dataclass(frozen=True)
class RelationProbePlanReceipt:
    """Portable typed closure for one persisted probe plan.

    Runtime objects stay the authority while planning.  This receipt prevents
    their JSON projection from becoming an untyped second authority: every
    plan field is covered by one content digest and the exact recipe/stack
    identity later joined to the typed execution result.
    """

    stack_path: str
    recipe_id: str
    base_recipe_id: str
    plan_sha256: str

    def __post_init__(self) -> None:
        if not self.stack_path or not self.recipe_id or not self.base_recipe_id:
            raise ValueError("a relation-plan receipt needs exact identities")
        if (len(self.plan_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.plan_sha256)):
            raise ValueError("a relation-plan receipt needs a SHA-256 digest")

    @classmethod
    def from_payload(cls, payload: Any) -> "RelationProbePlanReceipt":
        if not isinstance(payload, dict):
            raise TypeError("a relation-plan receipt covers one mapping")
        recipe = payload.get("recipe")
        base = payload.get("base_recipe")
        if not isinstance(recipe, dict) or not isinstance(base, dict):
            raise ValueError("a relation-plan receipt needs both exact recipes")
        stack = payload.get("stack_path")
        recipe_id = recipe.get("recipe_id")
        base_id = base.get("recipe_id")
        if recipe.get("flags", {}).get("relation_stack_path") != stack:
            raise ValueError("a relation-plan receipt binds recipe to stack")
        covered = {key: value for key, value in payload.items()
                   if key != "receipt"}
        return cls(str(stack or ""), str(recipe_id or ""), str(base_id or ""),
                   _canonical_hash(covered))

    @classmethod
    def from_dict(cls, row: Any) -> "RelationProbePlanReceipt":
        if not isinstance(row, dict) or set(row) != {
                "stack_path", "recipe_id", "base_recipe_id", "plan_sha256"}:
            raise ValueError("relation-plan receipt schema is closed")
        return cls(str(row["stack_path"]), str(row["recipe_id"]),
                   str(row["base_recipe_id"]), str(row["plan_sha256"]))

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RelationProbeResolution:
    """Closed result of the supported recurrent-relation probe planner.

    ``absent`` means only that this particular supported planner found no ready
    plan.  It is never a semantic negative about the architecture.
    """

    status: str  # resolved | partial | unresolved | absent | failed
    plans: tuple[RelationProbePlan, ...] = ()
    issues: tuple[RelationProbeIssue, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {
                "resolved", "partial", "unresolved", "absent", "failed"}:
            raise ValueError("relation-probe resolution status is closed")
        if any(not isinstance(row, RelationProbePlan) for row in self.plans):
            raise TypeError("relation-probe plans are typed")
        if any(not isinstance(row, RelationProbeIssue) for row in self.issues):
            raise TypeError("relation-probe issues are typed")
        paths = tuple(row.stack_path for row in self.plans)
        issue_paths = tuple((row.stack_path, row.kind) for row in self.issues)
        if len(paths) != len(set(paths)):
            raise ValueError("each exact stack has at most one probe plan")
        if len(issue_paths) != len(set(issue_paths)):
            raise ValueError("each stack has at most one issue of each kind")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires a typed failure kind")
        if self.status == "resolved":
            if not self.plans or self.issues or self.failure_kind:
                raise ValueError("resolved carries plans and no unresolved evidence")
        elif self.status == "partial":
            if not self.plans or not self.issues or self.failure_kind:
                raise ValueError("partial carries both plans and unresolved evidence")
        elif self.status == "unresolved":
            if self.plans or not self.issues or self.failure_kind:
                raise ValueError("unresolved carries issues and no plan")
        elif self.status == "absent":
            if self.plans or self.issues or self.failure_kind:
                raise ValueError("absent carries no candidate or global failure")
        else:
            if (self.plans or self.issues or self.failure_kind not in _FAILURE_KINDS
                    or not self.failure_detail):
                raise ValueError("failed carries exactly one closed global failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plans": [row.to_dict() for row in self.plans],
            "issues": [dataclasses.asdict(row) for row in self.issues],
            "failure_kind": self.failure_kind,
            "failure_detail": self.failure_detail,
            "semantic_negative": False,
        }


def _issue(container: Any, kind: str, detail: str,
           calls: tuple[Any, ...] = ()) -> RelationProbeIssue:
    return RelationProbeIssue(
        container, kind, detail,
        tuple(_child_path(container.path, child)
              for child in container.children),
        calls,
    )


def _ordered_containers(inventory: Any) -> tuple[Any, ...]:
    return tuple(row for row in inventory.modules
                 if _class_key(row.class_ref) in _ORDERED_CONTAINER_TYPES)


def resolve_relation_probes(
    index: ProgramIndex,
    inventory: Any,
    execution: Any,
) -> RelationProbeResolution:
    """Return every evidence-qualified recurrent-relation probe plan.

    The successful execution is the capability/input authority.  The instance
    tree is the exact construction/address authority.  ProgramIndex source proof
    is the mechanism authority.  The three are joined, never substituted.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("relation probing requires ProgramIndex")
    if not _external_type(
            inventory, "physics.instance_inventory", "InstanceInventory"):
        raise TypeError("relation probing requires InstanceInventory")
    if not _external_type(
            execution, "physics.execution_observation", "ObservationResult"):
        raise TypeError("relation probing requires ObservationResult")
    if (execution.provenance is not None
            and execution.provenance != inventory.provenance):
        return RelationProbeResolution(
            "failed", failure_kind="provenance_mismatch",
            failure_detail=(
                "inventory and execution result cite different construction "
                "provenance"
            ),
        )
    if execution.status != "ok" or execution.observation is None:
        failure = execution.failure
        detail = (
            f"{failure.kind}:{failure.stage}:{failure.detail}"
            if failure is not None else "execution produced no observation"
        )
        return RelationProbeResolution(
            "failed", failure_kind="base_execution_failed",
            failure_detail=detail,
        )
    observation = execution.observation
    if (execution.provenance != inventory.provenance
            or observation.provenance != inventory.provenance
            or execution.recipe != observation.recipe):
        return RelationProbeResolution(
            "failed", failure_kind="provenance_mismatch",
            failure_detail=(
                "inventory, execution result and observation must cite the same "
                "construction provenance and recipe"
            ),
        )

    inventory_by_path = {row.path: row for row in inventory.modules}
    calls_by_path: dict[str, list[Any]] = {}
    for call in observation.module_calls:
        calls_by_path.setdefault(call.path, []).append(call)
    observation_sha256 = _observation_hash(observation)

    plans: list[RelationProbePlan] = []
    issues: list[RelationProbeIssue] = []
    for container in _ordered_containers(inventory):
        members = tuple(_child_path(container.path, child)
                        for child in container.children)
        if len(members) < 2:
            issues.append(_issue(
                container, "container_protocol_invalid",
                "the supported relation probe requires at least two direct members"))
            continue
        if (_class_key(container.class_ref)
                == ("torch.nn.modules.container", "ModuleList")):
            expected = tuple(str(index) for index in range(len(container.children)))
            if container.children != expected:
                issues.append(_issue(
                    container, "container_protocol_invalid",
                    "the exact ModuleList children are not a dense runtime sequence"))
                continue
        if not _is_at_or_below(container.path, observation.recipe.target_path):
            issues.append(_issue(
                container, "target_not_ancestor",
                "the successful recipe target does not contain this container"))
            continue

        member_calls = tuple(
            call for member in members for call in calls_by_path.get(member, ()))
        counts = {member: len(calls_by_path.get(member, ())) for member in members}
        if any(count == 0 for count in counts.values()):
            issues.append(_issue(
                container, "partial_invocation",
                "not every direct member was positively invoked by the base recipe",
                member_calls))
            continue
        if any(count != 1 for count in counts.values()):
            issues.append(_issue(
                container, "repeated_invocation",
                "the current observer cannot collapse repeated member invocations",
                member_calls))
            continue
        ordered_calls = tuple(calls_by_path[member][0] for member in members)
        mismatches = tuple(
            call.path for call in ordered_calls
            if call.path not in inventory_by_path
            or inventory_by_path[call.path].class_ref != call.class_ref
        )
        if mismatches:
            issues.append(_issue(
                container, "call_class_mismatch",
                "execution class evidence disagrees with the exact inventory member",
                ordered_calls))
            continue
        if tuple(row.index for row in ordered_calls) != tuple(sorted(
                row.index for row in ordered_calls)):
            issues.append(_issue(
                container, "execution_order_conflict",
                "direct members did not execute in stored-child order",
                ordered_calls))
            continue

        plans.append(RelationProbePlan(
            container=container,
            member_paths=members,
            member_calls=ordered_calls,
            base_recipe=observation.recipe,
            observation=observation,
            observation_sha256=observation_sha256,
            inventory_config_sha256=inventory.provenance.config_sha256,
            index_fingerprint=index.fingerprint,
        ))

    plans.sort(key=lambda row: row.stack_path)
    issues.sort(key=lambda row: (row.stack_path, row.kind, row.detail))
    if plans and issues:
        status = "partial"
    elif plans:
        status = "resolved"
    elif issues:
        status = "unresolved"
    else:
        status = "absent"
    return RelationProbeResolution(status, tuple(plans), tuple(issues))


__all__ = [
    "RelationProbeIssue",
    "RelationProbePlan",
    "RelationProbePlanReceipt",
    "RelationProbeResolution",
    "resolve_relation_probes",
]
