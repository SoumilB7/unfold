"""U9-A — exact inventory of root-owned source components.

This is an address boundary, not a modality or mechanism classifier.  It joins
the SourceBundle's already-resolved component addresses to the U3 construction
graph.  A component is ``active`` only when an exact config-scope construction
and installation proves ownership.  Merely declaring a nested config/source is
``declared_unused`` and cannot create a tower, projector, or fusion route.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    resolve_component_root,
)
from .config_scoped_owner import resolve_config_constructed_root
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan


_ENTRY_STATUSES = frozenset({
    "active", "declared_unused", "ambiguous", "failed", "unavailable",
})


@dataclass(frozen=True)
class ComponentOwnerEntry:
    """One exact component address and its root-ownership disposition."""

    component_key: str
    config_path: tuple[str, ...]
    declared_architecture: str | None
    status: str
    component_root: ComponentRootResolution | ConstructedComponentRoot | None = None
    rival_spans: tuple[SourceSpan, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if not self.component_key:
            raise ValueError("a component-owner entry needs a component key")
        expected_path = () if self.component_key == "root" else tuple(
            self.component_key.split("."))
        if self.config_path != expected_path or any(not part for part in self.config_path):
            raise ValueError("component key and exact config path must agree")
        if self.status not in _ENTRY_STATUSES:
            raise ValueError(f"unknown component-owner status {self.status!r}")
        if any(not isinstance(span, SourceSpan) for span in self.rival_spans):
            raise TypeError("component-owner rivals carry exact source spans")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a typed failure kind")
        if self.status == "active":
            if self.component_root is None or self.rival_spans or self.failure_kind:
                raise ValueError("active carries one exact component root only")
            if self.component_key == "root":
                if not isinstance(self.component_root, ComponentRootResolution) \
                        or self.component_root.status != "resolved":
                    raise ValueError("the active root is a resolved D0 address")
            elif not isinstance(self.component_root, ConstructedComponentRoot) \
                    or self.component_root.config_path != self.config_path:
                raise ValueError("an active child is its exact constructed root")
        elif self.status == "ambiguous":
            if len(self.rival_spans) < 2 or self.component_root is not None \
                    or self.failure_kind:
                raise ValueError("ambiguous preserves at least two exact rival spans")
        elif self.status in {"failed", "unavailable"}:
            if not self.failure_kind or self.component_root is not None \
                    or self.rival_spans:
                raise ValueError("failed/unavailable carries typed failure only")
        elif self.component_root is not None or self.rival_spans or self.failure_kind:
            raise ValueError("declared-unused carries no active structural proof")


@dataclass(frozen=True)
class ComponentOwnerInventory:
    """Closed, canonically ordered inventory for one SourceBundle."""

    root: ComponentRootResolution
    entries: tuple[ComponentOwnerEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root, ComponentRootResolution):
            raise TypeError("component inventory carries its D0 root result")
        if any(not isinstance(item, ComponentOwnerEntry) for item in self.entries):
            raise TypeError("component inventory entries are typed")
        keys = tuple(item.component_key for item in self.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("component inventory is unique and canonically ordered")
        root_entries = tuple(item for item in self.entries
                             if item.component_key == "root")
        if len(root_entries) != 1:
            raise ValueError("component inventory contains exactly one root entry")
        root_entry = root_entries[0]
        if self.root.status == "resolved":
            if root_entry.status != "active" or root_entry.component_root != self.root:
                raise ValueError("resolved D0 root is the active root entry")
        elif root_entry.status == "active":
            raise ValueError("an unresolved D0 root cannot be active")

    def entry(self, component_key: str) -> ComponentOwnerEntry | None:
        return next((item for item in self.entries
                     if item.component_key == component_key), None)

    @property
    def active(self) -> tuple[ComponentOwnerEntry, ...]:
        return tuple(item for item in self.entries if item.status == "active")


def resolve_component_inventory(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    config_selector=None,
) -> ComponentOwnerInventory:
    """Classify every non-pipeline source component through exact ownership.

    Pipeline slots are sibling models resolved by the diffusion/compound
    adapter and belong to U10.  U9 inventories the root model's own nested
    config components only; it never reinterprets a pipeline slot as a child.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("component inventory requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("component inventory requires a SourceBundle")
    root = resolve_component_root(index, bundle, "root")
    entries = [_root_entry(root)]
    pipeline = tuple(bundle.pipeline_components or ())
    keys = set((bundle.component_files or {}).keys()) \
        | set((bundle.component_architectures or {}).keys())
    keys.discard("root")
    # Pipeline components are sibling models owned by U10.  Their recursively
    # exposed descendants (``text_encoder.vision_config``) remain inside that
    # sibling too; treating them as U9 root children would falsely label them
    # declared-unused merely because the root denoiser never constructs them.
    keys = {
        key for key in keys
        if not any(key == slot or key.startswith(f"{slot}.")
                   for slot in pipeline)
    }
    for component_key in sorted(keys):
        path = tuple(component_key.split("."))
        architecture = (bundle.component_architectures or {}).get(component_key)
        if root.status != "resolved":
            entries.append(ComponentOwnerEntry(
                component_key, path, architecture, "unavailable",
                failure_kind="root_unresolved",
                failure_detail=(
                    f"root component address is {root.status}; nested ownership "
                    "cannot be proven"),
            ))
            continue
        result = resolve_config_constructed_root(
            index, bundle, root, path, config_selector=config_selector)
        if result.status == "resolved":
            entries.append(ComponentOwnerEntry(
                component_key, path, architecture, "active",
                component_root=result.candidate.component_root,
            ))
        elif result.status == "ambiguous":
            spans = tuple(dict.fromkeys(
                span for rival in result.rivals for span in rival.spans))
            entries.append(ComponentOwnerEntry(
                component_key, path, architecture, "ambiguous",
                rival_spans=spans,
            ))
        elif result.status == "failed":
            entries.append(ComponentOwnerEntry(
                component_key, path, architecture, "failed",
                failure_kind=result.failure_kind,
                failure_detail=result.failure_detail,
            ))
        else:
            entries.append(ComponentOwnerEntry(
                component_key, path, architecture, "declared_unused"))
    return ComponentOwnerInventory(root, tuple(sorted(
        entries, key=lambda item: item.component_key)))


def _root_entry(root: ComponentRootResolution) -> ComponentOwnerEntry:
    if root.status == "resolved":
        return ComponentOwnerEntry(
            "root", (), root.declared_architecture, "active",
            component_root=root)
    if root.status == "ambiguous":
        return ComponentOwnerEntry(
            "root", (), root.declared_architecture, "ambiguous",
            rival_spans=tuple(candidate.span for candidate in root.candidates))
    if root.status == "failed":
        return ComponentOwnerEntry(
            "root", (), root.declared_architecture, "failed",
            failure_kind="parse_failure",
            failure_detail="; ".join(
                f"{item.kind}: {item.detail}" for item in root.parse_failures))
    return ComponentOwnerEntry(
        "root", (), root.declared_architecture, "unavailable",
        failure_kind="root_absent",
        failure_detail="the bundle declares no unique root architecture")


__all__ = [
    "ComponentOwnerEntry", "ComponentOwnerInventory",
    "resolve_component_inventory",
]
