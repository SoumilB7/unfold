"""U2 projection receipts — the render half of the fact/projection contract.

A ``ProjectionReceipt`` is emitted by the ACTUAL render consumer that draws a
fact (the op/chip/card that puts the value on the page).  Where COR-2's
projection *obligations* say "this config occurrence was consumed into this
exact fact target," a receipt says "this exact fact target was drawn on this
surface at this node."  Net 2 joins the two — occurrence -> target -> receipt —
so a consumed structural value with no drawn witness is caught, and a drawn
claim that references no consumed/registered fact is a fabrication.

Coverage is owner/mechanism-SCOPED, never a global boolean: a scope is
"receipted" once its render consumer has been migrated to emit receipts.  Net 2
blocks obligations inside a receipted scope and leaves every other scope
advisory, so migrating one mechanism (the scheduler, next) can never make an
unrelated transformer or VAE obligation suddenly blocking.

This module is the authoritative typed channel; ``RenderEvent.facts_projected``
(the older key-set witness for net #13) stays as temporary compatibility.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionReceipt:
    """One drawn fact: the render surface's proof it projected a target."""

    fact_key: str            # "owner.fact", e.g. "root.vision.projector_out_features"
    owner: str               # the fact owner, e.g. "root.vision"
    mechanism: str           # the decision scope, e.g. "projector_out_width"
    surface: str             # render view name, e.g. "ops_vision_projector"
    node_path: tuple[str, ...]   # block path to the drawn node
    projection_kind: str     # "op" | "chip" | "card" | "prose"
    value_status_hash: str   # hash of (drawn value, evidence status)

    def scope(self) -> tuple[str, str]:
        return (self.owner, self.mechanism)

    def to_dict(self) -> dict:
        return {
            "fact_key": self.fact_key, "owner": self.owner,
            "mechanism": self.mechanism, "surface": self.surface,
            "node_path": list(self.node_path),
            "projection_kind": self.projection_kind,
            "value_status_hash": self.value_status_hash,
        }


def receipts_from_projects(projects, surface: str,
                           node_path: tuple[str, ...] = ()) -> tuple:
    """Build typed receipts from a surface's ``projects`` descriptors.

    Each descriptor is ``{owner, fact|fact_key, mechanism, projection_kind?,
    value?, status?}``.  Shared by every render consumer (graph drills and the
    architecture facts-only channel) so a receipt is built one way."""
    out = []
    for spec in projects or ():
        owner = str(spec["owner"])
        leaf = str(spec.get("fact") or spec["fact_key"])
        fact_key = leaf if leaf.startswith(owner + ".") else f"{owner}.{leaf}"
        out.append(ProjectionReceipt(
            fact_key=fact_key, owner=owner, mechanism=str(spec["mechanism"]),
            surface=surface, node_path=tuple(node_path),
            projection_kind=str(spec.get("projection_kind") or "op"),
            value_status_hash=value_status_hash(
                spec.get("value"), str(spec.get("status") or "unknown"))))
    return tuple(out)


def value_status_hash(value, status: str) -> str:
    """Stable short hash of a fact's drawn value and its evidence status — the
    handle a value-conformance net uses to prove the drawing did not drift from
    the ledgered value.  Never identity-derived (no class/model name)."""
    return hashlib.sha256(f"{status}\x00{value!r}".encode()).hexdigest()[:16]


def _claims():
    """The ONE registry: migrated claims.  A claim carrying a projection policy
    IS a receipted scope — there is no second table to drift."""
    from .registry import MIGRATED_SCOPES
    return MIGRATED_SCOPES


def receipted_scopes(claims=None) -> frozenset:
    """Scopes whose render consumer emits receipts, DERIVED from the claim
    register: exactly those claims that declare a projection policy."""
    return frozenset(c.scope() for c in (claims or _claims()) if c.projection)


def policy_for(owner: str, mechanism: str, claims=None):
    """The registered projection policy for a scope, or None."""
    for claim in (claims or _claims()):
        if claim.scope() == (owner, mechanism):
            return claim.projection
    return None


class _ReceiptedScopes(frozenset):
    """``RECEIPTED_SCOPES`` kept as a module constant for existing callers, but
    always the DERIVED set — never an independently maintained list."""


RECEIPTED_SCOPES: frozenset = _ReceiptedScopes(receipted_scopes())


def is_receipted_scope(owner: str, mechanism: str,
                       receipted_scopes=RECEIPTED_SCOPES) -> bool:
    return (owner, mechanism) in receipted_scopes


def join_obligation_receipts(obligations, receipts,
                             receipted_scopes=RECEIPTED_SCOPES) -> dict:
    """Join consumption obligations to render receipts, per scope.

    ``obligations`` are the serialized rows from
    ``ir.extras.config_access.projection_obligations`` (each carries source
    occurrence, target owner+key, and mechanism).  ``receipts`` are the typed
    receipts unioned from the render log.

    Applicability is decided UPSTREAM: an obligation exists only for a
    source-proven consumption under its true component owner, so a receipted
    (owner, mechanism) obligation UNCONDITIONALLY owes a matching receipt —
    a missing one is a blocking miss (the migrated consumer failed to draw a
    value it proved and consumed).  Absence of render output is never taken as
    proof of non-applicability.  Obligations outside a receipted scope produce
    no finding — they remain the advisory read-but-not-yet-receipted census.
    """
    by_target: dict[tuple[str, str, str], list] = {}
    for receipt in receipts:
        by_target.setdefault(
            (receipt.owner, receipt.fact_key, receipt.mechanism), []).append(receipt)
    findings: list[str] = []
    receipted_targets: set[tuple[str, str, str]] = set()
    for ob in obligations:
        owner = ob["target"]["owner"]
        key = ob["target"]["key"]
        mechanism = ob.get("mechanism", "")
        if (owner, mechanism) not in receipted_scopes:
            continue
        fact_key = f"{owner}.{key}"
        source = f"{ob['source']['component']}:{ob['source']['path']}"
        candidates = by_target.get((owner, fact_key, mechanism)) or []
        if not candidates:
            findings.append(
                f"{source} -> {owner}.{key} is in receipted scope "
                f"{owner}/{mechanism} but no render surface emitted a matching "
                "projection receipt (a consumed structural value the diagram "
                "does not draw)")
            continue
        # U2.1a: existence is not proof.  The receipt must come from an ALLOWED
        # surface and kind, and must carry the fingerprint the CONSUMPTION
        # expected — otherwise the drawing drifted from the consumed value or a
        # foreign surface cleared the obligation.
        policy = policy_for(owner, mechanism)
        expected = ob.get("expected_value_status_hash") or ""
        if not expected:
            # A receipted scope whose CONSUMPTION recorded no expected
            # fingerprint cannot be validated at all — any matching receipt
            # would clear it.  The consumption must pass status=; until it
            # does, the scope is unproven and BLOCKS.
            findings.append(
                f"{source} -> {owner}.{key} is in receipted scope "
                f"{owner}/{mechanism} but the consumption recorded NO expected "
                "value/status fingerprint (pass status= at consume) — a receipt "
                "cannot be validated against a missing expectation")
            continue
        accepted = []
        for receipt in candidates:
            if policy and receipt.surface not in policy.allowed_surfaces:
                findings.append(
                    f"{source} -> {owner}.{key}: receipt came from surface "
                    f"{receipt.surface!r}, which {owner}/{mechanism} is not "
                    f"allowed to be projected onto "
                    f"(allowed: {sorted(policy.allowed_surfaces)})")
                continue
            if policy and receipt.projection_kind not in policy.allowed_kinds:
                findings.append(
                    f"{source} -> {owner}.{key}: receipt declares projection "
                    f"kind {receipt.projection_kind!r}, not allowed for "
                    f"{owner}/{mechanism} (allowed: {sorted(policy.allowed_kinds)})")
                continue
            if policy and tuple(receipt.node_path) not in policy.allowed_node_paths:
                findings.append(
                    f"{source} -> {owner}.{key}: receipt names node path "
                    f"{tuple(receipt.node_path)!r}, not a registered consumer "
                    f"node for {owner}/{mechanism} "
                    f"(allowed: {sorted(policy.allowed_node_paths)})")
                continue
            if receipt.value_status_hash != expected:
                findings.append(
                    f"{source} -> {owner}.{key}: the drawn value/status "
                    f"fingerprint {receipt.value_status_hash!r} does not match "
                    f"the fingerprint {expected!r} recorded when the value was "
                    "consumed — the drawing drifted from the consumed value")
                continue
            accepted.append(receipt)
        if accepted:
            receipted_targets.add((owner, fact_key, mechanism))
        elif not findings:
            findings.append(
                f"{source} -> {owner}.{key}: no valid receipt "
                f"for {owner}/{mechanism}")
    return {"findings": sorted(set(findings)),
            "receipted_targets": sorted(receipted_targets)}


def fabrication_findings(receipts, registered_keys, claimed_targets,
                         debt_keys) -> list[str]:
    """Reverse-fabrication: every emitted receipt must reference a fact that is
    a registered ledger fact, a declared migration-claim target, or a
    registered (shrinking) typed-debt entry.  A receipt for an unregistered
    target is a drawn claim with no evidence behind it."""
    findings: list[str] = []
    for receipt in receipts:
        target = (receipt.owner, receipt.fact_key.rsplit(".", 1)[-1])
        if (receipt.fact_key in registered_keys
                or target in claimed_targets
                or target in debt_keys):
            continue
        findings.append(
            f"receipt for {receipt.fact_key} on surface {receipt.surface!r} "
            "references no registered fact, migration-claim target, or typed "
            "debt entry — a drawn claim with nothing behind it")
    return sorted(set(findings))


__all__ = [
    "ProjectionReceipt", "value_status_hash", "receipts_from_projects",
    "RECEIPTED_SCOPES", "receipted_scopes", "policy_for",
    "is_receipted_scope", "join_obligation_receipts",
    "fabrication_findings",
]
