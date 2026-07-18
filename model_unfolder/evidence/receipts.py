"""U2-R5 projection receipts — the render half of the fact/projection contract.

A ``ProjectionReceipt`` is emitted INSIDE the actual projector (the code that
draws the value on a surface).  Where a consumption obligation says "this config
occurrence was consumed into this exact fact target," and the typed fact says
"this is the value and its evidence status," a receipt says "this exact fact was
drawn on this canonical surface, at this structural target, by this projector."
The validator joins all three:

    consumed occurrence -> exact target fact -> registered fact definition
      -> allowed projection route -> receipt from the actual projector
      -> matching fact value/status hash

**FactDefinition is the sole route authority** (§5.5): a fact declares its
``projection_routes`` and the receipted-scope set and every validation rule
derive from the REGISTRY.  The old ``ProjectionPolicy`` on ``MigrationClaim`` is
deleted — a claim binds a source occurrence to a fact; the fact owns where it
may project.

The EXPECTED hash originates from the typed fact and the consumption — never
from a renderer-created ``projects=[{value, status}]`` descriptor.  The receipt
records what WAS drawn; the fact records what SHOULD have been drawn; the
validator refuses a mismatch in either direction, and refuses a consumption
whose own fingerprint disagrees with the fact (drift between the two upstream
authorities is a finding, not a tie to break).

Coverage is owner/mechanism-SCOPED, never a global boolean, so migrating one
mechanism can never make an unrelated obligation suddenly blocking.
``RenderEvent.facts_projected`` (the key-set witness for net #13) stays as
shadow compatibility until parity is proven for each migrated route.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProjectionReceipt:
    """One drawn fact: the actual projector's proof of what it projected (§5.5)."""

    fact_id: str             # "<owner>.<fact_key>" — the ledger key it cites
    owner: str               # the fact owner, e.g. "root.vision"
    fact_key: str            # the fact leaf, e.g. "projector_out_features"
    mechanism: str           # the decision scope, e.g. "projector_out_width"
    fact_value_status_hash: str   # hash of (drawn value, cited status)
    surface: str             # ONE canonical surface: ir|spec|opgraph|block|card|html|json|params|conformance
    structural_target: str   # the drawn structural node, e.g. "vision_projector"
    projector_symbol: str    # the ACTUAL projector (module.function) that emitted
    node_ids: tuple = ()     # drawn node identity within the surface
    # R5-vet: HOW the fact appears on the surface — validated against the
    # route's projection_kinds (a declared-but-unchecked field is decoration).
    projection_kind: str = ""
    output_hash: "str | None" = None
    # U2-R5 (context validation): stamped by the RENDER CONTEXT itself when the
    # receipt is recorded — a receipt smuggled from another parse/render carries
    # a foreign token and cannot clear this render's obligations.
    context_token: str = ""

    def scope(self) -> tuple[str, str]:
        return (self.owner, self.mechanism)

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id, "owner": self.owner,
            "fact_key": self.fact_key, "mechanism": self.mechanism,
            "fact_value_status_hash": self.fact_value_status_hash,
            "surface": self.surface,
            "structural_target": self.structural_target,
            "projector_symbol": self.projector_symbol,
            "node_ids": list(self.node_ids),
            "output_hash": self.output_hash,
        }


def value_status_hash(value, status: str) -> str:
    """Stable short hash of a fact's value and its evidence status — the handle
    the validator uses to prove the drawing did not drift from the ledgered
    fact.  Never identity-derived (no class/model name)."""
    return hashlib.sha256(f"{status}\x00{value!r}".encode()).hexdigest()[:16]


def receipts_from_projects(projects, *, surface: str, structural_target: str,
                           projector_symbol: str, node_ids: tuple = (),
                           projection_kind: str = "", fact_rows=None) -> tuple:
    """Build typed receipts from a projector's ``projects`` descriptors.

    Called INSIDE the actual projector (the code drawing the value) — never from
    an upstream helper that guesses whether drawing happened.  Each descriptor is
    ``{owner, fact|fact_key, mechanism, value}``: the DRAWN value only.  The
    evidence STATUS is cited from ``fact_rows`` (the parse's ledgered
    ``fact_provenance``) — one author, the typed fact; the renderer never
    derives or invents a tier.  The hash binds (drawn value, ledgered status),
    so a drawn value that drifted from the fact fails the join, and a missing
    fact hashes against "unknown" and fails the join's fact check first."""
    out = []
    fact_rows = fact_rows or {}
    for spec in projects or ():
        owner = str(spec["owner"])
        leaf = str(spec.get("fact") or spec["fact_key"])
        fact_id = leaf if leaf.startswith(owner + ".") else f"{owner}.{leaf}"
        row = fact_rows.get(fact_id)
        # R5-vet: the renderer may NEVER supply a status.  Missing ledgered fact
        # -> the cited status is the explicit non-status "unledgered", whose
        # hash can never match a real fact; the join's missing-fact check
        # blocks first anyway.  (The old fallback to spec["status"] let the
        # descriptor certify itself.)
        cited_status = (str(row.get("status"))
                        if isinstance(row, dict) and row.get("status")
                        else "unledgered")
        out.append(ProjectionReceipt(
            fact_id=fact_id, owner=owner, fact_key=fact_id.rsplit(".", 1)[-1],
            mechanism=str(spec["mechanism"]), surface=str(surface),
            structural_target=str(structural_target),
            projector_symbol=str(projector_symbol),
            node_ids=tuple(node_ids), projection_kind=str(projection_kind),
            fact_value_status_hash=value_status_hash(
                spec.get("value"), cited_status)))
    return tuple(out)


def stamp_context(receipts, context_token: str) -> tuple:
    """The RENDER CONTEXT stamps its own token onto every receipt it records —
    the projector cannot forge another context's token, so a receipt can only
    clear obligations joined against the render that actually drew it."""
    return tuple(replace(r, context_token=context_token) for r in receipts)


# --------------------------------------------------------------------------- #
# Route authority — derived from the FactDefinition registry, nowhere else
# --------------------------------------------------------------------------- #

def projection_routes_by_fact() -> dict:
    """Every registered route KEYED BY ITS FACT — the matched route must belong
    to the exact FactDefinition the obligation targets, never merely another
    fact sharing an owner/mechanism scope (R5-vet)."""
    from .registry import REGISTRY
    out: dict = {}
    for key, definition in REGISTRY.items():
        routes = tuple(getattr(definition, "projection_routes", ()) or ())
        if routes:
            out[key] = routes
    return out


def projection_routes():
    """Flat view of every registered route (scope derivation)."""
    routes = []
    for fact_routes in projection_routes_by_fact().values():
        routes.extend(fact_routes)
    return tuple(routes)


def receipted_scopes(routes=None) -> frozenset:
    """Scopes whose facts declare projection routes — the migrated set,
    DERIVED from the registry (never an independently maintained list)."""
    return frozenset(r.scope() for r in (routes if routes is not None
                                         else projection_routes()))


def scope_is_receipted(owner: str, mechanism: str, scopes) -> bool:
    """Normalized membership: a ``layers[i]`` route pattern covers every
    concrete index, so future per-layer routes need no new machinery."""
    from .registry import _normalize_owner
    return ((owner, mechanism) in scopes
            or (_normalize_owner(owner), mechanism) in scopes)


def is_receipted_scope(owner: str, mechanism: str, scopes=None) -> bool:
    return scope_is_receipted(owner, mechanism,
                              scopes if scopes is not None
                              else receipted_scopes())


class _LazyReceiptedScopes:
    """``RECEIPTED_SCOPES`` kept as a module constant for existing callers, but
    ALWAYS the registry-derived set at the moment of use (never a snapshot that
    could go stale against the one authority)."""

    def __contains__(self, scope) -> bool:
        return scope in receipted_scopes()

    def __iter__(self):
        return iter(sorted(receipted_scopes()))

    def __len__(self) -> int:
        return len(receipted_scopes())


RECEIPTED_SCOPES = _LazyReceiptedScopes()


# --------------------------------------------------------------------------- #
# The validator
# --------------------------------------------------------------------------- #

def join_obligation_receipts(obligations, receipts, facts=None, *,
                             context_token: str = "",
                             scopes=None, routes=None) -> dict:
    """Join consumption obligations, typed facts, and render receipts (§5.5).

    ``obligations`` are the serialized rows from
    ``ir.extras.config_access.projection_obligations``.  ``facts`` is the parse's
    ``fact_provenance`` mapping (``{"owner.key": {"value":…, "status":…}}``) —
    the EXPECTED value/status originates HERE and at the consumption, never from
    the renderer.  ``receipts`` are the typed receipts from THIS render's log;
    ``context_token`` is that render's own token, so a foreign receipt is
    rejected before any other field is considered.

    Applicability is decided UPSTREAM: an obligation exists only for a
    source-proven consumption under its true owner, so a receipted
    (owner, mechanism) obligation UNCONDITIONALLY owes a valid receipt.
    Absence of render output is never proof of non-applicability.  Obligations
    outside a receipted scope produce no finding — they remain the advisory
    census.
    """
    live_scopes = scopes if scopes is not None else receipted_scopes()
    routes_by_fact = (routes if routes is not None
                      else projection_routes_by_fact())
    facts = facts or {}
    from .registry import owner_matches_pattern

    by_target: dict[tuple[str, str, str], list] = {}
    for receipt in receipts:
        by_target.setdefault(
            (receipt.owner, receipt.fact_id, receipt.mechanism), []).append(receipt)

    findings: list[str] = []
    receipted_targets: set[tuple[str, str, str]] = set()
    for ob in obligations:
        owner = ob["target"]["owner"]
        key = ob["target"]["key"]
        mechanism = ob.get("mechanism", "")
        if not scope_is_receipted(owner, mechanism, live_scopes):
            continue
        fact_id = f"{owner}.{key}"
        source = f"{ob['source']['component']}:{ob['source']['path']}"
        # R5-vet: an empty join context DISABLES nothing — it blocks.  A
        # receipted join that cannot say which render it is joining against
        # cannot accept any receipt.
        if not context_token:
            findings.append(
                f"{source} -> {fact_id} is in receipted scope "
                f"{owner}/{mechanism} but the join carries NO render-context "
                "token — a context-less join cannot validate any receipt")
            continue
        # R5-vet: the routes must belong to THIS fact's definition — never to
        # another fact that happens to share the (owner, mechanism) scope.
        scope_routes = tuple(
            r for r in routes_by_fact.get(key, ())
            if r.mechanism == mechanism
            and owner_matches_pattern(owner, r.owner_pattern))
        if not scope_routes:
            findings.append(
                f"{source} -> {fact_id} is treated as receipted but no "
                f"FactDefinition declares a projection route for "
                f"{owner}/{mechanism} — the registry is the sole route "
                "authority and it is silent")
            continue
        # THE EXPECTED HASH: from the typed FACT.  A consumption that never
        # became a ledgered fact cannot be receipted — that absence is itself
        # the finding, never a pass.
        fact_row = facts.get(fact_id)
        if not isinstance(fact_row, dict) or "value" not in fact_row:
            findings.append(
                f"{source} -> {fact_id} is in receipted scope "
                f"{owner}/{mechanism} but NO typed fact is ledgered for it — "
                "a consumption that never became a fact cannot be receipted")
            continue
        expected_fact = value_status_hash(
            fact_row.get("value"), str(fact_row.get("status")))
        expected_consumption = ob.get("expected_value_status_hash") or ""
        if not expected_consumption:
            findings.append(
                f"{source} -> {fact_id} is in receipted scope "
                f"{owner}/{mechanism} but the consumption recorded NO expected "
                "fingerprint (pass status= at consume) — a receipt cannot be "
                "validated against a missing expectation")
            continue
        if expected_consumption != expected_fact:
            findings.append(
                f"{source} -> {fact_id}: the consumption's fingerprint "
                f"{expected_consumption!r} disagrees with the typed fact's "
                f"{expected_fact!r} — the fact drifted from its own consumption")
            continue

        candidates = by_target.get((owner, fact_id, mechanism)) or []
        if not candidates:
            findings.append(
                f"{source} -> {fact_id} is in receipted scope "
                f"{owner}/{mechanism} but no projector emitted a matching "
                "receipt (a consumed, ledgered fact the diagram does not draw)")
            continue
        accepted = []
        for receipt in candidates:
            # R5-vet: internal coherence FIRST — a receipt whose fact_id does
            # not equal owner.fact_key is malformed, and its fact_key must be
            # the obligation's exact target key (the by_target index alone let
            # a wrong fact_key ride a correct fact_id).
            if receipt.fact_id != f"{receipt.owner}.{receipt.fact_key}":
                findings.append(
                    f"{source} -> {fact_id}: receipt is MALFORMED — fact_id "
                    f"{receipt.fact_id!r} does not equal owner.fact_key "
                    f"{receipt.owner!r}.{receipt.fact_key!r}")
                continue
            if receipt.fact_key != key:
                findings.append(
                    f"{source} -> {fact_id}: receipt cites fact_key "
                    f"{receipt.fact_key!r}, not the obligation's target "
                    f"{key!r}")
                continue
            if receipt.context_token != context_token:
                findings.append(
                    f"{source} -> {fact_id}: receipt carries a FOREIGN render-"
                    "context token — a receipt from another parse/render cannot "
                    "clear this one's obligations")
                continue
            matched_route = None
            for route in scope_routes:
                if receipt.surface != route.surface:
                    continue
                if receipt.structural_target != route.structural_target:
                    continue
                if route.node_paths and tuple(receipt.node_ids) not in route.node_paths:
                    continue
                # R5-vet: the KIND participates — a route that allows an "op"
                # does not thereby allow a "prose" claim of the same value.
                if receipt.projection_kind not in route.projection_kinds:
                    continue
                # R5-vet: EXACT projector-symbol membership — "any nonempty
                # symbol" validated nothing.
                if receipt.projector_symbol not in route.projector_symbols:
                    continue
                matched_route = route
                break
            if matched_route is None:
                findings.append(
                    f"{source} -> {fact_id}: receipt (surface="
                    f"{receipt.surface!r}, target={receipt.structural_target!r}, "
                    f"nodes={tuple(receipt.node_ids)!r}, "
                    f"kind={receipt.projection_kind!r}, "
                    f"projector={receipt.projector_symbol!r}) matches no "
                    f"registered projection route for {owner}/{mechanism}")
                continue
            if receipt.fact_value_status_hash != expected_fact:
                findings.append(
                    f"{source} -> {fact_id}: the drawn value/status fingerprint "
                    f"{receipt.fact_value_status_hash!r} does not match the "
                    f"typed fact's {expected_fact!r} — the drawing drifted from "
                    "the ledgered fact")
                continue
            accepted.append(receipt)
        if accepted:
            receipted_targets.add((owner, fact_id, mechanism))
    return {"findings": sorted(set(findings)),
            "receipted_targets": sorted(receipted_targets)}


def fabrication_findings(receipts, facts, claimed_targets) -> list[str]:
    """Reverse-fabrication, R5-vet strengthened.

    A registered LEAF NAME is not evidence — ``root.ghost.projector_out_
    features`` shares a registered leaf and used to pass.  Every emitted receipt
    must now reference:

    * an ACTUAL LEDGERED FACT (``facts`` is this parse's fact_provenance) — or a
      declared migration-claim target / registered typed-debt entry for
      unmigrated channels;
    * an owner the fact's DEFINITION permits (normalized owner_patterns);
    * a registered projection route the receipt actually matches (owner,
      mechanism, surface, target, kind, symbol) when the definition declares
      routes.
    """
    from .registry import REGISTRY, owner_matches_pattern
    findings: list[str] = []
    facts = facts or {}
    for receipt in receipts:
        target = (receipt.owner, receipt.fact_key)
        # Soumil's final vet: pending config_read/classification DEBT can
        # NEVER authorize a receipt — a pending INPUT classification is not
        # permission to draw an architectural OUTPUT.  A receipt cites an
        # actual typed fact or an exact migration-claim target; nothing else.
        if receipt.fact_id not in facts and target not in claimed_targets:
            findings.append(
                f"receipt for {receipt.fact_id} on surface {receipt.surface!r} "
                "references no LEDGERED fact, migration-claim target, or typed "
                "debt entry — a drawn claim with nothing behind it")
            continue
        definition = REGISTRY.get(receipt.fact_key)
        if definition is not None:
            if not any(owner_matches_pattern(receipt.owner, pattern)
                       for pattern in definition.owner_patterns):
                findings.append(
                    f"receipt for {receipt.fact_id}: owner {receipt.owner!r} is "
                    f"outside {receipt.fact_key!r}'s registered owner patterns "
                    f"{sorted(definition.owner_patterns)}")
                continue
            routes = tuple(getattr(definition, "projection_routes", ()) or ())
            if routes and not any(
                    owner_matches_pattern(receipt.owner, r.owner_pattern)
                    and receipt.mechanism == r.mechanism
                    and receipt.surface == r.surface
                    and receipt.structural_target == r.structural_target
                    and receipt.projection_kind in r.projection_kinds
                    and receipt.projector_symbol in r.projector_symbols
                    for r in routes):
                findings.append(
                    f"receipt for {receipt.fact_id} matches none of "
                    f"{receipt.fact_key!r}'s registered projection routes")
    return sorted(set(findings))


__all__ = [
    "ProjectionReceipt", "value_status_hash", "receipts_from_projects",
    "stamp_context", "projection_routes", "receipted_scopes", "routes_for",
    "RECEIPTED_SCOPES", "is_receipted_scope", "join_obligation_receipts",
    "fabrication_findings",
]
