"""Migration-claim validation and corpus coverage — ONE shared implementation.

Fifth directive (U0/U1 close): claims match EXACTLY —

    owner + mechanism + config occurrence
        -> target owner + fact key + sink kind

There is no cross-mechanism target union: every consumption inside a claimed
scope is judged strictly under ITS OWN mechanism's binding for that exact
path.  A consumption that is untagged, tagged with a mechanism that declares
no binding for the path, or aimed at the wrong fact/sink kind is a violation.

Anti-vacuity is corpus-level and BINDING-level: every declared path-to-target
binding must be observed and target-matched on at least one (real or
synthetic) witness.  The real corpus gate and every poison call the same two
functions below — the gate cannot drift from what the poisons prove.
"""
from __future__ import annotations

from typing import Any, Iterable


def _triple(target) -> tuple[str, str, str]:
    return (target.owner, target.fact_key, target.structural_sink_kind)


def _event_triple(event) -> tuple[str, str, str]:
    target = event.projection_target
    if target is None:
        return (event.fact_owner or event.component,
                event.fact_key or event.canonical, "fact")
    return _triple(target)


def validate_claims(events: Iterable[Any], scopes: Iterable[Any],
                    *, classified_paths: set[tuple[str, str]] = frozenset(),
                    ) -> list[dict]:
    """Per-parse claim validation — one structured row per claim.

    Row shape: scope, claimed_by, mechanism, bindings (each with its OWN
    observed_events / target_matches), row totals, violations.
    """
    events = list(events)
    # (owner, path) -> {mechanism: binding-target triple}
    declared: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = {}
    for claim in scopes:
        for binding in claim.bindings:
            declared.setdefault((claim.owner, binding.config_path), {})[
                claim.mechanism] = _triple(binding.target)
    ignored_paths = {(e.component, e.config_path)
                     for e in events if e.intent == "ignored"}

    def _lawful_consumption_exists(key: tuple[str, str]) -> bool:
        """Some consumption at this occurrence matches ITS OWN mechanism's
        declared binding — the only thing that excuses a present read."""
        mechanisms = declared.get(key, {})
        for e in events:
            if (e.component, e.config_path) != key or e.intent != "consumed":
                continue
            expected = mechanisms.get(getattr(e, "mechanism", ""))
            if expected is not None and _event_triple(e) == expected:
                return True
        return False

    rows: list[dict] = []
    for claim in scopes:
        violations: list[str] = []
        binding_rows: list[dict] = []
        observed_total = 0
        matches_total = 0
        for binding in claim.bindings:
            key = (claim.owner, binding.config_path)
            scope_events = [e for e in events
                            if (e.component, e.config_path) == key]
            expected = _triple(binding.target)
            observed = len(scope_events)
            matches = 0
            for e in scope_events:
                if not getattr(e, "path_exact", False):
                    violations.append(
                        f"{claim.owner}/{claim.mechanism}: inexact read of "
                        f"{e.config_path!r} — a claimed scope may not read "
                        "through the bare funnel")
                    continue
                if e.intent == "consumed":
                    mechanism = getattr(e, "mechanism", "")
                    if mechanism == claim.mechanism:
                        actual = _event_triple(e)
                        if actual == expected:
                            matches += 1
                        elif actual[:2] != expected[:2]:
                            violations.append(
                                f"{claim.owner}/{claim.mechanism}: "
                                f"{binding.config_path!r} consumed into WRONG "
                                f"fact {actual[0]}.{actual[1]} (declared "
                                f"{expected[0]}.{expected[1]}) — "
                                "source-to-target drift")
                        else:
                            violations.append(
                                f"{claim.owner}/{claim.mechanism}: "
                                f"{binding.config_path!r} consumed with WRONG "
                                f"sink kind {actual[2]!r} (declared "
                                f"{expected[2]!r})")
                    elif mechanism not in declared.get(key, {}):
                        label = mechanism or "<untagged>"
                        violations.append(
                            f"{claim.owner}/{claim.mechanism}: "
                            f"{binding.config_path!r} consumed under mechanism "
                            f"{label!r} which declares no binding for this "
                            "path — wrong mechanism")
                    # else: another claim's mechanism owns this consumption and
                    # judges it under ITS binding — never a union here.
                elif (e.present and e.intent in {"inspected", "bound"}
                        and not _lawful_consumption_exists(key)
                        and key not in ignored_paths
                        and key not in classified_paths):
                    violations.append(
                        f"{claim.owner}/{claim.mechanism}: present read of "
                        f"{e.config_path!r} is not consumed under any declared "
                        "mechanism binding, scoped-ignored, or precisely "
                        "classified")
            observed_total += observed
            matches_total += matches
            binding_rows.append({
                "path": binding.config_path,
                "target_owner": binding.target.owner,
                "target_key": binding.target.fact_key,
                "target_kind": binding.target.structural_sink_kind,
                "observed_events": observed,
                "target_matches": matches,
            })
        rows.append({
            "scope": f"{claim.owner}/{claim.mechanism}",
            "claimed_by": claim.claimed_by,
            "bindings": binding_rows,
            "observed_events": observed_total,
            "target_matches": matches_total,
            "violations": sorted(set(violations)),
        })
    return rows


def binding_id(scope: str, binding_row: dict) -> str:
    return (f"{scope}::{binding_row['path']} -> "
            f"{binding_row['target_owner']}.{binding_row['target_key']}"
            f"[{binding_row['target_kind']}]")


def audit_claim_coverage(rows_by_witness: dict[str, list[dict]],
                         scopes: Iterable[Any]) -> dict:
    """Corpus/binding-level anti-vacuity — the ONE gate function.

    Every declared path-to-target binding must be observed AND target-matched
    on at least one witness whose claim row is violation-free.  Per-witness
    zero observations stay lawful; a binding satisfied NOWHERE is vacuous and
    is returned in ``unwitnessed``.
    """
    needed: dict[str, None | str] = {}
    for claim in scopes:
        scope = f"{claim.owner}/{claim.mechanism}"
        for binding in claim.bindings:
            needed[binding_id(scope, {
                "path": binding.config_path,
                "target_owner": binding.target.owner,
                "target_key": binding.target.fact_key,
                "target_kind": binding.target.structural_sink_kind,
            })] = None
    witness_violations: dict[str, list[str]] = {}
    for witness, rows in rows_by_witness.items():
        for row in rows:
            if row["violations"]:
                witness_violations.setdefault(witness, []).extend(
                    row["violations"])
                continue
            for binding_row in row.get("bindings") or []:
                if (binding_row["observed_events"] > 0
                        and binding_row["target_matches"] > 0):
                    key = binding_id(row["scope"], binding_row)
                    if key in needed and needed[key] is None:
                        needed[key] = witness
    return {
        "unwitnessed": sorted(k for k, v in needed.items() if v is None),
        "satisfied": {k: v for k, v in needed.items() if v is not None},
        "witness_violations": witness_violations,
    }


__all__ = ["validate_claims", "audit_claim_coverage", "binding_id"]
