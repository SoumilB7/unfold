"""H4 (early slice) — the fact-provenance rule + generalized identity-table
negative controls.

The plan (§11) lands the H4 EARLY SLICE right after H1+H2, before H3, so U6
cannot add another bypass while the full dataflow taint net (item 2) is built
later.  This slice proves two things mechanically:

* item 1 — a structural fact that CLAIMS code/derived evidence may not cite an
  identity field or a declared class-vocabulary table as its deciding source;
  the config_declared / class_default tiers stay exempt (reading the
  checkpoint's own declaration is lawful);
* item 6 — the known evasions (rename, relocate Python<->YAML, intermediate
  enum, helper-hidden) cannot slip past the shape net.

Item 2 (§16.6) has now LANDED: the dataflow/taint net catches a class-name
comparison whose branch WRITES A STRUCTURAL SINK — closing the substring-nonmarker
boundary the early slice documented.  It distinguishes UNLAWFUL identity->structure
(a structural sink) from LAWFUL code-shape (returns a role string/bool), so the
clean production tree stays at zero identity debt (the net is preventive).
"""
from __future__ import annotations

from model_unfolder.evidence.identity_guard import (
    scan_fact_provenance_identity,
    scan_identity_source,
    scan_identity_yaml_source,
)


# --- item 1: the fact-provenance rule ----------------------------------------

def _prov(status, source):
    return {"unet.stage.attention_cell": {"value": "transformer2d",
                                          "status": status, "source": source}}


def test_code_fact_citing_identity_field_is_flagged():
    for source in ("config:model_type", "architectures", "config:_class_name",
                   "repo_id"):
        findings = scan_fact_provenance_identity(_prov("code_proven", source))
        assert any(f.kind == "fact_provenance_identity" for f in findings), source


def test_code_and_config_and_derived_are_also_covered():
    assert scan_fact_provenance_identity(_prov("code_and_config", "config:model_type"))
    assert scan_fact_provenance_identity(_prov("derived", "architectures"))


def test_code_fact_citing_declared_class_table_is_flagged():
    # A code-proven fact must not be decided by reading a class-marker table
    # (those tables lawfully decide only config_declared facts).
    for table in ("dit_class_markers", "scheduler_flow_matching_markers",
                  "drill_class_markers"):
        assert scan_fact_provenance_identity(_prov("code_proven", table)), table


def test_declared_tiers_reading_their_own_declaration_are_lawful():
    # The causal-LM role read of architectures[] and the class-default hydration
    # channel are the lawful config-declared path (I-2, G-4) — NOT violations.
    assert scan_fact_provenance_identity(_prov("config_declared", "architectures")) == []
    assert scan_fact_provenance_identity(_prov("config_declared", "config:_class_name")) == []
    assert scan_fact_provenance_identity(_prov("class_default", "model_type")) == []


def test_real_reader_sources_do_not_false_positive():
    for source in ("decoder_ffn_gated_from_files", "config:hidden_act",
                   "attention_causality_from_files", "norm_math", "file.py:123"):
        assert scan_fact_provenance_identity(_prov("code_proven", source)) == [], source


def test_generic_architecture_word_is_not_the_architectures_field():
    # A reader named with the generic word "architecture" must not match the
    # "architectures" identity field (exact-segment matching, not substring).
    assert scan_fact_provenance_identity(
        _prov("code_proven", "block_architecture_reader")) == []


def test_non_evidence_statuses_and_empty_are_ignored():
    assert scan_fact_provenance_identity(_prov("asserted", "model_type")) == []
    assert scan_fact_provenance_identity(_prov("config_declared", "model_type")) == []
    assert scan_fact_provenance_identity({}) == []
    assert scan_fact_provenance_identity(None) == []


# --- item 6: generalized identity-table negative controls ---------------------

def test_intermediate_enum_table_is_caught():
    """map class identity -> an intermediate enum BEFORE structure: still a
    class-keyed table, still caught by the shape net."""
    findings = scan_identity_yaml_source(
        "cell_role:\n  UNetMidBlock2DCrossAttn: cross\n  ResnetBlock2D: self\n")
    assert any(f.kind == "identity_table" for f in findings)


def test_helper_hidden_table_is_caught():
    """hide the table behind a helper function: the literal is still found."""
    findings = scan_identity_source(
        "def pick_cell(cls):\n"
        "    table = {'SimpleCrossAttnDownBlock': 'plain',\n"
        "             'CrossAttnDownBlock2D': 'transformer2d'}\n"
        "    return table.get(cls, 'transformer2d')\n",
        path="model_unfolder/adapters/diffusor/helper.py",
    )
    assert any(f.kind == "class_keyed_literal" for f in findings)


def test_generator_over_class_markers_with_domain_word_is_caught():
    """a generator/any() over a class name that contains a domain marker is
    caught by the existing domain predicate."""
    findings = scan_identity_source(
        "def kind(block_class):\n"
        "    if any(m in block_class.lower() for m in ('vision', 'audio')):\n"
        "        return {'mixer': 'x'}\n",
        path="model_unfolder/evidence/newdet.py",
    )
    assert any(f.kind == "class_identity_branch" for f in findings)


def test_substring_nonmarker_to_structural_sink_is_now_caught():
    """H4 item 2 (§16.6) — the taint net LANDED, so the boundary the early slice
    documented now FLIPS: a substring test on a class name that is NOT a domain
    marker ("SimpleCrossAttn" in cls), whose branch WRITES A STRUCTURAL SINK, is
    caught as ``class_identity_branch`` (identity fabricating structure)."""
    findings = scan_identity_source(
        "def kind(cls):\n"
        "    if 'SimpleCrossAttn' in cls:\n"
        "        return {'cell': 'plain_cross'}\n"
        "    return {'cell': 'transformer2d'}\n",
        path="model_unfolder/adapters/diffusor/newdet.py",
    )
    assert any(f.kind == "class_identity_branch" for f in findings)


def test_taint_catches_class_name_to_spec_constructor():
    findings = scan_identity_source(
        "def build(cls):\n"
        "    if 'Flux' in cls:\n"
        "        return AttentionSpec(kind='mha', num_heads=8)\n",
        path="model_unfolder/adapters/diffusor/x.py")
    assert any(f.kind == "class_identity_branch" for f in findings)


def test_taint_does_not_flag_lawful_code_shape_returning_a_role():
    """The crucial non-false-positive: classifying a RESOLVED class to a ROLE
    string/bool (the RMSNorm->norm code-shape category) is NOT a structural sink,
    so it is lawful and uncaught — production stays clean."""
    for body in ("        return 'norm'\n", "        return True\n"):
        findings = scan_identity_source(
            "def role(name):\n"
            "    if 'norm' in name.lower():\n" + body,
            path="model_unfolder/evidence/patterns.py")
        assert not any(f.kind == "class_identity_branch" for f in findings)


def test_taint_exempts_a_typed_display_wrapper():
    findings = scan_identity_source(
        "@identity_display\n"
        "def label(cls):\n"
        "    if 'Flux' in cls:\n"
        "        return {'kind': 'flux'}\n")
    assert not any(f.kind == "class_identity_branch" for f in findings)


def test_taint_net_adds_no_debt_to_the_clean_production_tree():
    """H4 is PREVENTIVE: the real tree has zero identity->structural-sink flows
    (all class-name comparisons are lawful code-shape), so landing the net keeps
    identity debt at zero — it guards the future, it does not manufacture debt."""
    from model_unfolder.evidence.identity_guard import scan_identity_debt
    assert scan_identity_debt() == []


# --- item 1: the corpus invariant (real facts, not synthetic) -----------------

def test_renderer_parser_dependency_firewall():
    """H4 (§16.6) — the renderer/parser firewall: the renderer consumes the IR,
    never raw config or the parser.  No ``renderers/`` module may import the
    parser/adapter layer or a config accessor — so a lying render can only come
    from a wrong IR, never from the renderer re-reading and re-guessing config.
    Currently clean; this gate keeps it so."""
    import ast
    import pathlib

    import model_unfolder

    renderers = pathlib.Path(model_unfolder.__file__).parent / "renderers"
    forbidden_names = {"get_config_value", "note_access", "config_access",
                       "resolve_source_files"}
    violations: list[str] = []
    for path in renderers.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.module and ("adapters" in node.module
                                    or node.module.endswith("parser")
                                    or node.module.endswith("transformer.debug")):
                    violations.append(f"{path.name}:{node.lineno} imports {node.module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "adapters" in alias.name or alias.name in forbidden_names:
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
    assert not violations, (
        "renderer/parser firewall breached — the renderer must consume the IR, "
        "not raw config/source:\n" + "\n".join(violations))


def test_no_corpus_fact_cites_identity_as_its_deciding_source():
    """The fact-provenance rule as a BLOCKING corpus invariant: every blessed
    fixture must parse so that no code/derived structural fact cites an identity
    field or a class-vocabulary table as its source.  A hit here is a genuine
    identity leak to REPORT to Soumil, never a test to relax."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from model_unfolder import unfold
    from model_unfolder.sable import load_corpus

    sink = io.StringIO()
    problems: list[str] = []
    for fname, fix in load_corpus():
        with redirect_stdout(sink), redirect_stderr(sink):
            ir = unfold(fix["config"]).ir
        prov = (ir.extras or {}).get("fact_provenance") or {}
        for finding in scan_fact_provenance_identity(prov):
            problems.append(f"{fname}: {finding.detail}")
    assert not problems, "\n".join(problems[:20])
