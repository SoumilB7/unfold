"""U5 terminal-consumer dependency firewall and exact debt quarantine."""
from __future__ import annotations

from model_unfolder.evidence.consumer_firewall import (
    ConsumerAccess,
    ConsumerAccessGroup,
    group_consumer_accesses,
    scan_consumer_accesses,
    scan_consumer_source,
)
from model_unfolder.evidence.structural_debt import (
    STRUCTURAL_DEBT,
    unbacked_debt_rows,
    unrowed_consumer_reads,
)


def _scan(source: str, module="model_unfolder/renderers/html/poison.py"):
    return scan_consumer_source(source, module=module)


def test_every_live_terminal_consumer_group_has_one_exact_debt_row():
    live = {group.key for group in group_consumer_accesses()}
    rows = {row.writer_key for row in STRUCTURAL_DEBT
            if row.sink_kind == "consumer_read"}
    assert live == rows
    assert unrowed_consumer_reads() == []


def test_consumer_quarantine_is_non_vacuous_and_spans_all_four_surfaces():
    accesses = scan_consumer_accesses()
    assert accesses
    assert {item.consumer for item in accesses} == {
        "renderer", "json", "params", "conformance",
    }
    assert {item.kind for item in accesses} >= {
        "backward_import", "raw_extras", "raw_config", "source_reopen",
        "spec_default",
    }
    assert not any(item.kind == "semantic_bucket" for item in accesses)
    assert not any(item.kind == "semantic_default" for item in accesses)


def test_poison_semantic_default_cannot_resolve_an_unknown():
    direct = _scan(
        "def card(placement):\n"
        "    return placement or 'pre'\n"
    )
    mapped = _scan(
        "def card(placement):\n"
        "    where = {'pre': 'before'}\n"
        "    return where.get(placement) or where['pre']\n"
    )
    assert [(item.kind, item.target) for item in direct] == [
        ("semantic_default", "pre"),
    ]
    assert any(item.kind == "semantic_default" and item.target == "pre"
               for item in mapped)


def test_poison_semantic_default_has_no_get_conditional_or_signature_escape():
    found = _scan(
        "def a(spec):\n"
        "    return spec.get('kind', 'mha')\n"
        "def b(placement):\n"
        "    return 'post' if placement is None else placement\n"
        "def c(norm_kind='rmsnorm'):\n"
        "    return norm_kind\n"
    )
    assert {(item.kind, item.target) for item in found} == {
        ("semantic_default", "mha"),
        ("semantic_default", "post"),
        ("semantic_default", "rmsnorm"),
    }


def test_poison_semantic_default_cannot_escape_by_renaming_the_selector():
    found = _scan(
        "def a(x):\n"
        "    return x or 'pre'\n"
        "def b(doc, x):\n"
        "    return doc.get(x, 'mha')\n"
        "def c(x):\n"
        "    return 'post' if x is None else x\n"
        "def d(x='rmsnorm'):\n"
        "    return x\n"
    )
    assert {(item.kind, item.target) for item in found} == {
        ("semantic_default", "pre"),
        ("semantic_default", "mha"),
        ("semantic_default", "post"),
        ("semantic_default", "rmsnorm"),
    }


def test_exact_semantic_canonicalization_preserves_unknown_control():
    found = _scan(
        "def normalize(storage):\n"
        "    return 'split_qkv' if storage == 'split' else storage\n"
    )
    assert not any(item.kind == "semantic_default" for item in found)

    defaulting = _scan(
        "def default(placement):\n"
        "    return placement if placement else 'pre'\n"
    )
    assert [(item.kind, item.target) for item in defaulting] == [
        ("semantic_default", "pre"),
    ]


def test_poison_renderer_direct_backward_import_blocks():
    found = _scan(
        "from model_unfolder.adapters.transformer import parser as hidden\n"
        "def draw(ir):\n    return hidden.parse(ir)\n"
    )
    assert any(item.kind == "backward_import" and "adapters" in item.target
               for item in found)


def test_poison_innocently_named_internal_bridge_cannot_launder_dependency():
    found = _scan(
        "from model_unfolder.innocent_bridge import canonical_looking_value\n"
        "def draw(ir):\n    return canonical_looking_value(ir)\n"
    )
    assert any(item.kind == "backward_import"
               and "innocent_bridge" in item.target for item in found)


def test_poison_renderer_direct_raw_config_and_source_reopen_block():
    found = _scan(
        "def draw(config, path):\n"
        "    source = path.read_text()\n"
        "    return config.get('hidden_size'), source\n"
    )
    assert {item.kind for item in found} >= {"raw_config", "source_reopen"}


def test_poison_renderer_receipt_dto_import_is_lawful_control():
    found = _scan(
        "from model_unfolder.evidence.receipts import ProjectionReceipt\n"
        "def draw(value):\n    return value\n"
    )
    assert not any(item.kind == "backward_import" for item in found)


def test_poison_raw_extras_direct_alias_and_subscript_are_exact():
    found = _scan(
        "def draw(ir):\n"
        "    x = ir.get('extras') or {}\n"
        "    y = x.get('modalities') or {}\n"
        "    return y['fusion']\n"
    )
    targets = {item.target for item in found if item.kind == "raw_extras"}
    assert "extras" in targets
    assert "extras.modalities" in targets
    assert "extras.modalities.fusion" in targets


def test_poison_attribute_style_raw_reads_are_not_an_escape():
    extras = _scan("def draw(ir):\n    return ir.extras.modalities.fusion\n")
    config = _scan(
        "def draw(config):\n    return config.hidden_size\n",
        module="model_unfolder/expanded/poison.py",
    )
    assert any(item.target == "extras.modalities.fusion" for item in extras)
    assert any(item.kind == "raw_config"
               and item.target == "config.hidden_size" for item in config)


def test_poison_raw_extras_crosses_a_local_helper_boundary():
    found = _scan(
        "def helper(payload):\n"
        "    return payload.get('family')\n"
        "def draw(ir):\n"
        "    return helper((ir.get('extras') or {}).get('render') or {})\n"
    )
    assert any(item.symbol == "helper" and item.target == "extras.render.family"
               for item in found)


def test_poison_raw_extras_crosses_an_exact_self_method_boundary():
    found = _scan(
        "class Drawer:\n"
        "    def helper(self, payload):\n"
        "        return payload.get('family')\n"
        "    def draw(self, ir):\n"
        "        return self.helper(ir.extras)\n"
    )
    assert any(item.symbol == "Drawer.helper"
               and item.target == "extras.family" for item in found)


def test_poison_dynamic_raw_extras_key_never_becomes_a_bare_excuse():
    found = _scan(
        "def draw(extras, key):\n    return extras.get(key)\n",
        module="model_unfolder/expanded/poison.py",
    )
    assert any(item.target == "extras.<dynamic>" for item in found)


def test_poison_json_global_semantic_evidence_bucket_blocks():
    found = _scan(
        "def renamed_selector(evidence):\n"
        "    return evidence.get('detections')\n"
        "def build(evidence):\n"
        "    return {'trace': {'code_finding_ids': renamed_selector(evidence)}}\n",
        module="model_unfolder/expanded/poison.py",
    )
    assert any(item.kind == "semantic_bucket" for item in found)


def test_json_empty_finding_ids_is_honest_interim_control():
    found = _scan(
        "def build():\n    return {'trace': {'code_finding_ids': []}}\n",
        module="model_unfolder/expanded/poison.py",
    )
    assert not any(item.kind == "semantic_bucket" for item in found)


def test_poison_json_truthiness_cleanup_is_counted_until_fact_is_tristate():
    found = _scan(
        "def build(spec):\n"
        "    return {'bias': spec.get('bias') or None}\n",
        module="model_unfolder/expanded/poison.py",
    )
    assert any(item.kind == "truthy_cleanup" and item.target == "bias"
               for item in found)


def test_poison_conformance_reopen_and_raw_config_helpers_block():
    found = _scan(
        "import ast\n"
        "def check(target, path):\n"
        "    src = path.read_text()\n"
        "    tree = ast.parse(src)\n"
        "    return _model_type(target), tree\n",
        module="model_unfolder/evidence/conformance.py",
    )
    assert {item.kind for item in found} >= {"source_reopen", "raw_config"}


def test_conformance_may_receive_graph_dto_but_not_build_one():
    found = _scan(
        "from model_unfolder.evidence.program_index import (\n"
        "    ProgramIndex, build_program_index,\n"
        ")\n",
        module="model_unfolder/evidence/conformance.py",
    )
    backward = {item.target for item in found
                if item.kind == "backward_import"}
    assert "model_unfolder.evidence.program_index.ProgramIndex" not in backward
    assert "model_unfolder.evidence.program_index.build_program_index" in backward


def test_poison_conformance_direct_config_read_and_helper_propagation_block():
    found = _scan(
        "def helper(payload):\n"
        "    return payload.get('num_heads')\n"
        "def check(target):\n"
        "    return helper(target)\n",
        module="model_unfolder/evidence/conformance.py",
    )
    assert any(item.symbol == "helper"
               and item.target == "config.num_heads" for item in found)


def test_group_fingerprint_changes_when_a_path_or_duplicate_read_is_added():
    one = ConsumerAccess("m.py", "f", "renderer", "raw_extras", "extras.a", 1)
    two = ConsumerAccess("m.py", "f", "renderer", "raw_extras", "extras.b", 2)
    before = group_consumer_accesses([one])[0]
    after = group_consumer_accesses([one, two])[0]
    assert before.fingerprint != after.fingerprint
    assert before.key != after.key
    duplicate = group_consumer_accesses([one, one])[0]
    assert duplicate.targets == ("extras.a", "extras.a")
    assert duplicate.fingerprint != before.fingerprint


def test_group_closure_rejects_forged_fingerprint_and_unsorted_targets():
    try:
        ConsumerAccessGroup("m", "f", "renderer", "raw_extras",
                            ("b", "a"), "0" * 16)
    except ValueError:
        pass
    else:
        raise AssertionError("forged consumer group was accepted")
    try:
        ConsumerAccess("m", "f", "renderer", "made_up", "x", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown consumer access kind was accepted")


def test_poison_parameter_default_is_ast_based_not_source_text_based():
    found = _scan(
        "from model_unfolder.ir import AttentionSpec as CompletelyRenamed\n"
        "def renamed_formula(spec: CompletelyRenamed):\n"
        "    unusual = int(spec.head_dim) or 17\n"
        "    return unusual\n",
        module="model_unfolder/params.py",
    )
    assert any(item.kind == "spec_default"
               and item.target == "attention.head_dim=>unusual"
               for item in found)


def test_shrink_rule_strands_row_when_consumer_group_changes():
    row = next(row for row in STRUCTURAL_DEBT
               if row.sink_kind == "consumer_read")
    live = {group.key for group in group_consumer_accesses()}
    assert unbacked_debt_rows((row,), census_keys=live) == []
    live.remove(row.writer_key)
    assert unbacked_debt_rows((row,), census_keys=live) == [row]
