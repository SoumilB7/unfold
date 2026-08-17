from pathlib import Path
import importlib.util

from model_unfolder.evidence.legacy_reader_quarantine import (
    LEGACY_SEMANTIC_READERS,
    PARSE_AUTHORITY_SITES,
    observed_legacy_parse_caller_fingerprint,
    observed_reader_implementation_fingerprint,
    legacy_reader_quarantine_problems,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_semantic_reader_quarantine_is_exact_and_cannot_grow():
    assert len(LEGACY_SEMANTIC_READERS) == 17
    assert legacy_reader_quarantine_problems(ROOT) == ()


def test_every_quarantined_reader_has_one_future_owner_and_deletion_condition():
    assert all(row.migration_unit.startswith("U")
               and row.reason
               and row.deletion_condition.startswith(row.migration_unit)
               for row in LEGACY_SEMANTIC_READERS)


def test_every_evidence_parse_site_has_one_explicit_authority_class():
    assert len(PARSE_AUTHORITY_SITES) == 25
    assert len({row.site for row in PARSE_AUTHORITY_SITES}) == 25
    legacy = [row for row in PARSE_AUTHORITY_SITES
              if row.category == "legacy_model_source"]
    assert len(legacy) == 14
    assert all(row.deletion_unit and row.reason for row in legacy)
    assert {row.category for row in PARSE_AUTHORITY_SITES} == {
        "central_program_index",
        "address_bootstrap",
        "repository_audit",
        "test_guard",
        "legacy_model_source",
    }


def test_reader_implementations_and_legacy_parse_callers_are_frozen():
    from model_unfolder.evidence import legacy_reader_quarantine as quarantine

    assert observed_reader_implementation_fingerprint(ROOT) \
        == quarantine.LEGACY_READER_IMPLEMENTATION_FINGERPRINT
    assert observed_legacy_parse_caller_fingerprint(ROOT) \
        == quarantine.LEGACY_PARSE_CALLER_FINGERPRINT


def test_generated_u3_inventory_is_current():
    script = ROOT / "scripts" / "generate_u3_reader_inventory.py"
    spec = importlib.util.spec_from_file_location("u3_inventory_generator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert (ROOT / "docs" / "U3_CURRENT_READER_INVENTORY.md").read_text(
        encoding="utf-8") == module.render()


def test_poison_new_raw_files_reader_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "poison.py").write_text(
        "def invented_mechanism_from_files(files): return None\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        item == "unregistered legacy semantic reader: "
                "invented_mechanism_from_files"
        for item in problems)


def test_poison_new_caller_of_existing_reader_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    adapter = tmp_path / "model_unfolder" / "adapters"
    evidence.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (evidence / "patterns.py").write_text(
        "def attention_score_scaling_from_files(files): return None\n",
        encoding="utf-8")
    (adapter / "poison.py").write_text(
        "def new_consumer(files):\n"
        "    return attention_score_scaling_from_files(files)\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        item.startswith("attention_score_scaling_from_files caller drift:")
        and "model_unfolder/adapters/poison.py:new_consumer" in item
        for item in problems)


def test_poison_qualified_or_aliased_reader_call_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    adapter = tmp_path / "model_unfolder" / "adapters"
    evidence.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (evidence / "patterns.py").write_text(
        "def attention_score_scaling_from_files(files): return None\n",
        encoding="utf-8")
    (adapter / "poison.py").write_text(
        "from model_unfolder.evidence.patterns import "
        "attention_score_scaling_from_files as hidden_reader\n"
        "import model_unfolder.evidence.patterns as patterns\n"
        "def aliased(files):\n"
        "    return hidden_reader(files)\n"
        "def qualified(files):\n"
        "    return patterns.attention_score_scaling_from_files(files)\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    caller_problem = next(
        item for item in problems
        if item.startswith("attention_score_scaling_from_files caller drift:"))
    assert "model_unfolder/adapters/poison.py:aliased" in caller_problem
    assert "model_unfolder/adapters/poison.py:qualified" in caller_problem


def test_poison_moving_reader_into_class_scope_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "patterns.py").write_text(
        "class Hidden:\n"
        "    def attention_score_scaling_from_files(self, files):\n"
        "        return None\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        item.startswith("attention_score_scaling_from_files definition drift:")
        and "Hidden.attention_score_scaling_from_files" in item
        for item in problems)


def test_poison_new_evidence_ast_parse_site_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "poison.py").write_text(
        "import ast\n"
        "def second_source_parser(text):\n"
        "    return ast.parse(text)\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        "unregistered evidence ast.parse site:" in item
        and "poison.py" in item
        and "second_source_parser" in item
        for item in problems)


def test_poison_reader_body_change_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "patterns.py").write_text(
        "def attention_score_scaling_from_files(files):\n"
        "    return 'changed semantic answer'\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        item.startswith("legacy reader/helper implementation drift:")
        for item in problems)


def test_poison_new_legacy_parse_helper_caller_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "poison.py").write_text(
        "def new_source_consumer(files):\n"
        "    return _parse_defs(files)\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    assert any(
        item.startswith("legacy parse-authority caller drift:")
        for item in problems)


def test_poison_aliased_ast_parse_site_is_blocking(tmp_path):
    evidence = tmp_path / "model_unfolder" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "poison.py").write_text(
        "import ast as syntax\n"
        "from ast import parse as parse_source\n"
        "other_parse = syntax.parse\n"
        "def imported_alias(text):\n"
        "    return parse_source(text)\n"
        "def assigned_alias(text):\n"
        "    return other_parse(text)\n",
        encoding="utf-8")
    problems = legacy_reader_quarantine_problems(tmp_path)
    parse_problems = tuple(
        item for item in problems
        if item.startswith("unregistered evidence ast.parse site:"))
    assert any("imported_alias" in item for item in parse_problems)
    assert any("assigned_alias" in item for item in parse_problems)
