"""S2 pure-speed laws: exact segments and call-local/content-keyed memoization."""
from __future__ import annotations

import ast
import json
import os
import pathlib
import statistics

from model_unfolder import everchanging
from model_unfolder.evidence import component_owner, decoder_block, program_index
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import SourceId


_VOCAB = {
    "config_roots": frozenset({"config"}),
    "activation_dispatch": frozenset(),
    "activation_calls": frozenset(),
    "container_classes": frozenset({"ModuleList"}),
}

_REPO = pathlib.Path(__file__).resolve().parents[1]
_BUDGETS = _REPO / "verification" / "latency_budgets.json"
_REQUIRED_TARGETS = {
    "llama-7b",
    "deepseek-v3",
    "qwen2-vl-7b-instruct",
    "pixart-sigma-xl-2-1024-ms",
    "stable-diffusion-3-5-large",
    "qwen3-omni-30b",
    "musicgen-small",
}


def _walker(text: str):
    sid = SourceId("/virtual/model.py", program_index.content_fingerprint(text),
                   component_key="root")
    tree = ast.parse(text)
    return program_index._SourceWalker(
        sid, text, tree, config_vocab=_VOCAB,
        factory_names=frozenset()), tree


def test_source_segments_match_cpython_for_unicode_multiline_crlf_and_formfeed():
    texts = (
        "x = 'λ'\ny = call(\n    x,\n    key='é',\n)\n",
        "x = 1\r\ny = (x +\r\n     2)\r\n",
        "x = 1\f + 2\ny = x\n",
    )
    for text in texts:
        walker, tree = _walker(text)
        for node in ast.walk(tree):
            expected = ast.get_source_segment(text, node) or ""
            assert walker._seg(node) == expected, (type(node).__name__, text)


def test_source_line_table_is_built_once_and_get_source_segment_is_never_called(
        monkeypatch):
    calls = 0
    real_split = program_index._split_source_lines

    def counted(text):
        nonlocal calls
        calls += 1
        return real_split(text)

    monkeypatch.setattr(program_index, "_split_source_lines", counted)
    monkeypatch.setattr(
        ast, "get_source_segment",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("quadratic source splitting returned")))
    walker, _tree = _walker(
        "class Model:\n"
        "    def __init__(self, config):\n"
        "        self.layers = ModuleList([Block(config) for _ in range(4)])\n"
        "    def forward(self, x):\n"
        "        return self.layers[0](x)\n")
    walker.run()
    assert calls == 1


def _root_fixture(tmp_path, architecture="Model"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "modeling.py"
    path.write_text(
        "class Model:\n"
        "    def __init__(self, config):\n"
        "        self.value = config.value\n")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture=architecture,
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture})
    return program_index.build_program_index(bundle), bundle


def test_component_root_memo_is_index_local_and_prefix_exact(tmp_path, monkeypatch):
    index, bundle = _root_fixture(tmp_path)
    calls = 0
    real = component_owner.resolve_owner_graph

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(component_owner, "resolve_owner_graph", counted)
    first = component_owner.resolve_component_root(index, bundle, "root")
    second = component_owner.resolve_component_root(index, bundle, "root")
    assert second is first and calls == 1

    explicit = component_owner.resolve_component_root(
        index, bundle, "root", root_param_prefixes={"config": ()})
    assert explicit.status == "resolved" and calls == 2

    other_index, other_bundle = _root_fixture(tmp_path / "other")
    component_owner.resolve_component_root(other_index, other_bundle, "root")
    assert calls == 3


def test_component_root_declared_address_change_cannot_hit_stale_cache(tmp_path):
    index, bundle = _root_fixture(tmp_path)
    assert component_owner.resolve_component_root(
        index, bundle, "root").status == "resolved"
    bundle.component_architectures["root"] = "Missing"
    assert component_owner.resolve_component_root(
        index, bundle, "root").status == "absent"


def test_decoder_candidate_memo_separates_bundle_and_selector_identity(
        tmp_path, monkeypatch):
    index, bundle = _root_fixture(tmp_path)
    calls = 0
    sentinel = object()

    def counted(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(
        decoder_block, "_decoder_block_candidates_for_config_uncached", counted)
    select_a = lambda _path: None
    select_b = lambda _path: None
    assert decoder_block.decoder_block_candidates_for_config(
        index, bundle, (), allow_root_stage=True,
        config_selector=select_a) is sentinel
    assert decoder_block.decoder_block_candidates_for_config(
        index, bundle, (), allow_root_stage=True,
        config_selector=select_a) is sentinel
    assert calls == 1
    decoder_block.decoder_block_candidates_for_config(
        index, bundle, (), allow_root_stage=True,
        config_selector=select_b)
    assert calls == 2
    twin = SourceBundle(**{
        field: getattr(bundle, field)
        for field in bundle.__dataclass_fields__
    })
    decoder_block.decoder_block_candidates_for_config(
        index, twin, (), allow_root_stage=True,
        config_selector=select_a)
    assert calls == 3


def test_everchanging_parse_is_content_keyed_and_results_are_isolated(
        tmp_path, monkeypatch):
    import yaml

    root = tmp_path / "everchanging"
    domain = root / "test"
    domain.mkdir(parents=True)
    path = domain / "vocab.yaml"
    path.write_text("items: [one, two]\n")
    old_mtime = path.stat().st_mtime_ns
    calls = 0
    real = yaml.safe_load

    def counted(text):
        nonlocal calls
        calls += 1
        return real(text)

    monkeypatch.setattr(everchanging, "_DIR", root)
    monkeypatch.setattr(yaml, "safe_load", counted)
    everchanging.clear_load_cache()
    first = everchanging.load("test", "vocab")
    second = everchanging.load("test", "vocab")
    assert calls == 1 and first == second
    first["items"].append("mutated")
    assert everchanging.load("test", "vocab") == {"items": ["one", "two"]}

    path.write_text("items: [red, blue]\n")
    os.utime(path, ns=(old_mtime, old_mtime))
    assert everchanging.load("test", "vocab") == {"items": ["red", "blue"]}
    assert calls == 2


def test_latency_budgets_are_numeric_and_derived_from_the_recorded_baseline():
    document = json.loads(_BUDGETS.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["library_imports_in_budget"] is False
    for key in ("program_index_after_imports", "end_to_end_cold"):
        record = document[key]
        assert set(record["samples_seconds"]) == _REQUIRED_TARGETS
        assert all(len(samples) == 3
                   for samples in record["samples_seconds"].values())
        medians = {
            target: statistics.median(samples)
            for target, samples in record["samples_seconds"].items()
        }
        assert medians == record["medians_seconds"]
        measured_max = max(medians.values())
        budget = record["budget_seconds"]
        assert isinstance(budget, (int, float)) and not isinstance(budget, bool)
        assert measured_max < budget <= measured_max * 1.5
