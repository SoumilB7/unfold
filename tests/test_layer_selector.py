"""U8-A — exact, mechanism-neutral per-layer selector poisons."""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.layer_selector import (
    LayerSelectionDecision,
    LayerSelectorResolution,
    SelectedConstructionCandidate,
    resolve_layer_selector,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ConstructionSiteId,
    SourceId,
    SourceSpan,
    SymbolId,
    build_program_index,
)


def _build(tmp_path, body, *, prefix="", name="modeling_selector.py"):
    path = tmp_path / name
    header = "class Dense: pass\nclass Sparse: pass\nclass Other: pass\n"
    if prefix:
        header += f"{prefix}\n"
    source = (
        header
        + "class Root:\n"
        + "    def __init__(self, config, layer_idx):\n"
        + textwrap.indent(textwrap.dedent(body).strip(), " " * 8)
        + "\n")
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(
        index, bundle, "root", root_param_prefixes={"config": ()})
    assert root.status == "resolved"
    callable_symbol = SymbolId(root.occurrence.root.source, "Root.__init__")
    return index, root, callable_symbol


def _selector(values, *, kinds=None):
    kinds = kinds or {}

    def select(path):
        path = tuple(path)
        return (path in values, values.get(path),
                kinds.get(path, "config_declared"))
    return select


def _resolve(case, values, *, target="choice", indices=(0, 1, 2, 3),
             prefix=()):
    index, root, callable_symbol = case
    return resolve_layer_selector(
        index, root, root.occurrence, callable_symbol, target, indices,
        "layer_idx", config_selector=_selector(values), config_prefix=prefix)


def _selected_names(result):
    return tuple(
        decision.selected_candidates[0].candidate.symbol.qualified_name
        for decision in result.decisions)


def test_membership_selector_preserves_every_mixed_layer(tmp_path):
    case = _build(tmp_path, """
        if layer_idx in config.sparse_layers:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(case, {("sparse_layers",): [1, 3]})
    assert result.status == "resolved"
    assert _selected_names(result) == ("Dense", "Sparse", "Dense", "Sparse")
    assert {operand.path for decision in result.decisions
            for operand in decision.operands} == {("sparse_layers",)}


def test_threshold_modulo_boolean_and_local_alias_are_exact(tmp_path):
    case = _build(tmp_path, """
        periodic = (layer_idx + 1) % config.period == 0
        routed = layer_idx >= config.first and periodic
        if routed:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(case, {("period",): 2, ("first",): 1})
    assert result.status == "resolved"
    assert _selected_names(result) == ("Dense", "Sparse", "Dense", "Sparse")


def test_short_config_list_is_unresolved_not_defaulted(tmp_path):
    case = _build(tmp_path, """
        if config.pattern[layer_idx]:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(case, {("pattern",): [False, True]}, indices=(0, 1, 2))
    assert result.status == "incomplete"
    assert [item.state for item in result.decisions] == [
        "selected", "selected", "unresolved"]
    assert len(result.decisions[2].unresolved_sites) == 2


def test_literal_registry_dispatch_selects_exact_candidate_edge(tmp_path):
    case = _build(tmp_path, """
        self.choice = CHOICES[config.pattern[layer_idx]]()
    """, prefix="CHOICES = {0: Dense, 1: Sparse}")
    result = _resolve(case, {("pattern",): [0, 1, 0, 1]})
    assert result.status == "resolved"
    assert _selected_names(result) == ("Dense", "Sparse", "Dense", "Sparse")
    assert {item.selected_candidates[0].candidate_index
            for item in result.decisions} == {0, 1}


def test_dynamic_registry_key_is_unresolved(tmp_path):
    case = _build(tmp_path, """
        self.choice = CHOICES[choose(layer_idx)]()
    """, prefix="CHOICES = {0: Dense, 1: Sparse}")
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "incomplete"
    assert result.decisions[0].state == "unresolved"


def test_two_unconditional_sites_are_ambiguous_and_neither_is_ranked(tmp_path):
    case = _build(tmp_path, """
        self.choice = Dense()
        self.choice = Sparse()
    """)
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "ambiguous"
    assert result.decisions[0].state == "ambiguous"
    assert {item.candidate.symbol.qualified_name
            for item in result.decisions[0].selected_candidates} == {
                "Dense", "Sparse"}


def test_same_class_at_two_sites_stays_occurrence_ambiguous(tmp_path):
    case = _build(tmp_path, """
        self.choice = Dense()
        self.choice = Dense()
    """)
    result = _resolve(case, {}, indices=(0,))
    selected = result.decisions[0].selected_candidates
    assert result.status == "ambiguous" and len(selected) == 2
    assert selected[0].candidate.symbol == selected[1].candidate.symbol
    assert selected[0].site_id != selected[1].site_id


def test_missing_config_operand_is_typed_unresolved(tmp_path):
    case = _build(tmp_path, """
        if layer_idx >= config.first:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "incomplete"
    assert result.decisions[0].state == "unresolved"
    assert result.decisions[0].operands == ()


def test_class_default_provenance_is_preserved_not_called_checkpoint(tmp_path):
    case = _build(tmp_path, """
        if layer_idx >= config.first:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    index, root, callable_symbol = case
    result = resolve_layer_selector(
        index, root, root.occurrence, callable_symbol, "choice", (0, 1),
        "layer_idx", config_selector=_selector(
            {("first",): 1}, kinds={("first",): "class_default"}))
    assert result.status == "resolved"
    assert {operand.source_kind for decision in result.decisions
            for operand in decision.operands} == {"class_default"}


def test_config_prefix_is_owner_scoped(tmp_path):
    case = _build(tmp_path, """
        if layer_idx in config.sparse_layers:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(
        case, {("text_config", "sparse_layers"): [1]},
        indices=(0, 1), prefix=("text_config",))
    assert result.status == "resolved"
    assert {operand.path for decision in result.decisions
            for operand in decision.operands} == {
                ("text_config", "sparse_layers")}


def test_present_but_unused_config_never_enters_operands(tmp_path):
    case = _build(tmp_path, """
        if layer_idx >= config.first:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """)
    result = _resolve(
        case, {("first",): 1, ("model_type",): "tempting-token"},
        indices=(0, 1))
    assert {operand.path for decision in result.decisions
            for operand in decision.operands} == {("first",)}


def test_unsupported_region_blocks_completeness_even_with_a_visible_site(tmp_path):
    case = _build(tmp_path, """
        try:
            self.choice = Dense()
        except ValueError:
            pass
    """)
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "incomplete"
    assert result.coverage_gaps


def test_unsupported_region_blocks_an_empty_census_from_claiming_absence(tmp_path):
    case = _build(tmp_path, """
        try:
            do_something_dynamic()
        except ValueError:
            pass
    """)
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "incomplete"
    assert result.candidates == () and result.decisions == ()
    assert result.coverage_gaps


def test_single_symbol_less_candidate_is_unresolved_not_selected(tmp_path):
    case = _build(tmp_path, "self.choice = external.Builder()")
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "incomplete"
    assert result.decisions[0].state == "unresolved"


def test_absent_target_is_honest_absence(tmp_path):
    case = _build(tmp_path, "self.other = Dense()")
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "absent"
    assert result.candidates == () and result.decisions == ()


def test_foreign_owner_and_foreign_index_cannot_be_laundered(tmp_path):
    first = _build(tmp_path, "self.choice = Dense()", name="first.py")
    second = _build(tmp_path, "self.choice = Sparse()", name="second.py")
    index, _root, callable_symbol = first
    _other_index, other_root, _other_callable = second
    result = resolve_layer_selector(
        index, other_root, other_root.occurrence, callable_symbol,
        "choice", (0,), "layer_idx", config_selector=_selector({}))
    assert result.status == "failed"
    assert result.failure_kind in {"owner_not_in_index", "callable_not_owned"}


def test_broken_component_file_is_rejected_at_d0_before_selection(tmp_path):
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("class Root:\n def __init__(self, config, layer_idx): pass\n")
    bad.write_text("class Hidden(:\n")
    bundle = SourceBundle(
        source="local", files=(str(good), str(bad)),
        component_files={"root": (str(good), str(bad))},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "failed"
    symbol = next(item.symbol for item in index.classes
                  if item.symbol.qualified_name == "Root")
    with pytest.raises(ValueError):
        resolve_layer_selector(
            index, root, root.occurrence if root.occurrence else
            # Unreachable placeholder: the resolver must reject D0 first.
            _fake_occurrence(symbol),
            SymbolId(symbol.source, "Root.__init__"), "choice", (0,),
            "layer_idx", config_selector=_selector({}))


def _fake_occurrence(symbol):
    from model_unfolder.evidence.component_owner import OwnerOccurrenceId
    return OwnerOccurrenceId(symbol)


def test_result_closure_rejects_candidate_not_in_census(tmp_path):
    case = _build(tmp_path, "self.choice = Dense()")
    result = _resolve(case, {}, indices=(0,))
    assert result.status == "resolved"
    chosen = result.decisions[0].selected_candidates[0]
    foreign_source = SourceId("/foreign.py", "f" * 64, component_key="root")
    foreign_symbol = SymbolId(foreign_source, "X")
    foreign_span = SourceSpan(foreign_source, 1, 0, 1, 1)
    foreign_site = ConstructionSiteId(
        foreign_symbol, SymbolId(foreign_source, "X.__init__"), foreign_span, 0)
    forged = SelectedConstructionCandidate(
        foreign_site, chosen.candidate_index, chosen.candidate)
    decision = LayerSelectionDecision(0, "selected", (forged,))
    with pytest.raises(ValueError):
        LayerSelectorResolution(
            "resolved", result.owner, result.owner_symbol,
            result.selector_callable, result.target,
            result.layer_index_parameter, result.candidates, (decision,))


def test_result_closure_rejects_an_open_failure_vocabulary(tmp_path):
    case = _build(tmp_path, "self.choice = Dense()")
    _index, root, _callable = case
    with pytest.raises(ValueError):
        LayerSelectorResolution(
            "failed", root.occurrence, failure_kind="plausible_new_default",
            failure_detail="must not become an unreviewed failure lane")


def test_complete_rename_changes_names_not_selector_shape(tmp_path):
    first = _build(tmp_path, """
        choose_sparse = layer_idx in config.schedule
        if choose_sparse:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """, name="before.py")
    second = _build(tmp_path, """
        arbitrary = layer_idx in config.schedule
        if arbitrary:
            self.choice = Sparse()
        else:
            self.choice = Dense()
    """, name="after.py")
    before = _resolve(first, {("schedule",): [0, 2]}, indices=(0, 1, 2))
    after = _resolve(second, {("schedule",): [0, 2]}, indices=(0, 1, 2))
    assert _selected_names(before) == _selected_names(after) == (
        "Sparse", "Dense", "Sparse")
