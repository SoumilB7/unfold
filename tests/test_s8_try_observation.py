"""Try clause boundaries expose rebinding/finally poisons without interpreting them."""
from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    SourceSpan, SymbolId, build_program_index, clear_program_index_source_cache,
    portable_source_index_fingerprint,
)


def indexed(tmp_path, source):
    path = tmp_path / "wrapper.py"
    path.write_text(textwrap.dedent(source))
    bundle = SourceBundle(source="path", files=(str(path),), component_files={"root": (str(path),)})
    return build_program_index(bundle), bundle


def contains(outer, inner):
    return (outer.line, outer.col) <= (inner.line, inner.col) and \
        (inner.end_line, inner.end_col) <= (outer.end_line, outer.end_col)


def test_handler_target_and_all_clause_boundaries_are_exact(tmp_path):
    index, _ = indexed(tmp_path, """
        class Wrapper:
            def forward(owner, value, enabled):
                if enabled:
                    try:
                        value = invoke(owner, value)
                    except (TypeError, library.Failure) as owner:
                        value = recover(owner)
                    except:
                        value = fallback(value)
                    else:
                        value = success(value)
                    finally:
                        value = finish(value)
                return value
    """)
    method = index.callables[0]
    row, = index.try_observations_in(method.symbol)
    assert row.owner == method.owner
    assert row.statement.enclosing_callable == method.symbol
    assert row.statement.span == row.span
    assert row.guard[0].kind == "if"
    assert row.guard[0].test.name == "enabled"
    first, bare = row.handlers
    assert first.bound_name == "owner"
    assert first.exception.kind == "tuple"
    assert first.exception.children[1].kind == "attribute"
    assert first.exception.children[1].name == "Failure"
    assert bare.exception is None and bare.bound_name is None
    assert first.span.line == 7 and first.body_span.line == 8
    assert row.body_span.line == 6
    assert row.else_span.line == 12 and row.finally_span.line == 14
    regions = (row.body_span, first.body_span, bare.body_span, row.else_span, row.finally_span)
    calls = index.calls_in(method.symbol)
    assert [call.callee.name for call in calls] == ["invoke", "recover", "fallback", "success", "finish"]
    assert all(contains(region, call.span) for region, call in zip(regions, calls))
    gap, = index.unsupported_execution_in(method.symbol)
    assert gap.construct_kind == "try" and gap.span == row.span and gap.guard == row.guard


def test_finally_return_and_parent_write_remain_locatable(tmp_path):
    index, _ = indexed(tmp_path, """
        def wrapper(owner, value):
            try:
                return delegated(owner, value)
            finally:
                owner.child = replacement
                return alternate
    """)
    row, = index.try_observations
    assert not row.handlers and row.else_span is None
    body_return, final_return = index.return_observations_in(row.enclosing_callable)
    assert contains(row.body_span, body_return.span)
    assert contains(row.finally_span, final_return.span)
    binding, = index.bindings_in(row.enclosing_callable)
    assert contains(row.finally_span, binding.span)
    assert binding.targets[0].name == "child"
    assert len(index.unsupported_execution_in(row.enclosing_callable)) == 1


def test_nested_try_and_nested_function_do_not_share_callable_ownership(tmp_path):
    index, _ = indexed(tmp_path, """
        def wrapper(owner):
            try:
                try:
                    run(owner)
                except RuntimeError as error:
                    recover(error)
                def nested(owner):
                    try:
                        return owner()
                    finally:
                        cleanup()
            finally:
                outer_cleanup()
    """)
    outer = next(row for row in index.callables if row.symbol.qualified_name == "wrapper")
    nested = next(row for row in index.callables if row.symbol.qualified_name == "wrapper.nested")
    first, second = index.try_observations_in(outer.symbol)
    third, = index.try_observations_in(nested.symbol)
    assert contains(first.body_span, second.span)
    assert third.enclosing_callable != first.enclosing_callable
    assert len(index.try_observations) == 3
    assert index.try_observations_in(SymbolId(outer.symbol.source, "absent")) == ()


def test_source_cache_and_fingerprints_include_exact_handler_target_bytes(tmp_path):
    original = "def wrapper(owner):\n try:\n  run(owner)\n except Failure as owner:\n  recover(owner)\n"
    clear_program_index_source_cache()
    first, bundle = indexed(tmp_path, original)
    cached = build_program_index(bundle)
    assert cached.try_observations == first.try_observations
    assert cached.fingerprint == first.fingerprint
    assert cached.try_observations[0].handlers[0].bound_name == "owner"
    changed, _ = indexed(tmp_path, original.replace("as owner:", "as other:"))
    assert changed.try_observations[0].handlers[0].bound_name == "other"
    assert first.fingerprint != changed.fingerprint
    assert portable_source_index_fingerprint(first) != portable_source_index_fingerprint(changed)
    relocated = tmp_path / "other_checkout"
    relocated.mkdir()
    same, _ = indexed(relocated, original)
    assert portable_source_index_fingerprint(first) == portable_source_index_fingerprint(same)


def test_except_star_is_not_promoted_to_regular_try_semantics(tmp_path):
    index, _ = indexed(tmp_path, """
        def wrapper(owner):
            try:
                run(owner)
            except* Failure as owner:
                recover(owner)
    """)
    assert not index.parse_failures
    assert index.try_observations == ()
    assert any(row.construct_kind == "unknown_statement" for row in index.unsupported_execution)


@pytest.fixture
def complete_try(tmp_path):
    index, _ = indexed(tmp_path, """
        def wrapper(owner):
            try:
                run(owner)
            except First as error:
                recover(error)
            except:
                fallback()
            else:
                success()
            finally:
                cleanup()
    """)
    return index.try_observations[0]


def test_typed_clause_records_reject_cross_source_and_wrong_order(complete_try):
    row = complete_try
    alien = replace(row.span.source, content_fingerprint="f" * 64)
    with pytest.raises(ValueError):
        replace(row, body_span=replace(row.body_span, source=alien))
    with pytest.raises(ValueError):
        replace(row, handlers=tuple(reversed(row.handlers)))
    with pytest.raises(ValueError):
        replace(row, else_span=row.body_span)
    with pytest.raises(ValueError):
        replace(row, statement=replace(row.statement, ordinal=1), span=row.body_span)
    with pytest.raises(ValueError):
        replace(row.handlers[0], body_span=row.finally_span)
    with pytest.raises(ValueError):
        replace(row.handlers[0], exception=replace(row.handlers[0].exception,
                                                  span=replace(row.handlers[0].exception.span, source=alien)))


def test_typed_clause_records_do_not_erase_required_region_or_handler_binding(complete_try):
    row = complete_try
    with pytest.raises(ValueError):
        replace(row, handlers=(), finally_span=None)
    with pytest.raises(ValueError):
        replace(row, handlers=())  # Else requires at least one handler.
    with pytest.raises(ValueError):
        replace(row.handlers[1], bound_name="owner")  # Bare except has no target.
    with pytest.raises(ValueError):
        replace(row.handlers[0], bound_name="owner.child")
    with pytest.raises(TypeError):
        replace(row, handlers=list(row.handlers))
    with pytest.raises(TypeError):
        replace(row, finally_span=None, body_span=None)
    with pytest.raises(ValueError):
        replace(row, body_span=SourceSpan(row.span.source, row.span.end_line + 1, 0,
                                        row.span.end_line + 1, 4))
