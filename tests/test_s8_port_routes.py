"""Call boundary wiring is not input/output semantic dependency."""
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.local_port_routes import read_local_port_route


def route(tmp_path, body):
    path = tmp_path / "source.py"
    path.write_text("class Arbitrary:\n    def forward(self, value, saved, condition):\n" +
                    textwrap.indent(textwrap.dedent(body).strip() + "\n", "        "))
    index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                             component_files={"root": (str(path),)}))
    forward = index.callables[0]
    call = next(row for row in index.calls_in(forward.symbol) if row.callee.name == "consume")
    return read_local_port_route(index, forward, call.args[0], call.span, call.guard)


def test_optional_transform_preserves_call_boundary_and_bypass(tmp_path):
    found = route(tmp_path, """
        side = saved[-1]
        if condition:
            value, side = arbitrary(value, side)
        consume(side)
    """)
    choice = found.value
    assert choice["kind"] == "conditional"
    assert choice["when_false"]["kind"] == "selection"
    assert choice["when_false"]["source"] == {"kind": "formal", "formal": "saved"}
    call = choice["when_true"]
    assert call["kind"] == "call_result" and call["result_slot"] == [1]
    assert call["mechanism"] == "unresolved"
    assert call["arguments"][1]["route"] == choice["when_false"]
    assert found.spans


def test_changed_result_slot_changes_route(tmp_path):
    left = route(tmp_path, "value, side = arbitrary(value, saved)\nconsume(side)").value
    right = route(tmp_path, "side, value = arbitrary(value, saved)\nconsume(side)").value
    assert left["result_slot"] == [1]
    assert right["result_slot"] == [0]


def test_unknown_helper_never_becomes_identity(tmp_path):
    found = route(tmp_path, "side = ignore(saved)\nconsume(side)").value
    assert found["kind"] == "call_result"
    assert found["arguments"][0]["route"] == {"kind": "formal", "formal": "saved"}
    assert found["mechanism"] == "unresolved"
    assert "depends_on" not in found


def test_loop_carrier_is_not_original_formal_each_iteration(tmp_path):
    found = route(tmp_path, """
        for unit in self.units:
            side = saved[-1]
            saved = saved[:-1]
            consume(side)
    """).value
    assert found["kind"] == "selection"
    assert found["source"]["kind"] == "loop_carried"


def test_missing_local_evidence_is_limited(tmp_path):
    assert route(tmp_path, "consume(unbound)").value["kind"] == "unresolved"


def test_rhs_reads_before_its_own_assignment(tmp_path):
    found = route(tmp_path, "value = consume(value)").value
    assert found == {"kind": "formal", "formal": "value"}


def test_loop_carrier_with_outside_alias_is_not_replaced_by_seed(tmp_path):
    found = route(tmp_path, """
        state = value
        for item in saved:
            consume(state)
            state = update(item)
    """).value
    assert found["kind"] == "loop_carried"
    assert found["initial_route"] == {"kind": "formal", "formal": "value"}


def test_loop_binding_overrides_formal_identity(tmp_path):
    found = route(tmp_path, "for value in saved:\n    consume(value)").value
    assert found["kind"] == "unresolved"


def test_with_binding_does_not_retain_original_formal(tmp_path):
    found = route(tmp_path, "with manager() as value:\n    consume(value)").value
    assert found["kind"] == "unresolved"


def test_starred_unpack_has_no_guessed_fixed_result_slot(tmp_path):
    found = route(tmp_path, "first, *middle, value = helper(saved)\nconsume(value)").value
    assert found["kind"] == "unresolved"


@pytest.mark.parametrize("body", [
    "for value in saved:\n    pass\nconsume(value)",
    "for value, item in saved:\n    pass\nconsume(value)",
    "for *value, item in saved:\n    pass\nconsume(value)",
    "with manager() as value:\n    pass\nconsume(value)",
    "state = value\nfor state in saved:\n    pass\nconsume(state)",
    "state = value\nwith manager() as state:\n    pass\nconsume(state)",
])
def test_region_target_rebinding_outlives_the_region(tmp_path, body):
    found = route(tmp_path, body)
    assert found.value["kind"] == "unresolved"
    assert found.spans


def test_unrelated_loop_target_does_not_overwrite_formal(tmp_path):
    found = route(tmp_path, "for item in saved:\n    pass\nconsume(value)").value
    assert found == {"kind": "formal", "formal": "value"}


@pytest.mark.parametrize("region", [
    "for value in saved:\n    pass",
    "with manager() as value:\n    pass",
])
def test_guaranteed_literal_assignment_closes_prior_region_rebinding(tmp_path, region):
    found = route(tmp_path, region + "\nvalue = 17\nconsume(value)").value
    assert found == {"kind": "literal", "value": 17}


def test_assignment_after_loop_can_restore_an_unrebound_formal(tmp_path):
    found = route(tmp_path, "for value in saved:\n    pass\nvalue = condition\nconsume(value)").value
    assert found == {"kind": "formal", "formal": "condition"}


def test_optional_assignment_does_not_close_prior_region_rebinding(tmp_path):
    found = route(tmp_path, "for value in saved:\n    pass\nif condition:\n    value = 17\nconsume(value)").value
    assert found["kind"] == "unresolved"


def test_completed_loop_retains_seed_and_exact_final_body_value(tmp_path):
    found = route(tmp_path, '''
        state = value
        for item in saved:
            state = discarded(item)
            state = final(item)
        consume(state)
    ''').value
    assert found['kind'] == 'loop_result'
    assert found['initial_route'] == {'kind': 'formal', 'formal': 'value'}
    assert found['iteration_result']['kind'] == 'call_result'
    # The final result's argument is the loop target, never the old formal.
    assert found['iteration_result']['arguments'][0]['route']['kind'] == 'unresolved'


def test_completed_loop_cannot_choose_one_optional_final_write(tmp_path):
    found = route(tmp_path, '''
        state = value
        for item in saved:
            if condition:
                state = left(item)
            else:
                state = right(item)
        consume(state)
    ''').value
    assert found['kind'] == 'loop_result'
    assert found['initial_route'] == {'kind': 'formal', 'formal': 'value'}
    choice = found['iteration_result']
    assert choice['kind'] == 'conditional'
    assert choice['when_true']['kind'] == 'call_result'
    assert choice['when_false']['kind'] == 'call_result'


def test_optional_augmented_update_retains_operator_and_bypass(tmp_path):
    found = route(tmp_path, '''
        state = value
        if condition:
            state += saved
        consume(state)
    ''').value
    assert found['kind'] == 'conditional'
    assert found['when_false'] == {'kind': 'formal', 'formal': 'value'}
    update = found['when_true']
    assert update['kind'] == 'inplace_operation' and update['operator'] == '+'
    assert update['operands'] == [{'kind': 'formal', 'formal': 'value'},
                                   {'kind': 'formal', 'formal': 'saved'}]


def test_sequential_optional_writes_preserve_both_guards(tmp_path):
    found = route(tmp_path, '''
        state = value
        if condition:
            state = first(saved)
        if saved:
            state = second(value)
        consume(state)
    ''').value
    assert found['kind'] == 'conditional'
    assert found['when_true']['arguments'][0]['route'] == {'kind': 'formal', 'formal': 'value'}
    assert found['when_false']['kind'] == 'conditional'
    assert found['when_false']['when_true']['arguments'][0]['route'] == {'kind': 'formal', 'formal': 'saved'}
    assert found['when_false']['when_false'] == {'kind': 'formal', 'formal': 'value'}


@pytest.mark.parametrize('transfer', ['break', 'continue'])
def test_loop_transfer_cannot_promote_lexically_last_write(tmp_path, transfer):
    found = route(tmp_path, f'''
        state = value
        for item in saved:
            state = early(item)
            if condition:
                {transfer}
            state = late(item)
        consume(state)
    ''').value
    assert found['kind'] == 'loop_result'
    assert found['iteration_result']['kind'] == 'unresolved'
    assert 'control transfer' in found['iteration_result']['reason']
