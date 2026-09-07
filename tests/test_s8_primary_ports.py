"""Primary region boundaries preserve source history without helper semantics."""
import textwrap

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_primary_ports import read_primary_regions


def regions(tmp_path, body):
    path = tmp_path / "model.py"
    path.write_text("class Cell:\n def forward(self, state, conditioning, items, enabled):\n" +
                    textwrap.indent(textwrap.dedent(body).strip() + "\n", "  "))
    index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                              component_files={"root": (str(path),)}))
    return read_primary_regions(index, index.callables[0], "state")[0]


def test_other_formal_history_survives_region_boundary(tmp_path):
    rows = regions(tmp_path, "conditioning = normalize(conditioning)\nstate = block(state, conditioning)")
    assert rows[0]["route"]["arguments"][1]["route"]["kind"] == "call_result"


def test_auxiliary_definition_does_not_borrow_later_state_version(tmp_path):
    rows = regions(tmp_path, "conditioning = normalize(state)\nstate = first(state)\nstate = block(state, conditioning)")
    historical = rows[1]["route"]["arguments"][1]["route"]["arguments"][0]["route"]
    assert historical["kind"] == "unresolved"
    assert rows[1]["route"]["arguments"][0]["route"] == {"kind": "region_input"}


def test_conditioning_merge_before_loop_retains_call_routes(tmp_path):
    rows = regions(tmp_path, "conditioning = embed(conditioning)\nif enabled:\n conditioning = adapt(conditioning)\nfor item in items:\n state = block(state, conditioning)")
    side = rows[0]["route"]["iteration_result"]["arguments"][1]["route"]
    assert side["kind"] == "conditional"
    assert side["when_true"]["kind"] == side["when_false"]["kind"] == "call_result"


def test_conditioning_entry_cannot_skip_inner_rebinding(tmp_path):
    rows = regions(tmp_path, "conditioning = embed(conditioning)\nfor item in items:\n with manager() as conditioning:\n  pass\n state = block(state, conditioning)")
    loop = rows[0]["route"]
    assert loop["kind"] == "unresolved" or loop["iteration_result"]["kind"] == "unresolved"


def test_loop_keeps_current_region_seed_and_zero_iteration(tmp_path):
    rows = regions(tmp_path, "state = first(state)\nfor item in items:\n state = block(state)")
    loop = rows[1]["route"]
    assert loop["kind"] == "loop_result"
    assert loop["initial_route"] == {"kind": "region_input"}
    carried = loop["iteration_result"]["arguments"][0]["route"]
    assert carried["kind"] == "loop_carried"
    assert carried["initial_route"] == {"kind": "region_input"}


def test_loop_target_and_with_rebinding_are_explicit_regions(tmp_path):
    for middle in ("for state in items:\n pass", "with manager() as state:\n pass"):
        rows = regions(tmp_path, "state = first(state)\n" + middle + "\nstate = last(state)")
        assert len(rows) == 3
        assert rows[1]["route"]["kind"] == "unresolved"
        assert not rows[1]["receives_previous_state"]


def test_conditional_replacement_does_not_become_identity(tmp_path):
    rows = regions(tmp_path, "if enabled:\n state = 7\nstate = last(state)")
    assert rows[0]["route"]["when_true"] == {"kind": "literal", "value": 7}
    assert rows[0]["route"]["when_false"] == {"kind": "region_input"}


def test_optional_augmented_update_is_retained(tmp_path):
    rows = regions(tmp_path, "for item in items:\n state = block(state)\n if enabled:\n  state += extra(item)")
    route = rows[0]["route"]["iteration_result"]
    assert route["kind"] == "conditional"
    assert route["when_true"]["kind"] == "inplace_operation"
    assert route["when_false"]["kind"] == "call_result"


def test_nested_else_update_retains_both_calls_and_update(tmp_path):
    rows = regions(tmp_path, "for item in items:\n if enabled:\n  state = first(state)\n else:\n  state = second(state)\n  if conditioning:\n   state += extra(item)")
    branch = rows[0]["route"]["iteration_result"]
    assert branch["kind"] == "conditional"
    assert branch["when_true"]["kind"] == "call_result"
    assert branch["when_false"]["kind"] == "conditional"
    assert branch["when_false"]["when_true"]["kind"] == "inplace_operation"
    assert branch["when_false"]["when_false"]["kind"] == "call_result"


def test_method_receiver_and_constant_overwrite_are_distinct(tmp_path):
    rows = regions(tmp_path, "state = state.flatten()\nstate = 7")
    assert rows[0]["route"]["receiver"]["route"] == {"kind": "region_input"}
    assert rows[0]["receives_previous_state"]
    assert not rows[1]["receives_previous_state"]


def test_short_circuit_named_write_is_not_ignored(tmp_path):
    rows = regions(tmp_path, "if enabled and (state := 7):\n state = block(state)")
    assert rows[0]["route"]["kind"] == "unresolved"
