"""Primary region boundaries preserve source history without helper semantics."""
import textwrap
import pytest

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


def test_conditioning_expression_keeps_addition_and_bypass(tmp_path):
    rows = regions(tmp_path, "conditioning = embed(conditioning)\nconditioning = conditioning + augment(items) if enabled else conditioning\nfor item in items:\n state = block(state, conditioning)")
    side = rows[0]["route"]["iteration_result"]["arguments"][1]["route"]
    assert side["kind"] == "conditional"
    assert side["when_true"]["kind"] == "source_operation"
    assert side["when_true"]["operands"][0]["kind"] == "call_result"
    assert side["when_false"]["kind"] == "call_result"


def test_expression_guard_rebinding_does_not_preserve_old_operand(tmp_path):
    rows = regions(tmp_path, "state = block(conditioning) if (conditioning := 7) else state")
    assert rows[0]["route"]["kind"] == "unresolved"


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



def test_primary_reader_declaration_enters_catalogue_and_requires_connection():
    """Exercise the actual reader declaration at the closed ledger boundary."""
    from model_unfolder.evidence.context import FactLedger
    from model_unfolder.evidence.facts import EvidenceFact
    from model_unfolder.evidence.reconciliation import FACT_CLAIM_REQUIREMENTS
    from model_unfolder.evidence.unet_primary_ports import UNetPrimaryPortProof

    owner, _, key = UNetPrimaryPortProof.fact_id.rpartition('.')
    fact = EvidenceFact(key=key, owner=owner, value={"regions": []},
                        status="code_proven", claim_kind=UNetPrimaryPortProof.claim_kind,
                        claim_readers=UNetPrimaryPortProof.reader_symbols)
    ledger = FactLedger()
    ledger.record_typed(fact)
    assert ledger.typed[fact.ledger_key()] is fact
    assert FACT_CLAIM_REQUIREMENTS[key] == UNetPrimaryPortProof.claim_kind == "connection"


def test_actual_primary_proof_passes_summary_and_fact_qualification(tmp_path):
    from dataclasses import replace
    import hashlib
    from model_unfolder.evidence.component_owner import resolve_component_root
    from model_unfolder.evidence.context import FactLedger
    from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
    from model_unfolder.evidence.document import prepare_document
    from model_unfolder.evidence.reconciliation import reconcile, ProjectionFactCitation
    from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
    from model_unfolder.evidence.unet_primary_ports import read_unet_primary_ports
    from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
    from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
    from physics.instance_inventory import InstanceInventory, ModuleNode, PackageVersion, Provenance, ResolvedClass, SourceFile

    source = '''from torch.nn import ModuleList
class Unit:
    def forward(self, value, side=None): return value, (value,)
def build(token):
    if token == 'one': return Unit()
    return Unit()
class Root:
    def __init__(self, config):
        self.left = ModuleList([])
        self.right = ModuleList([])
        for token in config.left:
            self.left.append(build(token))
        for token in config.right:
            self.right.append(build(token))
    def forward(self, value):
        saved = (value,)
        for first in self.left:
            value, branch = first(value)
            saved += branch
        for second in self.right:
            side = saved[-1:]
            value = second(value, side)
        return value
'''
    source = source.replace('        saved = (value,)\n',
                            '        value = preprocess(value)\n' + '\n' * 90 + '        saved = (value,)\n')
    path = tmp_path / 'model.py'; path.write_text(source)
    bundle = SourceBundle(source='test', architecture='Root',
                          component_files={'root': (str(path),)}, component_architectures={'root': 'Root'})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, 'root')
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(index, bundle, root, topology).require_value()
    graph = read_unet_stage_execution(construction, bundle, root).require_value()
    cls = ResolvedClass('fixture.model', 'Root')
    provenance = Provenance((PackageVersion('fixture', 'test'),),
        (SourceFile('fixture.model', 'model.py', hashlib.sha256(path.read_bytes()).hexdigest()),),
        hashlib.sha256(b'{}').hexdigest(), cls, 'fixture.model.Root', 'fixture.model.Root(config)', {},
        {'python': '3.12', 'platform': 'test', 'hash_seed': '0', 'network': 'denied',
         'hf_hub_offline': '1', 'transformers_offline': '1', 'diffusers_offline': '1'})
    inventory = InstanceInventory(1, provenance, (ModuleNode('', cls, cls.module, (cls,), (), (), {}, ()),), (), ())
    table = reconcile(model='fixture', inventory=inventory, observations=(),
                      config_document=prepare_document({}, merge=False), program_index=graph.index)
    bindings = RuntimeSourceBindings(table, inventory, graph.index)
    fact = read_unet_primary_ports(graph, bindings)
    ledger = FactLedger(); ledger.record_typed(fact)
    citation = ProjectionFactCitation(fact)
    assert citation.summary.evidence_refs == tuple(sorted(set(citation.summary.evidence_refs)))
    assert len(citation.summary.evidence_refs) > 1
    with pytest.raises(ValueError, match='semantic kind'):
        replace(fact, claim_kind='existence')
