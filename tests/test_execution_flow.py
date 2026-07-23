"""U3 Phase 3 — addressed invocation resolver poisons.

Resolves WHO each call site in the EXPLICIT execution owner's forward invokes, by
joining to exact OwnerGraph child occurrences + the B2 ContainerInventory.  Never
selects the owner; never fabricates a runtime index; a sliced iterable never binds
a loop; ModuleDict/ModuleList/Sequential execution stays unresolved.
"""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    OwnerOccurrenceId,
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.execution_flow import (
    AddressedInvocation,
    ExternalAddressedInvocation,
    InvocationResolution,
    RepeatedInvocationTemplate,
    UnresolvedInvocation,
    resolve_addressed_invocations,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    CallObservation,
    CallSiteId,
    ExprNode,
    SourceId,
    SourceSpan,
    SymbolId,
)


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(files, arch="Wrapper"):
    flat = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(source="local", files=tuple(flat),
                        component_files={k: tuple(v) for k, v in files.items()},
                        component_architectures={"root": arch})


def _pipeline(tmp_path, body, arch="Wrapper", use_root=False, source=None):
    """D0 -> B1 -> B2 -> Phase 3, at B1's model-stage occurrence (or the D0 root
    when use_root)."""
    src = (source or _MODEL).replace("        # BODY", body)
    files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = pi.build_program_index(_bundle(files, arch))
    cr = resolve_component_root(idx, _bundle(files, arch), "root")
    b1 = resolve_declared_model_stage(idx, cr)
    occ = cr.graph.root.occurrence if use_root else b1.occurrence
    inv = resolve_container_inventory(idx, cr, occ)
    res = resolve_addressed_invocations(idx, cr, occ, inv)
    return idx, cr, occ, inv, res


_MODEL = """
    class Attn:
        def __init__(self, config): pass
    class Norm:
        def __init__(self, config): pass
    class Block:
        def __init__(self, config): pass
    class Other:
        def __init__(self, config): pass
    MODELS = {"a": Attn, "b": Other}
    class BaseModel:
        def __init__(self, config):
            self.attn = Attn(config)
            self.norm = Norm(config)
            self.layers = ModuleList([Block(config) for _ in range(config.n)])
            self.seq = Sequential(Block(config), Block(config))
            self.experts = ModuleDict({})
            if config.flag:
                self.rival = Attn(config)
            else:
                self.rival = Other(config)
        def forward(self, x):
        # BODY
        def _helper(self, x):
            return x
    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = BaseModel(config)
"""


def _callee(cr, inv):
    return cr.graph.node_for(inv.callee_owner_occurrence).symbol.qualified_name


# --------------------------------------------------------------------------- #
# Direct self.<field> -> exact graph child; same module twice -> two identities
# --------------------------------------------------------------------------- #

def test_direct_self_field_resolves_to_graph_child(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.attn(x)")
    assert res.status == "resolved"
    (a,) = res.addressed
    assert isinstance(a, AddressedInvocation) and _callee(cr, a) == "Attn"
    assert a.caller_occurrence == occ and a.callee_owner_occurrence.root == occ.root


def test_same_module_called_twice_is_two_identities(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            h = self.attn(x)
            return self.attn(h)""")
    assert len(res.addressed) == 2
    assert len({a.call_site for a in res.addressed}) == 2          # distinct CallSiteIds
    assert {_callee(cr, a) for a in res.addressed} == {"Attn"}     # same callee class
    assert res.addressed[0].callee_owner_occurrence == res.addressed[1].callee_owner_occurrence


def test_external_primitive_call_has_a_separate_exact_construction_identity(tmp_path):
    source = ("    from torch.nn import LayerNorm\n" + _MODEL).replace(
        "self.norm = Norm(config)", "self.norm = LayerNorm(config.n)")
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            h = self.norm(x)
            for layer in self.layers:
                h = layer(h)
            return h""", source=source)
    (external,) = res.external_addressed
    assert isinstance(external, ExternalAddressedInvocation)
    assert external.construction.external_reference.qualified_target == \
        "torch.nn.LayerNorm"
    assert external.construction.occurrence.parent == occ
    assert external.call_site in res.call_sites
    assert not any(item.call_site == external.call_site for item in res.unresolved)


def test_shadowed_external_constructor_stays_unresolved(tmp_path):
    source = ("    from torch.nn import LayerNorm\n"
              "    LayerNorm = replacement\n" + _MODEL).replace(
        "self.norm = Norm(config)", "self.norm = LayerNorm(config.n)")
    idx, cr, occ, inv, res = _pipeline(
        tmp_path, "            return self.norm(x)", source=source)
    assert res.external_addressed == ()
    assert any(item.reason == "rival_or_unresolved_child"
               for item in res.unresolved)


# --------------------------------------------------------------------------- #
# Symbolic ModuleList loop -> template (never N); sliced/indexed -> unresolved
# --------------------------------------------------------------------------- #

def test_modulelist_loop_is_a_repeated_template(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            return x""")
    (t,) = res.templates
    assert isinstance(t, RepeatedInvocationTemplate) and t.container.field == "layers"
    assert len(t.container.element_sites) == 1                     # ONE template, not N
    assert t.element_template in t.container.element_sites


def test_sliced_iterable_loop_preserves_the_exact_base_container(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for layer in self.layers[: 2]:
                x = layer(x)
            return x""")
    (template,) = res.templates
    assert template.iteration_kind == "sliced"
    assert template.container.field == "layers"
    assert template.loop.iterable.kind == "subscript"
    assert template.loop.iterable.children[1].kind == "slice"


def test_builtin_enumerate_binds_the_second_tuple_target(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""")
    (template,) = res.templates
    assert template.iteration_kind == "enumerated"
    assert template.element_target.name == "layer"
    assert template.container.field == "layers"


def test_builtin_enumerate_over_slice_preserves_both_protocol_steps(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for index, layer in enumerate(self.layers[: 2]):
                x = layer(x)
            return x""")
    (template,) = res.templates
    assert template.iteration_kind == "enumerated_sliced"
    assert template.element_target.name == "layer"
    assert template.loop.iterable.children[1].kind == "subscript"


def test_reversed_enumerate_targets_do_not_bind_by_a_familiar_spelling(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for layer, index in enumerate(self.layers):
                x = layer(x)
            return x""")
    assert res.templates == ()
    assert any(u.reason == "local_or_free_name_call" for u in res.unresolved)


def test_locally_shadowed_enumerate_is_not_a_builtin_protocol(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            enumerate = self._helper
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""")
    assert res.templates == ()
    assert any(u.reason == "local_or_free_name_call" for u in res.unresolved)


def test_module_shadowed_enumerate_is_not_a_builtin_protocol(tmp_path):
    src = _MODEL.replace(
        "    class Attn:",
        "    enumerate = object()\n    class Attn:",
    ).replace("        # BODY", """
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""")
    files = {"root": (_write(tmp_path, "m.py", src),)}
    bundle = _bundle(files)
    idx = pi.build_program_index(bundle)
    cr = resolve_component_root(idx, bundle, "root")
    b1 = resolve_declared_model_stage(idx, cr)
    inv = resolve_container_inventory(idx, cr, b1.occurrence)
    res = resolve_addressed_invocations(idx, cr, b1.occurrence, inv)
    assert res.templates == ()
    assert any(binding.name == "enumerate"
               for binding in idx.module_bindings_in(b1.occurrence.root.source))


def test_indexed_access_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.layers[0](x)")
    assert res.addressed == () and res.templates == ()
    assert any(u.reason == "indexed_access_unproven" for u in res.unresolved)


def test_moduledict_items_loop_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for k, m in self.experts.items():
                x = m(x)
            return x""")
    assert res.templates == ()
    assert any(u.reason in ("loop_iterable_not_a_cited_container", "local_or_free_name_call")
               for u in res.unresolved)


# --------------------------------------------------------------------------- #
# Container / Sequential / helper / non-owner calls stay unresolved
# --------------------------------------------------------------------------- #

def test_sequential_called_directly_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.seq(x)")
    assert any(u.reason == "container_execution_unproven" for u in res.unresolved)


def test_helper_method_call_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self._helper(x)")
    assert any(u.reason == "field_is_not_a_constructed_child" for u in res.unresolved)


def test_non_owner_call_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return torch.cat([self.attn(x)])")
    assert any(u.reason == "non_owner_call" for u in res.unresolved)
    assert len(res.addressed) == 1                                 # the inner self.attn resolves


def test_rival_field_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.rival(x)")
    assert res.addressed == ()
    assert any(u.reason in ("rival_or_unresolved_child", "rival_child_occurrences")
               for u in res.unresolved)


# --------------------------------------------------------------------------- #
# Never selects an owner; consistency of inventory + D0; failures
# --------------------------------------------------------------------------- #

def test_explicit_diffusion_root_selection(tmp_path):
    src = _MODEL.replace("        # BODY", "            return self.attn(x)")
    # a root with no base_model_prefix -> B1 absent; caller passes the D0 root
    src = src.replace('base_model_prefix = "model"', "pass").replace(
        "self.model = BaseModel(config)", "self.attn = Attn(config)")
    files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = pi.build_program_index(_bundle(files))
    cr = resolve_component_root(idx, _bundle(files), "root")
    occ = cr.graph.root.occurrence                                  # explicit selection
    inv = resolve_container_inventory(idx, cr, occ)
    res = resolve_addressed_invocations(idx, cr, occ, inv)
    assert res.status in ("resolved", "absent")


def test_mismatched_inventory_and_bad_types_are_rejected(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.attn(x)")
    other = OwnerOccurrenceId(SymbolId(SourceId("/x.py", "fp", component_key="root"), "Ghost"))
    other_inv = resolve_container_inventory(idx, cr, occ)
    object.__setattr__(other_inv, "owner_occurrence", other)       # forge a mismatched inventory
    with pytest.raises(ValueError):
        resolve_addressed_invocations(idx, cr, occ, other_inv)
    with pytest.raises(TypeError):
        resolve_addressed_invocations(idx, cr.graph, occ, inv)     # not a D0
    with pytest.raises(TypeError):
        resolve_addressed_invocations(idx, cr, occ, object())      # not an inventory


def test_owner_not_in_graph_fails(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, "            return self.attn(x)")
    ghost = OwnerOccurrenceId(SymbolId(SourceId("/x.py", "fp", component_key="root"), "Ghost"))
    ghost_inv = resolve_container_inventory(idx, cr, occ)
    object.__setattr__(ghost_inv, "owner_occurrence", ghost)
    res2 = resolve_addressed_invocations(idx, cr, ghost, ghost_inv)
    assert res2.status == "failed" and res2.failure_kind == "owner_not_in_graph"


def test_every_call_site_is_partitioned_once(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            h = self.attn(x)
            for layer in self.layers:
                h = layer(h)
            return torch.cat([h])""")
    sites = ([a.call_site for a in res.addressed]
             + [t.call_site for t in res.templates]
             + [u.call_site for u in res.unresolved])
    assert len(sites) == len(set(sites))                           # disjoint partition


# --------------------------------------------------------------------------- #
# Closure forgeries
# --------------------------------------------------------------------------- #

def _occ(name="BaseModel"):
    return OwnerOccurrenceId(SymbolId(SourceId("/m.py", "fp", component_key="root"), name))


def _site():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    return CallSiteId(fn, __import__("model_unfolder.evidence.program_index",
                                     fromlist=["SourceSpan"]).SourceSpan(fn.source, 5), 0)


def test_addressed_invocation_requires_non_empty_child_occurrence():
    occ = _occ()
    with pytest.raises(ValueError):
        AddressedInvocation(_site(), occ, occ, None, (), ())        # empty callee occurrence


def test_resolution_partition_closure():
    occ = _occ()
    with pytest.raises(ValueError):                                 # failed w/o kind
        InvocationResolution("failed", occ)
    with pytest.raises(ValueError):                                 # resolved w/o owner+callable
        InvocationResolution("resolved", occ)
    with pytest.raises(ValueError):                                 # detail w/o kind
        InvocationResolution("failed", occ, failure_detail="x")


# --------------------------------------------------------------------------- #
# Correction poisons: heterogeneous/unresolved container, census + partition
# --------------------------------------------------------------------------- #

def _from_src(tmp_path, src, arch="Wrapper"):
    files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = pi.build_program_index(_bundle(files, arch))
    cr = resolve_component_root(idx, _bundle(files, arch), "root")
    b1 = resolve_declared_model_stage(idx, cr)
    occ = cr.graph.root.occurrence if b1.status != "resolved" else b1.occurrence
    inv = resolve_container_inventory(idx, cr, occ)
    return idx, cr, occ, inv, resolve_addressed_invocations(idx, cr, occ, inv)


def test_heterogeneous_container_loop_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _from_src(tmp_path, """
        class A:
            def __init__(self, config): pass
        class B:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.mixed = ModuleList([A(config), B(config)])
            def forward(self, x):
                for m in self.mixed:
                    x = m(x)
                return x
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.templates == ()                                     # element_sites[0] never chosen
    assert any(u.reason == "heterogeneous_or_unresolved_container_elements"
               for u in res.unresolved)


def test_unresolved_element_construction_loop_is_unresolved(tmp_path):
    idx, cr, occ, inv, res = _from_src(tmp_path, """
        class BaseModel:
            def __init__(self, config):
                self.blocks = ModuleList([make_block(config) for _ in range(config.n)])
            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.templates == ()
    assert any(u.reason == "heterogeneous_or_unresolved_container_elements"
               for u in res.unresolved)


def test_unresolved_call_alongside_valid_calls_partition_census(tmp_path):
    idx, cr, occ, inv, res = _from_src(tmp_path, """
        class Attn:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.attn = Attn(config)
            def forward(self, x):
                a = self.attn(x)
                b = get_thing()(x)
                return torch.cat([a, b])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert len(res.addressed) == 1                                  # self.attn resolves
    assert res.unresolved                                           # the others do not
    # exact partition of the complete call-site census:
    bucket = ([i.call_site for i in res.addressed]
              + [t.call_site for t in res.templates]
              + [u.call_site for u in res.unresolved])
    assert set(bucket) == set(res.call_sites) and len(bucket) == len(set(bucket))
    assert len(res.call_sites) == len(list(idx.call_sites_in(res.callable_symbol)))


def test_omitted_call_site_partition_is_rejected():
    occ = _occ()
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    s1 = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    s2 = CallSiteId(fn, SourceSpan(fn.source, 2), 0)
    # census has two sites but no bucket covers them -> partition inequality
    with pytest.raises(ValueError):
        InvocationResolution("resolved", occ, occ.root, fn, call_sites=(s1, s2))


def test_duplicate_census_is_rejected():
    occ = _occ()
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    s = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    with pytest.raises(ValueError):
        InvocationResolution("resolved", occ, occ.root, fn, call_sites=(s, s))


def test_absent_invocation_resolution_carries_no_call_sites_or_graph():
    occ = _occ()
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    s = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    with pytest.raises(ValueError):                        # absent-with-call-sites
        InvocationResolution("absent", occ, occ.root, call_sites=(s,))
    with pytest.raises(ValueError):                        # absent-with-callable
        InvocationResolution("absent", occ, occ.root, callable_symbol=fn)


def test_forged_unresolved_caller_is_rejected():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    site = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    call = CallObservation(None, fn, 0, ExprNode(kind="name", name="f"), None,
                           (), (), (), SourceSpan(fn.source, 1))
    with pytest.raises(TypeError):
        UnresolvedInvocation(site, "not-an-occurrence", "reason", call, ())
    # a forged call site that is not CallSiteId.of(call) is rejected
    wrong = CallSiteId(fn, SourceSpan(fn.source, 9), 0)
    with pytest.raises(ValueError):
        UnresolvedInvocation(wrong, _occ(), "reason", call, ())


def test_unresolved_sibling_owner_cannot_enter_resolution():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    site = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    call = CallObservation(None, fn, 0, ExprNode(kind="name", name="f"), None,
                           (), (), (), SourceSpan(fn.source, 1))
    owner = _occ("BaseModel")
    sibling = _occ("Sibling")
    unresolved = UnresolvedInvocation(site, sibling, "reason", call, call.guard)
    with pytest.raises(ValueError):
        InvocationResolution(
            "resolved", owner, owner.root, fn, (site,), unresolved=(unresolved,))


def test_invocation_guard_must_equal_authoritative_call_guard():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "BaseModel.forward")
    span = SourceSpan(fn.source, 1)
    site = CallSiteId(fn, span, 0)
    call = CallObservation(None, fn, 0, ExprNode(kind="name", name="f"), None,
                           (), (), (("actual",),), span)
    with pytest.raises(ValueError):
        UnresolvedInvocation(site, _occ(), "reason", call, (("forged",),))


def test_failed_invocation_resolution_rejects_census_and_callable_payload():
    owner = _occ()
    fn = SymbolId(owner.root.source, "BaseModel.forward")
    site = CallSiteId(fn, SourceSpan(fn.source, 1), 0)
    with pytest.raises(ValueError):
        InvocationResolution(
            "failed", owner, owner.root, fn, (site,),
            failure_kind="owner_not_in_graph")


def test_repeated_template_round_trips_to_call_and_loop(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            return x""")
    (template,) = res.templates
    assert template.call_site == CallSiteId.of(template.call)
    assert template.guard == template.call.guard
    with pytest.raises(ValueError):
        replace(template, guard=())
    with pytest.raises(ValueError):
        replace(template, iteration_kind="enumerated")
    with pytest.raises(ValueError):
        replace(template, element_target=ExprNode(kind="name", name="other"))


def test_exception_target_cannot_launder_shadowed_enumerate_as_builtin(tmp_path):
    idx, cr, occ, inv, res = _pipeline(tmp_path, """
            try:
                x = x
            except Exception as enumerate:
                x = x
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""")
    assert res.templates == ()
    assert any(region.construct_kind == "try"
               for region in idx.unsupported_execution_in(res.callable_symbol))
