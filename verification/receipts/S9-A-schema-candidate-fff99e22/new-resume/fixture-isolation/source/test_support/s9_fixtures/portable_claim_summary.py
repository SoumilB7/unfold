"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import hashlib
import json
from model_unfolder.evidence import program_index as PI
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.instance_population_claim import read_constructor_defaults
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from physics.instance_inventory import InstanceInventory, ModuleNode, PackageVersion, Provenance, ResolvedClass, SourceFile


def _defaults(directory, *, code=None, auxiliary='HELPER = 1\n', owner=None,
              provenance='fixture.helper'):
    """A real static constructor proof with an explicitly reconciled tiny DTO."""
    directory.mkdir()
    model = directory / 'model.py'
    model.write_text(code or 'class Root:\n    def __init__(self, mode="silu"): pass\n')
    helper = directory / 'helper.py'
    helper.write_text(auxiliary)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    helper_digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    components = {'root': (str(model),)}
    if owner is not None:
        components[owner] = (str(helper),)
        external = ()
    else:
        external = (PI.SourceFileNode(PI.SourceId(str(helper), helper_digest,
                      external=True, external_provenance=provenance)),)
    index = PI.build_program_index(SourceBundle(
        source='path', files=tuple(p for paths in components.values() for p in paths),
        component_files=components), external_nodes=external)
    cls = ResolvedClass('fixture.model', 'Root')
    config = {}
    document = prepare_document(config, merge=False)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    environment = {'python': '3.12', 'platform': 'test', 'hash_seed': '0',
                   'network': 'denied', 'hf_hub_offline': '1',
                   'transformers_offline': '1', 'diffusers_offline': '1'}
    origin = Provenance((PackageVersion('fixture', 'local'),),
        (SourceFile('fixture.model', 'model.py', digest),), config_sha, cls,
        'fixture.model.Root', 'fixture.model.Root(config)', {}, environment)
    inventory = InstanceInventory(1, origin, (
        ModuleNode('', cls, cls.module, (cls,), (), (), {}, ()),), (), ())
    table = reconcile(model='synthetic', inventory=inventory, observations=(),
                      config_document=document, program_index=index)
    bindings = RuntimeSourceBindings(table, inventory, index)
    fact = read_constructor_defaults(bindings, document)
    assert fact is not None
    return index, document, fact
