"""Reproduce a same-class spatial occurrence misbinding on the initial tree.

Run from the immutable initial checkout. No pytest, test imports, worker,
production mutation, or baseline write. The construction inventory is an
explicit synthetic DTO, as in the existing runtime-binding boundary controls.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_stage_selection import read_unet_stage_selection
from model_unfolder.evidence.unet_stage_operands import read_unet_selected_stage_operands
from model_unfolder.evidence.unet_stage_constructor_operands import read_unet_selected_stage_constructor_operands
from model_unfolder.evidence.unet_selected_stage_children import read_unet_selected_stage_children
from model_unfolder.evidence.unet_cell_mechanism import read_unet_cell_mechanisms
from model_unfolder.evidence.unet_selected_child_execution import read_unet_selected_child_execution
from model_unfolder.evidence.unet_selected_spatial import read_unet_selected_spatial_operations
from model_unfolder.evidence.unet_claims import read_unet_spatial_claims
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from model_unfolder.evidence.reconciliation import reconcile
from physics.instance_inventory import (
    InstanceInventory, ModuleNode, PackageVersion, Provenance, ResolvedClass, SourceFile,
)

source = next(ast.literal_eval(node.value) for node in ast.parse(
    (ROOT / 'tests/test_unet_selected_child_execution.py').read_text()).body
    if isinstance(node, ast.Assign) and node.targets[0].id == 'SOURCE')
source = source.replace(
    'self.spatial = ModuleList([Spatial(width, convolution)])',
    'self.spatial = ModuleList([Spatial(width, convolution, stride=2), Spatial(width, convolution, stride=1)])')
document = {'kinds': ['a'], 'widths': [32], 'counts': [1], 'spatial': [True], 'convolution': [True]}

with tempfile.TemporaryDirectory(prefix='s8-review-spatial-') as directory:
    path = Path(directory) / 'model.py'
    path.write_text(textwrap.dedent(source))
    bundle = SourceBundle(source='test', architecture='Root',
                          component_files={'root': (str(path),)}, component_architectures={'root': 'Root'})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, 'root')
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(index, bundle, root, topology).require_value()
    execution = read_unet_stage_execution(construction, bundle, root).require_value()
    cells = read_unet_stage_cells(execution, bundle).require_value()
    prepared = prepare_document(document, merge=False)
    selection = read_unet_stage_selection(construction, root, DocumentBinding('root', (), prepared)).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    constructor = read_unet_selected_stage_constructor_operands(factory).require_value()
    children = read_unet_selected_stage_children(constructor, cells).require_value()
    mechanisms = read_unet_cell_mechanisms(cells).require_value()
    executed = read_unet_selected_child_execution(children).require_value()
    spatial = read_unet_selected_spatial_operations(executed, mechanisms).require_value()

    def node(path, name, children=(), attributes=None):
        module = 'torch.nn.modules.container' if name == 'ModuleList' else 'fixture.model'
        cls = ResolvedClass(module, name)
        return ModuleNode(path, cls, module, (cls,), children, (), attributes or {}, ())

    root_class = ResolvedClass('fixture.model', 'Root')
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    provenance = Provenance(
        (PackageVersion('fixture', 'local'),), (SourceFile('fixture.model', 'model.py', source_hash),),
        config_hash, root_class, 'fixture.model.Root', 'fixture.model.Root(config)', {},
        {'python': '3.12', 'platform': 'synthetic', 'hash_seed': '0', 'network': 'denied',
         'hf_hub_offline': '1', 'transformers_offline': '1', 'diffusers_offline': '1'})
    inventory = InstanceInventory(1, provenance, (
        node('', 'Root', ('down', 'up')), node('down', 'ModuleList', ('0',)),
        node('up', 'ModuleList'), node('down.0', 'Stage', ('units', 'spatial')),
        node('down.0.units', 'ModuleList', ('0',)), node('down.0.units.0', 'Unit'),
        node('down.0.spatial', 'ModuleList', ('0', '1')),
        node('down.0.spatial.0', 'Spatial'), node('down.0.spatial.1', 'Spatial'),
    ), (), ())
    table = reconcile(model='synthetic same-class spatial occurrences', inventory=inventory,
                      observations=(), config_document=prepared, program_index=spatial.index)
    fact = read_unet_spatial_claims(spatial, RuntimeSourceBindings(table, inventory, spatial.index))
    result = {
        'checkout_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source': textwrap.dedent(source), 'source_sha256': source_hash,
        'inventory_kind': 'synthetic boundary DTO, not a claimed actual model construction',
        'source_construction_strides': {'down.0.spatial.0': 2, 'down.0.spatial.1': 1},
        'reader_positive_operation_count': len(spatial.spatial_operations),
        'reader_issues': [{'kind': row.kind, 'detail': row.detail} for row in spatial.issues],
        'projected_fact': fact.value,
        'misbound_stride_one_occurrence': fact.value['down.0.spatial.1'],
    }
    assert len(spatial.spatial_operations) == 1
    assert fact.value['down.0.spatial.1']['effect'] == 'reduce'
    assert fact.value['down.0.spatial.1']['operand'] == 2
    print(json.dumps(result, sort_keys=True, indent=2))
