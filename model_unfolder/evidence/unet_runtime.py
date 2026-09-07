"""UNet reader orchestration over the reconciled runtime denominator.

There is no structural projection here. The adapter consumes these existing
reader results into the canonical fact layer and IR. Each dependency stays the
exact object supplied by its producing reader.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .component_owner import resolve_component_root
from .diffusion_root import read_diffusion_root_topology
from .document import DocumentBinding
from .reader_result import ReaderResult
from .reconciliation import reconcile
from .runtime_source import RuntimeSourceBindings
from .unet_attention_source import read_unet_runtime_attention_sources
from .unet_cell_mechanism import read_unet_cell_mechanisms, read_unet_stage_join_connections
from .unet_nested_mechanism import read_unet_nested_mechanisms, read_unet_runtime_nested_ffns
from .unet_root_preprocess import read_unet_root_preprocessing
from .unet_selected_child_execution import read_unet_selected_child_execution
from .unet_selected_spatial import read_unet_selected_spatial_operations
from .unet_selected_stage_children import read_unet_selected_stage_children
from .unet_stage_cells import read_unet_stage_cells
from .unet_stage_construction import read_unet_stage_construction
from .unet_stage_constructor_operands import read_unet_selected_stage_constructor_operands
from .unet_stage_execution import read_unet_stage_execution
from .unet_stage_operands import read_unet_selected_stage_operands
from .unet_stage_selection import read_unet_stage_selection


@dataclass(frozen=True)
class UNetRuntimeEvidence:
    bindings: RuntimeSourceBindings
    root: object
    reader_results: tuple[tuple[str, ReaderResult], ...]

    def result(self, name):
        return dict(self.reader_results).get(name)

    def value(self, name):
        result = self.result(name)
        return result.value if result is not None and result.has_value else None


def investigate_unet_runtime(*, model, inventory, observations, document, bundle, index):
    """Repoint D/E/F at exact runtime classes; retain unsuccessful attempts."""
    root = resolve_component_root(index, bundle, "root")
    results = []

    def run(name, reader, *args, **kwargs):
        result = reader(*args, **kwargs)
        results.append((name, result))
        return result.value if result.has_value else None

    topology = run("topology", read_diffusion_root_topology, index, root)
    construction = (run("construction", read_unet_stage_construction,
                        index, bundle, root, topology)
                    if topology is not None and topology.kind == "u_shaped" else None)
    execution = (run("execution", read_unet_stage_execution, construction, bundle, root)
                 if construction is not None else None)
    source_index = execution.index if execution is not None else index
    table = reconcile(model=model, inventory=inventory, observations=observations,
                      config_document=document, program_index=source_index)
    bindings = RuntimeSourceBindings(table, inventory, source_index)
    if execution is None:
        return UNetRuntimeEvidence(bindings, root, tuple(results))
    cells = run("cells", read_unet_stage_cells, execution, bundle, runtime_bindings=bindings)
    if cells is not None:
        run("stage_joins", read_unet_stage_join_connections, cells)
    selection = run("selection", read_unet_stage_selection, construction, root,
                    DocumentBinding("root", (), document))
    factory = (run("factory_operands", read_unet_selected_stage_operands, selection)
               if selection is not None else None)
    operands = (run("constructor_operands", read_unet_selected_stage_constructor_operands, factory)
                if factory is not None else None)
    children = (run("children", read_unet_selected_stage_children, operands, cells)
                if operands is not None and cells is not None else None)
    mechanisms = (run("mechanisms", read_unet_cell_mechanisms, cells)
                  if cells is not None else None)
    nested = (run("nested", read_unet_nested_mechanisms, mechanisms)
              if mechanisms is not None else None)
    child_execution = (run("child_execution", read_unet_selected_child_execution, children)
                       if children is not None else None)
    if child_execution is not None and mechanisms is not None:
        run("spatial", read_unet_selected_spatial_operations, child_execution, mechanisms)
    preprocessing = (run("preprocessing", read_unet_root_preprocessing, selection, root)
                     if selection is not None else None)
    if (selection is not None and preprocessing is not None
            and nested is not None and child_execution is not None):
        run("attention_sources", read_unet_runtime_attention_sources,
            selection, preprocessing, nested, child_execution, root)
    if cells is not None:
        bindings = cells.runtime_bindings
    if nested is not None:
        bindings = replace(bindings, index=nested.index)
        run("nested_ffns", read_unet_runtime_nested_ffns, nested, bindings)
    return UNetRuntimeEvidence(bindings, root, tuple(results))
