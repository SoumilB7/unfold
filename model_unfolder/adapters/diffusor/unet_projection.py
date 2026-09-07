"""Thin canonical projection of qualified UNet facts.

No source, config, or module-name classification occurs in this consumer.
Construction cards describe containment; execution edges require their own
connection facts and cannot be inferred from the order of these cards.
"""
from ...block_schema import Block
from ...ir import ModelIR
from ...opgraph import ffn_region
import json
from .blocks import diffusion_loop_blocks, diffusion_loop_edges, diffusion_loop_region


def _block_id(path):
    return "instance_" + path.replace(".", "__") if path else "denoiser"


def project_unet(*, facts, handoffs, name, architecture, table=None):
    modules = facts["root.denoiser.constructed_modules"].value
    shapes = facts["root.denoiser.constructed_parameter_shapes"].value
    relations = facts["root.denoiser.constructed_stage_relations"].value
    population_key = "root.denoiser.constructed_modules"
    shape_key = "root.denoiser.constructed_parameter_shapes"
    relation_key = "root.denoiser.constructed_stage_relations"
    ffn_key = "root.denoiser.ffn_mechanisms"
    ffns = facts[ffn_key].value if ffn_key in facts else {}
    join_key = "root.denoiser.stage_join_connections"
    joins = facts[join_key].value if join_key in facts else {}
    primitive_key = "root.denoiser.runtime_primitives"
    primitives = facts[primitive_key].value if primitive_key in facts else {}
    connection_key = "root.denoiser.cell_connections"
    connections = facts[connection_key].value if connection_key in facts else {}
    context_key = "root.denoiser.context_connections"
    contexts = facts[context_key].value if context_key in facts else {}
    arithmetic_key = "root.denoiser.cell_arithmetic"
    arithmetic = facts[arithmetic_key].value if arithmetic_key in facts else {}
    defaults_key = "root.denoiser.declared_constructor_defaults"
    defaults = facts[defaults_key].value if defaults_key in facts else {}
    spatial_key = "root.denoiser.spatial_mechanisms"
    spatial = facts[spatial_key].value if spatial_key in facts else {}
    context_formals = list(dict.fromkeys(row["source_formal"] for row in contexts.values()))
    context_ids = {formal: f"unet_context_{number}" for number, formal in enumerate(context_formals)}
    dispositions = ({row.provenance.instance_path: row for row in table.occurrences}
                    if table is not None else {})

    def card(path):
        module = modules[path]
        children = [card(f"{path}.{child}".lstrip(".")) for child in module["children"]]
        parameters = [(key, row) for key, row in shapes["parameters"].items()
                      if key.rpartition(".")[0] == path]
        chips = [f"{shapes['by_module'][path]:,} parameters in subtree"]
        disposition = dispositions.get(path)
        if disposition is not None and disposition.execution.kind == "execution_unresolved":
            chips.append("Execution: " + disposition.execution.reason_class)
            chips.append(disposition.execution.detail or disposition.execution.reason)
        chips.extend(f"{key.rpartition('.')[2]}: " + " × ".join(map(str, row["shape"]))
                     for key, row in parameters)
        block: Block = {
            "id": _block_id(path), "kind": "opaque", "role": "constructed",
            "label": path.rpartition(".")[2], "title": module["class_name"],
            "description": (
                "Constructed occurrence: " + path + ". "
                "The children below show containment. Their order does not establish execution order."),
            "facts": chips, "source_instance_path": path,
            "source_fact_keys": [population_key, shape_key],
            "source_component": "root", "source_owner": module["class_name"],
        }
        if children:
            block.update(children=children, view="constructed_children")
        if path in primitives:
            primitive = primitives[path]
            block.update(kind=primitive["kind"], label=primitive["label"],
                         title=primitive["label"],
                         description="Operation established by the exact constructed framework type. Caller connections require their own source proof.")
            block["source_fact_keys"].append(primitive_key)
        if path in connections:
            connection = connections[path]
            nodes = {}
            for key, row in connection["calls"].items():
                meaning = primitives.get(row["member"], {})
                nodes[key] = {"id": _block_id(path) + "__" + key,
                              "target": _block_id(row["member"]),
                              "kind": meaning.get("kind", "opaque"),
                              "label": meaning.get("label", "Constructed child call"),
                              "guard": row["guard"]}
            block.update(view="runtime_cell_connections",
                         detail={"connection_calls": nodes,
                                 "connections": connection["connections"]})
            block["facts"].append(connection["unresolved"])
            block["source_fact_keys"].append(connection_key)
            if any(row["member"] in primitives for row in connection["calls"].values()):
                block["source_fact_keys"].append(primitive_key)
        if path in ffns:
            mechanism = ffns[path]
            ffn = {"kind": "dense", **mechanism}
            namespace = _block_id(path) + "__op_"
            operations = []
            for op in ffn_region(ffn, None).ops:
                if op.kind == "input":
                    continue
                operations.append({
                    "id": namespace + op.id, "kind": op.kind,
                    "role": "operation", "label": op.label or op.fn or op.id,
                    "title": op.label or op.fn or op.id,
                    "description": "Operation on the source-proven returned FFN computation.",
                    "source_fact_keys": [ffn_key],
                })
            block.update(kind="ffn", role="ffn", label="Feed-forward", view="runtime_ffn",
                         title="Source-proven feed-forward computation",
                         detail={"ffn": ffn, "op_namespace": namespace},
                         children=operations + children)
            block["facts"].extend(["Activation: " + mechanism["activation"],
                                    "Projection storage: " + mechanism["projection_mode"]])
            for field in ("input_projection", "output_projection"):
                weight = shapes["parameters"].get(mechanism[field] + ".weight")
                if weight is not None:
                    block["facts"].append(field.replace("_", " ") + ": "
                                          + " × ".join(map(str, weight["shape"])))
            block["source_fact_keys"].append(ffn_key)
        if path in arithmetic:
            mechanism = arithmetic[path]
            operands = []
            for number, label in enumerate(("Primary input branch", "Transformed branch")):
                operand_id = _block_id(path) + f"__return_operand_{number}"
                operands.append(operand_id)
                children.append({"id": operand_id, "kind": "unknown", "role": "unresolved_branch",
                                 "label": label, "title": label,
                                 "description": "This operand of the return addition is proven. Its complete internal route remains under investigation.",
                                 "facts": ["investigation_missing · branch lineage · owner: S8"],
                                 "source_fact_keys": [arithmetic_key], "resolved": False})
            detail = dict(block.get("detail", {}))
            detail.setdefault("connections", [])
            detail.setdefault("connection_calls", {})
            detail["return_arithmetic"] = {"operands": operands, "scale": mechanism["return_scale"]}
            block.update(view="runtime_cell_connections", detail=detail, children=children)
            block["source_fact_keys"].append(arithmetic_key)
            for injection in mechanism["conditioning"]:
                block["facts"].append(("Conditional " if injection["conditional"] else "")
                                      + injection["operation"] + " side input from " + injection["source_formal"])
        if path in joins:
            routes = []
            by_path = {child.get("source_instance_path"): child for child in children}
            for number, row in enumerate(joins[path]):
                parents = {target.rpartition(".")[0] for target in row["targets"]}
                if len(parents) != 1 or next(iter(parents)) not in by_path:
                    continue
                target = by_path[next(iter(parents))]
                operands = []
                for slot in row["operand_slots"]:
                    operand_id = _block_id(path) + f"__join_{number}_operand_{slot}"
                    operands.append(operand_id)
                    children.append({
                        "id": operand_id, "kind": "unknown", "role": "unresolved_input",
                        "label": "Operand source unresolved", "title": "Concat operand lineage",
                        "resolved": False,
                        "description": row["input_lineage_reason"],
                        "facts": ["investigation_missing · owner: S8"],
                        "source_fact_keys": [join_key],
                    })
                routes.append({"operands": operands, "target": target["id"]})
                target["facts"].append("Source-proven concat output feeds the repeated child call")
                target["source_fact_keys"].append(join_key)
            if routes:
                block.update(view="runtime_stage_connections", detail={"join_routes": routes}, children=children)
                block["source_fact_keys"].append(join_key)
        if path in contexts:
            row = contexts[path]
            block["facts"].extend(["External context input proven", row["reason"],
                                    "investigation_missing · cross-attention query role · owner: S8"])
            block["source_fact_keys"].append(context_key)
            block.update(view="runtime_context_connection", detail={
                "source": context_ids[row["source_formal"]],
                "source_label": "External context input",
                "target_formal": row["target_formal"]})
        if path in spatial:
            row = spatial[path]
            label = "Spatial reduction" if row["effect"] == "reduce" else "Spatial resize"
            block.update(label=label, title=label)
            block["facts"].append("Source-proven spatial primitive: " + row["primitive"])
            if row["operand"] is not None:
                block["facts"].append("Stride: " + str(row["operand"]))
            else:
                block["facts"].append("Resize direction: investigation_missing")
            block["source_fact_keys"].append(spatial_key)
        return block

    stage_paths = (relations["producer_stages"] + relations["intermediate_stages"]
                   + relations["consumer_stages"])
    stage_cards = [card(path) for path in stage_paths]
    for block in stage_cards:
        block["label"] = block["source_instance_path"]
        block["source_fact_keys"].append(relation_key)
        block["facts"].append("Cross-attention role: investigation_missing · owner: S8")
    contained = set(stage_paths)
    other_paths = [path for path in modules
                   if path and not any(path == parent or path.startswith(parent + ".")
                                       for parent in contained)
                   and path.rpartition(".")[0] == ""
                   and path not in {relations["producer_field"], relations["consumer_field"]}]
    other_cards = [card(path) for path in other_paths]
    other_cards.append({
        "id": "unet_skip_bank", "kind": "opaque", "role": "skip_accumulator",
        "label": "Saved skip outputs", "title": "Source-proven skip accumulator",
        "description": "Repeated producer calls contribute bypass outputs to an accumulator. Later calls receive a selected portion. Individual producer-to-consumer pairings remain under investigation.",
        "facts": ["Accumulation and selected-input route proven", "Exact tensor-to-cell pairing: investigation_missing · owner: S8"],
        "source_fact_keys": [relation_key],
    })
    skip_routes = ([{"source": _block_id(path), "target": "unet_skip_bank"}
                    for path in relations["producer_stages"]]
                   + [{"source": "unet_skip_bank", "target": _block_id(path)}
                      for path in relations["consumer_stages"]])
    other_cards.extend({
        "id": context_ids[formal], "kind": "source", "role": "context_input",
        "label": "External context input", "title": "Root input: " + formal,
        "description": "Required root input transported by the proven source route to the selected context argument.",
        "facts": ["Connection proven; cross-attention query role remains under investigation"],
        "source_fact_keys": [context_key],
    } for formal in context_formals)
    context_routes = list({(context_ids[row["source_formal"]], _block_id(row["stage"])): {
        "source": context_ids[row["source_formal"]], "target": _block_id(row["stage"])}
        for row in contexts.values()}.values())
    geom = dict(handoffs)
    geom.update({
        "denoiser_family": "source_projected",
        "denoiser_label": ["U-Net", "Denoiser"],
        "denoiser_title": "Constructed U-Net denoiser",
        "denoiser_desc": "Exact constructed stages and parameter shapes. Open execution relations remain visible.",
        "denoiser_children": stage_cards + other_cards,
        "denoiser_view": "unet_constructed",
        "output_domain": "unknown", "block_conditioning": None,
        "suppress_conditioning_source": True,
    })
    render = {
        "family": "diffusion", "layout": "unet_pipeline", "theme": "teal",
        "denoiser_view": "unet_constructed",
        "loop_blocks": diffusion_loop_blocks(geom),
        "loop_edges": diffusion_loop_edges(geom), "loop_region": diffusion_loop_region(),
    }
    # The old loop builder has a denoiser-kind display default. Supply the
    # explicit view and citations at the actual block boundary.
    for block in render["loop_blocks"]:
        if block["id"] == "denoiser":
            block.update(view="unet_constructed", source_instance_path="",
                         source_fact_keys=[population_key, shape_key, relation_key])
            if contexts:
                block["source_fact_keys"].append(context_key)
            if defaults:
                block["source_fact_keys"].append(defaults_key)
                block["facts"] = list(block.get("facts") or ()) + [
                    f"Declared class default · {key}: {json.dumps(row['value'])} (checkpoint omitted)"
                    for key, row in defaults.items()]
    return ModelIR(
        name=name, architecture=architecture, vocab_size=0, hidden_size=None,
        max_position_embeddings=None, tie_word_embeddings=None, layers=[],
        extras={"render": render, "unet": {
            "stage_relations": relations,
            "stage_block_ids": {path: _block_id(path) for path in stage_paths},
            "other_block_ids": [_block_id(path) for path in other_paths],
            "context_routes": context_routes,
            "skip_routes": skip_routes,
            "parameter_shapes": shapes,
        }},
        warnings=[row["reason"] for row in relations["unresolved_relations"]],
    )
