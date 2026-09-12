# Four post-render IR surface differences

The current capture wrappers snapshot after first HTML render. The unchanged preservation comparator snapshots and deep-copies structural IR before rendering its Diagram. These are different capture moments.

Existing metadata._block_lookup calls _ensure_declared_op_cards for reachable ops cards. A card with view=ops, declared detail.ops, and no children receives children = cards_from_region(ops_region(detail.ops)). The VAE mid card declares its three existing operations without children before this lazy projection. This helper, its lookup caller, VAE card producer, cards_from_region and ops_region have identical accepted83140/currentfb1 ASTs (recorded in result.json).

Diagram.to_ir caches a dict whose extras refer to ModelIR.extras; metadata mutates those reachable card dictionaries. Thus later Diagram.to_ir/ir.to_dict reflects generated children. split_structural_ir deep-copies before returning, so the comparator's early snapshot remains stable.

capture_current captures Diagram only after the original to_html returns, then save_diagram and surfaces serialize its mutated IR. capture_examples keeps object references from capture_unfold but serializes them only after _render_rows renders every row. Neither wrapper retained a complete actual pre-render IR snapshot. Graph/Region records do not substitute for that surface.

Actual affected VAE cards and byte hashes are retained. This receipt does not call their post-render hash an actual pre-render result; owner independently verifies any narrowly derived expected-hash recovery. No model/render run, product change or blessing was performed.
