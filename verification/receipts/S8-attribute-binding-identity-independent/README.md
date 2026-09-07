# Independent frozen occurrence correction — ACCEPT within scope

Reviewed frozen worker hashes, confirmed by executor:

- physics/attribute_bindings.py: 5bd7302f421e3c6786391dbdabd341bb0b93a3b6da26ba9e5b4ba5002feb0f04
- physics/instance_inventory.py: e7c8931be7b6cdf84c0e67d87b9f8b5434eb5a5ac3897ff2521464735baacb6f

Exact bytes retained in source/. Historical RETURN receipt is unchanged. Commands from unfold-pkg:

    python3 verification/receipts/S8-attribute-binding-identity-independent/replay.py
    python3 verification/receipts/S8-attribute-binding-identity-independent/installed_class.py

Both original independent contamination classes replayed against the correction; PASS:

- A first fallback that changes second then raises retains its failure, before/after state and actual fallback function. Its registered slot change is explicit. The next lookup is unresolved because the observation session is contaminated; it has no child occurrence address. Frozen second remains [2,2] although the actual replacement is [3,3].
- A successful fallback that changes stages[0] retains the original stages container address, but exact iteration is refused with modulelist_slots_differ_from_construction. No replacement receives stages.0. The frozen stage remains [2,2] and actual replacement [3,3].

Source inspection confirms the inventory supplies original object identities and ordered registered slots, _is_constructed_occurrence checks exact identity plus current registered reachability, and _iteration compares original name/order/identity before assigning occurrence addresses. Failed/mismatched observed lookups retain state and contaminate subsequent requests through the shared session.

The actual installed-class/MRO positive also PASS: the selected forward is the real peft wrapper, config the builtin ConfigMixin property getter, and conv_in/down_blocks remain one-time observed custom-fallback children with actual super anchor 1. Exact ModuleList slots retain the child and None. The tiny fixture allocates an object of the installed class with nn.Module initialization and small child modules; it does not run the full SDXL constructor or forward.

No additional lookup syntax was explored. No pytest, model campaign, production edit, commit or blessing occurred. The previous 14-control authority review is retained; this replay covers only the two returned identity defects and the installed positive, as assigned. Before/after production fingerprints match, including after the installed-class check. Source-consumer future binding, wrapper-control closure and S8 completion remain separate obligations.
