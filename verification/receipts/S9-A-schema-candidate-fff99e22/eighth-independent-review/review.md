# Eighth frozen S9-A independent source review

Disposition: RETURN for two concrete activation-registry protocol gaps. No reviewer tests/models/generators were run. No runtime success or output blessing is inferred. The exact frozen sources and manifest links are retained alongside this review.

## A1: Selected activation binding and class mutation remain outside the use closure

`activation_registry.py::_registry_use_closure` guards only registry, backing mapping and factory names. `_returned_activation` then certifies the selected local class's original declaration and forward. An explicit module-level `Chosen.forward = replacement`, or rebinding `Chosen` before the literal mapping is constructed, is an allowed Assign whose receiver is ignored by the closure. The original class remains in `index.classes`, so `_plain_class` can still certify it. The same issue applies to an explicitly mutated imported selected activation address inside the captured source closure.

This is not a request for arbitrary external-package mutation proof. It is missing treatment of explicit writes/escapes to the selected callable identity in the very source files the capability already inspects. Extend the bounded closure to the exact selected class/import address, its bindings and relevant importing consumers; reject unsupported mutation/rebinding/escape. Add an actual source fixture with a selected local class whose forward is changed after declaration. Registry/backing/factory-only poisons do not cover this.

## A2: Complete lexical statement shape does not establish a synchronous positional call protocol

`_plain_class` permits AsyncFunctionDef. `_instantiated_lookup` and `_returned_activation` require only parameter counts plus the nested statement shapes. Thus an async __getitem__ can satisfy Assign/Assign/Return, and an async forward can satisfy one Return, while actual invocation returns a coroutine rather than the certified activation result. Likewise `def __getitem__(self, *, key)` or `def forward(self, *, value)` has two parameters but cannot receive the protocol's positional key/input.

Require the exact synchronous FunctionDef declaration and supported positional parameter kinds (including any needed distinct role names) at both boundaries. ModuleStatement.kind and ParamRecord.kind already retain the needed syntax; no new interpreter or neutral index field is required. Actual async and keyword-only source poisons should remain unavailable.

## Accepted bounded corrections

E1: the expert swish reader now requires the actual sigmoid call as a direct product child, the same Name operand inside/outside sigmoid, and the same guard; it retains that product span. The gate*sigmoid(up*alpha) source poison preserves resolved storage but withholds activation. This closes the specifically reported different-lane case, without accepting aliases or claiming general formula equivalence.

The declared_kind fallback is metadata only. ReaderProjectionClaimProof authenticates the issued result/index/document before the finite projector may raise typed ReaderClaimUnavailable. The catch requires the real witness's finite owner/key declaration; it stores claim_kind/readers but explicitly removes proof and document token. EvidenceFact refuses a token without proof, and presentation only summarizes validated proof objects. The parser first retains the original value/status row, then invokes this path. No class3 attempt or qualification seal is manufactured.

The six diffusion reader files remain byte-identical to the independently accepted seventh D1/D2 correction. Their acceptance is unchanged. The neutral class decorators/keywords and module statement/name-receiver/context records retain syntax without interpreting it; the new registry implementation makes bounded use of those records. Its mutation checks cover the previously reported registry/backing/factory cases, but A1/A2 prevent accepting positive activation-registry authority yet.

No review of moving main-tree repairs is implied. Further review should target the exact next frozen correction rather than relabel this candidate.
