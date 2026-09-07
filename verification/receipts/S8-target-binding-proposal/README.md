# Proposed bounded target-binding observation

Status: proposal for owner review; no production change is included.

The current source-port view does not bind eight SDXL bookends or individual loop members to its call boundaries. `reproduce.py` demonstrates why promoting a matching source method name would be wrong: the indexed `Cell.helper` has no parent writes, but a process-local replacement makes the actual lookup call `replace_child`. The class address, source bytes, and instance attribute census remain unchanged. The constructed child returns 6 before that call and 103 afterward. This standalone Python/source-index probe is not a model execution result or a full inventory replay.

Correction from independent review: `ProgramIndex.ClassRecord.body_assigns` already records direct class-body assignment syntax. Such assignments must be checked by any source lookup closure; they are not an absent observation. The remaining gap is the actual selected callable/descriptor and executable code at the worker boundary, which the existing class address, MRO and `vars(instance)` census do not record.

## Observation contract

Add optional, explicitly requested attribute lookups to the existing instance worker boundary. Each request names an existing owner occurrence path and an attribute address. These strings locate evidence and never classify architecture. Each result has exactly one disposition:

- `registered_child`: the default supported Python/Module lookup selects the very object registered under that child slot; retain owner path, attribute, child occurrence path, and the lookup-implementation witness.
- `plain_bound_method`: default supported lookup selects a plain Python function bound to this owner. Retain the actual defining class/function address, exact source file hash and location, and a worker check that the current executable code matches code compiled from those exact source bytes. Instance shadowing, custom lookup, descriptors, absent source, and code/source disagreement cannot produce this disposition.
- `non_callable`: an observed scalar/None slot, with the existing typed scalar representation.
- `unresolved`: a concrete reason identifying the unsupported lookup boundary. No execution or exhaustion claim is manufactured.

For a requested repeated child container, also record whether its actual type and iteration callable are the exact closed supported Torch `ModuleList` implementation captured before model imports. Only this positive witness permits joining its ordered registered children to the source loop's successive elements. A same-named custom container, changed `__iter__`, instance override, or absent witness leaves the loop target unbound. Zero iterations, guards and early exits remain source facts; iterator identity does not establish branch execution.

The observation is made at construction. It does not prove the slot remains bound until a later call. That second obligation stays with a bounded source reader: retain the existing R1 alias/container/storage/delete refusal, follow only a positively bound helper method to check parent mutation/escape, and reject unclosed helper dispatch. No return-value semantics are inferred from a no-parent-write result.

## Consumer and growth boundary

The primary-port proof may annotate a call boundary with an actual occurrence only after the lookup witness, source reaching target, and parent stability all agree. Helper input/result ports may be expanded from the already indexed method while preserving opaque internal calls and guards. The canonical architecture fact remains `primary_state_ports`; the existing module cards and existing IR are reused. Loop bindings retain each actual occurrence and optional branch alternatives rather than drawing source order as execution.

Expected footprint: one small worker observation module, optional request/result fields on the existing inventory, and one bounded source binding helper. Remove the wrapper-only fallback for every target that becomes proven; keep it only for actual unresolved boundaries. Existing S7 artifacts remain historical and are regenerated for final comparison freshness. No per-model hook, new architecture IR, class namespace census, arbitrary alias interpreter, or new execution requirement is proposed.

Required counterexamples: class-body helper replacement; instance helper shadow; custom lookup/descriptor; code object changed while the file is unchanged; helper parent write/escape; child slot replaced before use; changed iterator/type; loop target rebound in the body; ordinary direct target and exact iterator positive controls. Historical R1 controls must stay rejected.
