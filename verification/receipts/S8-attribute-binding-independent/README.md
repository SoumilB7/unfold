# Independent requested-lookup worker review — RETURN

Scope: frozen neutral worker and inventory extension only; no production source-consumer acceptance, model campaign, pytest, code edits, commits or output blessings. Reviewed working tree on audio-composite-support, frozen by executor:

- physics/attribute_bindings.py: 2fcb2b32c53061c9ced783bd7d6295d32e3dde2cb80e5711e734b11d30a5f377
- physics/instance_inventory.py: 5da02e89ec8ac6ac5264fade1ffd13fb6da5b581450331dbc038750933d77a9b

Exact bytes retained under source/. Run from unfold-pkg:

    python3 verification/receipts/S8-attribute-binding-independent/replay.py
    python3 verification/receipts/S8-attribute-binding-independent/installed_class.py

Fourteen supplied standalone controls PASS. Independent tiny installed-class/MRO fixture PASS: actual forward is the peft wrapper, config is ConfigMixin.config's builtin property getter, conv_in/down_blocks are explicitly observed custom fallback results with actual super anchor 1; iteration preserves alias slot and None. This fixture uses only object allocation, nn.Module initialization and small Linear/ModuleList children, not SDXL construction or forward execution. Exact observations are in installed-class.json.

## Blocking finding: observation-time replacements reuse frozen construction addresses

Earliest join producer: physics/instance_inventory.py:645–648 retains original occurrence_objects but passes only the current owner to lookup observation. physics/attribute_bindings.py:404–409 derives a child_path from attribute spelling without checking the selected object against that original occurrence. _iteration at lines 292–293 similarly derives each occurrence_path from the current registered slot name.

Two executable counterexamples are retained in replay.py/results.json:

1. First requested fallback replaces self.second, a Linear(2,2), with Linear(3,3), then raises. Its unresolved result at attribute_bindings.py:395–396 drops before/after state. The next request emits observed_registered_child child_path=second, observed_state_changed=False, registered_slots_unchanged=True. Frozen second has weight shape [2,2]; the actual selected object has [3,3].
2. A successful first lookup replaces self.stages[0] with Linear(3,3). Both observations report unchanged bounded state and owner registered slots. The later exact ModuleList iteration assigns the replacement occurrence_path=stages.0, although frozen stages.0 remains Linear(2,2). This does not require a custom iterator or new supported lookup syntax.

The bounded before/after fields are not themselves falsely described: the documented shallow snapshots cannot detect every descendant change. The defect is the positive occurrence address crossing from the post-observation object graph into the frozen construction graph without identity evidence.

Minimum correction: validate every selected child and non-None iterator slot against the original occurrence object at that exact path; otherwise emit a typed limitation. Preserve the frozen construction denominator/shapes. Failed or mismatching custom fallback observations must retain their recorded contamination context, or later observations must conservatively be marked contaminated. No new supported lookup form or architectural IR is needed.

## Supported conclusions and limits

Actual plain-function code/source matching, refusal of executable mismatch, wrapped-metadata separation, bounded closure unknowns, builtin identity, exact property selection without executing its getter, actual super anchor, exact ModuleList iterator/length and serialization passed the stated controls. Unresolved closure cells remain unknown. A property getter witness does not establish its return value. The super record identifies the available next fallback; it does not prove the source invokes it. Custom fallback results remain one-time observations and cannot establish future call selection from parent stability alone. Function defaults and arbitrary globals are not captured by these records and cannot be inferred from source declarations. Source selection, mutation/escape closure, argument interpretation and actual architectural connection proofs remain separate reader obligations.

RETURN applies to the construction-address defect above. No claim is made that the S8 product or demonstration is complete. Review artifacts only; executor may commit them with the eventual correction receipt.
