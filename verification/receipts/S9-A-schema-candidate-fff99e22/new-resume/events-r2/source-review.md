# Event return correction — source only

Frozen25 correctly returned a runtime name collision: the new nested helper _ffn_fact_status was overwritten by the existing FFN-schedule string assignment in parse(). The helper is now _ffn_status_with_selector_default, with only its definition and two calls renamed. The old status scalar and all calculation, event, fact, and reader logic are unchanged. A stdlib AST inventory of the whole parse lexical scope finds exactly one binding of the new helper name: its definition.

The ordinary synthetic fixture directly constructs PreparedDocument without provenance. Config access explicitly defines an empty provenance map as unestablished and forbids interpreting it as checkpoint_declared (config_access.py lines 107–111). The old assertion invented an origin that the fixture never supplied. Both ordinary synthetic event assertions now require exactly the empty string, while default cases still require exactly class_default. This is a strict unknown-origin preservation check, not acceptance of multiple origins or a production provenance rewrite. Actual MLA/partial controls continue through config_to_ir production preparation.

No tests, models, or product imports were run. Both files parse with stdlib AST. Previous handoff and frozen25 are unchanged; source review cannot replace root runtime verification.
