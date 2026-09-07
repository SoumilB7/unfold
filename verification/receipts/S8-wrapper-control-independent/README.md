# Independent wrapper-control prerequisite assessment

Ruling: the refusal is correct. A finite neutral source-record extension can close this specific address gap; a general exception interpreter or new architectural IR is unnecessary. This assessment does not authorize implementation or accept S8 completion.

Run from unfold-pkg:

    python3 verification/receipts/S8-wrapper-control-independent/replay.py

Independent replay reproduces the saved pre-refusal false positive (conditional_wrapper) and the current correct unresolved result. results.json records exact reader/index hashes and actual installed source hash. No pytest or model construction/execution ran. The script imports only synthetic wrapper functions for source/code observations.

## Why current records cannot support the old conclusion

ProgramIndex._visit_stmt, program_index.py:1697–1706, emits one UnsupportedExecutionRegion('try'), then visits body, handlers, else and finally under the same outer guard. Their call/binding observations do not say which exception clause contains them.

The callable IdentifierObservation contract at lines 346–366 explicitly excludes string-valued exception capture targets. The visit_ExceptHandler at line 1267 belongs to a different module-binding collector; it does not repair the callable census. Therefore except Exception as owner supplies no callable store record for owner. The red fixture's captured(owner, value) received the exception object, yet the previous reader accepted the original bound owner. Existing unsupported status must be respected until this missing clause information is available.

Actual installed diffusers/utils/peft_utils.py, SHA256 78d1748aeec2a730f639bdeddc5c988c9f16e389c420d5ad8b150a50978c26dc, differs in a bounded, syntactically observable way: its wrapper's try at lines 313–320 has no handlers and no else; forward_fn(self, *args, **kwargs) occurs in the body at 315; conditional unscale_lora_layers(self, lora_scale) is in finally at 320. This distinction does not require knowing whether a particular exception occurs.

## Minimum lawful record

A neutral TryObservation, analogous to LoopObservation, can carry:

- owner, enclosing_callable, StatementId, outer guard, whole statement SourceSpan;
- body_span;
- ordered handler clauses, each with clause span, exception_type as ExprNode or None, bound_name as the exact AST field or None, and body_span;
- else_span and finally_span, optional when absent.

The AST handler-name string can be retained explicitly as a clause field with clause-level provenance. Do not manufacture an ast.Name, a binding value, or an exact identifier-token span from it. If the existing IdentifierObservation contract is extended instead, actual token-aware span capture is required. Exception types remain structured ExprNode values, not parsed diagnostic strings.

Spans and ordered clauses suffice to associate existing calls/bindings/transfers with clauses and detect nested boundaries. No new architectural role, chosen branch, exception match, executable edge, or completed-return flag belongs in this record. Keep unsupported execution status for semantics outside the reader's admitted form; adding syntax coverage is not universal execution closure.

Strict minimum for the current witness is even smaller operationally: a reader may require no handlers and no else. Handler clause presence then rejects the red poison without interpreting its target at all. Retaining ordered handler targets is a small neutral completeness measure, not permission to begin supporting handler flow.

## Bounded reader contract and non-goals

The first consumer can admit only the observed simple try/finally form after exact runtime callable/source/closure matching. It must retain which delegation belongs to the body and which effects belong to finally. Conditions excluding unclosed parent-exposing calls remain explicit source branch conditions; a current local USE_PEFT_BACKEND value is not a deployment or all-path proof. The wrapper also edits kwargs before delegation, so target identity does not imply byte-for-byte input transparency.

Handler bodies, try-else, except-star, nested unsupported controls, finally returns/raises/breaks/continues or writes affecting a claimed return remain limited unless separately supported. In particular, a call occurring in a try body does not establish that the wrapper returns that call's value: finally can replace or abort the return. Supporting the actual no-handler form requires no handler selection, exception-type evaluation, raising-call prediction, general Python interpreter, all-path execution closure, or model/class-name exception.

## Stop-rule disposition

U11_UNET_EXECUTION_PLAN.md §9 explicitly says to stop and report when the one ProgramIndex needs a new neutral observation record. The executor correctly reported that prerequisite and retained a refusal. The rule does not say every neutral-record implementation must receive a separate user permission prompt. The owner can assess this concrete bounded source contract against the already authorized S8 work; any expansion of architectural semantics, preservation approval or support-set policy remains a separate decision. This receipt supplies that assessment and does not make the owner's implementation decision.
