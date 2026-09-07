# Independent review: the SDXL root primary-state route

Scope: read-only inspection of the exact installed root implementation and existing neutral index records. This is a source-premise review for the executor, not acceptance of a new production reader, the completed S8 demonstration, or any output delta. No pytest, model execution, production edits, schema extension, blessing, or commit was performed.

Reviewed index implementation: immutable checkpoint `d3a4d61732233a14ee67917f48fe5ec11d41f10e` at `/private/tmp/unfold-s8-followup-review`. Installed source: `/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/diffusers/models/unets/unet_2d_condition.py`, SHA-256 `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`. The fresh ordinary SDXL inventory reports that same root runtime source hash.

## Conclusion

The existing records contain enough syntax to establish a bounded, guarded chain of **call input and result ports** through the primary state. That does not establish that an opaque call's result semantically depends on its input. It does not select a guard outcome, establish an actual loop iteration, or bind a loop target to an individual runtime module by itself. Those are separate obligations.

A lawful overview can therefore show the constructed down, mid and up stage groups, with a primary-state route through explicit repeat and conditional boundaries and visible optional updates. A direct, unconditional `conv_in → down → mid → up → conv_out` chain would omit real source computation. Assigning individual stage-to-stage tensor edges solely from the order of registered modules would also overclaim.

## Smallest source-port chain

All line numbers below refer to the exact installed root source above. The variable names locate expressions; they are not evidence of mechanism.

| Region | Positive source premise | Boundary that must remain |
| --- | --- | --- |
| Input projection, 1110 | The result of `self.conv_in(sample)` is assigned to the primary state. | Binding the call target to the constructed projection needs exact runtime member and unchanged-call evidence. Preprocessing can already replace the incoming state at 1080 and 1098; a route starting at the public input must retain these. |
| Down repetition, 1138–1159 | Both alternatives at 1145 and 1155 pass the incoming state as `hidden_states` and assign result slot 0 to the state and slot 1 to a residual result. | Loop entry/zero-iteration exit and loop carry must be explicit. In the non-cross branch, 1157 optionally updates the state with an opaque `pop(0)` result before the next iteration or exit. The other branch passes optional additional residuals into the call. |
| Skip bank, 1137, 1159, 1161–1170 | The initial bank contains the input-projection result; each down call's second result is appended. The ControlNet branch rebuilds the bank with additions. | This is a separate state channel. Do not route primary state through the bank update or erase the optional bank transformation. Exact per-stage skip pairing needs its own slice/count proof. |
| Mid conditional, 1173–1192 | When the outer guard is taken, both alternatives at 1175/1184 pass the entering state to the mid call and assign its result to the state. | Preserve the outer bypass. Within the taken branch, the adapter guard can update the state at 1192. Both inner alternatives remain guarded; the spelling `has_cross_attention` proves no attention mechanism. |
| Mid residual update, 1194–1195 | The ControlNet branch adds the supplied mid residual to the current state and assigns the result back. | This branch is outside the outer mid-presence guard. It cannot be hidden inside the mid call or skipped in the route to the up loop. |
| Up repetition, 1198–1226 | Both alternatives at 1210/1221 pass the current state as `hidden_states`, along with a separately sliced residual tuple, and assign their call result to the state. | Explicit entry/zero-iteration exit and result-to-next-iteration carry. `enumerate(self.up_blocks)` alone is not a proof of an individual stage's execution. The bank slice/drop at 1201/1202 remains a separate input route. |
| Output normalization conditional, 1229–1231 | The taken branch routes state into `conv_norm_out`, then its result into `conv_act`, then assigns that result as state. | The untaken branch carries the incoming state unchanged. Do not silently draw normalization as unconditional. |
| Output projection, 1232 | `conv_out` consumes the merged state from the preceding conditional. | The value comes from that merge, not directly from the last syntactic up call. |

The finite primary-state transfer is thus: input-projection result → down repeat with optional adapter update → optional mid call with optional adapter update → optional mid residual addition → up repeat → optional normalization/activation → output-projection input. Each call is an opaque boundary until its own reader establishes a stronger mechanism claim.

## What the current neutral index provides

`premises.json` records the selected `BindingObservation` rows, their guards, the root loops, and the existing dataflow operator records. `BindingObservation.value` for an augmented assignment contains only the right-hand side. At 1157 and 1192, interpreting that value as the new state would incorrectly discard the old state. The existing same-statement `DataflowObservation` records preserve `aug:+`; a bounded reader can use those records without changing the index schema.

`LoopObservation` records the target, iterable, guard and lexical body extent. It does not by itself establish loop execution or iteration order. A recurrence proof must join the incoming state, every state write in the body, the branch merges, the carry and the zero-iteration exit. A proof of straight-line reaching definitions cannot be reused across the loop boundary unchanged.

The index explicitly marks short-circuit guards at 1113, 1139, 1142, 1156, 1174, 1188–1190 and 1209 as unsupported execution expressions carrying calls. These predicates can remain opaque. Their syntactic alternatives may still be represented, but their truth values and side-effect behavior cannot be invented. Target-binding proofs must separately account for calls which could invalidate a binding; recognizing an exhaustive `if`/`else` is not permission to assume arbitrary guard calls are harmless.

## Target binding is separate from port wiring

A call's syntax establishes which argument expression enters which call port and which assigned value receives which result slot. To identify a call as a particular constructed stage also requires the runtime occurrence, its exact class/source, the container/member binding, and applicable unchanged-forward/ancestor/alias checks. For loop targets, connecting a particular iteration to a particular member requires supported container iteration semantics and no unresolved target or container rebinding. Registration order, class equality and card order are insufficient.

Consequently a partial but honest proof can retain `down-loop call`, `mid call` or `up-loop call` as an opaque target while showing its established port route. It must not relabel that route as a proven affine, attention, conditioning or skip mechanism. Calls such as `pop()` likewise expose a result port, not an automatic semantic dependence from all of their arguments to that result.

## Minimal review poisons for the proposed reader

These are obligations for the executor's bounded implementation, not a request for a general interpreter:

- Removing the optional update at 1157 or 1192 must change the route proof; inserting `sample = opaque(sample)` after a stage call must introduce its opaque boundary.
- A loop that executes zero times must preserve the incoming state through a visible loop merge. An in-loop use of the previous state cannot always resolve to the pre-loop seed.
- A replacement assignment in one branch must remain that branch's value, including where the other branch retains the old value.
- Rebinding the loop target or changing its container must prevent an unsupported individual-module attribution.
- Replacing a guard with an opaque call must not make either alternative certain; replacing a call with a constant-returning helper must not preserve a stronger input-to-output semantic-dependence claim.

## S8 product intent

The full-stage intent does not require solving every execution path or exhausting the original 240 unknown execution rows. It requires the constructed stage content to be visible and its claimed connections to have evidence. A guarded repeat route linking actual down/mid/up groups, plus the separately established skip and conditioning routes, satisfies that architectural direction without pretending every optional operation executes. Individual stage edges require the additional iteration-to-occurrence join described above. Unknown target or route segments should remain explicit at the relevant boundary rather than produce disconnected stage collections or unqualified arrows.

This review supports implementing that bounded route from the current records. Acceptance still requires inspection of the actual proof constructor, refusal controls, qualified facts and generated HTML on an immutable candidate.

## Reproduce

Run in `/private/tmp/unfold-s8-followup-review`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-root-primary-route-review/inspect_premises.py --source /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/diffusers/models/unets/unet_2d_condition.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-root-primary-route-review/premises.json
```

Exit code: 0. This records syntax, not a runtime recipe or complete mechanism proof.
