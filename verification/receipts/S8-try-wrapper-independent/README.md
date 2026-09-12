# Independent neutral Try / wrapper review — producer ACCEPT, consumer RETURN

Reviewed frozen sources:

- program_index.py: 54379ba1c4187f65654e4153f25794cee0e1a17121ab4fad819a0ef37b5b5a11
- unet_wrapper_binding.py: f8789d6e63fbd8ca347be402cf8370788b3ddb4792d47f5baa5387bff1ce4b97
- unet_call_binding.py: 2f7680f28a73f0e1249570e68c4e2092950ee0fe325dadbf145171324ba43448 (only imported exact_callable in scope)

Exact bytes retained under source/. Replays from unfold-pkg:

    python3 verification/receipts/S8-try-wrapper-independent/replay_wrapper.py
    python3 verification/receipts/S8-try-wrapper-independent/replay_try_actual.py

## Neutral producer ACCEPT within scope

TryObservation preserves owner/callable/statement, outer guards, ordered handler clauses and their structured exception expressions, raw AST binding names, and exact body/else/finally spans. Existing UnsupportedExecutionRegion('try') is retained. Handler targets do not masquerade as ordinary assignment values or invented identifier-token spans. Except-star remains unknown. Independent source/order/handler-erasure construction poisons reject; repeated build keeps records and changed target bytes change the portable source fingerprint. Calls and finalizer writes remain locatable in the appropriate source regions. These records make no exception selection or execution guarantee.

## Wrapper RETURN: chosen exclusion can also exclude delegation

At the frozen consumer's delegation selection (lines 66–79), the reader never retains or checks call.guard. At lines 115–139 it chooses a false condition to suppress a parent-exposing call, then returns conditional_wrapper.

Counterexample:

    if enabled:
        before(owner)
        return captured(owner, value)

The emitted condition is enabled=False. Executing exactly that branch returns None and never delegates. No branch both excludes the unclosed parent call and performs delegation. This is a concrete contradiction in the returned conditional proof, not a demand for all-path execution closure. wrapper-results.json retains exact source, guards, output condition and tiny synthetic execution events.

Minimum correction: reject guarded delegation in this first bounded consumer, or establish and retain compatible delegation guards before choosing exclusions. The first option preserves actual SDXL, whose delegation is unguarded inside the admitted try body. No new neutral record is needed.

Controls matter: a compatible else-delegation works with enabled=False; nested else exposure is safely excluded by selecting outer=False. Thus the current always-False selection is not, by itself, a demonstrated inversion of an else guard. The defect is missing compatibility with the delegation's own guard. Handler-target rebinding, finalizer return, and unconditional finalizer parent exposure correctly refuse. Conditional before/finalizer effects preserve both separate source conditions.

## Actual positive and limits

Using the saved S8-actual-lookup-positive inventory, current exact installed source bytes were checked against every saved source hash, then reindexed and passed through the current wrapper reader. The real SDXL wrapper remains conditional_wrapper with two unresolved USE_PEFT_BACKEND=False source conditions. No constructor or model execution reran; this is an independent static replay of persisted worker evidence.

No pytest, production edits, commits or blessings. Actual wrapper positivity is not a runtime claim that those conditions hold and not a proof of input transparency. New root invocation logic and lookup/helper closure were outside this frozen review and require separate assessment. The executor was notified of the concrete RETURN and released to make the bounded correction only after the above replays completed. Historical bytes and findings remain preserved here.
