# Independent wrapper correction / lookup consumers — partial ACCEPT, RETURN

Frozen source bytes are retained in source/; fingerprints.json records identical hashes before and after all replays:

- wrapper: 6f55779e38c91b9a322b3155c2beff22e102981f7fe32f4b0e72abc2eaa59930
- lookup/helper: 399fc757456deefe91590951b0fdca378fc39c88d99cfdbe7a3ce56cddb10fca
- cell connections: 2ac9cbe708cefb91091e509172d163dd65cd60a96b5cbc47c1765feefd544682
- call binding: 2f7680f28a73f0e1249570e68c4e2092950ee0fe325dadbf145171324ba43448

## ACCEPT: bounded wrapper correction

replay_wrapper.py confirms guarded delegation now refuses, including the original self-excluding branch. A compatible guarded else-delegation also remains limited, consistently with the bounded contract. Unguarded delegation with conditional before/finally effects stays positive; handler capture, finalizer control replacement and unconditional finalizer parent exposure refuse. replay_try_actual.py retains the neutral producer acceptance and independently rebuilds the real saved SDXL wrapper from matching current source bytes: conditional_wrapper, two source exclusions. No new deployment value is asserted.

## RETURN L1: canonical hasattr does not close its stored receiver lookup

Earliest producer: unet_lookup_closure.py:207–217 admits hasattr(self.__dict__[storage], name) as an allowed prefix call based on canonical builtin identity and syntax. That does not establish that the addressed object's attribute lookup is harmless or cannot mutate the owner.

replay_lookup_receiver.py uses an actual tiny worker-constructed fixture with ConfigProbe(self) in the addressed storage. Its __getattr__ changes owner.child from Linear(2,2) to Linear(3,3) when an external flag changes, then raises AttributeError. Initial observation has observed_state_changed=False and registered_slots_unchanged=True. The source reader returns conditional_registered_child. On a later lookup, the earlier config-return condition is still excluded, yet the prefix hasattr replaces the child and super selects the replacement. The emitted condition therefore does not establish the claimed original occurrence.

The test does not ask to support this receiver class. It shows that the current positive reader lacks a required premise. Minimum correction: establish the actual receiver's supported lookup/effect behavior through bounded evidence, or retain a typed limitation. Do not infer that behavior from storage spelling, class name, one successful lookup or builtin hasattr identity alone. This is separate from the accepted worker construction-address correction: the worker correctly captured the initial object graph; the false claim concerns a later source-authorized lookup.

## RETURN L2: inherited helper exclusion contradicts an inner target binding

Earliest producer: unet_call_binding.py:182 checks only exclusions discovered in the current method's skipped calls. Lines 187–199 append inherited helper conditions and emit the target without checking its own source guard against those conditions.

replay_helper_conditions.py uses an actual tiny worker fixture:

    def helper(self, value):
        if enabled:
            ignored = self.opaque  # getter source is unresolved
            return self.child(value)
        return value

Parent-stability closure excludes enabled=True and retains enabled=False. The root call to helper and later root child may remain conditionally bound. However, pending traversal also emits a constructed target for the child inside helper, whose own guard is enabled=True, carrying the inherited enabled=False condition at the identical source address. The emitted route is contradictory.

Minimum correction: suppress a target inside an inherited excluded guard, or establish compatible call conditions. No additional syntax or condition evaluator is needed for this exact same-source contradiction. This is the same contract as the wrapper correction at a different producer boundary.

## Saved actual positive and scope

replay_saved_helpers.py reuses S8-actual-lookup-positive's saved inventory, checks every current indexed source hash against its saved source hash, and independently rebuilds the source index and reader results. All four actual root helpers close conditionally: get_time_embed, get_class_embed, get_aug_embed, process_encoder_hidden_states. No identical-source guard contradiction was found in those actual target rows. This confirms the ordinary positive remains; it does not cancel either counterexample or establish that every emitted condition is sufficient. saved-helper-results.json contains exact targets and limitations.

Run from unfold-pkg:

    python3 verification/receipts/S8-lookup-wrapper-followup-independent/replay_wrapper.py
    python3 verification/receipts/S8-lookup-wrapper-followup-independent/replay_try_actual.py
    python3 verification/receipts/S8-lookup-wrapper-followup-independent/replay_lookup_receiver.py
    python3 verification/receipts/S8-lookup-wrapper-followup-independent/replay_helper_conditions.py
    python3 verification/receipts/S8-lookup-wrapper-followup-independent/replay_saved_helpers.py

The negative replays intentionally record the current false-positive results for review. Historical receipts were not overwritten. No pytest, full model construction or model campaign, production edit, commit or blessing. Only tiny synthetic objects/functions were executed. Real product rendering and source-condition presentation remain owner/executor review obligations.
