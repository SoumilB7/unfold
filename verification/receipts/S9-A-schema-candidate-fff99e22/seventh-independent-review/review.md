# Seventh frozen S9-A independent source review

Disposition: RETURN for the newly stamped expert swish formula boundary; ACCEPT the bounded diffusion D1/D2 corrections and norm/width source portions described below. No tests or models were run by this reviewer, no output is blessed, and known runtime failures are not relabelled as passes. The independently reviewed bytes are copied beside this receipt and checked against the seventh manifest where listed.

## D1/D2 corrected

`diffusion_stack.py:640–653` now authenticates exact upstream witness type, actual issued result, original producer callable, identical index and prepared document, then invokes that witness's input validator. Block, stream, conditioning and bookend witnesses recursively traverse the retained dependency chain before projecting. Bookends require all three dependencies. The block validates its own deciding inputs after its stack dependency. This closes the prior scope-only bypass.

`capture_diffusion_selector` now snapshots nested dict/list/tuple values before returning the original selector object unchanged. The actual upstream foreign-selector under ambient A poison and actual post-read mutable-guard poison are present in test_diffusion_reader_claims.py:156–201, alongside a matching-input positive control. Typed registered defaults remain subject to their constructor/path/omission validation. These are source-control observations, not reviewer-executed results.

Schema identity checks and projection routing still retain actual block/stream/conditioning/bookend results. Active lanes remain filtered before source-ordered owner assignment. Existing position/rotation output is not newly admitted by the qualifying router.

## E1: Swish applied-function identity is not established by product reachability

`ExpertStorageClaimWitness.project` at expert_storage.py:338–354 newly certifies the complete non-dispatch activation formula as applied_function. Its existing `_swish_formula_evidence` at lines800–874 checks a sigmoid argument is a multiplication with one numeric operand and that the sigmoid reaches a storage-proven product. It does not compare the other sigmoid argument operand with the gate multiplied outside the sigmoid. The surrounding `_lane_product_binding` at1448–1467 only requires both lane dependencies and a multiplication; it cannot supply that equality.

Concrete source falsifier, not executed here: in the existing `_literal_swish_expert` fixture, replace `gate * torch.sigmoid(gate * self.alpha)` with `gate * torch.sigmoid(up * self.alpha)`. The latter can retain two-lane storage/product flow yet is not the claimed gate swish formula. Clip classification also follows the sigmoid input and would not repair the missing same-operand relation.

Required bounded correction: prove the actual sigmoid nonnumeric operand and outer multiplicative gate have the same versioned source operand, plus the claimed offset/clamp sides, or leave this formula's applied_function proof unavailable. Keep independently proved storage and width available. Do not change pre-existing output values as an incidental stamping fix. Add an actual producer poison for the changed sigmoid lane; no label-only or copied-DTO test substitutes for it.

## Bounded norm and width points accepted at source

Decoder norm retains every selected candidate, actual examined invocation census and exact primitive construction/classification. It checks unanimous positive family and projects presence_only, preserving the positive-census limitation. Embedding norm retains an unguarded primitive and its actual versioned def-use edge into the repeated child; its completeness is local to that positive edge, not whole-CFG absence. The seventh norm fixture's reported completeness failures remain runtime failures pending corrected fixtures and a run; they do not justify changing the proof to complete.

FFN width authenticates its original mechanism invocation/index/document and invokes the upstream finite gate projection to validate selected operands. Expert width authenticates its original storage invocation. Both compare all recorded arithmetic premises with the supplying document, retain checkpoint/class-default distinctions and exact paths, and project the whole integer value. Existing width producers check output/input dimensions and retain merged premises. No new false value/owner/document binding was found in these bounded portions. Storage relation certification is separate from E1's applied-function formula problem.

Activation registry authority remains explicitly excluded: its known module mutation/call-use hole is being repaired separately. No acceptance of the seventh positive registry implementation is implied by this review.
