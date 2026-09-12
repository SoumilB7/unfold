# Sixth frozen S9-A independent source review

Disposition: RETURN for two concrete input-binding gaps. This is a source review, not an executed adversarial reproduction or an output approval. No tests or models were run by this reviewer. The reported 918 focused passes and 39-model dry census do not resolve these gaps.

## D1: Authentic upstream selection is not reconciled downstream

`diffusion_block.py:416` checks the stack result's issued identity/index/document scope, then validates only the block witness's own selector reads. It does not validate the retained stack witness's reads. The custom diffusion `config_guard_selector` is not one of the generic callback names captured by the central rail. Therefore an actual stack reader invoked under ambient document A with a selector returning B's deciding operands may issue a scoped result; its direct stack projection would reject B, but an actual downstream block/stream projection can pass the scope check without reaching that rejection. Conditioning and bookend projections likewise check dependency scope rather than recursively reconciling the deciding inputs of the actual dependency chain.

Required correction: a bounded shared validation of the retained actual stack→block→stream dependency inputs before any downstream finite projection. Preserve actual result identity and validate every recorded selected and unselected rival operand. A copied-result poison is insufficient: add an actual upstream wrong-selector-under-ambient-document case, then an actual downstream reader projection that must refuse it.

## D2: Captured mutable selector values are not snapshots

`diffusion_stack.py:628–635` stores the exact returned `selected` object in `DiffusionOperandRead` and returns that same object to the reader. Lists/dicts or guarded tuple payloads remain mutable. Later validation therefore observes a possibly changed value rather than the value used for selection. A frozen dataclass does not freeze its nested payload.

Required correction: retain an immutable/value snapshot of ordinary selector data at the read boundary while returning the original object unchanged. Preserve the exact typed registered-default evidence under its own validation. Add a mutable-selector poison proving mutation after selection cannot rewrite the deciding evidence.

## Bounded accepted points

The schema retains the actual four ReaderResult objects and requires value identity with the projected inventories. Source ordering and inactive-lane filtering are explicit in stream selection. Bookend duplicate slots require the same operation, operand, construction/source/checkpoint address; this is narrower than accepting equal values alone. Finite projections bind whole composite values and decisive spans. These local checks do not compensate for D1 or D2.

The sixth FFN activation fallback explicitly returns typed unavailable for raw dispatch lacking exact activation-registry membership evidence, including otherwise lowercase names. This conservatively closes the earlier R5 normalization false-authority path in this revision. A later main-tree registry extension is outside this review.

`projection_ir.py:486–514` does not route the existing rotation/position fact to a qualifying reader witness. Thus this revision does not add rotation→RoPE claim authority; the pre-existing unqualified output remains outside that acceptance.

No artifact blessing, owner exception, runtime validation, or completion claim is made. Further leaf review is deferred until the two dependency-input corrections are available in a stable revision.
