# Bounded S9-A renderer presentation firewall review

ACCEPT at source the four-symbol renderer-only exception. No tests/models were run by this reviewer; the eighth candidate's earlier backward-import failures remain recorded until root reruns the corrected source.

PresentationChip and UnknownValueAnnotation are typed display records. PresentationChip.from_facts validates and snapshots existing typed facts; it does not construct architectural facts, resolve source, or infer model geometry. Its decoded references are display transport and bind against current producer ledger metadata. UnknownValueAnnotation retains a closed state/reason and exact path, without issuing an investigation or fact proof.

chips_from_block decodes existing producer chip records and validates exact card paths/aliases/cited fact keys. unknown_annotations_from_ir reads current IR envelopes and unresolved-value records, validates their schema paths/states, and deserializes display-only reasons. It does not run readers or author architectural values. Existing renderer uses type-check, render and record these transport objects.

The firewall allowlist contains exactly PresentationChip, UnknownValueAnnotation, chips_from_block and unknown_annotations_from_ir, tested by exact fully-qualified equality and only when consumer == renderer. It does not admit the whole presentation module or a prefix. Existing evidence/reader import policy otherwise remains intact. The added test rejects infer_architecture, wildcard imports and presentation.readers.resolve while admitting the four symbols. Thus the exception permits S9-A's specified annotation transport rather than upstream fact interpretation.

This is source acceptance for the root-authored bounded fix, not runtime or artifact approval. Moving later source is not covered by these pins.
