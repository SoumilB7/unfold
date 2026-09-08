# Bounded per-witness ledger recovery proposal

**The proposed proof is lawful with the boundaries below.** It can avoid redundant
historical model parses for witnesses whose exact committed canonical ledger is
recovered. No proposal execution or queue change occurred during this review.

The actual accepted-source Bloom capture recovers its immutable expected ledger
hash `cb0f1c08fb94057f4400c55e893d680c3a6d428d45f1f9582942b2d7edd1a520`.
Its saved old/current comparison has one changed value: the global declaration
list at `config_access.projection_coverage.receipted_scopes`, with exactly twelve
additions and no removals. Source pins and exact additions are recorded here.
`parser.py:368` authors this list from the global sorted scopes;
`evidence/receipts.py:172` derives them from registered projection routes. A
denoiser scope in this global list is not evidence that a particular witness
contains a denoiser.

For each independently captured current non-UNet ledger C:

1. Pin its actual bytes, exact executable input identity, comparator and the
   immutable expected manifest row. Verify the current scope list equals the
   independently established old list plus those exact twelve additions, with
   the expected ordering and no duplicate/removal/extra change.
2. Deep-copy C to D and replace only that one list with the actual old Bloom
   list. Retain an exact structural diff proving this is the only edit.
3. Serialize D with the unchanged existing comparator `_canon_bytes` and require
   its SHA-256 to equal this witness's immutable expected ledger hash.

On equality, under the same SHA-256 commitment assumption used by the existing
preservation gate, D recovers the whole **committed canonical ledger document**.
Every other canonical value matches that committed document, so the sole list
delta is proved per witness. Bloom supplies a candidate old list; the individual
expected hash supplies the witness-specific verification. No inference that
"all models probably changed the same way" is needed.

Limits are material:

- D is explicitly a **derived counterfactual**, never an actual old parse,
  historical execution receipt, source witness or newly qualified mechanism.
- This recovers canonical ledger values, not historical Python object identities,
  original pretty-printed byte formatting, the full IR, SVG, Sable observations
  or execution environment. Those surfaces retain their own required evidence.
- Preserve actual C untouched. Do not put this transformation in the comparator,
  baseline normalizer or product. The unchanged gate still sees the actual delta.
- Equality gives a concrete delta explanation; it grants no baseline approval.
  Owner review and Soumil's no-self-bless boundary remain intact.
- A mismatch is unresolved, not a near pass. Retain the failed derivation and
  invoke the already-reviewed actual old-source recovery/investigation fallback.
  Do not expand the edit, strip additional fields or adjust the expected hash.
- SDXL keeps its actual old/new differential; this proposal does not replace the
  user's UNet mechanism, projection and end-to-end demonstrations.

The benefit is avoiding the remaining non-UNet historical model constructions when exact
content commitments can already prove the narrowly proposed ledger delta. The
fallback remains necessary for any witness with additional or different changes.
