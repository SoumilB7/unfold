# Final derivation tool pre-use verdict

**ACCEPT bounded static implementation**, script SHA
`1bfcd6ac720d144db0ac619da0930e9d0581831ba2fae64bbee312e0b731e6e0`,
contract SHA `bdd1fbf02418b265f7a0512120d35d595529b783f175dd3dd8543df55942485c`.

Independently checked the contract's exact 41 old scopes and 53 current scopes
against the actual retained Bloom ledgers/delta. The comparator source and
expected-manifest hashes equal the pinned unchanged files. SDXL is excluded.

The script checks witness/input identity from the retained capture record,
canonical actual ledger bytes and their recorded hash, exact ordered current
list with no duplicate accommodation, and exactly twelve reviewed additions.
Only the list is replaced in a deep copy. Restoring that path must recover the
entire actual structure. The derived whole canonical hash must then equal this
witness's immutable expected ledger hash.

Actual bytes are copied unchanged and pinned before/after in `finally`. A
derived match is explicitly not an old run, mechanism proof, green gate or
blessing. Mismatch requires separately executed actual old-source recovery;
the tool merely records that command and its approved pins. No extra field
normalization or baseline update exists.

Downstream use must require successful process completion **and** the final pin
check, not merely an intermediate printed match or result file. Missing files,
invalid preparation pins or a final mutation may terminate nonzero before a
structured RED result; they must remain failures. Capture provenance still comes
from the previously reviewed actual-capture wrappers, not from this derivation.

No model, test, render or derivation-tool execution was performed in this review.
The receipt records static acceptance only; actual per-witness match results and
owner delta review remain required.
