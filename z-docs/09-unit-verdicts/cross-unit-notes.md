# Cross-unit notes

Observations from the per-unit audits that affect more than one unit.

# Cross-unit notes (U0–U2)

- **Tracker split-brain from the start**: §21 shows U0/U1 "awaiting review"
  and U2 "PENDING"; DONE marks live in the hardening plan §17.2 and the U2
  summary; z-docs holds a third copy.
- **Recall was never a gate here either**; every "honesty" transition was
  justified by zero *corpus* delta on 25–26 witnesses. The instrument U10
  lacked was absent from day one.
- **Corpus blind spots repeat**: MusicGen (composite) was missing until R9 and
  hid the `AutoConfig` mutation bug; one multimodal witness through U1/U2.
- **U1's `config_facts.yaml` incident is U10's mistake in reverse**: U1 added
  config-authored detail (caught as drift); U10 removed source-proven detail
  (accepted as honesty). Both passed the same gate, which detects *change* but
  not *direction*.

# Cross-unit notes (U4–U5)
- **The U10 pattern is a U4 pattern.** U4 step 6 licensed demote-now /
  re-prove-later; §7 forbade it the same day. Three provable facts were
  dropped under step 6 (all restored by U6/U7), MLA cache was not.
- **Silence mechanism**: `config_accessed_unprojected` is `blocking=False`.
- **Fixture signatures are not a detail metric**; a per-fact witness diff
  (re-parse at two trees) should be a standard receipt.
- Doc/tree disagreements: U4 row "no push" (pushed); U5 "82 rows" is 83; U5
  rows self-assigned still open.

# Cross-unit notes (U6–U7)
- **The U10 pattern first appeared in U6.** `e758be4` collapsed six
  embedded-encoder/MusicGen attention drills onto the unresolved stub while
  the receipt asserted byte-identical galleries — uncheckable at the time.
- Two acceptance gaps explain both the BLOOM DTO and the U6 collapses: no
  fact ⇒ projection qualification until U8, and hash-based tracking that
  treats "distinct → identical" as a count change rather than a loss. U7 fixed
  the label-collision half; the collapse-onto-stub half is still unguarded.
- All four cited receipt directories under `/private/tmp/model-unfolder-verification/`
  are gone; the units' test-count claims are not reproducible from the repo.
