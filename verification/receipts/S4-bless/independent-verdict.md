# S4 independent review and product-delta authorization

- step: S4 recall and unknown-rate gates
- implementation reviewed: `8f9862ed2a0937f34f862c0564a22bf59542a914`
- pre-bless receipt: `d7f78f17ca`
- independent review: `z-docs/11-execution/S4-review.md`
- reviewer verdict: **ACCEPT** (2026-09-04)
- reviewer: Claude (owner; independent of the executor)
- arbiter: Soumil
- arbiter verdict: **I approve the S4 product delta: Granite's 2 and
  SDXL's 11 unresolved-evidence notices, plus the 29 Sable-lock rows.**
- authorization date: 2026-09-04

The authorization is closed. It permits exactly:

1. the canonical `sable` surface to change for all 29 preservation witnesses;
2. `ir`, `expanded`, and `html_meta` additionally to change for
   `granite-3-0-8b-instruct`;
3. `ir`, `expanded`, and `html_meta` additionally to change for
   `stable-diffusion-xl-base-1-0`.

It does not permit a changed witness denominator, input identity, tool version,
view sequence/hash, gallery, parameter surface, ledger surface, or any other
canonical surface. The candidate manifest was generated separately and promoted
only after an exact closed-set comparison against this authorization.
