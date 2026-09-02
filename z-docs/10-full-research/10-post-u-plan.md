# The plan after the U plan

Written 2026-09-02. Everything here follows from `08-findings-register.md` and
`09-judgment.md`. It is organised so that the goal — complete structure,
honest mechanism, exact values, a diagram a person can learn from — is the
thing each phase reports against, not a ledger. Each phase names what it
ships to a user.

## §0 Decisions the author must make first (one page, before any code)

1. **Confirm the three tiers** as the product contract: tier 1 structure is
   *complete* and a missing element is a blocking finding; tier 2 mechanism is
   *honest* and the only tier where "unknown" is a legitimate result, with
   provable negatives proven; tier 3 values exact and sourced. (`00` §8, E9.)
2. **Name the user**: the learner in a notebook / the Space, with the auditor
   as the second reader. Write the one-page product spec: what "included"
   means, the journey, where it ends, what a chip means.
3. **Adopt the physics split** (`07`): tier 1 from a meta-device instance +
   FakeTensor forward trace; tier 2 from the existing static forward readers
   re-pointed at exact class instances; tier 3 from config paths + instance
   attributes. Decide the trust rule for custom-code repos (opt-in; installed
   library code is already executed today via `AutoConfig`).
4. **Release cadence**: every phase below ends with a PyPI release and a Space
   redeploy, or it is not done.

## §1 Phase 0 — stop the bleeding (days, ships v0.3.0)

- Rotate the two leaked tokens; scrub both notebooks; add `*.ipynb` secrets
  scan. (D9)
- Fix `_SourceWalker._seg` (once-per-file line split) and memoise
  `decoder_block_candidates_for_config` / `resolve_component_root` /
  `everchanging.load`: cold 25–114 s → single-digit seconds. (A7, `02` §3)
- One version string; README truthfulness (diffusion is shipped; SSM rows out
  or honest); Space caption and error transparency (typed refusals shown);
  regenerate `examples/` at HEAD with chips visible; fix `deepseek-v3.html`.
  (E2, E5, E6)
- Put `z-docs/`, `PROTOCOL.md`, `done/` under version control; delete the
  dead `preservation_manifest.json`; archive the 12 superseded docs; regenerate
  the code map. (D3, `06` keep/retire)
- Remove the three consumer unknown→known conversions, the label sniff, the
  invented literals, the phantom expert boxes, the placeholder tower for bare
  diffusion configs; make a zero-layer IR carry a warning. (B16, B17)

## §2 Phase 1 — see what you forgot (weeks 1–3, ships v0.3.1)

The missing half of verification, all blocking:
- **No waiver without a chip**: an unknown-typed fact may waive conformance
  only if the drawing carries a visible unresolved node. (B6)
- **Reverse conformance**: every executed op in the forward closure is drawn
  or chipped, for every component (encoders, towers, denoisers, UNet, VAE).
- **Directed recall ratchet** against the blessed baseline: a view/fact that
  was proven may not become unresolved without a named re-proof record;
  labels stored in the fixture signature. (B1, B2)
- **Independent oracle net**: meta instance + FakeTensor trace as a Sable
  witness — static owner graph must agree with the instantiated module tree;
  drawn op order must agree with the traced call order. (D5, `07`)
- **Independent, persisted visual verdict** required by `bless`: rubric +
  pixel↔truth trace + before/after diff from a fleet that did not implement
  the change; self-marked CLEAN refused. (D4)
- Flip `config_accessed_unprojected`/`asserted_facts` to blocking; wire
  `census.py --check`; `zero_asserted_census` must not swallow exceptions;
  unit liveness in the debt register; receipts stored in-repo; coverage test
  on real witnesses. (B10, B12, D2, D6)
- **Publish the coverage denominator**: per model proven / flagged / silent,
  silent = 0 required; per catalogue family the count of witnesses. (`05` §4)

## §3 Phase 2 — restore the structure (weeks 2–6, ships v0.4.0)

Ordered by measured leverage (`E` §5, `03` top-10):
1. Prune guarded / config-selected rival invocations with the consumed config
   value before readers run (T5, GPT-2, Falcon, MPT, Phi-3, Mistral, Mixtral).
2. Tier-1 from the instance: build the component/stack/block skeleton from the
   meta-device module tree (PixArt's real class, SD3.5's ×38, every guarded
   `__init__` branch), keep provenance by mapping instance paths to source
   spans; retire the ownership substrate that this replaces (`07`: bucket A,
   ≈55k lines → deleted or demoted to provenance lookup).
3. Wire the import closure into the DiT readers, lift the mixin-lane refusal,
   pass `canonical_import`, partial-cell projection, ordered heterogeneous
   container execution (`FeedForward.net`), two-lane FFN proof.
4. Restore every lost item with an owner: encoder attention drills (CLIP
   `WithProjection` factory walk, UMT5 per-layer bias), T5 loop-carried bias,
   Qwen2-VL vision cell (guarded/comprehension attention call; fused
   reshape/permute/unbind unpack), Qwen3.5 mask (branch-aware tainting) and
   position (`position_ids[1:]`), GLM partial RoPE, DeepSeek yarn scale and
   `q_a/kv_a_layernorm`, MLA cache, Gemma-2 `embed_scale`, Granite
   multipliers, CLIP `position_embedding`, MusicGen's T5 tower view.
5. Prove negatives (QK-norm absence, plain LayerNorm, computable widths);
   accept `Conv1D` and slice-unpack in the affine/storage protocols;
   `GELU(tanh)` labelled honestly.

Exit: diffusion tier-1 100 % on the 15 witnesses; out-of-corpus sample ≥ 90 %
proven with 0 silent; DeepSeek score scale correct.

## §4 Phase 3 — finish the migration on the new physics (weeks 5–10, ships v0.5.0)

U11–U15 re-scoped: UNet/VAE/scheduler structure from the instance tree with
mechanism readers on exact classes (the 14 unwired `unet_*` readers are
either wired or deleted); the five quarantined readers deleted; the 82
consumer-firewall rows and the 279-row register driven to zero or to lawful
display rows; YAML reduced to syntax; `parse()` split into the
observe → bind → interpret → project stages; duplicate helpers unified; one
unknown vocabulary.

## §5 Phase 4 — the learner product (in parallel from Phase 1, design-led)

- The one-page spec becomes the acceptance test: journey per archetype,
  bundling rules, where drills end, per-block parameter counts and shapes,
  the causal mask drawn, the *why* cards (the Neurons/Transformers labs'
  narratives, revived as content, not code).
- Her Eyes as a blocking per-bless persona over the fleet's diff report.
- `unfold-npm` reborn as a **renderer-only** JS package over engine JSON
  (drop the JS engine mirror); the Space serves cached engine JSON in
  seconds, not a 30-s spinner.
- Release every phase; announce what changed for a *reader*, not a ledger.

## §6 Governance that survives contact with agents

- One tracker table (unit | status | release | proven/flagged/silent), machine
  receipts in-repo, commit template enforced by a hook, no append-only rows.
- Per unit, two questions the arbiter asks that neither model wrote: "what is
  the product losing this week?" and "what are you both not measuring?" —
  answered with the coverage numbers.
- Bless = fleet verdict + before/after pixels + arbiter approval, or nothing.

## §7 What this plan does not do

It does not reopen the doctrine (it narrows one over-application), it does not
promise every Hub repo (bounded-product law stands), and it does not pause the
honesty gates for speed. It changes where truth about *existence* comes from,
adds the half of verification that was missing, and puts a reader back at the
end of the pipeline.

---

## Amendment 2026-09-02 (after the counter-review)

§3-2 ("tier-1 from the instance … retire the ownership substrate … deleted
in the same unit") is **withdrawn**. The instance tree enters as a
construction inventory and disagreement oracle in shadow mode; it becomes
tier-1 authority per family only after the disagreement matrix is understood,
with the old path kept for differential testing; deletion is responsibility
by responsibility, never in the replacement's unit. The ≈55k-line figure is a
heuristic classification, not a deletion list. Torch/library execution is an
accepted dependency; remote code runs isolated. Binding order is now
`12-execution-order.md` v2.
