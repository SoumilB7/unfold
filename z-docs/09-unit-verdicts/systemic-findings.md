# Systemic findings — the campaign judged against its own goal

The goal (`../01-product/next-product-architecture.md`): a general evidence
compiler for neural architecture — a new model is accurate when its mechanisms
are already understood, a new mechanism costs one interpreter, unknown stays
visible, and every surface projects one canonical fact. This file judges the
campaign as a whole, independent of any single unit, and states how it would
be done from here.

## 1. What is unambiguously right

1. **The doctrine.** Code as the only mechanism authority, config as operands,
   typed unknown, one-way dependency direction, deletion-in-the-same-unit,
   symmetric trust. It is correct and it is rare. The transformer decoder
   core proves it works: 14/14 witnesses fully resolved at `HEAD`, zero
   `wiring_unresolved`, MLA/GQA/gated-delta/hybrid stacks all proven from
   source with no family branch anywhere in the tree.
2. **Soundness enforcement.** Identity guard, consumer firewall, structural
   debt register with deletion conditions, qualification (fact ↔ spec), op /
   nested / wiring conformance, reverse-fabrication poisons, isolated-worktree
   receipts with fingerprints. Since U4 there is no code path from "unknown"
   to "conventional default", and that has held through six further units.
3. **The migration discipline.** Shadow-first slices, exact counterexample
   matrices, refusal to special-case (BERT, Lumina, StableLM, DBRX were all
   solved at shared boundaries), explicit owner reassignment when a unit
   cannot close a row.

## 2. The two things it overlooked

### 2.1 A self-imposed index boundary was mistaken for honesty

The ProgramIndex indexes the model's own modeling file(s). Real architectures
are spread across files: Diffusers keeps `BasicTransformerBlock`,
`JointTransformerBlock` and the `Attention` container every DiT uses in shared
modules; Transformers keeps shared processors and helpers likewise. Python
follows imports; the index did not. When a block class lived outside the
bundle the readers correctly reported "unresolved" — and the process
classified that as *honesty* rather than *blindness*.

Consequences, measured (`recall-trajectory.md`):

- U10: PixArt and SD3.5 denoisers went from correct, conformance-verified
  skeletons to blank boxes; Mochi and Wan lost their block drills; per-layer
  attention kind is unresolved on essentially every diffusion denoiser layer
  because head geometry lives in the external container.
- U6: distinct CLIP/UMT5/T5 encoder attention drills and MusicGen's decoder
  attention collapsed onto the unresolved stub; approved on an uncheckable
  "galleries byte-identical" claim; some still missing at HEAD.
- U9: Qwen2-VL's vision cell (an ordinary pre-norm attention+MLP block in the
  model's own file) dropped to opaque because one child reader cannot see an
  attention call inside a guarded list comprehension, and the projection then
  discarded the FFN and norm it *had* resolved; the removal was described to
  Soumil as "two fabricated cell drills" — both were correct.
- U8: Qwen3.5's provable causal-mask and partial-RoPE legends became
  "unresolved"; the T5 encoder split shipped 23 layers drawn *without* a
  relative bias the source applies — a silent negative in six galleries.

**This gap was known from the start.** U3's binding boundary 8 (2026-07-19)
required shared Diffusers files to enter as explicit external source nodes;
`build_program_index(bundle, external_nodes=())` was built with that comment
and no caller has ever passed `external_nodes`. On U3's third day all 15
diffusion roots failed model-stage resolution with `mro_incomplete` because
their bases live outside the bundle — the symptom was typed as a failure and
routed around, not escalated.

The legacy `_augment_diffusion_files` was a *correct* exact import-follower
(AST, bounded, called-names only). It was labelled a "whole-file union" and
scheduled for deletion instead of being promoted into the substrate. U11-A1
then rebuilt the same capability (`evidence/import_source.py`, general,
demand-driven) — for UNet readers only, one week after U10 shipped the
regression. Import closure belonged in U3 as substrate; it arrived as a U11
side-effect and is still not applied to the DiT path.

### 2.2 Soundness is ratcheted; recall is not — and the plan licensed the loss

Master plan §20.7 (U4) step 6 reads: *"If a known-good model becomes unknown,
fix its evidence reader in the matching U6–U13 slice; do not restore the global
default."* The §7 lost-detail procedure, written the same day (2026-07-28),
says the opposite: never call the generic output honest until re-proof has
been attempted. The plan's sentence won in practice — U4 dropped the final
norm on all 26 witnesses, Gemma-2's activation and GPT-OSS's clamp (silently:
the one net that noticed, `config_accessed_unprojected`, is still
`blocking=False`), U8/U9/U10 repeated the shape, and the two texts were never
reconciled.


Every gate asks "did we claim something false?" None asks "did we lose
something we had proven?" A conformance-verified view turning into
`unresolved` is invisible to every net, so it surfaces only as one line in a
28-gallery approval batch. The lost-detail rule exists as prose (§7 of the U3
procedure audit, written after the DeepSeek/GLM episode) and has now been
violated by three consecutive units. Prose rules that are not gates do not
survive contact with a 60-minute exhaustive lane.

### 2.3 What the experiments added (see `experimental-confirmation.md`)

- Soundness is not perfect: DeepSeek-V3's score scale is a **wrong**
  `code_proven` fact (yarn `mscale²` ignored); GPT-2/MPT out-of-corpus cells
  are drawn without residual connections; T5 standalone loses its encoder
  silently.
- Proven facts are sometimes not drawn (Qwen2-VL text FFN, MusicGen's T5
  tower) and provable negatives are drawn as unknown (QK-norm chips on seven
  decoders, DBRX LayerNorm).
- The DiT collapse has two causes — the bundle boundary *and* a by-design
  refusal of own-file `source_mixin_delegate` attention lanes — and closure
  alone restores skeletons, not cells.
- Measured distance from intent: transformer core broad and correct bar one
  fact; out-of-corpus transformers **75 %** of provable facts proven; diffusion
  denoisers **33 %**; PixArt/SD3.5 **0 %**.
- The single highest-leverage general fix is upstream of all of this:
  **prune guarded / config-selected rival invocations with the already-consumed
  config value before readers run** — it zeroes T5 and empties GPT-2, Falcon,
  MPT, Phi-3, Mistral and Mixtral drills today.

## 3. Cost profile

| measure | value |
|---|---|
| `evidence/` size | 14.3k lines / 25 modules (pre-U0) → 92.6k / 142 (`HEAD`), 6.5× |
| net production growth in U10 alone | +11.6k lines (against the finish-line rule "sustained growth without deletion is a stop signal") |
| execution docs | 20.5k lines in `unfold-pkg/docs/` + 9.4k in `z-docs/`; per-slice prose records now exceed the code they describe in several units |
| committed-tree receipt | ~87 min wall (full lane 63, focused 24, preservation 18, authority 12); ~8–10 per unit |
| `unfold()` latency at `HEAD` | cold: Llama-7B 29–51 s, PixArt 65 s, Qwen2-VL 101 s, DeepSeek-V3 12 s (source resolution + index build); warm: 0.6 s (in-process cache only — nothing is persisted across processes, so every CLI/notebook run pays the cold cost) |
| witness corpus | 29 models; diffusion inner mechanisms unresolved on 15/15; coverage denominator against the declared support set is not measured anywhere |

The ratchet infrastructure earns its cost. The narrative documentation and
the static-only method's latency do not.

## 4. The blind spot in the method

Static-only analysis was chosen (product contract: "model code is not
executed"). The master plan §22.5 already permits an opt-in runtime probe;
nobody built it. Instantiating a model on the `meta` device — no weights, no
forward pass, CPU, milliseconds — yields the exact module tree, parameter
shapes and which guarded branches were constructed: precisely the *occurrence
ownership* the U3 substrate spends ~10k lines reconstructing statically. Used
not as authority but as a corroboration net ("static owner graph must agree
with the instantiated module tree"), it would have caught PixArt instantly
(the instantiated model contains 28 `BasicTransformerBlock`s; the static index
said "unresolved"), and it resolves the DBRX checkpoint-shape ambiguity that
static analysis provably cannot. Remote-code models stay opt-in.

## 5. How it would be done from here

0. **Prune guarded / config-selected rival invocations before readers run**
   (using the consumed config value: `add_cross_attention`, `_attn_implementation`,
   `is_decoder`, `sliding_window is None`, per-layer list guards). Measured as
   the largest single gap in `experimental-confirmation.md` §3; general, no
   per-model code.
1. **Promote demand-driven import closure into the substrate** — *and* lift
   the `source_mixin_delegate` refusal in `diffusion_block.py:299`, pass
   `canonical_called_import_target` into the attention-lane census, and let a
   cell project every resolved sub-fact independently (partial-cell); make every
   reader use the closure (the code exists). Re-run the U10 readers on the 15 diffusion
   witnesses; expect PixArt/SD3.5/Mochi/Wan skeletons, ×N counts and most
   inner kinds to return with no doctrine change. Do this **before** U11-A2
   deletes `_augment_diffusion_files` and before U11 goes further — U11 is
   building on the same boundary.
2. **Add a blocking recall net**: any fact or view that was
   conformance-verified in the blessed baseline may not become `unresolved`
   without a named re-proof record. This is §7 in mechanical form.
3. **Add the meta-instantiation corroboration net** (opt-in, corpus-only, no
   remote code): static owner graph ≡ instantiated module tree.
4. **Restore the deferred provable detail with owners**: Qwen3.5 mask and
   partial-RoPE legends and the T5 loop-carried relative bias (U8), the
   Qwen2-VL vision cell (U9 — two general fixes: attention-interface calls
   inside `if`/comprehension/`torch.cat`, and a *partial-cell* projection
   that never discards a resolved FFN/norm because a sibling fact failed),
   HunyuanVideo/Lumina refiners and AuraFlow's learned position (U10), the
   gpt-oss `swiglu_limit` chip (U4).
5. **Collapse the documentation planes**: one tracker table, machine
   receipts (`receipt.json` already exists), `z-docs` as the only prose;
   retire per-slice narratives.
6. **Publish the coverage denominator**: proven-fact ratio per official
   architecture class in the supported Transformers/Diffusers versions, as the
   headline metric instead of test counts.
7. Then resume U11–U15.
