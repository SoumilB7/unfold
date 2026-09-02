# CONFIG ABLATION CENSUS — "eradicate the config and see what actually works"
*(2026-07-11 · Soumil's experiment, run post-Unit-1. Method: each witness parsed
in four quadrants — A full (config+code) · B code-blocked (config-only) ·
C config→numbers+address-only (code present) · D BOTH removed (no evidence at
all). Normalized-structure diffs via the identity-guard machinery. Witnesses:
llama-7b, gemma-2-2b, phi-2, internlm-7b (remote-code), t5-base, Mixtral.
Script in the appendix; raw table in the session log.)*

## THE ONE-LINE VERDICT
**Position is the ONLY fact family with tri-state honesty discipline.** When
every channel of evidence is removed (quadrant D), `position_kind` alone falls
to honest "unknown" — while **mask, gated, activation, norm_kind,
norm_placement, projection_mode, bias, window, routing and tie_word_embeddings
all silently assert a default value.** That default layer — not config-reading
itself — is the rot the run_77 campaign kept hitting.

## WHAT THE QUADRANTS PROVED
1. **The code rail is real and load-bearing (A vs B).** Blocking code with the
   config intact strips 200–965 structure paths per model: position
   application, projection/storage modes, Mixtral's entire router+expert
   complex (965 paths), norm placement (gemma sandwich), parallel residual
   (phi). internlm's RoPE box is code-proven — same config, code blocked →
   box gone, honest-unknown banner returns. Unit 1 made this rail reach
   remote-code repos.
2. **The remaining config dependencies are mostly DOCTRINE-LEGIT value
   channels (A vs C).** Activation choice (`ACT2FN[config.hidden_act]` — the
   code proves the dispatch, config supplies which), bias flags
   (`Linear(bias=config.attention_bias)`), window sizes, tie flag, per-layer
   schedules (gemma's alternation). These are checkpoint VALUES feeding
   code-proven expressions — keep them; they are not the enemy.
3. **The enemy is quadrant D — defaults asserted at zero evidence:**
   - `mask="causal"` asserted for EVERYTHING incl. T5's bidirectional encoder.
   - `gated=True` asserted (phi-2's truth is False — only the landed
     code-reader saves it in A; remove evidence and the fabrication returns).
   - `activation=silu`, `norm_placement="pre"` asserted.
   - `norm_kind` asserted BOTH ways from the eps-SPELLING channel: t5 D-parse
     says layernorm (truth rmsnorm), PhiMoE/Persimmon-class says rmsnorm
     (truth layernorm) — a spelling, not evidence.
   - `tie_word_embeddings=False` asserted when the flag is absent (gemma/t5
     truth is True).
   - `projection_mode`/`bias` partially null out instead of declaring unknown.
4. **Fabrication cascades exist:** t5's `gated` flips with the NORM evidence
   (rmsnorm⇒gated heuristic chaining off norm_kind) — one asserted default
   feeding another reader's heuristic. Kill the default layer and the cascade
   dies with it.

## WHAT THIS SETS AS THE U2 ACCEPTANCE METRIC
Extend position's tri-state to EVERY structural fact family (the B5 provenance
enum: code_proven / config_declared / class_default / asserted / unknown), and
render unknowns pale-honest instead of asserting. **The D-quadrant parse is the
permanent measuring stick: a no-evidence parse must contain ZERO asserted
structural facts — today it contains ~10 families per model; target 0.** This
census script becomes the net.

## APPENDIX — the harness (reusable)
Quadrants: A `config_to_ir(cfg)` · B patch `evidence.context.resolve_source_files`
→ empty bundle (consumer namespace, not sources — the from-import trap) ·
C `numbers_only(cfg)` = address keys verbatim (`model_type, architectures,
_repo_id, _name_or_path, auto_map`) + int/float fields only (strings, bools,
dicts, lists stripped — bools are structural signals, numbers are checkpoint
values) · D = B∘C. Facts compared: mask, gated, activation, norm_kind,
norm_placement, position_kind, projection_mode, bias, scores_scaled,
window_size, parallel_residual, routing, tie_word_embeddings, via
`identity_guard._normalized_structure` + fact-value collection. Verdicts:
value identical in A and D = DEFAULT-AS-FACT; D=unknown/absent =
evidence-backed; changed = evidence-driven (classify which channel by B vs C).
