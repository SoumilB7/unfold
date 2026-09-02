# Baseline generation audit — HEAD `d778d46`, 2026-09-02

Purpose (Soumil, after S1): "confirm we are actually ok with what we achieve,
and that where we are is generating, in all possible model cases, what it's
supposed to." This directory is the measured answer at the tree **before**
S2–S4 change anything, so every later step has a denominator to be compared
against. It is the ancestor of `coverage.json` (S4) and of the S7 disagreement
matrix.

## Method (script: `gen_audit.py`, copy in `../../10-full-research/` at S6)

For each model, in a fresh process with `HF_HUB_OFFLINE=1`:

1. **Product parse** through the corpus route (`_coerce → ParseContext.build →
   config_to_ir → Diagram.to_ir`), the same path `sable()` audits. From the IR:
   - `num_layers`, the **distinct layer groups** by a 20-slot fact signature
     (attention kind / mixer / heads / kv / head_dim / rope / position / QK-norm
     / mask / scaling; FFN kind / activation / gated / width / experts /
     routing; norm kind / placement / residual topology);
   - the **unknown rate**: how many of those slots are `None` / `unknown` /
     `unresolved` / `ambiguous` across all layers;
   - opaque blocks, unresolved model blocks, loop blocks, cross-layer edges,
     warnings, tying, hidden size, the layer-0 drawn block kinds.
2. **Sable** (`render_images=False`): oracle state, distinct view count and
   labels, every check with findings (blocking or not), passing checks.
3. **Meta-instance inventory** (torch meta device, no weights): class, params,
   module count, every top-level `ModuleList` stack with its element class and
   distinct child signatures, shared-parameter groups. Typed failure if the
   library cannot build it.
4. **First shadow comparison**: IR layer count vs the instance's main stack;
   IR distinct groups vs instance distinct signatures; tying vs parameter
   identity.

Per-model JSON in `models/`; the roll-up table and the reading of it in
`BASELINE_GENERATION_AUDIT.md` (written when the batches finish).

## Sets

- **A** — 14 corpus transformer witnesses (blessed).
- **B** — 15 corpus diffusion witnesses (blessed).
- **C** — 15 out-of-corpus models from the local HF cache, chosen for
  difficulty: Qwen3.5-27B full (multimodal), Qwen3.6-35B-A3B (hybrid MoE),
  MiniMax-M2, LFM2, Jamba, GLM-4-9B, DeepSeek-Coder-V2-Lite (MLA+MoE),
  SANA 1.5, HunyuanDiT, Qwen3-Omni, Qwen3-VL-235B, Command-A, GPT-NeoX-20B,
  Seed-OSS-36B, SD v1.4 (UNet).

## What "ok" means here (the contract, `14` §B B1)

- Tier 1: every structural element and relation present, or a **visible**
  unresolved marker; a silent omission is the failure we are looking for.
- Tier 2: every mechanism slot either proven or visibly unknown; the unknown
  rate is allowed to be non-zero, silence is not.
- Tier 3: values exact and sourced (not measured here beyond presence).

The audit cannot see pixels; it sees the IR the pixels are projected from and
the product's own nets. Pixel-level truth remains the fleet's job (Her Eyes).
