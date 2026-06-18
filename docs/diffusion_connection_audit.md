# Diffusion connection-fidelity audit (Sable · code-truth · auto-depth)

*Same method as `docs/llm_connection_audit.md`: read the real `diffusers` source
(installed 0.38.0), recurse through every drill to the description-only leaves, and
conform each level's **connections** (arrows / ports / merges / splits) against the
block `forward()`, both directions.*

## Verdict at a glance

Diffusion fidelity is **uneven by denoiser family**:

| Level | Family | Fidelity |
|---|---|---|
| **Pipeline** (noise → t → prompt → encoders → latent → denoiser → scheduler → VAE → image) + the `until t=0` denoise loop | all | ✅ exact |
| **Text encoders** drill (`embed → self-attn → ffn → norm → add`) | all | ✅ (transformer encoder layer) |
| **VAE decoder block** (`norm1 → conv1 → norm2 → conv2 → ⊕ residual [+upsample]`) | all | ⚠ minor — missing the **nonlinearity** nodes (SiLU between norm/conv) and upsample is drawn after conv vs before; otherwise matches `ResnetBlock2D` |
| **UNet denoiser** (`conv_in → down/mid/up stages → ResNet(norm→conv→temb→norm→conv→⊕) + Transformer(self + cross attn)`) | SDXL, SD1.5, Kandinsky | ✅ **structural** — drills the full UNet, matches `UNet2DConditionModel` / `ResnetBlock2D` (incl. temb injection) |
| **DiT / MMDiT denoiser block** | FLUX, SD3/3.5, PixArt, Wan, HunyuanVideo, Sana, Lumina, AuraFlow, CogView | ⚠ **annotation-rich, structurally simplified** — see below |

## The DiT/MMDiT block gap (the real finding)

Our DiT block renders as a generic transformer layer — `norm → attn → ⊕ → norm → ffn → ⊕`
plus `adaln_cond` / `text_cond` side blocks — with a **rich attention `variant`
annotation** (e.g. FLUX: *"Joint Attention — MM-DiT (dual-stream)… two streams keep
separate Q/K/V and MLPs; only the attention is joint; modulated by the timestep via
AdaLN"*). The structure the `forward()` actually wires is **not drawn**:

Conformed against `FluxTransformerBlock` / `FluxSingleTransformerBlock` (transformer_flux.py)
and `JointTransformerBlock` (attention.py):

1. **Dual stream not structural.** Real double-stream blocks run **two parallel streams**
   (image `hidden_states` + text `encoder_hidden_states`), each with its own AdaLN, FFN,
   and gated residuals, meeting only at the **joint attention** (`attn(hidden, encoder)
   → (attn_out, context_attn_out)`). We draw one stream + a description. *(code → structure:
   the second stream, its norms/FFN/residuals, and the two-into-one attention merge are
   missing.)*
2. **Single vs double stream IS distinguished, but single-stream wiring is approximate.**
   FLUX's 19 double-stream layers render sequential (`norm→attn→⊕→norm→ffn→⊕`) and the 38
   single-stream layers render parallel (`norm → attn ∥ ffn → ⊕`) — so the split is correct.
   But the real single-stream `forward` is `norm → (attention ∥ MLP) → **concat** → gated
   **proj_out** → ⊕` (one shared output projection over the concatenated attn+mlp), which we
   draw as two separate paths into a plain combined add — close, but the concat + shared
   proj_out (and its AdaLN gate) are not drawn.
3. **AdaLN modulation gates not wired.** Real residuals are **gated**: `h = h + gate_msa *
   attn`, `h = h + gate_mlp * ff`, with `norm2` modulated `*(1+scale)+shift` — all from the
   timestep (`temb`). We draw plain residual `⊕`s and an `adaln_cond` side leaf; the gate ×
   on each residual and the shift/scale on the norm are described, not connected.
4. **Cross-attention DiTs** (PixArt/HunyuanDiT): text enters via a **cross-attention**
   sub-block (`attn2` to encoder states) + AdaLN-single; we render generic self-attention.

None of this is *wrong wiring* of what we draw — it's **missing structure**: the MMDiT
block is the one place the diffusion side is at "v1 annotation" depth while the UNet side
is structural. (This matches the standing diffusion-adapter scope note.)

## Recursive leaf conformance (where we DO drill)

Every drill that bottoms out at leaves reconciles with source:
- **UNet ResNet** leaves `norm1 → conv1 → temb → norm2 → conv2 → ⊕` ✅ (`ResnetBlock2D`, default time-embedding-norm path).
- **UNet Transformer** drills to self-attention `q/k/v → scores → softmax → ⊙ → concat → o_proj` ✅ (+ cross-attention to text — present).
- **VAE decoder** leaves `norm1 → conv1 → norm2 → conv2 → ⊕ [+upsample]` ⚠ (nonlinearity nodes absent).
- **Text-encoder** leaves `embed → self-attn(drills) → ffn → norm → add` ✅.
- **DiT attention** drills to `q/k/v → (RoPE where axial) → scores → softmax → ⊙ → concat → o_proj` ✅ at the SDPA level — but the *joint/dual-stream* wiring above it is annotation-only.

## Per-family block conformance (read every major family's `forward()` — the "no missing out" pass)

Reading each denoiser block's `forward` in diffusers 0.38 (the sandwich-Gemma lesson),
the families cluster into **three topologies**, and two carry structural surprises:

| Family | Block `forward` topology | We draw | Gap |
|---|---|---|---|
| **FLUX-double, SD3/3.5, AuraFlow, HunyuanVideo** | **MM-DiT dual-stream**: `norm1`(img) ∥ `norm1_context`(txt) → **joint `attn(img,txt)→(out,ctx)`** → gated residuals → per-stream FFN (`ff` / `ff_context`) | one stream + "dual-stream" label + AdaLN `×` gates (new) | ❌ second (text) stream — its norm/FFN/gated residuals — not drawn |
| **Sana, PixArt, Wan, CogVideoX, Mochi, LTX, Allegro, HunyuanDiT, Lumina** | **Cross-attention DiT**: `norm1→self-attn→×gate→⊕` → `norm2→**cross-attn(attn2 to text K/V)**→⊕` → `norm3→FFN→×gate→⊕` — **three** sublayers | **two** sublayers (self-attn + FFN) + a `text_cond` side rail | ❌ **the whole cross-attention sublayer is missing** — the largest diffusion class, incl. most video DiTs |
| **FLUX-single** | parallel `attn ∥ MLP → concat → gated proj_out → ⊕` | parallel `attn ∥ ffn → ⊕` | ⚠ concat + shared gated proj_out not drawn |

Universal across **every** DiT family: each sublayer residual is **AdaLN-gated**
(`h = h + gate · sublayer(modulated norm)`), confirming the AdaLN `×` is the right
universal connector. Family-specific notes: Lumina gates cross-attn with a `tanh` gate;
Sana/Wan/PixArt drive AdaLN from a shared `scale_shift_table` (AdaLN-single) vs per-block.

**Two surprises ranked by breadth:**
1. **Missing cross-attention sublayer** — affects the *largest* class (PixArt + Sana + the
   video DiTs Wan/CogVideoX/Mochi/LTX/Allegro + Hunyuan-DiT + Lumina). These blocks have a
   distinct `attn2` cross-attention to the encoded text, with its own residual, that we fold
   away into a single self-attention + side rail.
2. **Representative collapse** — the architecture view shows **one** dominant block, so for
   mixed stacks (FLUX 19+38, AuraFlow 4+32) only the single-stream block renders; the
   double-stream block (and its AdaLN gates) isn't shown. Both block types should render
   (the LLM side already shows "N block types").

## Progress

- ✅ **Cross-attention sublayer (DONE).** Cross-attention DiTs (PixArt · Sana · Wan ·
  CogVideoX · Mochi · LTX · Allegro · Hunyuan-DiT · Lumina) now render the real **three
  sublayers** — `norm → self-attn → ×gate → ⊕` · `norm → cross-attn(to text) → ⊕` · `norm
  → FFN → ×gate → ⊕` — with the cross-attention as its own clickable, solid block (its own
  drill `view:"cross_attention"`: image Q ∥ text K/V), its own residual, and the text
  conditioning re-pointed to it. Gates sit on self-attn + FFN only (cross-attn is a plain
  add — matches `SanaTransformerBlock`/`PixArt`/`WanTransformerBlock`). Rendered + pinned
  (`test_cross_attn_dit_has_three_sublayers_and_adaln_gates`).
- ✅ **AdaLN `×` gates** on the double-stream / cross-attn DiT residuals (`gate_msa`,
  `gate_mlp`) — Tier-2 connectors driven by the timestep (the universal DiT conditioning
  mechanism), per BLOCK_STANDARD §5.4.
- ✅ **Standard** §5.4 — dual-stream / cross-attn / single-stream / AdaLN / VAE rules.

### Still to land (in progress)
- **Show both block types** (so MM-DiT's double-stream block + its AdaLN gates render, not
  only the dominant single-stream representative).
- **Dual-stream two-column** (image ∥ text sharing one joint attention) + **single-stream
  concat → gated proj_out**.

## Fix shape (when you want it) — bring the DiT block to UNet-level structural depth

Additive, mirroring how the UNet denoiser already drills:

1. **A real MMDiT block view** (`view:"mmdit_block"`): two lanes (image · text) sharing one
   **joint attention** node (two-in / two-out), each lane with its own AdaLN-modulated FFN
   and **gated** (`×`) residuals. Reuse the Tier-2 `gate_mul` glyph for the AdaLN gates and
   the existing parallel/lane engine.
2. **A single-stream block view** for FLUX/SD3 single blocks: `norm → (attention ∥ MLP) →
   concat → ×gate → proj_out → ⊕`. Distinguish the two by layer index (config gives the
   double/single split) so the `× N` frame shows both regions.
3. **Wire AdaLN**: the `adaln_cond` block feeds `×` gates on the attention/FFN residual
   joins and shift/scale on the pre-FFN norm (Tier-3 → a real connection from temb).
4. **Cross-attention DiT** (PixArt/Hunyuan): add the `cross_attention` sub-block to the
   text encoder states (the transformer adapter already has `cross_attention` + the
   `encoded text` side source — reuse it).
5. **VAE**: add the `nonlinearity` (SiLU) node between norm and conv; move upsample ahead
   of conv1 to match `ResnetBlock2D`.

UNet (the most-served legacy diffusion) is already structural; this brings the DiT/MMDiT
frontier to the same bar.
