# Experimental confirmation — rendering the witnesses to test the audit

Five read-only experiments (2026-08-31, tree `a00ae48`, transformers 5.12.1,
diffusers 0.38.0). Every claim in this chapter was re-tested by rendering the
models with the production `sable()`/`unfold()` into scratch directories,
reading the produced PNGs/IR/expanded JSON, and checking them against the
installed Hugging Face source. No blessed artifact, manifest or repository
file was touched. Renders of blessed witnesses reproduced the blessed PNGs
byte-for-byte, so findings apply to the blessed galleries too.

| experiment | scope | witnesses rendered |
|---|---|---|
| A | diffusion denoisers + import-closure probe | 15 diffusion witnesses |
| B | transformer / multimodal + Qwen2-VL / Qwen3.5 root-cause probes | 14 witnesses |
| C | embedded encoders, T5 bias, MTP; historical trees `74c50ec^`, `e758be4`, `37f3b1b` | 10 witnesses × 3 trees |
| D | out-of-corpus generalisation | 13 models not in the corpus |
| E | gates, recall blindness, closure API, receipts, debt register | — |

## 1. Claim verdicts

| audit claim | verdict | what the experiment showed |
|---|---|---|
| PixArt / SD3.5 denoisers are blank boxes while the source specifies ×28 / ×38 blocks | **CONFIRMED** | `layers: []`, no ×N, `denoiser_structure_unresolved`; both render the *same* PNG; all 25 Sable nets green on the blank box (A) |
| cause = block classes imported from Diffusers' shared files, outside the one-file bundle | **CONFIRMED, and incomplete** | bundle = exactly one denoiser file (`sources.py:529-613`); `resolve_called_import_source` resolves `BasicTransformerBlock` / `JointTransformerBlock` in one hop in scratch; **the identical `BasicTransformerBlock` PixArt leaves blank is fully drawn inside SDXL** through the UNet readers' closure (A, E). But closure alone is not enough: SD3.5 gets its ×38 skeleton back yet cells stay `materialization_blocked` (joint dual-state lane ≠ "primary-plus-cross"; Diffusers `FeedForward.net` is a heterogeneous `ModuleList`; the merged-index `Attention` needs `canonical_import` passed); PixArt additionally needs the config-guard selector for three guarded `transformer_blocks` init branches (`rival_container_records`) (A) |
| per-layer attention kind unresolved on every DiT because head geometry lives in the external container | **PARTIAL — second cause found** | true for CogVideoX/Qwen-Image/Hunyuan/Lumina/AuraFlow (external `Attention`; geometry *does* reach the IR when called with `heads=`/`dim_head=` names). **FLUX.1/FLUX.2/Wan/LTX/PRX have their attention class in their own file** and fail because `diffusion_block.py:299-310` refuses every `source_mixin_delegate` lane by design (A) |
| transformer decoder core fully resolved | **CONFIRMED, with caveats** | 12/12 decoders: typed attention kind on every layer, zero `wiring_unresolved`, all nine geometry spot-checks CORRECT (Gemma-2 sandwich/softcap, GPT-OSS sinks/formula, DBRX 48/8/128, Granite /128, MLA widths, BLOOM 128, StableLM rot 16/pass 48). Caveats: DBRX `norm_kind unknown` ×40 for a plain `nn.LayerNorm`; "QK norm unresolved" chips on 7/12 decoders whose source has no q/k norm (B) |
| Qwen2-VL vision cell lost to a guarded-comprehension attention call; readers had resolved FFN and norm | **CONFIRMED, with a second blocker** | vision tower `failed` only on the attention family; FFN/norm/boundary ops resolved. Cause 1: `attention_child.py:479,498` skip guarded calls. Rewriting the guard in scratch restores cell topology; then cause 2: `attention_storage` rejects the fused `qkv → reshape/permute/unbind` unpack (B) |
| Qwen3.5 mask and partial RoPE provable but unresolved | **CONFIRMED, root causes corrected** | the ternary mask selector *is* consumed; the schedule fails because `hidden_states` is reassigned in the sibling `linear_attn(..., attention_mask=…)` branch, so two actuals carry the mask token (branch-insensitive tainting, `attention_mask.py:1427-1455`). Position fails because `coordinate_origin` rejects the `position_ids[1:]` slice under a guard (`position_coordinate.py:69-133`); the half-turn itself resolves (B) |
| DeepSeek MLA cache unresolved | **CONFIRMED** | `cached: None` ×61 despite `past_key_values.update` at `modeling_deepseek_v3.py:455` (B) |
| U6 collapsed CLIP/UMT5/T5 encoder + MusicGen attention drills onto the stub; some still missing | **CONFIRMED, worse than stated** | reproduced on the `74c50ec^` and `e758be4` trees: pre-U6 all distinct full drills; U6 tree `attn == encoder_*_op_selfattn` on SD3.5/AuraFlow/Wan/MusicGen, SDXL CLIP-G == five `unet_*__selfattn`. **The audit's "SD3.5 `encoder_0` restored in U10" is wrong**: at HEAD it is byte-identical to the U6 stub — only the *name* reappeared when U10 dropped the denoiser `attn` view. None of the collapsed drills has been re-proven (C) |
| U8's T5 `g1` layers drawn without the relative bias and without a chip | **CONFIRMED, mechanism located** | `modeling_t5.py:612,718,730,745,297-315`; parser types `position_kind='unknown'` for index > 0 (`parser.py:1764-1771`); `opgraph.py:834-836` adds the bias lane only for `alibi|relative_bias` and **emits no chip for `unknown`**; the `g0` card even says "computed once by the first layer and shared down the stack". Six witnesses, byte-identical `g1` PNGs (C) |
| MTP removal was a genuine correction | **CONFIRMED** | no `nextn`/MTP construction in `modeling_deepseek_v3.py` or `modeling_glm4_moe.py`; the key exists only in the hub config (C) |
| no recall gate | **REFINED** | the corpus tests *do* go red on `view drift: 4 → 4` / `hash mismatch`, but every message is undirected — identical for a loss and an improvement — and `bless` clears it recording only a superseded hash. A hash-identical render with a lost code oracle passes entirely (E) |
| hash-set collapse blindness | **CONFIRMED** | fixture stores a sorted set of hashes with no labels; two drills merging = count −1, nothing named (E) |
| `config_accessed_unprojected` advisory | **CONFIRMED** | `sable.py:551-555` `blocking=False`; Granite and SDXL fixtures fail it today (E) |
| `external_nodes` never passed; DiT readers never call the closure | **CONFIRMED** | production callers of `resolve_called_import_source`: `unet_nested_mechanism`, `unet_stage_construction`, `selected_composite_ffn` only (E) |
| receipts unreproducible | **CONFIRMED** | 9 of 10 cited receipt directories absent; only U10's exists (E) |
| orphaned owners | **PARTIAL** | U9 (DONE) still owns 19 rows, U5 (DONE) 2; DBRX theta mis-scoped to U11; "U3-G"/"R3" exist only in prose, never in the register (its validator forbids them) (E) |

## 2. Findings the audit did not have

**Soundness is not perfect either.**
- **WRONG `code_proven` fact**: DeepSeek-V3's score scale is drawn `QK^T/√192` and marked code-proven; `modeling_deepseek_v3.py:409-415` multiplies `self.scaling` by yarn `mscale²` (≈1.87/√192 for the fixture's `factor: 40`). The first drawn contradiction of source found in the corpus; every gate passed (B).
- **Proven but undrawn** (fact→projection losses): Qwen2-VL's text FFN has `gated=true / split / silu / 18944` all `code_proven` in the ledger yet renders "Feed-forward mechanism unresolved" — and the blessed `her_eyes_review.md` approves that view as "Textbook SwiGLU" while the blessed PNG is the placeholder (B). MusicGen's T5 conditioning tower is fully proven in the IR (12×`T5Block`, MHA 12/12/64, RMSNorm, relative bias) and has no view and no chip (C).
- **Unknown over-applied to provable negatives**: "QK norm unresolved" on Llama/Gemma-2/BLOOM/DBRX/GPT-OSS/Granite (no q/k norm exists); DBRX LayerNorm unknown; DeepSeek shared-expert width and GLM dense width unknown though computable (B).
- **Silent absences**: GLM-4.5 partial RoPE drawn as full RoPE (StableLM's identical mechanism renders correctly); Gemma-2 `embed_scale`, Granite `embedding_multiplier`/`logits_scaling`; DeepSeek `q_a_layernorm`/`kv_a_layernorm` missing from the MLA drills; CLIP `position_embedding` no longer drawn; Qwen3.5/Qwen2-VL attention drills draw no positional op and no in-drill chip (B, C).
- **Live gate hole in diffusion**: the blanket `_DIFFUSION_SOURCE_CLOSURE_PATHS` excusal (`structural_debt.py:762-785`) already lets `config_field_audit`, `config_standing_unconsumed` and every conformance net pass on a blank denoiser — the "passes every net forever" case is live today, not after U11-A2 (A).
- **Three CLIP towers stubbed by a factory gap**: `CLIPTextModel` towers resolve; every `CLIPTextModelWithProjection` tower stubs because the attention reader does not walk the `CLIPTextModel._from_config` nesting that the FFN reader does (C). UMT5 attention drops entirely because its per-layer bias is built inside the layer, so the relative-bias reader finds no `i == 0` expression and the whole mechanism fails with it (C).
- **Two-FFN blocks** (AuraFlow joint, FLUX.2, Qwen-Image img/txt) fall to "mechanism unresolved" because the projection demands uniqueness instead of proving both lanes share one mechanism; `GELU(approximate="tanh")` is labelled plain GELU (A).
- **Harness gaps**: preservation ignores a fixture's `source` field; `check_regression` never reports a lost oracle; the register never checks that a row's owner unit is still open (E).

## 3. Generalisation — 13 models outside the corpus (D)

No crashes or timeouts; **no fabricated mechanism found anywhere**. Where the syntax matches a corpus witness the result is essentially correct (LLaVA's Llama tower, Gemma-3's per-layer mask schedule with two templates, Mistral's GQA/RoPE/FFN). Genuinely new mechanisms (Chroma's guidance approximator, CogView4's partial-sequence RoPE, Qwen2.5-VL vision windows) stay visibly unknown — that half of the intent holds. The failure mode is that **known mechanisms written in a different surface syntax degrade to unknown, and one unresolved rival call zeroes an entire block**:

| gap | models hit | effect |
|---|---|---|
| rival/guarded invocation of the same child collapses every fact to `ambiguous` (GPT-2's guarded `crossattention` with `add_cross_attention=False`; T5's encoder+decoder invocations of one stack class; Falcon's `FALCON_ATTENTION_CLASSES[...]`; MPT/Phi-3 lanes) | 5 | T5 → **0/12** facts |
| mask/rope builder reached through an alias or per-layer guard (`mask_function = … if … else …`; `if self.use_rope`) | 4 | mask/position unresolved |
| QK-norm absence never proven | 7 | chip on every plain model |
| `getattr(cfg,"head_dim",None) or …` | 2 | `?` widths, blocking audit |
| `Conv1D` / fused-QKV slice unpack | 3 | empty attention drill |
| shared-file imports outside the bundle | 2 (Chroma, CogView4) | blank cells |
| residual add inside a child / helper geometry | 2 | GPT-2 and MPT drawn **without residual connections** |

Wrong-ish outputs: T5's encoder and cross-attention silently absent (a 2-layer decoder-only stack, no warning); GPT-2/MPT cells drawn as a straight chain; bare diffusion transformers draw VAE/scheduler/text-encoder scaffolds that are not in the config.

**Proven / provable typed-fact rows: 162/216 ≈ 75 % for the transformer adapter; ≈ 0 % for the two out-of-corpus diffusion transformers' repeated cells.**

## 4. Distance from intent, measured

| domain | proven / provable | worst case |
|---|---|---|
| transformer decoder core (12 corpus witnesses) | broad; geometry correct; one wrong proven fact | DeepSeek yarn scale |
| out-of-corpus transformers (11) | 162/216 ≈ 75 % | T5 0/12 |
| multimodal towers (Qwen2-VL, MusicGen) | far: vision cell opaque, text FFN lost despite proven facts, T5 tower proven-undrawn | — |
| embedded text encoders in diffusion pipelines | T5 layer 0 and `CLIPTextModel` towers proven; T5 layers 1–23 proven-with-fabricated-absence; `WithProjection` CLIP ×3, UMT5 ×2, MusicGen T5 provable-but-stubbed | — |
| diffusion denoisers (15) | **37/111 ≈ 33 %** (30 % excluding SDXL's UNet path) | PixArt 0/9, SD3.5 0/7 |

## 5. Ranked fixes, by measured leverage

1. **Prune guarded / config-selected rival invocations using the already-consumed config value** (`add_cross_attention=False`, `_attn_implementation`, `is_decoder`, `sliding_window is None` → alias-bound builders) before readers run — root of the two largest gaps; 7 of 11 out-of-corpus transformers; ≈45 ambiguous rows (D).
2. **Wire the existing import closure into `diffusion_stack` / `diffusion_block` and pass `canonical_called_import_target` into the attention-lane census**, *and* lift the `source_mixin_delegate` refusal in `diffusion_block.py:299` — the two halves of the DiT collapse (A).
3. **Render a chip for `position_kind == 'unknown'` in `opgraph.py:834`** — turns six silent negatives into honest unknowns in one render-side change (C); likewise chips for any unresolved fact currently drawn as absence.
4. **Never withhold a resolved sub-fact because a sibling failed** (partial-cell projection): restores Qwen2-VL's FFN/norm/residual and MusicGen's T5 tower immediately (B, C).
5. **Attention-child protocol**: accept mutually exclusive dispatch branches and comprehension call sites as one compute protocol; accept the single-producer reshape/permute/unbind Q/K/V unpack; add `Conv1D` and slice-unpack to the affine/storage protocol tables (B, D).
6. **Score-scaling reader consumes the `self.scaling` assignment chain** including conditional multiplications — fixes the one wrong proven fact (B).
7. **Prove negatives**: QK-norm absence, plain LayerNorm, computable widths — stop rendering provable negatives as unknown (B, D).
8. A directed recall net (proven → unresolved must cite a re-proof record), labels in the fixture signature, `check_regression` reporting a lost oracle, and unit-liveness in the debt register (E).

## 6. Effect on the verdicts

No grade changes. The experiments strengthened every unit finding they touched and added one soundness defect (DeepSeek scale) to the transformer core that the audit had rated clean; the U6 loss is larger than stated (nothing collapsed in U6 has come back); the U10 root cause is two causes, and the recommended fix is correspondingly two changes plus two small projection generalisations. Scratch artefacts: `scratchpad/exp{A,B,C,D,E}/`.
