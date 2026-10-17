# Pod E — Skeptical-user / source-truth audit of `model_unfolder`

Environment: `unfold-pkg` working tree, transformers 5.12.1, diffusers 0.38.0. HF hub network was reachable (no token). Outputs under `.../scratchpad/pods/E-output/`.

## Per-model verdict table

| Model | Input form | Result | Runtime | HTML/JSON |
|---|---|---|---|---|
| Llama-3-8B | raw dict (real `config.json`) | OK, but 1 missing computation | 16.1s (cold) | 62.9KB/20.7KB |
| Qwen3-8B | raw dict | OK, but 1 missing computation | 1.8s | 68.0KB/24.3KB |
| DeepSeek-V3 | raw dict (**real, current** HF `config.json`) | **CRASH** (unhandled `TypeError`) | 4.5s to crash | — |
| DeepSeek-V3 | hub id `"deepseek-ai/DeepSeek-V3"` | **CRASH** (different unhandled `ValueError`) | 11.7s to crash | — |
| DeepSeek-V3 | repo's own older-style fixture dict | OK, MLA shapes exact; 1 wrong/missing computation | fast | — |
| Gemma-2-9b | raw dict | OK, but architecture-collapse gap | 1.6s | 73.2KB/23.1KB |
| Gemma-2-9b | hub id `"google/gemma-2-9b"` | **CRASH** (same `ValueError` family) | 11.3s to crash | — |
| OLMo-2-1124-7B | raw dict | OK, no wrong claims found | 1.6s | 61.3KB/24.2KB |
| Mixtral-8x7B-v0.1 | raw dict | OK, but attention geometry entirely unresolved | 1.6s | 86.8KB/24.4KB |
| FLUX.1-dev | hub id | OK overall params, but majority of blocks unresolved | 38.5s | 279.3KB/42.1KB |
| Stable Audio Open 1.0 | local proxy dict (real repo gated, no token) | OK, attention mechanism unresolved | 3.6s | 112.5KB/20.7KB |
| LFM2-1.2B ("unseen" model) | raw dict | **CONFIDENTLY WRONG** claim | 9.8s | 65.3KB/23.9KB |

All HTML pages are self-contained except a Google Fonts `<link>` (decorative "Caveat" font) — functions offline with a font fallback.

## Confirmed wrong / crashing findings, with evidence

**1. DeepSeek-V3 crashes on its real, current-on-hub config (both input forms).**
Raw-dict path: `TypeError: can only join an iterable` at `model_unfolder/adapters/transformer/parser.py:1089` (`".".join(path) for path in config_paths`), because `_actual_checkpoint_path` (parser.py:1520) returns `None` for a class-default score-formula operand and that `None` is put straight into `config_paths` at parser.py:3740-3743. Root cause: the live DeepSeek-V3 config uses legacy `rope_scaling` (yarn, with `mscale`/`mscale_all_dim`/`beta_fast`/`beta_slow`), which the team's own `tests/sable_test_corpus/deepseek-v3.json` fixture does **not** use (it has the newer `rope_parameters` shape instead) — so the fixture masks the bug. Hub-id path: a *different* unhandled `ValueError: claim belongs to another or changed prepared document` at `evidence/reader_claims.py:537`, same score-formula call site (`parser.py:3701`). Both are raw internal exceptions with no actionable message — they violate the product's own bar for honest/actionable failures.

**2. Gemma-2-9b crashes via the headline `unfold("google/gemma-2-9b")` call.**
Same `ValueError: claim belongs to another or changed prepared document`, this time from the **embedding-scale** reader at `parser.py:5189`. Reproducible, deterministic. Llama-3-8B/Qwen3-8B/OLMo-2/Mixtral all succeeded via the identical hub-id code path, so this is model-specific, not a general hub-id problem — but it is a second flagship model (of the 8 named) broken on the one-liner the package's own docstring advertises.

**3. LFM2-1.2B ("unseen" model) — confidently wrong architecture claim, not an honest unknown.**
Real config: `layer_types` names only indices `[2,5,8,10,12,14]` as `"full_attention"`; the other 10/16 layers are `"conv"`. Confirmed in `transformers/models/lfm2/modeling_lfm2.py:406-411`: `self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"`; when false, the layer builds `Lfm2ShortConv` — a convolution block with no Q/K/V/attention at all. model_unfolder's output collapses all 16 layers into one `layer_group_0` with `signature.attention_kind = "gqa"` for every layer, and the rendered HTML shows uniform GQA blocks throughout (`GQA` label appears; `gqa` class 28×). The only trace of the truth is a warnings-panel entry noting `'layer_types'` and `'full_attn_idxs'` are "unread config fields" — a debt note, not a correction, and it never reaches the diagram itself. **62.5% of this model's layers are mislabeled as an attention mechanism they do not have.** Total params claimed (1.51B) is also well above the model's ~1.2B public size, consistent with costing conv layers as attention. This directly fails the "unseen model stays honest" promise: the tool doesn't render unknown, it renders a wrong uniform answer.

**4. The single most basic attention fact — score scaling — is never computed, for any model tested.**
Every one of Llama-3, Qwen3, Gemma-2, OLMo-2, Mixtral, LFM2 shows `"formula": "scaling formula unresolved"` on the `scores` node (`opgraph.py:600-604`), even though `head_dim` is known and printed elsewhere in the very same JSON, and the default `1/sqrt(head_dim)` requires no config lookup at all. Gemma-2's explicit `query_pre_attn_scalar=256` (a config-declared override) is *also* not surfaced. This is a systemic, 100%-reproducible missing computation on the most fundamental number in attention.

**5. Mixtral-8x7B: attention geometry entirely unresolved, ~1.3B params silently dropped.**
Output has `heads: {query:32, key_value:8, kv_groups:4, residual_width:4096}` — no `head_dim`, and projections show only `in_features`/`out_features` partially (`query.in_features` only, no `out_features`; `output.out_features` only, no `in_features`) with `assumptions: ["attention geometry unresolved — Q/K/V/O matrix terms omitted"]`. Mixtral's attention is architecturally identical to Llama-3's (same implicit `head_dim = hidden/heads = 128`), which resolved correctly — so this is a real, MoE-specific parser gap. Claimed total 45.36B vs. commonly cited 46.7B (~1.34B gap ≈ exactly one layer's worth of Q/K/V/O across 32 layers, 41.9M × 32 ≈ 1.34B) confirms the missing mass.

**6. DeepSeek-V3 (working fixture): dense-layer FFN width nulled despite being a plain top-level config field.**
`first_k_dense_replace=3` layers use ordinary `intermediate_size=18432` (present directly in config, unambiguous in `modeling_deepseek.py`). Output's `layer_group_0.ffn.intermediate_size` is `null`, and the operation graph's `gate_proj`/`up_proj`/`down_proj` nodes carry no width at all — a real rendering gap, mislabeled by the tool's own assumption text "ordinary/shared FFN inner width unknown" (it is not actually unknown). Costs ≈1.19B missing params (3 × 3 × 7168 × 18432).

**7. FLUX.1-dev: 38 of 57 blocks (the majority — all `single_transformer_blocks`) carry zero resolved mechanism.**
`layer_group_1` (38 layers) has `attention_kind: None, ffn_kind: None, norm_kind: "unknown", norm_placement: "unknown", residual_topology: "unknown"`, plus 38 `op_conformance` warnings that a source-proven elementwise-multiply is not drawn. Total-param figure (11.90B) is still plausible/close to FLUX's public "12B" figure, but the structural diagram omits the defining architectural feature (single-stream blocks) for the majority of the model.

## What worked correctly (spot-checked against oracle arithmetic/source)
- Llama-3-8B: GQA 32/8, head_dim 128, all Q/K/V/O shapes, total params **exactly** 8,030,261,248 (hand-computed match).
- Qwen3-8B: QK-norm per-head (extent=128, after reshape, before RoPE) — matches architecture; params 8.19B match.
- OLMo-2: MHA (32/32), QK-norm on the **full 4096-dim vector before reshape** (correctly distinguished from Qwen3's per-head norm), post-norm placement; params 7.30B match.
- Gemma-2-9b: Q/K/V/O shapes exact (4096/2048/2048/3584), embedding scale = `59.86651818838306` = exactly `sqrt(3584)`.
- Mixtral MoE routed-expert width (gate_up_proj 7168→4096 = 2×moe width) correct.
- DeepSeek-V3 MLA shapes (via the working fixture) match the task's own oracle exactly: q_a 7168→1536, q_b 1536→24576, kv_a 7168→576, kv_b 512→32768, o_proj 16384→7168.

## Verdict
The product does **not** yet reliably keep its "render only what's proven, unknowns visible" promise under real-world use. Against the exact 8 named flagship models plus one genuinely unseen architecture: 2 of 8 named models crash on the primary `unfold(hf_id)` API with raw, unactionable internal exceptions (DeepSeek-V3 both ways, Gemma-2-9b via id); the unseen-model test produces a confidently wrong uniform-attention claim across 62.5% of layers instead of an honest unknown; and even models that "work" carry systemic missing computations (attention scaling formula never computed anywhere; Mixtral attention geometry and DeepSeek dense-FFN width silently dropped; FLUX's majority block type unresolved). Where the tool does compute, its arithmetic is genuinely excellent and exact (Llama-3, Gemma-2, OLMo-2, Qwen3, DeepSeek MLA shapes all check out to the exact number/formula) — so the engine is not broadly wrong, but it is not yet safe to trust unattended on live, real-world configs.
