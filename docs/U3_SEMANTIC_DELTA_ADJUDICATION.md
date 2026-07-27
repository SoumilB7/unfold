# U3 Semantic Delta Adjudication

> **Status:** independent audit complete; code corrections verified; Soumil
> approved the nine `RMSNorm -> Norm` views on 2026-07-28; exact re-bless
> applied and inspected; final unchanged-tree bracket pending.
>
> **Authority:** this is the exact U3-C5 delta receipt referenced by
> `U3_COMPLETION_MASTER_PLAN.md`. It does not credit the semantic work to U3.

## 1. Why this audit exists

The neutral U3 closure is byte-identical (`7530467 -> 77aec7e`), but the branch
already carried semantic U6/U7/U9/U14 candidates whose outputs differed from the
last blessed U2 manifest. U3 cannot bless those changes merely because its own
substrate is neutral.

The audit therefore answered four separate questions:

1. which witnesses and surfaces differ;
2. which exact semantic commit first caused each difference;
3. whether the new claim is proved by the selected owner or is an honest
   unknown;
4. whether any downstream consumer strengthened that evidence improperly.

## 2. Verification-method correction

The first temporary historical-comparison harness created detached worktrees but
ran Python from the main checkout. Python could therefore import the current
package before the detached package. Those results were discarded.

Every retained comparison:

- changes the process working directory to the detached worktree;
- sets that worktree as `PYTHONPATH`;
- asserts `Path(model_unfolder.__file__).is_relative_to(worktree)`;
- renders the unchanged corpus inputs from the main checkout;
- compares hashes only after that import assertion passes.

This is a binding rule for future commit attribution. A worktree existing on
disk is not proof that the process imported it.

## 3. Exact current-versus-blessed matrix

Current drift before the small corrections in Section 7:

- **19 witnesses**;
- **47 canonical surfaces**;
- **9 named SVG views**.

After the Section 7 GPT-OSS assumption correction:

- **19 witnesses**;
- **46 canonical surfaces**;
- **9 named SVG views**.

The sole surface removed from the drift set is GPT-OSS `params`; its regenerated
hash is byte-identical to the blessed hash. Its `ir`, `ledgers`, `expanded`,
`html_meta` and `sable` surfaces remain changed for the separately adjudicated
expert-storage and learned-sink evidence.

| Witness | Changed canonical surfaces | Changed named view |
|---|---|---|
| auraflow-v0-3 | `ir`, `html_meta` | `encoder_0` |
| bloom | `ir`, `ledgers`, `sable` | — |
| cogvideox-5b | `ir`, `html_meta` | `encoder_0` |
| deepseek-v3 | `ir`, `ledgers`, `expanded`, `params`, `html_meta`, `sable` | — |
| fluxtransformer2dmodel | `ir`, `html_meta` | `encoder_1` |
| gemma-2-2b-it | `ledgers` | — |
| glm-4-5 | `ir`, `ledgers`, `expanded`, `params`, `html_meta`, `sable` | — |
| gpt-oss-20b | `ir`, `ledgers`, `expanded`, `params`, `html_meta`, `sable` | — |
| llama-7b | `ledgers` | — |
| ltx-video | `ir`, `html_meta` | `encoder_0` |
| mochi-1-preview | `ir`, `html_meta` | `encoder_0` |
| musicgen-small | `ir`, `ledgers`, `expanded`, `html_meta` | `conditioning_encoder` |
| olmo-2-1124-7b | `ledgers` | — |
| pixart-sigma-xl-2-1024-ms | `ir`, `html_meta` | `encoder_0` |
| qwen2-vl-7b-instruct | `ledgers` | — |
| qwen3-8b | `ledgers` | — |
| stable-diffusion-3-5-large | `ir`, `html_meta` | `encoder_2` |
| stablelm-2-1-6b | `ledgers` | — |
| wan2-2-t2v-a14b-diffusers | `ir`, `html_meta` | `encoder_0` |

The nine view changes are one mechanism class: a T5/UMT5 conditioning tower
formerly labelled `RMSNorm` is now labelled generic `Norm`. No layout, edge,
layer count, attention or FFN node changes in those nine views.

All nine were regenerated again through the production
`Diagram.save_images(..., highlight_clickable=True)` path and compared with
their blessed gallery counterparts. This matters because a first inspection
render omitted the Dable amber clickable overlay and was not a like-for-like
pixel presentation. With identical highlight settings, all nine preserve the
same geometry, connectivity, repetition count, labels other than the two norm
nodes, and clickable-node highlighting. The only architectural text delta is
`RMSNorm -> Norm` on the pre-attention and pre-FFN norm nodes.

## 4. Exact commit attribution

Detached-tree comparisons at the relevant boundaries produced this chain:

| Commit | True unit | Exact effect retained at current HEAD |
|---|---|---|
| `78ab271` | U6 candidate | owner-qualified attention-storage facts/provenance; evidence-ledger changes, no structural IR change |
| `80fc2c3` | U7 candidate | exact ordinary-FFN result; BLOOM becomes proven dense storage; hybrid MoE blocks stop borrowing a whole-file ordinary-FFN verdict |
| `d96c411` | U7 candidate | exact routed-expert `fused_gate_up` storage for DeepSeek-V3, GLM-4.5 and GPT-OSS |
| `6effc34` | U7 candidate | exact selected-block norm result; nested T5/UMT5 selection failures stop inheriting the old spelling-derived RMS claim |
| `a788288` | U6/U7 correction | rejects ambiguous nested norm ownership, applies exact projection-bias facts and keeps unresolved facts unknown |
| `306fd8e` | U6 candidate | GPT-OSS learned attention-sink provenance only; structural IR is unchanged |

The later neutral/output-routing commits `bb1b4b1`, `703274d`, `243f4a2`,
`4102e1c`, `7530467` and `77aec7e` did **not** cause the 19-witness preservation
matrix. `703274d` does make the already-implemented exact CLIP block readers
reachable in the standalone FLUX fixture; that is the separate stale-test
correction in Section 7.

## 5. Source and counterexample verdict

### 5.1 Ordinary and routed FFNs

- BLOOM's selected `BloomMLP` constructs `dense_h_to_4h`, applies GELU, then
  `dense_4h_to_h`: `projection_mode=dense` is source-proven.
- DeepSeek-V3 and GLM-4.5 construct one stacked `gate_up_proj` parameter and a
  separate stacked `down_proj`, split the gate/up lanes, and invoke only selected
  experts: `expert_projection_mode=fused_gate_up` is source-proven.
- GPT-OSS has the same stacked gate/up plus down storage, including its explicit
  expert biases.
- The ordinary-FFN reader deliberately abstains on hybrid routed blocks. It does
  not let routed experts certify the dense/shared lane.

Permanent controls cover dense, split-gated, fused experts, wrong-axis,
uninvoked, sibling-owner, rival and hybrid-abstention cases.

### 5.2 Normalization

- BLOOM and MusicGen decoder blocks construct `torch.nn.LayerNorm`.
- Llama, Gemma, DeepSeek-V3, GLM, GPT-OSS and Qwen2-VL selected decoder blocks
  prove RMS normalization from exact external protocol or implementation math.
- T5 and UMT5 implementations are in fact RMS-style (mean-square plus `rsqrt`,
  no mean subtraction), but the current nested text-tower path cannot yet prove
  the unique encoder-block occurrence because encoder/decoder candidates rival.

Therefore generic `Norm` is the only lawful current T5/UMT5 rendering. Restoring
`RMSNorm` from `layer_norm_epsilon` would be accurate by luck but would restore
the forbidden config-name inference. U9 owns the recursive tower occurrence
proof that may lawfully restore `RMSNorm`.

### 5.3 Projection bias and learned sinks

- MusicGen decoder FFN `fc1` and `fc2` are constructed with `bias=False`.
- CLIP encoder layers construct two exact `torch.nn.LayerNorm` calls.
- GPT-OSS constructs a learned per-head `sinks` parameter and concatenates it
  with attention logits before softmax.

The positive and negative real-model controls and the adversarial focused suite
are green.

## 6. Downstream-consumer audit

DeepSeek-V3 and GLM-4.5 now withhold the unproven ordinary/shared gated shape.
The parameter estimator consequently reports a two-projection **lower-bound
convention** and publishes the explicit assumption:

> FFN structure unknown — counted as 2 projections (a gated FFN would add
> hidden x inner per layer)

That is not an exact total. It remains X-02/U14 debt: U14 must publish a
range/partial result rather than one nominal total.

GPT-OSS has no shared expert, so its unknown ordinary/shared `gated` field does
not participate in any parameter formula. Publishing the same assumption there
was false metadata. Section 7 removes it and adds a zero-shared versus
one-shared poison.

## 7. Surgical corrections from this audit

1. `tests/test_diffusion.py::test_text_encoder_shows_real_config_dims` now
   expects the exact source-proven CLIP `LayerNorm` while retaining T5 as
   unknown. The old comment claimed neither tower had an exact block, which is
   no longer true.
2. `params.py` emits the ordinary/shared FFN unknown assumption for an MoE only
   when at least one shared expert exists. A routed-only MoE cannot be qualified
   by an unused mechanism field.
3. A synthetic poison proves:
   - zero shared experts: no ordinary/shared-gate assumption;
   - one shared expert: the assumption remains mandatory.

The table in Section 3 deliberately records the pre-correction attribution
matrix. The verified post-correction GPT-OSS row is `ir`, `ledgers`, `expanded`,
`html_meta`, `sable` (no `params` drift).

No family/model branch, config fallback, renderer exception or blessing was
added.

## 8. Decision and remaining gate

Technical verdict:

- **retain** the exact readers and their owner-bound facts;
- **do not forward-revert** to whole-file unions or config-name guesses;
- **accept as intentional candidates** the source-proven FFN/expert/bias/sink
  changes and the honest T5/UMT5 loss of specificity;
- **do not mark U6/U7/U9/U14 complete**—these are bounded pre-existing slices;
- **do not re-bless automatically**.

Soumil explicitly approved the nine `RMSNorm -> Norm` views on 2026-07-28.
The project-owned guarded `bless()` path then regenerated the nine complete
galleries and fixtures; the 26-witness preservation manifest was rebuilt from
production outputs.

The before/after artifact audit proves:

- exactly one PNG changed in each approved gallery, at the approved view name;
- no other PNG, review record or gallery file changed;
- every frozen config, fixture identity and source stayed equal;
- each fixture replaces exactly one view signature and carries its exact prior
  signature in `superseded_hash_signature`;
- the manifest changes 19 already-adjudicated witnesses, 55 surface hashes
  (the 46 semantic/evidence hashes plus nine gallery hashes), nine view hashes
  and nine fixture-input hashes;
- GPT-OSS `params` remains byte-identical to the previous blessed hash;
- no unexpected witness, surface, view or file entered the transition.

Before U3-C5 can turn green:

1. commit the inspected fixture/manifest transition;
2. rerun the unchanged-tree focused, U2-authority, preservation, full-suite and
   isolated-checkout bracket;
3. record the remaining U9 exact-T5-owner and U14 parameter-range debts without
   weakening either gate.

Until then, U3 neutral infrastructure is complete but its project-wide C5
release receipt remains held. U4 implementation must not use the held state as
permission to introduce more unreviewed output drift.
