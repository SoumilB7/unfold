# Findings register — every prior finding, accepted, voided or revised

Decided 2026-09-02 against the nine research passes (`00`–`07`, `11`). Sources
of the original claims: `S` `../09-unit-verdicts/systemic-findings.md`, `E`
`experimental-confirmation.md`, `F` `first-principles-judgment.md`, `M`
`method-and-plan-judgment.md`, `Ux` per-unit files. Evidence column names the
research file (and section) that decided the row.

## A. Physics and substrate

| id | finding | status | decision and evidence |
|---|---|---|---|
| A1 | ProgramIndex indexes only the model's own file; imported classes unresolved | **ACCEPT** | `02` §5: only `transformer_sd3.py` indexed; `build_program_index` called only with `external_nodes=()` (`context.py:227`); `import_roots` proves import references, never expands the index |
| A2 | U3 boundary 8 declared, scaffolded, never used; 15 diffusion roots failed B1 day 3 | **ACCEPT** | `06` decision log + `U3`; `02` confirms the parameter is dead |
| A3 | Import closure exists, used only by UNet readers | **ACCEPT, sharpened** | `01`: all 14 `unet_*` readers (8.6k lines) have **no production caller either** — U11 is half-landed; the closure is used by nothing that ships |
| A4 | Closure alone restores SD3.5's skeleton, not cells; PixArt needs the guard selector | **ACCEPT** | `E`; `02` §5 exact stop at `execution_flow.py:663` → `diffusion_stack.py:724` |
| A5 | `diffusion_block.py:299-310` refuses own-file mixin lanes | **ACCEPT, scoped** | `02`: latent for SD3.5 (behind A1); active for FLUX/Wan/LTX/PRX (`E` A) |
| A6 | Static-only is the wrong physics for *construction*; meta instantiation gives tree/shapes/branches in ms | **ACCEPT, with the latency argument removed** | `07`: meta works on all 5 targets, 0.04–0.53 s after imports; recovers DeepSeek's 3 dense + 58 MoE layers, fused expert shapes, SD3.5's `context_pre_only` schedule; FakeTensor trace resolves the Qwen3.5 ternary, T5 loop-carried bias, MoE routing; **PixArt's `_class_name: Transformer2DModel` is remapped at runtime to `PixArtTransformer2DModel` — the static bundle reads a file the runtime never constructs**. But the physics case must not rest on latency (see A7) |
| A7 | Cold `unfold()` 30–100 s; nothing persisted | **REVISE** | `02`: cold 25 / 96 / 114 s but **62–85 % is one quadratic bug** — `ast.get_source_segment` per node (`program_index.py:2355-2361`; 17k calls = 14 s Llama, 77k = 70 s SD3.5); warm 1.1–5.6 s; plus process-only per-component source cache and a 7–12 s `AutoConfig` import. A one-site fix, not intrinsic cost |
| A8 | `evidence/` 14k → 96k lines; U10 net +11.6k | **ACCEPT** | `01`: 267 modules / 133k total; readers 60.2k; `07` estimates ≈45–55k after the physics split |
| A9 *(new)* | Six import cycles held open by 379 function-local imports; 26 dead definitions; 129 duplicated private helpers (`_self_field` ×47, 26 bodies); 12 modules > 1,500 lines; `parse()` now 3,769 lines in a 4,996-line file | **NEW — ACCEPT** | `01` §health, `02` §1 |

## B. Recall, soundness and the gates

| id | finding | status | decision and evidence |
|---|---|---|---|
| B1 | No gate measures recall | **REVISE** | `05` §2: of 42 checks, recall-direction = 7 (2 advisory, 2 scoped); **exactly one blocking, per-model, code-comparing recall check** exists (`op_conformance` "missing" direction, one representative layer per group); every code-facing net compares against the same AST the parser read — **no independent oracle anywhere** |
| B2 | Hash-set signature, no labels | **ACCEPT, nuanced** | `05` §5: covered for the 29 witnesses by the ordered preservation manifest; not by `bless`/`check_regression` |
| B3 | U4 step 6 vs §7 same day, never reconciled | **ACCEPT** | `06` contradictions #1 |
| B4 | Verified detail lost and approved in U6/U8/U9/U10 | **ACCEPT** | `E`, `04` (SD3.5 header "LAYERS 0 · HIDDEN 0 · PARAMS ?") |
| B5 | Nothing collapsed in U6 came back | **ACCEPT** | `E` C |
| B6 | T5 g1 silent negative mechanism | **ACCEPT** | `E` C, `04` |
| B7 | DeepSeek score scale wrong `code_proven` | **ACCEPT** | `E` B |
| B8 | Proven-but-undrawn (Qwen2-VL FFN, MusicGen T5) | **ACCEPT** | `E`, `04` |
| B9 | Unknown over-applied to provable negatives | **ACCEPT** | `E` B; `03` (QK-norm absence never proven, 7 models) |
| B10 | Advisory nets | **ACCEPT** | `05` §1 |
| B11 | Blanket closure excusal greens blank denoisers | **ACCEPT** | `05` §5: 94 of 115 `classified:` rows are that expansion |
| B12 | Harness gaps (oracle, `source`, liveness) | **ACCEPT, extended** | `05`: also `zero_asserted_census` swallows every exception; `test_coverage.py` runs on 17 synthetic dicts, never real witnesses; `census.py --check` is wired into nothing |
| B13 | Silent absences list | **ACCEPT** | `E`, `03`, `04` |
| B14 | Out-of-corpus 75 %; T5 0/12; GPT-2/MPT no residuals; scaffolds | **ACCEPT** | `E` D; `03` §5 (ecosystem 33 % clean / 71 % with typed unknowns) |
| B15 | Rival/guarded invocations = largest general gap | **ACCEPT, ranked** | `03` top-10: #1 config-conditional mask-builder alias (Mistral/Qwen2 lineage, untested); #2 non-attention mixers make hybrids all-or-nothing; #3 subscript-slice fused QKV |
| B16 *(new)* | **Consumers still convert unknown → known and sniff labels**: `graph.py:105`, `op_render.py:287` (unknown kind → norm box), `metadata_modalities.py:494,668,897` (`or where['pre']`), `labels.py:787` ("prompt states" → Prompt else Vision, under a comment forbidding it), invented literals `30.0/48/0.1` in `blocks/model.py:120,165,268`, four phantom MoE expert boxes (`feed_forward.py:582-608`), fixed 15-cell sliding strip (`svg.py:345`), placeholder text-encoder tower for bare diffusion configs | **NEW — ACCEPT** | `04` §3. Contradicts U4's "no surface manufactures a mechanism" and U5's firewall scope |
| B17 *(new)* | Zero-layer diffusion IR ships with no warning (`has_value` branch on an empty projection); `encoder_panel.py:79-80` silent `except Exception: return {}`; encoder slot facts never reach `ir.extras.fact_provenance` | **NEW — ACCEPT** | `02` §4–5 |

## C. Doctrine

| id | finding | status | decision and evidence |
|---|---|---|---|
| C1 | Identity law over-applied to torch primitives | **ACCEPT** | `03`, `E` B |
| C2 | `type_roles.yaml` substring roles load-bearing in conformance | **ACCEPT** | `05` §5 |
| C3 | Ten vocabularies for unknown | **ACCEPT, counted** | `01`: three typed vocabularies with two *different* failure-kind sets (`facts.py:44` vs `reader_result.py:25`) plus an ad-hoc tail; 29 distinct strings |

## D. Process and method

| id | finding | status | decision and evidence |
|---|---|---|---|
| D1 | §13 template 1/487; 57 of August's 105 bodies empty | **REVISE** | `06`: template 1/487 ✓; **empty bodies 396/487, August 111/111** (my 57/105 was wrong) |
| D2 | 9 of 10 receipt dirs gone | **REVISE** | `06`: 30 cited, 22 gone, 3 recreated by this program's own runs |
| D3 | Documentation planes disagree; prose exceeds code | **ACCEPT, extended** | `06`: four planes; **`z-docs/` and `PROTOCOL.md` are not under version control**; 17 of 23 spot-checked "current state" claims stale; 12 docs to archive |
| D4 | Visual layer not binding; Her Eyes lapsed | **ACCEPT** | `05` §5, `06` (366 reviews total, 0 since 08-06; bless authority delegated/self-marked) |
| D5 | No third witness for recall | **ACCEPT** | `05` §2 ("no independent oracle") |
| D6 | Orphaned owners | **ACCEPT** | `05` |
| D7 | `parse()` god function | **ACCEPT** | `01`, `02` |
| D8 *(new)* | Nothing shipped since v0.2.17 (2026-07-07): 254 commits, +215k lines; `origin/main` 253 commits behind; Space pinned to 0.2.17 and asleep; PRs stopped at #13 | **NEW — ACCEPT** | `06`, `11` |
| D9 *(new)* | Plaintext HF tokens in `finalize.ipynb` and `done/tryrun.ipynb`, outside any `.gitignore` | **NEW — ACCEPT (security)** | `00`, `06`, `11` |

## E. Product

| id | finding | status | decision and evidence |
|---|---|---|---|
| E1 | User never named; auditor served, learner sold | **ACCEPT, deepened** | `00` §2 (six candidate users; current z-docs never mention newcomer/notebook/journey); `11` (the learner half was built in the Next.js labs and abandoned; the engine kept the drill and dropped the journey, worked example, animation, the *why*) |
| E2 | Version inconsistency | **ACCEPT** | `00` (0.2.17 / 0.2.15 / Space pin) |
| E3 | Diffusion 33 % proven; PixArt/SD3.5 0 % | **ACCEPT** | `E` A; `04` (blessed SD3.5 header shows 0 layers) |
| E4 | Stale Her Eyes approves a placeholder | **ACCEPT** | `04` |
| E5 *(new)* | README contradictions: "Diffusors — coming soon" beside a full diffusor adapter; SSM/RWKV rows vs "no SSM/Mamba"; DeepSeek params 669.8B/36.4B match neither README (~675B/~41B) nor paper (671B/37B); PixArt returns no estimate; Space caption "generated purely from the Hugging Face config map" is false; Space flattens every error | **NEW — ACCEPT** | `00` §3, `04` §4 |
| E6 *(new)* | `examples/*.html` three months stale, zero unresolved chips, `flux-1-dev.html` shows a DiT HEAD no longer produces, **`deepseek-v3.html` is actually Gemma-4-E2B** | **NEW — ACCEPT** | `04` §4 |
| E7 *(new)* | Every on-screen sentence is hand-authored in `adapters/*/blocks/` (279 `title` literals; ~850 presentation keys vs ~490 fact keys); no per-block param counts drawn; causal mask never drawn; the `*` on totals explained only by a hover title | **NEW — ACCEPT** | `04` §2–3 |
| E8 *(new)* | `unfold-npm` is a dead full JS mirror (broken import, 0 tests, unpublished, dropped RoPE nodes in its one parity test); the intended endpoint was in-browser | **NEW — ACCEPT** | `11` |
| E9 *(new)* | Three-tier framing (structure / mechanism / values) asserted by the auditor, not yet explicitly signed off | **NEW — OPEN** | `00` §8; resolved in `10-post-u-plan.md` §0 |

## F. Grades and verdicts — re-issued

| id | prior | status | re-issued |
|---|---|---|---|
| F1 | U0 C→B · U1 B− · U2 B+ · U3 B− · U4 B− · U5 B+ · U6 C+ · U7 A− · U8 B− · U9 B− · U10 A/C | **REVISE two** | **U4 → C+** (B16: consumers still default unknown→known, so "no surface manufactures a mechanism" is false); **U5 → B** (firewall does not cover semantic reconstruction or label sniffing, `04`); U11 provisionally **incomplete** (A3). Others stand |
| F2 | Project B−; plan B; method B+ | **REVISE** | see `09-judgment.md` — project **C+** (the product regressed and nothing shipped; the engine core stands), plan **B−** (no user, no recall, no release), method **B** |

---

## Amendment 2026-09-02

- **A6** downgraded from ACCEPT to **"promising, verified on a pilot"**: meta
  instantiation is the construction *inventory* and denominator; it authors
  tier-1 only through the reconciliation contract (`12` §1) after shadow
  comparison. Trace is positive evidence only.
- **`03` percentages** are a prioritisation map, not release metrics (many rows
  classified without rendering).
- **`06` "version-control done/"** narrowed: curate; secrets and generated
  artifacts excluded.
- **`12` v1** was stale (F4/F2c already at `9b3cb7b`/`9a4e1e5`); replaced by v2.
