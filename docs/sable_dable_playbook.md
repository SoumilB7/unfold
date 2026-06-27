# Sable + Dable Playbook — confirm a model's diagram against its real HF code

**Read this whole file once, then follow it top to bottom. Do exactly what each STEP says.
Run every command. Look at every image. Do not skip steps. Do not guess.**

/previews
it should have folders with model name -> images, html, report and Manifest

## 0. What you are doing (the one idea)

`model_unfolder` turns a HuggingFace model's `config.json` into an architecture diagram.
Your job is to **prove the diagram is correct** by checking it two ways:

1. **SABLE** = run 9 blocking automated nets plus the staged unread-config audit.
   Pass = every blocking net reports nothing; every config-audit warning is still triaged.
2. **DABLE** = look at the rendered **images** with your own eyes, AND read the model's
   **real HuggingFace (transformers or diffusers) code**, AND confirm the picture matches the code.

IF IMAGES OR CODE ARENT SEEN THE REVIEW IS NULL AND VIOD SO NEVER AND I MEAN NEVER MISS THAT PART

The single most important rule:

> **The HuggingFace `forward()` code is the ground truth. The diagram must match it —
> nothing the code does may be missing from the diagram, and nothing in the diagram may
> be absent from the code.**

If the diagram and the code disagree → that is a **FAIL**. Report it. Do not "fix" it by
imagining the code is wrong.

### Hard rules (never break these)
- **Never run `git commit`.** Soumil reviews and commits. You only inspect and report.
- **Never paste, print, or echo a HuggingFace token.** Authentication is automatic. Just
  call the commands below with no token argument.
- **Always run commands from the package directory:**
  `cd /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg`
- **Look at the PNG images for real** (open them / Read them as images). Reading the HTML
  or SVG text is NOT looking. A clean automated report is **necessary but not sufficient** —
  you must still look at pixels and read the code.
- **Save HTML inside the model folder:**
  `previews/<model_folder>/<model_folder>.html`, never alongside that folder.

---

## 1. Setup — pick the model and load it

Always `cd` first:

```bash
cd /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg
```

There are **three ways** to load a model. Use whichever fits what you were given.

**(A) By HuggingFace id** (a string like `"black-forest-labs/FLUX.1-dev"`). Use this when you
were given a model id. It downloads the config automatically.

```python
MODEL = "black-forest-labs/FLUX.1-dev"   # <-- put the id here
```

**(B) By a local config dict** (a `config.json` you already have). Use this for offline work.

```python
import json
MODEL = json.load(open("/path/to/config.json"))   # <-- a dict
```

**(C) By a blessed corpus fixture** (for practice / offline). These already exist in this repo
under `tests/sable_corpus/`. The real config is nested under the `"config"` key:

```python
import json
MODEL = json.load(open("tests/sable_corpus/mochi-1-preview.json"))["config"]
```

> In every command below, `MODEL` means "the thing you loaded above." It can be a string id
> or a dict — every command accepts both.

If you just want a hard model to practice on, use one of these fixtures (all hard, all good
test cases): `hunyuanvideo`, `flux-2-dev`, `fluxtransformer2dmodel`, `mochi-1-preview`,
`auraflow-v0-3`, `sana-1600m-1024px-diffusers`, `cogvideox-5b`, `wan2-2-t2v-a14b-diffusers`.

---

## 2. STEP A — SABLE (9 blocking nets + config coverage advisory)

Run this **exact** script. Replace the first line with your model from STEP 1.

```python
from model_unfolder.sable import sable

MODEL = "black-forest-labs/FLUX.1-dev"   # <-- your model (string id or dict)
OUT   = "/tmp/sable_out"                  # where images get written

report = sable(MODEL, source="local", outdir=OUT)
print(report.summary())
```

`report.summary()` prints something like:

```
SABLE · HunyuanVideo
  oracle: present
  mechanical: PASS  (14 distinct views, 14 PNGs)
    [       ok] click_coupling
    [       ok] dangling_connectors
    [       ok] unique_ref_ids
    [       ok] no_dotted_arrows
    [       ok] no_dotted_boundaries
    [       ok] config_field_audit  — coverage advisory — promote to blocking after owned-field backlog is zero
    [       ok] op_conformance
    [       ok] wiring_conformance
    [       ok] fact_conformance
    [       ok] label_lint
  visual review: PENDING  (inspect the gallery against report.rubric)
```

### How to read it — exact pass/fail rules

Check these in order. If any blocking net fails, the model **FAILS Sable** — write down which net and its
findings, then still continue to STEP B/C/D so your report is complete.

1. **`oracle: present`** — REQUIRED.
   - If it says `oracle: MISSING …`, the model's HuggingFace code is **not installed**, so the
     code-vs-diagram checks were **skipped**. This is a **blocked / NEEDS-HUMAN** result — you
     cannot confirm against code. Report it as: "oracle MISSING — conformance could not run."
     (You may still do the visual pass, but you cannot fully confirm.)

2. **Every blocking net shows `[       ok]`** — REQUIRED. A net with `FAIL (n)` lists its findings
   underneath (lines starting with `·`). Copy those findings verbatim into your report.

3. **Triage every `config_field_audit` warning.** It is currently displayed as `WARN`, not
   counted against `mechanical_passed`, because the existing owned-field backlog is being
   migrated incrementally. That is a compatibility concession, not permission to ignore it:
   an unread field that may alter architecture is `#*IMP*` and prevents an independently
   confirmed PASS until parsed, intentionally classified, or proven irrelevant from HF code.

What each net means (so you can describe a failure correctly):

| Net | What a finding means |
|---|---|
| `click_coupling` | a clickable block has no card in its immediate next-depth panel (a same-id card elsewhere cannot mask the broken link). |
| `dangling_connectors` | a `⊕` `×` `⊙` `‖` connector is drawn with a missing input (see §5). **P0.** |
| `unique_ref_ids` | two arrowheads/markers share an id → arrowheads can vanish in the browser. |
| `no_dotted_arrows` | a generated dataflow arrow uses a dotted stroke, which incorrectly reads as optional/uncertain flow. |
| `no_dotted_boundaries` | a region/highlight boundary uses a dotted stroke; all generated structural boundaries must be solid. |
| `config_field_audit` | a config field was never read by its owning parser. This is a staged advisory until the known backlog reaches zero; triage it manually now. |
| `op_conformance` | the diagram is **missing an op the code does**, or **drew an op the code never does** (a coarse op-KIND diff against the real `forward()`). |
| `wiring_conformance` | a conditioning input (text / timestep) is wired wrong vs the code. |
| `fact_conformance` | same op-kind but wrong **meaning** — e.g. drew NoPE when the code applies RoPE, or softmax when the code is linear attention. |
| `label_lint` | a label breaks the label rules (raw activation name on a block, nested parens, etc.). |

> `op_conformance`, `wiring_conformance`, `fact_conformance` are the three nets that compare
> the diagram to the **actual code**. They only run when `oracle: present`. A failure in any of
> them is the strongest possible signal that the picture does not match the code.

**Also confirm the variants rendered.** Some models have more than one block type (e.g. Flux:
dual-stream AND single-stream). The summary prints the number of distinct views. If you expect
two block types but see only one denoiser view, that is a **FAIL** (a non-dominant variant was
not rendered). You will verify this for real in STEP B.

---

## 3. STEP B — DABLE part 1: look at the images

The Sable run already wrote PNGs to `OUT` (e.g. `/tmp/sable_out`). List and read the manifest:

```bash
ls -1 /tmp/sable_out/*.png
cat /tmp/sable_out/MANIFEST.txt
```

The manifest looks like:

```
# 14 DISTINCT diagram views (architecture + every drill, to the leaves)
# amber border = clickable block (opens a card); no border = static port/glyph
00__architecture.png
01__encoder_0.png
03__denoiser.png
04__denoiser__1.png      <- a SECOND denoiser variant (good — both block types rendered)
07__attn.png
08__ffn.png
...
```

**Now OPEN AND LOOK AT EVERY PNG** (use the Read tool on each `.png` path — it shows you the
image). Do not skip any. For each image, check the visual rubric below. Mark any image that
violates a rule.

### Visual rubric — a FAIL if you see ANY of these
- a line/arrow passes **through** a block instead of around it.
- arrowheads collide, or an arrow ends in empty space (looks dangling).
- two boxes overlap, or a label overflows / clips outside its box.
- a caption or chip collides with the `×N` badge, a block, or the frame edge.
- two **different** ops show the **same label** in one view (reads as a duplicate).
- a pale/washed-out box that looks like it *should* drill into detail but doesn't (lazy),
  versus an honestly-unknown opaque node (acceptable).
- an arrow whose meaning is ambiguous — every arrow must read as the ONE real flow it is.
- the block does **not** read as the right mental model of the computation.

### The amber border
In these debug images, a **thin amber/orange border = the block is clickable** (it opens a
card or a deeper drill). **No border = a static port** (a plain in/out anchor) or a glyph.
- A big, named computation box (Attention, Feed-Forward, …) with **no amber border** is
  suspicious → note it.
- A connector glyph (`⊕ × ⊙ ‖`) is amber (clickable, opens a one-line card) but is a glyph,
  not a box — that is correct.

### Run the dangling-connector flag explicitly (do this too)
```python
from model_unfolder import unfold
d = unfold(MODEL)
print("wiring_problems:", d.wiring_problems())
```
This must print `wiring_problems: []`. **A non-empty list is a P0 FAIL** — it names every
connector drawn with a missing input. (This is the same thing the `dangling_connectors` net
checks; run it anyway as a double-check.)

---

## 4. STEP C — get the model's REAL HuggingFace code (the oracle)

You will now look at the actual modeling source so you can compare it to the images.

Run this **exact** script (it finds the code file, finds the real block class(es) the model
stacks, and prints each block's submodules and its full `forward()`):

```python
import ast, json
from pathlib import Path
from model_unfolder.evidence.sources import resolve_source_files
from model_unfolder.evidence.forward_ops import _field_types, _role_of, _method, _module_list_elems

def dump_block_code(cfg):
    files = resolve_source_files(cfg, source="local").files
    print("MODELING FILES:", [str(f) for f in files] or "NONE (oracle MISSING)")
    built = set()                      # the classes the model actually STACKS = the real blocks
    for p in files:
        tree = ast.parse(Path(str(p)).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                for _f, c in _module_list_elems(_method(n, "__init__")).items():
                    built.add(c)
    for p in files:
        src = Path(str(p)).read_text(); tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name in built:
                ft = _field_types(_method(n, "__init__"))
                subs = [(f, c, _role_of(c)) for f, c in ft.items() if _role_of(c)]
                print(f"\n===== BLOCK CLASS: {n.name} =====")
                print("  submodules (field, class, role):", subs)
                fwd = _method(n, "forward")
                if fwd:
                    print("  --- forward() ---\n" + ast.get_source_segment(src, fwd))

# This line works whether MODEL is a dict OR a string id (token is automatic):
from model_unfolder.parser import _coerce
cfg = MODEL if isinstance(MODEL, dict) else _coerce(MODEL)
dump_block_code(cfg)
```

The output tells you, for each real block:
- **`MODELING FILES`** — the actual `.py` file on disk. If it says `NONE`, the code is not
  installed → you cannot confirm against code → **NEEDS-HUMAN**, stop and report that.
- **`submodules`** — every sub-layer the block builds, with its **role**
  (`attention` / `ffn` / `norm` / `linear`). This is the list of named pieces.
- **`forward()`** — the real computation, line by line. **This is the ground truth.**

> Want to see even more of the file (e.g. the attention class internals, or a norm class)?
> Open the printed `.py` path directly with the Read tool and read the relevant class.

### How to read a `forward()` (what to extract from it)
Read it slowly and list, on paper:
1. **Every operation in order** — each `self.something(...)` call, each `+`, each `*`, each
   `torch.cat(...)`, each `.chunk(...)`/`.split(...)`, each `for` loop.
2. **Norm placement** — for each sub-layer (attention, FFN): is a norm applied BEFORE it
   (`x = self.norm(x); x = self.attn(x)`), AFTER it (`x = self.norm(self.attn(x))` before the
   add), or BOTH (a norm before and a norm after — a "sandwich")?
3. **Residual adds** — `hidden = hidden + something`. How many, and what do they add?
4. **Parallel vs sequential** — does the FFN take the **attention's output** (sequential), or
   does it take the **same input the attention took** (parallel — one norm feeds both)?
5. **Merges** — `torch.cat([a, b])` is a real concat (a `‖`). A reshape/`.view` is not.
6. **Gates** — a `* gate` multiply (a `×`). Note if the gate is folded into a norm instead
   (e.g. `hidden = hidden + self.norm(sublayer, gate)` — then there is NO separate `×`).

Keep this list. You will tick each item off against the images next.

---

## 5. STEP D — confirm the diagram MATCHES the code (both directions)

This is the heart of the job. An architecture is made of exactly **five kinds of thing**.
Check all five, in **both directions**, comparing your `forward()` list (STEP C) to the
images (STEP B).

The five things:
1. **Blocks** — the operation boxes (attention, FFN, norm, …).
2. **Arrows** — the flow lines connecting blocks.
3. **Repeating regions** — the `×N` loop frames (a `for` loop / `ModuleList` in code).
4. **Connectors** — fan-in glyphs: `⊕` (add), `×` (multiply/gate), `⊙` (apply values),
   `‖` (concat).
5. **Splitters** — fan-out: one value (a dot on a line) feeding several places.

**A repeat region must be earned.** Draw a repeat frame/pill only when the code has an
actual loop/`ModuleList` region and the declared count is greater than one (or honestly
unknown as `×N`). A one-off block (`repeat == 1`) gets neither a `×1`/semantic pill nor the
surrounding repeat frame. Put semantic names on the block/card, never in repeat chrome.

### Direction 1 — code → diagram (nothing the code does is missing)
For **each** operation in your STEP-C list, find it in the images:
- A `self.attn(...)` call → there must be an **Attention** block.
- A `self.ff(...)` / `FeedForward` → there must be a **Feed-Forward** block.
- A `self.norm(...)` → there must be a **Normalization** block in the right place
  (before/after the sub-layer, matching STEP C item 2).
- A `+` residual → there must be a **`⊕`** with the sub-layer output AND the skip as its two
  inputs.
- A `torch.cat([a, b])` → there must be a **`‖`** with **two** inputs (the two things joined).
- A `* gate` → there must be a **`×`** (unless the gate was folded into a norm — then a norm
  box, no `×`).
- A `for ... in self.layers` / `ModuleList` → there must be a **`×N`** loop frame.
- RoPE applied to Q/K in code (`apply_rotary_emb`, `image_rotary_emb`, `freqs_cis`) → the
  attention drill must show an **apply-RoPE** step on Q and K (NOT a "NoPE" chip).

**If the code does it and the diagram does not show it → FAIL.** Write: "code does X
(`file.py` line), diagram omits it."

### Direction 2 — diagram → code (nothing invented)
For **each** block, arrow, and connector in the images, find it in the code:
- Every box must correspond to a real `self.x(...)` call, tensor op, or config-gated line.
- Every `‖` / `×` / `⊙` / `⊕` must correspond to a real `cat` / `*` / attention-apply / `+`.

**If the diagram shows it and the code does not do it → FAIL.** Write: "diagram shows X, code
never does it."

### The connector rule (check every glyph)
Every `⊕` `×` `⊙` `‖` must have **TWO inputs visibly arriving at it**. Look at the image:
two arrows must reach each glyph.
- A `⊕` (residual add): the sub-layer output + the skip line. Both must be there.
- A `×` (gate): the thing being gated + the gate/conditioning source (e.g. timestep). Both.
- A `⊙` (apply values, in attention): the softmax scores + the V lane. Both.
- A `‖` (concat): the two lanes being joined. Both. (A `‖` with one input is a FAIL.)
- **Only allowed exception:** a `×` by a **labelled constant** — then the constant is written
  next to the glyph (e.g. `× 2.5`). If you see `× something` with no second arrow AND no
  written constant, that is a **FAIL** ("× what?").

### Heterogeneous models (more than one block type)
If STEP C printed **two or more** different block classes that the model stacks (e.g.
`FooTransformerBlock` and `FooSingleTransformerBlock`), then STEP B must have shown **a
separate denoiser/layer image for each one**. If a block type the code builds has **no
image**, that is a **FAIL** (an invisible variant).

---

## 6. Decision — PASS / FAIL / NEEDS-HUMAN

Decide with this table. Use the FIRST row that matches.

| Condition | Verdict |
|---|---|
| `oracle: MISSING`, or `MODELING FILES: NONE` | **NEEDS-HUMAN** — code not installed, cannot confirm against code. |
| Any blocking Sable net has findings (not all `[ ok ]`) | **FAIL** — list the net + findings. |
| `config_field_audit` warns on a plausibly architectural field and ownership is unresolved | **NEEDS-HUMAN** — record it under `#*IMP*`; do not call it a confirmed PASS. |
| `wiring_problems()` not empty | **FAIL (P0)** — list each dangling connector. |
| Any image violates the visual rubric (§3) | **FAIL** — name the image + which rule. |
| A code op is missing from the diagram, or a diagram op is absent from the code (§5) | **FAIL** — quote both sides. |
| A connector has a missing input (§5) | **FAIL** — name the glyph + the view. |
| A block type the code stacks has no image | **FAIL** — name the missing variant. |
| All blocking nets `ok`, config warnings triaged, oracle present, every image clean, every code op matched both ways, every connector has two inputs, every block type imaged | **PASS** |

When in doubt, do **not** call it PASS. Call it NEEDS-HUMAN and describe exactly what you
were unsure about.

---

## 7. What to report back (use this exact template)

```
SABLE + DABLE REPORT — <model name or id>

VERDICT: PASS | FAIL | NEEDS-HUMAN

1. SABLE (automated)
   oracle: present | MISSING
   nets: <list every blocking net and ok/FAIL; paste each FAIL verbatim>
   config_field_audit: ok | WARN <list and triage every unread dotted path>
   distinct views: <n>   PNGs: <n>
   wiring_problems(): []  | <the list>

2. DABLE — images
   <for each PNG you looked at: filename → "clean" or the rubric violation you saw>

3. DABLE — code (the oracle)
   modeling file(s): <paths printed in STEP C>
   block class(es) the model stacks: <names>
   per block, the forward() ops you found: <your STEP-C list, short>

4. CODE ↔ DIAGRAM reconciliation (the 5 things, both directions)
   blocks:      <matched? any code op missing / any diagram op invented?>
   arrows:      <every flow matches?>
   loops (×N):  <matches the ModuleList / for loop?>
   connectors:  <every ⊕ × ⊙ ‖ has two inputs? any missing?>
   splitters:   <fan-outs correct?>
   variants:    <every block type the code builds has an image?>

5. FINDINGS (if any) — one line each, with: what's wrong, the image OR the
   code line that proves it, and which direction (code→diagram or diagram→code).
```

---

## 8. Appendix — fast facts to sanity-check (optional but useful)

These are architecture facts the parser now **reads from the code** (not from a lookup table).
After STEP C, you can cross-check that the diagram got them right:

- **Norm placement** — read the layer `forward()`: pre-norm (norm before sub-layer),
  post-norm (norm after, before the add — e.g. OLMo-2), or **sandwich/double** (norm both
  sides — e.g. Gemma-2/3). The diagram's norm boxes must sit accordingly.
- **Parallel residual** — if attention and FFN take the **same** normed input and a single
  combined add (e.g. GPT-J, Phi, Cohere), the diagram must draw them **side by side into one
  `⊕`**, not stacked sequentially.
- **QK-norm** — if the attention builds `norm_q` / `norm_k` (RMSNorm or LayerNorm), the
  diagram must show a **"QK-Norm" chip** (it is a property/chip, not necessarily a drill box).
- **Single-stream fusion** (Flux-family) — read the single block's `forward()`:
  `torch.cat([attn, mlp])` → ONE shared projection = **concat_fused**; a fully fused parallel
  block = **parallel**; plain attn→FFN with no cat = **sequential**. The single-stream image
  must match.
- **Gate via norm** (Mochi) — if you see `hidden = hidden + self.norm(sublayer, gate)` (the
  gate folded into the norm), the diagram must show an **output norm box, NOT a `×`** there.
- **Conv FFN** (Sana) — if the block builds `self.ff = GLUMBConv`, the FFN drill must show a
  **gated conv Mix-FFN**, not a plain Linear MLP.

If any of these facts in the diagram disagrees with what the `forward()` actually does → that
is a **FAIL** (and the most valuable kind to catch).

---

### Reminder of the doctrine you are enforcing
**config.json → numeric/flag facts. The model's code (`forward()`) → the structure.** Your
whole job is to confirm the picture is faithful to that code. A green automated report is not
enough — you must read the code and look at the pixels, every time.


**IMPORTANT**
-> Do not only vet high-level connections. Be clinically cynical about the smallest mismatch:
if code does something the diagram does not show, or the diagram shows something the code
does not do, record it under `*IMP*`.

-> Evidence is mandatory. Every issue, even a slight suggestion, must cite both the HF code
location and the diagram image file.

-> IF IMAGES OR CODE ARE NOT SEEN, THE REVIEW IS NULL AND VOID. NEVER MISS THAT PART.

---

## 9. Anti-overlook protocol — how to beat a weaker "clean PASS"

This section exists because a lower-intelligence audit can produce a confident PASS while
missing the exact thing that matters. The failure mode is usually not stupidity in the
obvious path; it is stopping one layer too early. The diagram "looks like attention" or
"looks like MoE", so the audit accepts the family template and never checks the small
family-specific code signature hiding inside the leaf.

Run this protocol on every model whose report says `None`, `clean`, or `PASS`.

### 9.1 First invalidate bad provenance

Before architecture reasoning, prove the report belongs to the folder in front of you:

- The folder must contain the saved **HTML**, every rendered **PNG**, `MANIFEST.txt`, and
  `report.md`. Missing PNGs = Dable was not done. Missing HTML = cards/facts/click targets
  cannot be re-vetted from the artifact. Either case prevents a fully confirmed PASS.
- The report's image names must match the current folder. If the report mentions
  `13__sparse_indexer.png` but the folder contains `07__mla_indexer.png`, the report is
  stale. Stale report = **NEEDS REPORT REGEN** even if the model might be correct.
- Count PNGs against `MANIFEST.txt`. A "0 PNG" folder is a null/void review.
- Open the actual PNGs as images. Do not infer pixels from HTML/SVG.

Write a finding like:

```
#*IMP*
- Report provenance failure: report lists <old image>, current folder has <actual image>.
  Review cannot be a clean PASS until regenerated.
```

### 9.2 Never stop at the wrapper file

Modern multimodal models are wrappers. The truthful structure often lives in delegated
modules:

- `AutoModel.from_config(config.vision_config)` → follow it into the concrete vision model
  (`SiglipVisionModel`, `PixtralVisionModel`, etc.).
- Projectors are real modules too. Audit `MultiModalProjector.forward()`, not just the
  top-level `forward()`.
- Text backbones can be delegated separately from the multimodal wrapper. A correct text
  stack does not excuse an incorrect vision stack.

Examples of overlooked delegated leaves:

- SigLIP/PaliGemma patch embedding: code is Conv2d first, then flatten/transpose, then
  position embedding. A diagram saying `Flatten patches → Linear / Conv2d` is wrong.
- Pixtral/Mistral3 patch path: code is Conv2d → flatten → `ln_pre` → rotary position
  embeddings passed to attention. A block saying `Add positions rope 2d` fabricates an
  add if the code only computes RoPE embeddings.
- Mistral3 projector: code is `norm → patch_merger → linear_1 → act → linear_2`; a
  two-linear MLP diagram omits the patch merge and norm.

### 9.3 Audit attention between projection and scores

A weak audit sees Q/K/V and RoPE and stops. Do not stop there. For every attention class,
read from the Q/K/V projections down to the attention call and list every operation between:

- Q/K/V normalization (`q_norm`, `k_norm`, `v_norm`, `norm_q`, `norm_k`)
- RoPE or other position embedding application
- post-RoPE query/key scaling
- cache update
- query/key/value concatenation or splitting
- attention mask/sparse-indexer selection

Rules:

- QK-norm is Tier-3. It can be a chip/card fact, not necessarily a box.
- If the code also normalizes V, the card must not only say QK-Norm; the V norm is a
  separate code signature and must be surfaced or consciously abstracted.
- Post-RoPE query scaling is not "just RoPE"; it is its own attention fact.

Concrete signatures to look for:

```python
query_states = self.q_norm(self.q_proj(...))
key_states = self.k_norm(self.k_proj(...))
query = attn.norm_q(query)
key = attn.norm_k(key)
value_states = self.v_norm(value_states)
query_states = query_states * get_llama_4_attn_scale(...)
```

If the card/diagram does not surface that fact, write a code→diagram finding.

### 9.4 Audit router post-processing, not just Top-k

A router is rarely only `Linear → Top-k`. After top-k, look for:

- softmax before selection
- top-k weight renormalization
- scale factors (`routed_scaling_factor`, per-expert scale, constants drawn next to `×`)
- bias used for selection but not weight gathering
- grouped top-k steps

Concrete signatures:

```python
router_probs = softmax(...)
router_top_value, router_indices = torch.topk(...)
router_top_value /= router_top_value.sum(...)
topk_weights = topk_weights * self.routed_scaling_factor
top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
```

If the diagram has only `Top-k → weights`, it may be missing a real `renormalize` or scale
operation. `renormalize` is a thin box; a constant scale can be a labelled `×` connector.

### 9.5 Audit expert storage layout: fused is not split unless declared

Do not assume every gated expert is three separate `gate_proj`, `up_proj`, `down_proj`
modules. Many HF implementations store a fused `gate_up_proj` and split the result:

```python
self.gate_up_proj = nn.Parameter(...)
gate, up = linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
```

A diagram showing separate `Linear(gate)` and `Linear(up)` can be acceptable only if the
split view is explicitly declared as a conscious composite abstraction in YAML. If not,
it is diagram→code fabrication. The lower model usually misses this because the math is
equivalent on a whiteboard, but this library is supposed to be honest about the code
signature too.

### 9.6 Audit norm class and final norm separately

Norm placement and norm class are separate facts:

- A diagram can place a norm correctly but label it with the wrong class.
- Final model norm is easy to miss because it is outside the repeated block.

Check:

```python
self.ln_f = nn.LayerNorm(...)
self.norm = RMSNorm(...)
hidden_states = self.ln_f(hidden_states)
```

If the diagram says `Final RMSNorm` and code says `LayerNorm`, that is a diagram→code
failure even if every residual route is correct.

### 9.7 Audit block labels as aggressively as block topology

A diagram can be structurally correct and still violate the design rules:

- Block labels must be bare operation/module names.
- Dimensions, counts, head ratios, channel counts, token counts, "top-8", "768-d", etc.
  belong on cards as chips, not painted on block labels.
- `Linear / Conv2d` is usually a red flag: read code and name the real op.
- A connector glyph must be clickable with a one-line card. Do not use `static=True` to
  silence real connectors.

Treat label leakage as a real finding. It is not a prettier/later problem; it corrupts the
architecture reading by mixing operation identity with facts.

### 9.8 Audit sparse/indexer paths as real submodules

Sparse attention/indexer blocks are not badges. If the code has an indexer class, audit it
like attention:

- projections
- norms
- RoPE
- scoring nonlinearity
- weighting/head aggregation
- masking
- top-k
- cache/reuse path

Also check layer variants. If the code has `indexer_types[layer_idx] == "shared"` and a
`prev_topk_indices` path, the report must prove whether the config uses full/shared
variants and whether each variant has a rendered image.

### 9.9 Explain why the miss happened

Every addendum should include one sentence beginning:

```
Why the lower audit missed it:
```

Use this to make the report teach future auditors. Good reasons are:

- stopped at wrapper source and did not follow delegated `AutoModel`
- checked Q/K/V + RoPE but not lines after RoPE
- accepted conceptual MoE math and missed fused implementation
- trusted stale image names
- checked topology but not norm class / final norm
- checked visible PNG but not HTML/card facts, or vice versa

This sentence is not blame; it is a guardrail. It tells the next model exactly where its
attention should go when the diagram looks "obviously fine."

### 9.10 Required addendum format for re-vetting old reports

Append, do not erase the old report:

```md
---

## Codex independent re-vet addendum — <date>

VERDICT UPDATE: PASS CONFIRMED | FAIL | NEEDS-HUMAN | NEEDS REPORT REGEN

#*IMP*
- <image file> shows <diagram fact>, but <HF file>:<line> does <code fact>
  (<code→diagram or diagram→code>).

Why the lower audit missed it: <one precise reason>.
```

If no issue is found, still say what you checked:

```md
VERDICT UPDATE: PASS CONFIRMED

#*IMP*
- No additional discrepancy found. I opened <N> PNGs and checked <HF source>. The suspected
  pitfall (<QK norm / fused expert / delegated vision / router renorm>) does not apply or is
  already surfaced correctly.
```

---

## 10. Meta-Sable: audit the library approach, not only one model output

This is the step that stops the same class of issue from coming back under a new model name.
When a model has an IMP finding, do not immediately ask "what one-off option fixes this
repo?" Ask which layer of the library's approach failed:

1. **Config vocabulary failed** — the config declared the fact, but the parser did not read
   its spelling. Surgical fix: add an alias / type label / ignored-field decision in
   `everchanging/`, then add a corpus case.
2. **Config cannot know the fact** — HF hardcodes the fact in the modeling class, not the
   config. Surgical fix: add a source-evidence detector and mark the rendered fact as
   code-derived. Do not invent a config default.
3. **IR cannot carry the distinction** — the parser discovered the fact, but `AttentionSpec`,
   `FFNSpec`, `LayerSpec`, `extras`, or the layer-group `signature()` cannot express it, so
   the renderer collapses variants or drops it. Surgical fix: extend the IR contract first,
   then renderer/cards/tests.
4. **Renderer/card projection failed** — the IR has the fact, but one of the three projections
   (SVG, JSON, cards) does not surface it, or surfaces it in the wrong tier. Surgical fix:
   fix the projection from the op-graph; do not duplicate facts by hand.
5. **Conformance net is blind** — automated Sable passes because the mismatch is below its
   granularity. Surgical fix: add a new reusable evidence marker / fact-conformance rule /
   wiring rule / label lint rule, not a report-only warning.
6. **Human pixel review caught it** — the graph model is correct, but the rendered pixels
   mislead. Surgical fix: fix layout/routing and add a representative image/corpus lock.
7. **Report provenance failed** — stale image names, no HTML, no PNGs, or missing manifest.
   Surgical fix: regenerate the report; do not reason from stale artifacts.

The lower audit usually fails because it treats the rendered output as the whole truth. The
rendered output is the review target, but the **source of truth is a three-way contract**:

```text
HF config + HF modeling forward()
        ↓
parser / evidence detectors
        ↓
IR op graph
        ↓
SVG + JSON + cards + HTML + PNGs
```

If any arrow in that chain is unproven, the review is incomplete.

### 10.1 What the current code approach covers well

The current architecture is strongest when the model's structure is expressible as known
transformer/diffusion primitives plus declared config fields:

- decoder stacks with MHA/GQA/MQA/MLA attention, RoPE/NoPE/sliding/global masks, dense FFN,
  gated FFN, MoE, shared experts, MTP, PLE, cross-layer KV sharing, and common norm placements;
- diffusion transformer denoisers with dual-stream, single-stream, concat-joint, cross-attn,
  AdaLN gates, QK-norm, axial/3D RoPE, scheduler/VAE/text-encoder shell, and UNet skeletons;
- routing variants already represented in `FFNSpec.routing`: grouped selection, noaux bias,
  top-k renormalization, and routed scaling;
- visual/coupling safety nets: click coupling, dangling connector detection, duplicate
  `url(#id)` detection, label lint, view coverage, op conformance, wiring conformance, and
  fact conformance.

This is not "family-specific support" in the old brittle sense. The intended happy path is:
new config spelling → YAML vocabulary; new recurring code-only fact → source-evidence detector;
new reusable topology → IR extension and shared renderer view.

### 10.2 Where the current approach will struggle on a new model

These are not bugs by themselves; they are known stress points. A Sable/Dable review must
explicitly check them before saying a model is clean.

#### A. Source-only facts that config does not declare

HF often hides architecture in the modeling class:

- unconditional Q/K/V norms;
- post-RoPE query scaling;
- attention algorithm choice hidden in an attention processor;
- DiT FFN activation/gating;
- fused single-stream topology;
- projector internals;
- final norm class;
- router renormalization/scale;
- per-expert fused `gate_up_proj`;
- sparse-indexer variants.

If a fact is absent from config, absence is not a negative. The correct outcome is either
"source proves it and the diagram marks it code-derived" or "source unavailable, review is
NEEDS-HUMAN / oracle missing." Never turn config silence into a confident diagram claim.

#### B. Coarse conformance is not wired conformance

`op_conformance` compares a **presence set** of op kinds from `forward()` against the rendered
layer group. It is intentionally robust, but it cannot prove:

- exact order of operations;
- exact port/operand of a connector;
- whether Q, K, and V each receive the right norm/scale/RoPE;
- whether a helper function performs important work hidden behind one call;
- whether a norm is RMSNorm vs LayerNorm unless a fact rule checks it;
- whether router weights come from raw scores after biased selection;
- whether a conceptually equivalent split view is faithful to a fused implementation.

Therefore a clean conformance net means "no coarse op-kind mismatch found." It does **not**
replace reading the relevant `forward()` and tracing arrows/ports manually.

#### C. Resolver scope can stop too early

The source resolver usually finds the main Transformers/Diffusers modeling file and, for
diffusion, augments common sibling files like `attention.py`. That is not enough when the
architecture delegates into:

- `AutoModel.from_config(...)` vision/audio/text towers;
- separate projector classes;
- attention processor classes;
- helper functions outside the main block class;
- wrapper configs where the text model lives under `text_config`, `language_config`,
  `llm_config`, `thinker_config`, or another nested component;
- remote/custom modeling code not installed locally.

For multimodal reports, never stop after the text backbone. Follow every delegated tower and
projector that contributes visible blocks.

#### D. IR grouping can hide variants

`LayerSpec.signature()` decides which layer groups get a distinct rendered view. If a new
architectural distinction is parsed but not included in the signature, two different block
types can collapse into one image. That is a silent review killer: the missing variant is not
available for Dable inspection.

Every new structural fact must answer:

- does it change the layer group's signature?
- does it require a separate view variant?
- does `tests/test_coverage.py` exercise that variant?
- does Sable's image manifest show the expected number of distinct diagrams?

If the answer is unclear, assume the variant may be hidden.

#### E. Semantic modality paths are not full source conformance

Vision/audio/video paths are currently extracted mainly as semantic stages from config:
pixels → patch embedding → encoder → reduction/projector → tokens/fusion. That is useful,
but it is weaker than a block-by-block conformance pass over the actual tower/projector
`forward()`.

When reviewing a multimodal model, verify at least:

- patch embedding op order: `Conv2d` vs `Linear`, flatten order, class/register tokens;
- pre/post encoder norms;
- vision attention position scheme: learned table, 2D RoPE, multimodal RoPE, or none;
- vision MLP kind: dense, gated, fused gate-up, conv;
- projector internals: norm, patch merger, activation, pooling, resampler, concat;
- fusion mechanism: placeholder replacement, soft prefix tokens, unified grid stream,
  cross-attention states, or adapter layers.

If the diagram only shows a generic stage but the code has named substructure that affects
the mental model, report it as a structural gap.

#### F. Labels can pass topology but fail understanding

A diagram can be code-faithful and still teach the wrong thing if labels leak facts or hedge:

- `Linear / Conv2d` means the real op was not resolved.
- `CLIP (768-d)` puts dimensions on the block instead of the card.
- `Top-8` paints a config value onto an op label.
- `Add positions rope 2d` is usually a fabricated operation unless the code really adds a
  tensor at that point.

Label findings are not cosmetic. They indicate the codebase has not separated operation
identity from facts cleanly enough.

### 10.3 Codebase-level checks to run while vetting any model

Run these checks in addition to looking at the model's report.

#### 1. Inspect first-class unread config findings

Sable now captures parser accesses across nested parses and prints unread fields as dotted
paths under `config_field_audit`. Start with `report.summary()`. Use debug output only when
you need to trace the individual lookups:

```bash
MODEL_UNFOLDER_DEBUG=1 python - <<'PY'
from model_unfolder import unfold
MODEL = "replace/me"
unfold(MODEL).to_ir()
PY
```

Classify every unread key:

- architecture-bearing → add parser support or YAML alias;
- non-architectural metadata → add to `everchanging/transformer/ignored_fields.yaml`;
- unknown → mention in the report as a risk, do not silently ignore.

An explicitly positive architecture flag (`true`, a non-empty block list, a positive
count, or a named module type) is never an ignore candidate merely because the current
picture looks plausible. Follow it into the owning class and compare the operations it
enables. For example, `_vae_config.mid_block_add_attention: true` must be reconciled with
the decoder's `conv_in` and `mid_block` calls; otherwise a stage-only VAE picture can omit
an entire attention-bearing region while every high-level pipeline arrow still looks right.
Conversely, config silence is not proof that a source default is false: only render that
structure when config or source evidence establishes it.

Ownership is scoped. `_scheduler_config` and fetched `_text_encoder_configs` are opaque to
the parent diffusion denoiser audit because separate adapters/components own their internals;
`_vae_config` remains recursive because the diffusion renderer draws the VAE decoder. Do not
make an entire nested config opaque merely to remove warnings—only a real component boundary
earns that classification.

#### 2. Diff config facts against source facts

For every important block, make a two-column note:

| Question | Must be answered from |
|---|---|
| Does config declare it? | `config.json` / nested config |
| Does source hardcode or override it? | concrete `forward()` / `__init__` |
| Does IR carry it? | `Diagram(...).to_ir()` |
| Does SVG show it? | generated HTML/PNG |
| Does card explain it? | clicked card in HTML |

If the answer changes between columns, that is the surgical fix location.

#### 3. Check every same-op/different-semantics axis

These are invisible to simple op-kind matching and must be vetted manually or promoted into
`fact_conformance`:

- attention: softmax vs linear vs recurrent/SSM-like; RoPE vs learned absolute vs NoPE;
  q/k/v norm placement; post-RoPE scaling; cache update; sparse/indexer path;
- FFN: dense vs gated vs GeGLU/SwiGLU; fused gate-up vs split projections; activation clamp;
- MoE: score function, top-k source, grouped routing, noaux bias, renormalization, routed
  scale, shared experts, fine-grained expert scale;
- norms: class, placement, final norm, post-attention/post-FFN sandwich norms;
- diffusion: AdaLN gate as `×` vs gate folded into norm, dual/single/concat/cross-attn
  topology, text conditioning as rail vs joined sequence.

When you find a new recurring same-op/different-semantics miss, add a reusable conformance
marker. Do not leave it as "remember to check later."

#### 4. Verify the implementation surface, not only the visible architecture surface

For every finding, decide whether the codebase needs one of these surgical additions:

- YAML vocabulary: alias/type/label/abstraction/ignored field;
- IR field or signature change;
- parser/evidence detector;
- renderer view/card projection;
- conformance net;
- coverage corpus case;
- pixel/layout regression;
- report regeneration only.

If the report says "add option X", reject it until it is mapped into one of these surfaces.
Options without a surface are how support turns into clutter.

#### 5. Treat "PASS but no source/image" as null, not clean

A model folder with no PNGs, no HTML, stale manifest, or missing oracle is not a clean pass.
It is unreviewed. The report must say which artifact is missing and must not claim code
accuracy.

### 10.4 The recurring root problem, stated plainly

The library keeps needing new support because modern HF models are not described by config
alone. The config gives dimensions and many switches, but the architecture often lives in:

- Python defaults inside `__init__`;
- delegated submodules;
- helper functions;
- attention processors;
- fused parameter layouts;
- wrapper configs that hide the real text/vision model;
- pipeline/component splits in Diffusers.

So the scalable approach is not "add more per-model options." The scalable approach is:

1. keep config vocabulary data-driven in `everchanging/`;
2. promote recurring source-only facts into evidence detectors;
3. extend the IR only for reusable structural distinctions;
4. force every distinction through SVG, JSON, cards, HTML, PNG, and tests;
5. bless a corpus case so the same miss cannot recur silently.

Anything else is a local patch, not model support.
