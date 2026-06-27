# Sable + Dable Playbook — confirm a model's diagram against its real HF code

**Read this whole file once, then follow it top to bottom. Do exactly what each STEP says.
Run every command. Look at every image. Do not skip steps. Do not guess.**

## 0. What you are doing (the one idea)

`model_unfolder` turns a HuggingFace model's `config.json` into an architecture diagram.
Your job is to **prove the diagram is correct** by checking it two ways:

1. **SABLE** = run automated checks (7 "nets"). Pass = every net reports nothing.
2. **DABLE** = look at the rendered **images** with your own eyes, AND read the model's
   **real HuggingFace code**, AND confirm the picture matches the code.

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

## 2. STEP A — SABLE (the 7 automated nets)

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
    [       ok] op_conformance
    [       ok] wiring_conformance
    [       ok] fact_conformance
    [       ok] label_lint
  visual review: PENDING  (inspect the gallery against report.rubric)
```

### How to read it — exact pass/fail rules

Check these in order. If ANY fails, the model **FAILS Sable** — write down which net and its
findings, then still continue to STEP B/C/D so your report is complete.

1. **`oracle: present`** — REQUIRED.
   - If it says `oracle: MISSING …`, the model's HuggingFace code is **not installed**, so the
     code-vs-diagram checks were **skipped**. This is a **blocked / NEEDS-HUMAN** result — you
     cannot confirm against code. Report it as: "oracle MISSING — conformance could not run."
     (You may still do the visual pass, but you cannot fully confirm.)

2. **Every net shows `[       ok]`** — REQUIRED. A net with `FAIL (n)` lists its findings
   underneath (lines starting with `·`). Copy those findings verbatim into your report.

What each net means (so you can describe a failure correctly):

| Net | What a finding means |
|---|---|
| `click_coupling` | a clickable block opens a card/view that does not exist (broken link). |
| `dangling_connectors` | a `⊕` `×` `⊙` `‖` connector is drawn with a missing input (see §5). **P0.** |
| `unique_ref_ids` | two arrowheads/markers share an id → arrowheads can vanish in the browser. |
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
| Any Sable net has findings (not all `[ ok ]`) | **FAIL** — list the net + findings. |
| `wiring_problems()` not empty | **FAIL (P0)** — list each dangling connector. |
| Any image violates the visual rubric (§3) | **FAIL** — name the image + which rule. |
| A code op is missing from the diagram, or a diagram op is absent from the code (§5) | **FAIL** — quote both sides. |
| A connector has a missing input (§5) | **FAIL** — name the glyph + the view. |
| A block type the code stacks has no image | **FAIL** — name the missing variant. |
| All nets `ok`, oracle present, every image clean, every code op matched both ways, every connector has two inputs, every block type imaged | **PASS** |

When in doubt, do **not** call it PASS. Call it NEEDS-HUMAN and describe exactly what you
were unsure about.

---

## 7. What to report back (use this exact template)

```
SABLE + DABLE REPORT — <model name or id>

VERDICT: PASS | FAIL | NEEDS-HUMAN

1. SABLE (automated)
   oracle: present | MISSING
   nets: <list every net and ok/FAIL; for each FAIL, paste its findings verbatim>
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
