# S5 pre-bless release receipt

Status: **release candidate prepared; independent review, HTML-delta approval,
manifest bless, publication, and deployment are still withheld**.

- implementation tree: `fb3e2fa857b71653a4ffe622660dfaab893d4c4c`
- coordinator run: `7c00255875`
- lane schedule: strictly serial; bounded parallelism exists only inside one lane
- Space preparation commit: `ff0e40a` in the sibling `hf` repository, local and
  unpushed
- package/tag/deployment: not performed

The machine receipt and all original lane logs are in this directory. The
candidate manifest is deliberately stored here rather than promoted to
`tests/preservation_expected_manifest.json`; that promotion requires Soumil's
approval and an independent verdict.

## C-5 release evidence

The locally built wheel is `model_unfolder-0.3.0-py3-none-any.whl`, SHA-256
`ba2bd7c8ee6a5e05b27b3bfffef4a22d1ee38277a8bfc623ea6b0e7646baffaa`.
A fresh Python 3.12.10 virtual environment installed that wheel and its declared
dependencies from package metadata, then rendered all **29/29** reviewed frozen
witnesses. It did not contain PyTorch. The resolved environment used
`transformers==5.16.1`, `diffusers==0.40.0`, and
`huggingface_hub==1.30.0`; see `clean-install.json` and `pip-freeze.txt`.

This proves the built candidate. It does **not** claim that
`pip install model-unfolder==0.3.0` from PyPI works yet: Soumil has not published
the release. The post-publication clean install and the live five-model Space
check remain owner actions after acceptance.

The README is pinned to the exact `coverage.json` denominator: 29 reviewed
corpus witnesses, 15 unseen robustness witnesses, and **621 proven / 241
visibly flagged / 0 silent**. It names the known incomplete SD3.5/PixArt,
SDXL, Jamba, Qwen side-reader, and parameter-estimation surfaces. The version
literal is owned by `pyproject.toml`; runtime `__version__` reads installed
distribution metadata.

The public examples are regenerated only from eight frozen reviewed fixtures.
`scripts/generate_examples.py --check` is blocking in CI. The example manifest
records each fixture, rendered name, and content hash. `deepseek-v3.html`
contains `DeepSeek-V3` and `DeepseekV3ForCausalLM`, and does not contain the old
Gemma identity.

The Space requirement is unpinned and its app catches `UnfoldError` as a typed,
persistent refusal with the explicit statement that no fallback architecture
was drawn. Its local poison test passed. No Space push or deployment occurred.

## Exact preservation delta awaiting approval

The complete 29-witness candidate manifest differs from the currently blessed
manifest in exactly two places:

| witness | changed surface | evidence-level cause |
|---|---|---|
| `granite-3-0-8b-instruct` | `html_meta` only | the two already-approved `config_accessed_unprojected` rows are shown as one readable producer-authored summary, with both exact rows under a disclosure |
| `stable-diffusion-xl-base-1-0` | `html_meta` only | the eleven already-approved `config_accessed_unprojected` rows are shown as one readable producer-authored summary, with all eleven exact rows under a disclosure |

All other 27 witnesses are byte-identical. For Granite and SDXL, `ir`,
expanded JSON, exact warning strings, facts, views/SVGs, gallery pixels,
parameters, inputs, ledgers, and versions are byte-identical. The earlier CSS
implementation changed every witness and was rejected before review; styling
is now local to the disclosure nodes.

Public `to_ir()["warnings"]` remains `list[str]`. Each exact audit warning is a
real immutable string subtype carrying producer metadata only in memory. This
preserves equality, `startswith`, `join`, JSON, copy/deepcopy, and existing
consumer behavior while allowing the HTML renderer to group rows without
reading evidence extras or interpreting receipt prose.

## Committed-tree bracket

The serial isolated-worktree coordinator collected **4,066** tests. It passed:

- focused: **346 passed**;
- U2 authority: **44 passed**;
- exhaustive non-preservation remainder: every globally collected node ran
  exactly once; all batches passed;
- static: clean across 13 changed Python files;
- collection: **4,066**;
- fingerprint/quiescence: coordinator, source artifacts, and every lane were
  identical before/after.

Preservation was the sole red lane: **50 passed / 2 failed**, exactly Granite
and SDXL. This is the required no-self-bless stop, not an unexplained failure.

## Defects caught before this receipt

1. Global disclosure CSS moved `html_meta` for all 29 witnesses. The rules were
   replaced with disclosure-local styling; the delta contracted to two.
2. The first typed transport changed public warning rows from strings to dicts.
   The exhaustive suite found five existing-consumer failures. The transport
   was redesigned as string-compatible rather than weakening those tests.
3. The first immutable string subtype could not be reconstructed by
   `deepcopy`. Candidate-manifest generation caught it; an explicit
   reconstruction contract and poison now pin the fix.
4. The previous coordinator overlapped full and preservation pytest campaigns,
   contrary to executor law 9. Lanes are now strictly sequential, with a poison
   pinning the invocation order.

## Review boundary

Soumil/reviewer must approve or return the two-row HTML presentation delta.
On approval, an independent verdict must be persisted, the candidate manifest
must be regenerated and re-compared, and the final committed-tree bracket must
reach zero failures before any push. Soumil alone tags, publishes to PyPI, and
deploys the Space.
