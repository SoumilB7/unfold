# Initial independent S8 authority review — RETURN

Reviewer: `/root/independent_review`, independent of the original implementation and the current executor. Reviewed implementation: `bb48a31e336c1721f0a111862bd081acf390bb25`; immutable checkout used for reproductions: `/private/tmp/unfold-s8-stop-review`, HEAD `349c02d2fcc41139ed770a1fe72233da4d5f6250` (implementation plus its stop receipt).

This is an initial bounded review, not a final S8 verdict or permission to bless. The executor's primary checkout is changing; these findings and line references describe the named initial implementation. The reviewer changed no production code, tests, gates, or baselines and ran no pytest lane.

The subsequent bounded claim-contract audit is recorded in [supplementary-review.md](supplementary-review.md): R5 (FFN callable overrides) and R6 (same-class spatial occurrence misbinding), with standalone reproductions and persisted results.

The later working-source audit of the new local-port reader is recorded separately in [local-port-review.md](local-port-review.md): R7 (reaching-definition coverage) and R8 (starred-unpack slot), with the checked source hash. Those two findings describe that working file, not `bb48a31`.

## Reproduction

Run from that immutable checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/reproduce.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/results.json
```

The command exited 0. Assertions confirm that the four defects reproduce; this is a successful counterexample run, not a green implementation verdict. `results.json` records the checkout SHA, exact results, source fixtures, and equal tracked-file fingerprints before and after. The concat fixture reads the existing test's literal source strings as data; it never imports or executes the test module. Temporary fixtures are deleted after each probe.

## Findings

### R1 — stale member identity survives escaped `self`

Earliest false producer: `model_unfolder/evidence/unet_cell_connections.py:54–84`, particularly the literal-`self` argument check at lines 76–78. The member-stability check accepts:

```python
alias = self
helper(alias)
value = self.first(value)
return self.second(value)
```

The probe returns both `member_stays_bound: true` and `direct_edge_found: true`. An unexamined helper can replace `second`, so the source call no longer has a proven connection to the previously constructed occurrence. Close supported aliases and escapes (including container-mediated ones), or leave that affected connection unresolved. A replacement poison and a harmless alias control should distinguish the cases. This is the §1g/§1i boundary between an instance's construction identity and source-proven call wiring.

### R2 — serialized class names become applied-function authority

Earliest new semantic promotion: `model_unfolder/evidence/primitive_semantics.py:68–70` and `model_unfolder/evidence/runtime_source.py:112–116`. The former looks up `(module, qualname)` strings. The latter treats that result as a proven framework operation.

A custom `nn.Module` whose `forward` returns `value + 10`, with `__module__ = "torch.nn.modules.activation"` and `__qualname__ = "SiLU"`, is not the canonical `torch.nn.SiLU` type. Nevertheless, the probe receives `("activation", "SiLU", "silu")`; actual output at zero is 10, while canonical SiLU returns 0.

The existing S7 prefix is address metadata; the S8 promotion introduces the false mechanism claim. Establish actual type membership in the closed framework set in the isolated witness, then preserve that result in typed provenance. Names alone cannot certify it. Instance or class forward replacements must not inherit canonical operation authority merely from a class address. This is execution-order §0 identity and executor law 1, not a request for model-name exceptions.

### R3 — an unrelated concat can occupy a connection proof

Earliest false producer: `StageJoinConnection.__post_init__`, `model_unfolder/evidence/unet_cell_mechanism.py:64–73`. Its checks do not establish that its reaching-definition bindings terminate at its `join`. The new `UNetJoinClaimProof.__post_init__`, `unet_claims.py:185–196`, checks that the cited operation is concat but does not repair that missing link.

Fixture:

```python
ignored = cat([side, side], dim=0)
joined = cat([value, side], dim=1)
value = unit(joined)
```

Replacing the valid DTO's `join` with the unused concat succeeds while its original bindings remain attached. The persisted reproduction checks this DTO constructor; it does not claim a complete forged-`EvidenceFact` end-to-end run. Inspection establishes that the wrapper's current validation adds only the concat protocol check.

Validate the exact authoritative records and the full reaching-definition chain at the qualification boundary, or require a reader-issued proof sealed to that exact claim. Add the wrong-concat poison. A typed wrapper around an unrelated operation's existence cannot satisfy a connection claim under §1i.

### R4 — possible dependency becomes conditioning provenance

Earliest false producer: `_origin_snapshots` / `_expr_origins`, `model_unfolder/evidence/unet_cell_mechanism.py:242–270`, used by `_conditioning` at line 559. These carry syntactic argument origins through arbitrary helper results. The new semantic promotion is `unet_claims.py:354`; `unet_projection.py:137–139` presents an additive side input from the formal.

For `ignore(arg): return 7`, this forward body yields an additive-conditioning claim from `side`:

```python
hidden = value * 2
detached_side = ignore(side)
hidden = hidden + detached_side
return value + hidden
```

The helper explicitly discards its argument. Executions with `side=0` and `side=999` both return 10 for `value=1`. The source proves an argument entering the helper and a helper result entering addition; it does not prove that the returned value derives from `side`. Prove the supported helper return route, or expose its opaque boundary. Do not promote a may-dependency summary to qualified provenance.

## Scope and lawful continuation

A branch-aware opaque operation around the optional FreeU call is lawful without first proving `getattr` defaults. Known formal-to-call-argument and call-result-to-concat connections can surround an explicitly unresolved operation. Keep the branches and ports visible; never bypass that operation or assert that every returned value depends on every input.

The latest user instruction allows the original 240 SDXL execution-unresolved occurrences to remain class 1 with concrete reasons after the family investigation. Matching positive observations for 1,690 occurrences must remain positive; source mismatches require explicit disposition. Exhaustive all-path closure is not required. Cross-attention may be proven, chipped, or proven not constructed. The skip and conditioning structure still needs its required connection accounting.

Fact qualification scope must enumerate the facts actually projected by the UNet cutover. Unchanged outer CLIP/VAE/scheduler reader debt may remain explicitly S9-owned; it cannot silently qualify new UNet claims. All other unfinished C-8 evidence, differential enumeration, independent final review, and Soumil's output-delta approval remain required. No approval or re-bless is recorded here.
