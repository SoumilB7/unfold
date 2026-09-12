# S11 module map — carried open items

Status: planning record only; S11 implementation has not started. Source: S8.1 owner verdict on `c7ec5526` and Soumil's 2026-09-10 instruction. Owner: executor.

| Item | Present dependency | Intended boundary / closure evidence | State |
|---|---|---|---|
| N1 — cache hooks | `model_unfolder/evidence/context.py`, `document.py`, `sources.py` and `parser.py` import `physics.result_cache`; two imports are at module level | Consolidate the hooks at one parser-owned injection point, or a `model_unfolder.cache` boundary. Evidence modules should no longer depend directly on the physics cache package. Preserve exact cache identity/replay and reader authority. | OPEN — S11 |
| N1 — package namespace | The wheel ships a top-level `physics` package | Move it under `model_unfolder.physics`, updating packaging/imports and verifying installed-wheel imports from a neutral directory. | OPEN — S11 |
| N2 — temporary test boundary | S6 permits the five parent cache coordination hooks because of N1 | Retire that temporary allowlist with N1 while preserving rejection of parent model construction and private/unknown physics imports. | OPEN — S11 |

The local stale editable-install observation is context from the owner ruling, not an S8.1 implementation task. This map records future work only; no modules were moved or cache APIs redesigned during closure.
