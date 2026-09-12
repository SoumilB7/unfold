# S6 instance-oracle pilot artifacts

These files are experimental evidence only. Nothing under `physics/` is
imported by a production adapter, fact producer, IR builder, parameter
estimator, or renderer.

Generate the frozen eight-model set from the checked-in corpus configs:

```bash
python3 scripts/generate_s6_pilots.py
```

Each pilot directory contains the exact build request, a typed meta-device
inventory, and one or more recipe-qualified FakeTensor observations. Every
successful construction records the resolved runtime class, constructor route,
config hash, modeling-source hashes, MRO library versions, build flags, and
deterministic environment. Each observation additionally records its named
recipe and exact library versions.

The two non-successful whole-path observations are intentional evidence, not
skips:

- Qwen2-VL's full vision tower reaches data-dependent token grouping and is
  `ExecutionUnresolved`; its text path and vision block MLP have separate
  positive observations.
- DBRX's full forward reaches a router-dimension rejection in the exact
  Transformers 5.12.1/config pairing and is `ExecutionFailed`; its first-block
  attention has a separate positive SDPA observation.

`pilots/manifest.json` hashes every request, inventory, and observation file and
summarizes their typed statuses. A missing trace call is never negative
evidence, and no artifact in this directory is a production architecture fact.
