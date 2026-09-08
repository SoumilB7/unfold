# Actual UNet latency gate

The canonical committed-tree command remains `python3 scripts/verify_commit.py --commit HEAD --focus <affected-test>`. It now includes the mandatory serial `latency-unet` preflight lane, alongside every existing focused, authority, collection, static, full and preservation lane. The gate does not run models during ordinary pytest: `tests/test_s81_latency.py` checks synthetic receipt integrity and budgets only.

To run the same actual gate directly in an exclusive measurement slot:

```sh
python3 scripts/profile_s81_latency.py --repo /absolute/frozen/checkout --output /private/tmp/unique-latency-receipt
```

The output must be new and outside the measured checkout. No baseline is written. A failed outcome is retained and stops subsequent launches. Afterward, an artifact-only check is available:

```sh
python3 scripts/profile_s81_latency.py --check --output /private/tmp/unique-latency-receipt
```

The two exact input envelopes are SDXL base 1.0 from `tests/sable_test_corpus` and SD-v1-4 from `tests/unseen_model_configs`. Each target has three serial cold/warm pairs, twelve fresh Python processes in total. Every cold process uses a new empty **enabled result cache**. Its warm partner uses the exact populated directory in a fresh process. This is not an OS-cache-cold claim: initial source hashing can warm filesystem pages. The location override isolates the product's default cache; it does not enable an otherwise disabled product feature. A disabled or ineligible inventory cache cannot pass.

The seven original S2 targets, measured records and 9-second legacy cold budget remain unchanged. The separate `unet_instance` section declares the owner thresholds: cold median at most 30 seconds, warm median at most 9 seconds. These are acceptance thresholds, not invented measured baselines. All three samples in each mode must succeed; a median never hides a failure or missing sample.

The timer retains the S2 explicit import boundary: torch, transformers, diffusers and model_unfolder are imported outside the unfold budget. Every lazy import still performed by public `unfold`, cache identity computation, dependency capture and cache validation remains inside. The diagnostic context import and delegating request observer are conservatively included in the budget too; the direct public call gets a nested duration for transparency. Nothing runs a model or validates its cache before the timed call. CPU count and host load averages (1/5/15 minutes) are retained at sample and campaign boundaries outside the budget, with explicit unavailability where unsupported. They never normalize timings or discard a sample. Full child-process wall time includes startup, all imports, pinning, unfold, HTML and artifact recording. Actual public HTML generation is measured separately and the raw page is retained.

The checker reads actual input, pre-render IR, quantities, HTML, runtime pins and cache entry bytes. Cold must record an inventory store and no hit. Warm must record an actual hit at the same request key and immutable entry SHA as its own cold partner, with complete captured dependencies and successful request/result provenance. A timed observer calls the existing source-resolved request producer once, returns the identical request object, retains the exact input/prepared-checkpoint/resolved-address/request chain, and restores the producer in `finally`. Entries must match that actual requested call; no guessed envelope-to-component config translation or second parse is used. The exact full IR and shape-derived quantities must match within each pair. These checks do not substitute for the separate architectural evidence gates or promise that every model is cache eligible.

Receipts preserve source membership/content before and in `finally`, actual executed tool bytes, exact input envelopes and budget bytes, import origins, interpreter/explicit-library entry-file pins, full cached dependency manifests/results, diagnostics, raw logs and cache snapshots. Initial runtime pins cover the interpreter and explicit library entry modules; complete worker dependency seals are retained separately, as emitted and validated by the product inside the timed call. The gate does not label a partial loaded-module list as complete initial dependency coverage. False source/input/runtime equality, altered/missing artifacts, a wrong pair, missing hit or incomplete campaign is red even if a saved status says PASS.

Use one repository measurement process at a time. The coordinator launches this lane serially and its children serially; its existing full-suite worker flags retain their existing meaning. Run under the same allowed outer process environment as the repository's worker-based verification. No hidden cache disabling, eager warm-up, timing normalization or budget relaxation is part of this command.
