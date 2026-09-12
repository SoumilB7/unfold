# Exact approved staging audit

ACCEPT the scoped baseline installation in `installation-acceptance.json`: canonical manifest SHA256 `84f0df7ade2f311adefbbb14f8ac0732b941f98d6110723f6554530a1d14b390` and 362 exact staged file hashes. Actual final green verification remains required.

Read-only commands executed from unfold-pkg:

```
python3 verification/receipts/S8-approved-stage-independent-710da8e/audit.py
python3 verification/receipts/S8-approved-stage-independent-710da8e/accept_manifest.py
```

Both passed. No model, renderer, pytest or baseline writer was invoked by this reviewer. The second command calls only existing pure input hashing, gallery-byte hashing, HTML metadata normalization and canonical byte serialization helpers. It independently checks the completed actual canonical writer receipt (1,275.505 seconds), before/finally source, corpus, tool and selected external pins.

All 29 config/source/model identities are exact. Ten ledger-only fixture and gallery trees remain byte-identical. The other 19 fixtures match their exact approved actual report, independent verdict, labelled view sequence, mandatory actual offline reproduction and 294 current PNGs. The 18 non-UNet proof lists are unchanged. SDXL's old corpus proof list was empty; its exact 12 new root.denoiser keys are the previously reviewed construction, defaults, primitive, connection, FFN, spatial and arithmetic proofs. No previously recorded corpus fact is retired. Superseded signatures and historical sidecars remain preserved; prior Her Eyes files are retained separately and replaced with the current individually reviewed files.

The expected manifest changes exactly the 29 owner-reviewed surface sets, plus the deliberately regenerated 19 gallery surfaces. Those gallery hashes were independently recomputed from all actual gallery files, including current review sidecars. The manifest's changed mechanism/placement/evidence hashes match the actual previously reviewed captures. Bloom's ledger matches its actual736 capture and its HTML metadata matches the approved final page. All 19 ordered view sequences match the actual approved reports.

Flux2, HunyuanVideo, Lumina and SD3.5 retain their old pre-render IR hashes: the four separately disclosed lazy-VAE post-render capture-order artifacts were explicitly rejected as candidate IR baselines. SDXL's five changed non-gallery surface hashes exactly match the reviewed current SDXL capture.

The six changed example HTML files match the approved saved candidate bytes. PixArt and Qwen2VL HTML, the sealed hero PNG and all other example files remain byte-identical. Only the reviewed manifest metadata changes. `examples-audit.json` records exact before/after files.

This acceptance authorizes the existing scoped installer under Soumil's “yes go ahead.” It neither claims remaining visual suggestions are fixed nor authorizes source changes, support expansion, baseline normalization, push or release.

## Later exact whitespace supplement

`review-whitespace-supplement.json` separately accepts removal of one extra terminal LF from the maintained SDXL Her Eyes file after static verification flagged it. The original reviewed temporary file, original verdict and actual 1,275.505-second manifest remain unchanged. All 143 review rows are byte-identical. The maintained stage copy now matches the live one-LF-shorter file; the original installation acceptance remains historical.

The derived manifest SHA256 is `00bf924b13aeff81ac6bcfd0f842fda2a9922ce5da2a299d582e3d8709c3e2bc`. Only `/witnesses/stable-diffusion-xl-base-1-0/surfaces/gallery` changes, recomputed using the unchanged pure gallery witness and canonical byte helpers. `audit_review_whitespace.py` passed and verified every other installed baseline pin. This was no model, rendering, re-blessing or whole-manifest generation run.
