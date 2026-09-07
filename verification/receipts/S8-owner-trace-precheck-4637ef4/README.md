# Owner trace precheck — 4637ef4

This is an interim source/card/native-image check, not acceptance of final S8 traces. The root reviewer read the actual installed implementations, hashed those bytes, extracted the actual HTML cards and inspected native views 46 (FFN), 31 (downsample), and 26 (first residual cell chain) in `z-docs/12-design/S8/4637ef4-sdxl-visual/`.

- The FFN picture shows a fused affine projection, two-way split, GELU on one lane, multiplication with the bypass lane, then output affine. `activations.py` lines 117–123 contains projection and the source split/GELU/multiply return; the NPU alternative has a separate closed primitive proof in the existing input-transform reader. `attention.py` lines 1727–1743 constructs and calls the selected sequence. Actual card shows input weight 5120×640, output weight 640×2560 and 4,920,960 subtree parameters; direct weight/bias arithmetic agrees.
- The first residual cell's inspected fragment is GroupNorm → SiLU → Conv2d. `resnet.py` lines 326–340 establishes those calls in that order. Other branches remain explicitly under investigation on the card; the fragment is not a proof of the full residual computation. Actual subtree count 2,255,040 agrees with the recorded affine, convolution and normalization shapes.
- The first downsample drill shows the actual convolution. `downsampling.py` lines 101–115 declares stride 2 for the selected primitive and line 145 calls it. The card labels spatial reduction, stride 2 and 921,920 parameters; 320×320×3×3+320 agrees.

Exact actual card text, page hash and source hashes are in `evidence.json`. Execution chips on these ordinary-production cards say no recipe was attempted; the separately joined family verification observations are not silently inserted into the ordinary page. Static mechanism proof and observed execution remain separate authorities.

Final completion still requires the final candidate's archived proof citations, canonical fact-to-block/drill linkage, three written traces and actual before/after conditions. These interim images are not substituted for that final packet.
