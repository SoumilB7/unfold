# Trace card selection correction

The report previously overwrote canonical occurrence cards with later invocation aliases. On actual456 artifacts, all three selected stage aliases were absent from the denoiser SVG although the canonical stages were visibly present. Product and generator outputs were correct.

The reporter now selects an occurrence's own actual card, excludes target-link aliases, requires the exact fact citation for each claim card, and chooses a stage only when its ID occurs in the actual denoiser SVG. Actual456 replay now has zero linkage gaps in all three traces. Removing all visible stage IDs still produces three stage-overview gaps. This is artifact linkage, not semantic approval of the claims.

Original reports and campaign pins remain unchanged. The final frozen f664 model campaign continues using its original scripts; a separately pinned corrected report may read the same artifacts afterward. No model rerun or output blessing is implied. Exact source hash, complete corrected trace rows and standalone positive/negative replay are retained here.
