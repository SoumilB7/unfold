# S8.1 Linux proof portability — tested proposal, approval required

S8.1 is **not DONE**. [Linux run34453756753](https://github.com/SoumilB7/unfold/actions/runs/34453756753) failed on SDXL proof-summary index fingerprints. Namespace, census and source freshness passed; coverage/examples were skipped. The previous owner approval renewed S7 sources only and required all117 payloads unchanged. This new payload renewal therefore needs explicit owner approval.

The recorded 8-file and20-file source closures reproduce both exact Linux failures by changing only the installation prefix. The source candidate changes ten persisted summary producers to the existing portable seal, keeps exact local membership/validation, and memoizes that derived seal per frozen index. It changes six production files and two test files. The generic constructor's existing absolute evidence references are deliberately unchanged; none occur in these persisted model proofs.

The complete fresh SDXL producer differs from the committed model at **exactly200 index-fingerprint leaves** (8+192), with no other model semantic change. Fresh observation/relation semantics match. The independent verdict is in [independent/actual-proposal-review.json](independent/actual-proposal-review.json).

The proposed S7 renewal is exactly:

- `models/stable-diffusion-xl-base-1-0.json.gz`: those200 proof seals; SHA `0ed9ed525605741fd6b848f58f84ee0f1b867169587cb932fdd5aa72b307dbcd`.
- `matrix.json`: six corrected source stamps plus that model's raw/logical hash; SHA `47f372e4629711fd51676d3005a522cdd10d6ad2f31a0dd4d12c1360490014d1`.

All39 matrix summary rows and116 other payload files remain identical. Full leaf enumeration is [comparison/full-semantic-delta.json](comparison/full-semantic-delta.json); matrix enumeration is [comparison/matrix-delta.json](comparison/matrix-delta.json). Comparators and production artifacts have not been changed.

Isolated candidate verification: **79 focused controls**, static, actual S5 example check, **52/52 original preservation**. Exact source/artifact fingerprints match around each lane, including candidate-only files. These are candidate-overlay checks at base89f181f4, not a new committed-tree/Linux verdict. Full results/logs are retained here. No new latency baseline was measured: authorized43/13 stays; S10 target30/9 and S11 N1/N2 remain open.

After owner approval: install the pinned source candidate and these two artifact files under the reviewed guard, record the closing receipt, run the committed S7 freshness check and push the same branch. The full Linux lane must pass before DONE. S9 has not started.
