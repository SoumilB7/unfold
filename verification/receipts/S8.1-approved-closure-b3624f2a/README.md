# S8.1 approved closure — DONE-pending-Linux

Owner accepted accuracy and explicitly approved the enumerated output delta on 2026-09-10. Same `audio-composite-support` branch.

- `b3624f2a`: exact four-file guarded installation, with independent SDXL html_meta verdict. Compact example/manifest and S7 sources only; all117 S7 payloads and other preservation rows unchanged.
- `41a80392`: measured UNet baseline and pure budget checker/test update. All12 original samples and host records retained; ceil(1.3 × slowest median) gives43s cold/13s warm. Seven original S2 records unchanged.
- [Committed output checks](output-checks.json): S5 1passed57.85s; S7 1passed8.36s; original preservation52passed422.34s.
- [Committed budget checks](budget-checks.json):48passed1.03s, static2Pythonfilesclean. Every lane/tool/source/artifact fingerprint matches.
- [Independent local verdict](local-closure-verdict.json): ACCEPT_LOCAL_CLOSURE_PENDING_LINUX; no local blocker.

SDXL measured32.681/9.888s and SD-v1-4 31.403/8.816s cold/warm pass the owner-authorized43/13 gates. The original30/9 failures remain in the historical campaign. S10 open item: “static UNet reader cost, target30/9”, owner executor; remeasure at deletion and ratchet down. S11 module map carries N1 evidence-to-physics cache coupling/top-level physics namespace and N2 temporary import allowlist retirement. No S9 work started.

[Artifact map](artifact-map.json) retains exact installer, approval, before backups, original logs/results, source/test preparations, independent reviews, baseline metadata, sheet and decision page. [Integrity result](integrity.json) records lossless hashes/decoded-original equality. Full historical12sample artifacts are already committed in [the preceding receipt](../S8.1-owner-resubmission-actual-results/README.md); they are referenced rather than recopied.

This receipt is the local closure snapshot before push. Native Linux namespace/shadow verification at the pushed head remains required before DONE.
