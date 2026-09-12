Actual37 saved-HTML browser QA, complete. Source harness is pinned in source-manifest.json; results/actual-origins.json joins published bytes to actual root captures and passed source/artifact brackets. results/manifest.json pins runtime output. visual-verdict.json closes six cases and preserves W1/S9-D and W2/S9-C/L5.

The first close-audit.py correctly refused changed SDXL payload attributes. close-audit-r2.py explicitly accounts for the content-addressed selector and exact mount pair, retaining zero executable JS size delta and preserving all raw input pages. No broad output normalization or blessing.

Isolated Playwright requirements are the exact browser36/requirements.txt; no project Python/source imports. No runtime remains active from this browser lane.
