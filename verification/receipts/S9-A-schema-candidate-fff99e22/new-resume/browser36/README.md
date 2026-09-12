Saved candidate36 browser QA used an isolated temporary venv and fresh Chrome profiles. No project Python or source was imported or modified. The original and corrected observer runs are retained.

Reproduce with Python3 venv in a fresh temporary directory, install the exact requirements.txt there, then run qa-r2.py after directing its output to a new empty receipt directory. The script refuses to overwrite an existing result directory. Use installed Google Chrome at the recorded executable path. HTTPS font requests are deliberately blocked; screenshots use CSS fallback fonts.

Verdict: interactions pass, visual readability returned on three overflowing default chips. See visual-verdict.json and results-r2/. No output approval.
