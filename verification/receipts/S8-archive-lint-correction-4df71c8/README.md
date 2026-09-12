# Archived replay script lint correction

The original4df71c8 broad gate genuinely failed static lint on five unused imports in three archived scripts; full/preservation were correctly missing. This correction removes only those named standard-library import aliases from maintained runnable `.py` copies. Every exact executed source remains beside it as `.executed.py.txt`; original execution hashes and original final-render artifact index are preserved verbatim. source-map.json resolves historical filenames to those exact archives and separately pins the cleaned copies, which are not claimed to have executed in any historical campaign.

AST comparison proves no other statement, expression, import alias or source behavior changed. The unchanged static gate is rerun against all21 changed Python files from the failed commit. No production source, audit policy, baseline, test behavior or model output is altered. The next complete serial coordinator remains required.
