# Original 52-case computation checkpoint

The retained actual run on commit `3724cba4e12a206fdf812a45b4abc5fda09647f1` passed all **52 original preservation tests**. It retained the original renderer, debt registry, expected manifest, and comparator, with two pytest workers and an initially empty task-local result cache. All witness calls remained required. This checkpoint does not include or approve the later compact-renderer raw HTML deltas.

The four original streams are listed in `file-list.txt`: the exact result, preservation log, adjacent empty log, and executed preparation `run.py`. The result preserves the actual command, source/artifact/tool/selected-external before/after pins, host observations, and outcome. `qualification.json` recomputes those retained equality checks without running tests or reading extra external source files. The original log reports 52 passes in 551.06 seconds; this is correctness evidence, not a latency-budget result.

All four streams (31,237 original bytes) were archived by appending `.gzip` to each full logical filename, restored to separate files, and verified by SHA-256 and byte count. File/directory membership and original bytes remained unchanged. Map SHA-256: `974dcbac97789c19eaa2191f51f824dfc2e514c7ced04b5de9ff03db1635b26d`.

No model, test, renderer, or baseline writer was run during packaging. Referenced external paths remain pins; this archive does not import them as new streams or broaden the original qualification scope.
