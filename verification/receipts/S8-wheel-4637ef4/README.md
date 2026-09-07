# S8 offline wheel smoke — 4637ef4

PASS for the bounded packaging check. An exact `git archive HEAD` from frozen `4637ef4b77a8139326a8643b9f7eb68ab9d0f93c` was built in a scratch copy using `python3 -m pip wheel --no-deps --no-build-isolation --no-index`. The resulting wheel was installed into a separate target directory with `pip install --no-index --no-deps --target`. From outside the checkout, imports of the installed package, production UNet adapter, physics inventory, source override and primitive witnesses succeeded. All six physics Python modules are included.

Exact wheel hash, member inventory, metadata dependencies and installed import paths are in `result.json`; build/install/import logs are retained. This reuses the host dependency environment: it is not a clean dependency-resolution test, model execution, broad gate or release approval. No network, pytest, production edits or blessing occurred.
