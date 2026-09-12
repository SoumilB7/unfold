# Source override preserves source compiler flags

The supported rewrite produced matching static/runtime root bytes but lost every invocation target. Its captured forward function was correctly rejected by strict code/source equality: the custom loader inherited its own future-annotations compiler flag into modeling source that did not declare it. This is an execution difference despite identical file hashes.

The earliest correction adds dont_inherit=True to the one-time source loader's compile call. The strict worker equality check is unchanged. Two regression controls compare exact loaded code and annotations against a direct independent compile, for requested sources with and without their own future directive; both must also pass the actual worker function witness. Full source-override file:9 tests passed in20.22seconds, including unchanged/changed construction, wrong hash, unused, preloaded and duplicate addresses.

The no-override branch returns before this loader is instantiated, and saved-IR rendering does not execute it. Nevertheless the owner requires a complete fresh frozen final campaign rather than a mixed-revision exception to the report's source pin. The f664 ordinary/sparse/misleading passes and failed rewrite remain historical. No outputs were blessed or pushed.
