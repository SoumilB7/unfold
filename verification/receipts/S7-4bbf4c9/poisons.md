# S7 platform-stable artifact poisons

The blocking tests prove the following independent failures:

1. Changing the recomputed hero SVG input while retaining the reviewed PNG
   makes the manifest stale.
2. Changing the reviewed PNG bytes makes its independent seal stale.
3. Forging the PNG hash in the manifest cannot certify the checked-in bytes.
4. Changing any deterministic release HTML remains byte-blocking.
5. `--check` succeeds with the platform rasterizer replaced by a function that
   raises, proving CI verification cannot silently re-rasterize.
6. The workflow retains exactly one blocking example check and no longer
   installs the now-unused renderer.

The poison implementations are in `tests/test_s5_release.py` and
`tests/test_s7_artifacts.py`.
