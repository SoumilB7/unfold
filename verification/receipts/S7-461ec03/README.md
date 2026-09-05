# S7 Linux-workflow closure receipt

- verified commit: `461ec03987a9a4d594a794391afb6df870cb28f2`
- coordinator run: `ab630054f9`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- scope: install Ubuntu's `librsvg2-bin` before the already-blocking release
  example check, so the later S7 namespace lanes can execute
- product delta: **none**

| lane | result |
|---|---:|
| collection | 4,212 node ids; 4,116 runnable tests in the full bracket |
| focused S6/S7 | 320 passed |
| U2 authority | 44 passed |
| full | all 24 fresh-process batches passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | clean |

The new poison asserts that the native renderer install occurs exactly once and
before `scripts/generate_examples.py --check`. This prevents CI from skipping the
actual S7 namespace proof because an unrelated required binary is absent.

The machine record is `receipt.json`; exact compressed logs are in `lanes/`.
