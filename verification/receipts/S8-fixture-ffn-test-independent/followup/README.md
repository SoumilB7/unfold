# FFN assertion correction — actual end-to-end PASS

The broad a83d704 test reached the retired global phrase assertion after three legacy source-helper controls passed. The revised test retains those controls and checks a current runtime FFN card: its own occurrence address selects its cited fact value, gated GELU and fused gate/up projection agree with its detail, and all five operation IDs occur together in an actual returned SVG with GELU.

The first local revision searched only model blocks. The second included loop blocks. Both local attempts encountered an actual sandbox resource-monitor refusal; the separate actual diagnostic retains the precise `monitor_unavailable` sysctl error. These logs remain red and are not product mechanism failures. The final test uses the existing full-forest `lint._walk_blocks` traversal and ran with authorized system resource-monitor access:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests/test_code_evidence.py::test_unet_ffn_activation_anchored_to_declared_blocks -q
```

`unfold-s8-ffn-semantic-test-v3.log`: **1 passed in 120.16 seconds**. The included test/fixture snapshots and hashes were captured after the run; they are not a claim of an immutable whole-tree execution manifest. The reviewed source changes were not edited during this run. The shared fixture now supplies three exact corpus constructor fields; its preserved original incomplete input still awaits its separate actual constructor-failure capture and assertion.
