"""Standalone report integrity checks; no model runs or blessed outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from report_s8_demonstration import observation


def observe(cards):
    return observation({"extras": {"render": cards}}, {}, "<html></html>")


cards = {"cards": {
    "_call_6": {"id": "_call_6", "kind": "conv2d"},
    "_call_12": {"id": "_call_12", "kind": "activation"},
}}
persisted = json.loads(json.dumps(cards, sort_keys=True))
assert observe(cards) == observe(persisted)

forward = {"sequence": [cards["cards"]["_call_6"], cards["cards"]["_call_12"]]}
reverse = {"sequence": list(reversed(forward["sequence"]))}
assert observe(forward)["block_structure"] != observe(reverse)["block_structure"]
assert observe(forward)["signatures"] != observe(reverse)["signatures"]
assert observe(forward) == observe(json.loads(json.dumps(forward, sort_keys=True)))
print(json.dumps({"mapping_roundtrip": "PASS", "list_reorder_poison": "REJECTED", "list_roundtrip": "PASS"}, indent=2))
