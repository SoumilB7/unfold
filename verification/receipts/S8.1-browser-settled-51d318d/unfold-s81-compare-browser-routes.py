"""Compare actual functional-browser receipts; never normalize product HTML."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before = json.loads((args.before / "receipt.json").read_text())
    after = json.loads((args.after / "receipt.json").read_text())
    if before["status"] != "ok" or after["status"] != "ok":
        raise ValueError("Both actual browser runs must succeed")
    for key in ("harness_sha256", "steps_sha256", "viewport", "browser_version"):
        if before[key] != after[key]:
            raise ValueError("Different functional browser inputs: " + key)
    rows = []
    if len(before["clicks"]) != len(after["clicks"]):
        raise ValueError("Different actual click counts")
    for index, (old, new) in enumerate(zip(before["clicks"], after["clicks"])):
        same_route = (old["depth"], old["target"]["id"], old["expected_card"]) == (
            new["depth"], new["target"]["id"], new["expected_card"])
        same_cards = old["result"]["active_cards"] == new["result"]["active_cards"]
        images = [directory / f"click-{index:02d}.png" for directory in (args.before, args.after)]
        rows.append({"step": index, "route_equal": same_route,
                     "visible_card_text_identity_markup_equal": same_cards,
                     "screenshots_equal": images[0].read_bytes() == images[1].read_bytes(),
                     "screenshot_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
                     "before_dom_elements": old["result"]["dom_nodes"],
                     "after_dom_elements": new["result"]["dom_nodes"]})
    architecture_equal = before["readiness"]["architecture_nodes"] == after["readiness"]["architecture_nodes"]
    result = {"architecture_equal": architecture_equal, "steps": rows,
              "status": "pass" if architecture_equal and all(row["route_equal"] and row[
                  "visible_card_text_identity_markup_equal"] for row in rows) else "fail",
              "scope": "Actual visible cards at each native-click step, exact browser-serialized markup/text/identity/placement. Screenshot differences require visual review. DOM counts intentionally differ; no claim of all latent card behavior from these routes.",
              "inputs": [str(args.before), str(args.after)]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
