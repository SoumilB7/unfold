"""Bounded producer/SVG replay; no model construction or pytest."""
import hashlib
import json
from pathlib import Path

from model_unfolder.adapters.diffusor.blocks import diffusion_opaque_render_spec
from model_unfolder.renderers.html.views_diffusion import _build_loop_view, _stub_info
from model_unfolder.expanded.loop import build_sampling_loop

out = Path(__file__).resolve().parent
root = out.parents[2]
paths = [
    "model_unfolder/adapters/diffusor/blocks.py",
    "model_unfolder/adapters/diffusor/parser.py",
    "model_unfolder/renderers/html/views_diffusion.py",
    "model_unfolder/expanded/loop.py",
]
before = {p: (root / p).read_bytes() for p in paths}
if (out / "results.json").exists():
    expected = json.loads((out / "results.json").read_text())["source_hashes"]
    assert all(hashlib.sha256(data).hexdigest() == expected[p] for p, data in before.items()), \
        "Historical source changed; retain this receipt and use a new correction receipt"
render = diffusion_opaque_render_spec({
    "component_presence": {"scheduler": False, "vae": False, "text_encoders": False},
    "text_encoder_specs": [], "text_encoders": [], "vae": None,
})
ir = {"extras": {"render": render}}
svg = _build_loop_view(ir, _stub_info(), "opaque_scope")
result = {
    "scope": "Source-unresolved render producer used by failed UNet cutover; no full model run",
    "component_scope": render.get("component_scope"),
    "block_ids": [b["id"] for b in render["loop_blocks"]],
    "expanded_sampling_loop": build_sampling_loop(ir["extras"]) is not None,
    "visible_labels": {label: label in svg for label in ("Scheduler", "VAE", "Noise")},
    "source_hashes": {p: hashlib.sha256(data).hexdigest() for p, data in before.items()},
    "source_unchanged": all((root / p).read_bytes() == data for p, data in before.items()),
}
assert result["source_unchanged"]
for p, data in before.items():
    target = out / "source" / (p + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
(out / "overview.svg").write_text(svg)
(out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
