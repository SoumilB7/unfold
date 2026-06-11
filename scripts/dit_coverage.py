#!/usr/bin/env python3
"""DiT coverage audit: run every major diffusion-transformer release through
model-unfolder against its REAL HuggingFace configs and report failures.

For each repo this:
  1. unfolds by repo id (the loader fetches model_index.json + component
     configs, or the bare transformer config);
  2. renders HTML and validates click coupling;
  3. records hard failures, gated repos, IR warnings, unresolved ("?") facts,
     and basic shape sanity (layer count / denoiser family).

Run from anywhere:  python3 scripts/dit_coverage.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from model_unfolder import unfold                      # noqa: E402
from model_unfolder.block_schema import validate_click_coupling  # noqa: E402

# --- the major DiT lineage, by generation ----------------------------------
MODELS: dict[str, list[str]] = {
    "MMDiT image (SD3 family)": [
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-medium",
    ],
    "Flux family": [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Krea-dev",
        "shuttleai/shuttle-3-diffusion",
    ],
    "Cross-attn DiT (PixArt lineage)": [
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    ],
    "Next-gen image DiT": [
        "fal/AuraFlow-v0.3",
        "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        "Alpha-VLLM/Lumina-Image-2.0",
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "THUDM/CogView3-Plus-3B",
        "THUDM/CogView4-6B",
        "Qwen/Qwen-Image",
        "HiDream-ai/HiDream-I1-Full",
        "OmniGen2/OmniGen2",
    ],
    "Video DiT": [
        "THUDM/CogVideoX-5b",
        "genmo/mochi-1-preview",
        "Lightricks/LTX-Video",
        "hunyuanvideo-community/HunyuanVideo",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "rhymes-ai/Allegro",
    ],
    "U-Net baseline (must not regress)": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "kandinsky-community/kandinsky-3",
    ],
    "Audio DiT": [
        "stabilityai/stable-audio-open-1.0",
    ],
}


def probe(repo: str) -> dict:
    out: dict = {"repo": repo}
    try:
        d = unfold(repo)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "403" in msg:
            out["status"] = "GATED"
        elif "404" in msg:
            out["status"] = "404"
        else:
            out["status"] = "PARSE FAIL"
            out["error"] = f"{type(e).__name__}: {msg.splitlines()[0][:140]}"
            out["trace"] = traceback.format_exc().splitlines()[-3:]
        return out

    ir = d.to_ir()
    try:
        html = d.to_html(standalone=True)
    except Exception as e:
        out["status"] = "RENDER FAIL"
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
        out["trace"] = traceback.format_exc().splitlines()[-3:]
        return out

    coupling = validate_click_coupling(html)
    extras = ir.get("extras") or {}
    out.update({
        "status": "OK",
        "layers": len(ir.get("layers") or []),
        "family": ("unet" if extras.get("unet")
                   else "dit" if extras.get("diffusion") else "?"),
        "warnings": list(ir.get("warnings") or []),
        "coupling": coupling or None,
        "unknown_facts": html.count(">?<") + html.count("dim ?"),
        "encoders": [s.get("name") for s in
                     ((extras.get("diffusion") or {}).get("text_encoder_specs") or [])
                     ] or None,
    })
    return out


def main() -> int:
    failures = 0
    for section, repos in MODELS.items():
        print(f"\n=== {section} ===")
        for repo in repos:
            r = probe(repo)
            status = r["status"]
            if status == "OK":
                flags = []
                if r["coupling"]:
                    flags.append(f"COUPLING: {r['coupling']}")
                if r["warnings"]:
                    flags.append(f"warn={r['warnings']}")
                if r["unknown_facts"]:
                    flags.append(f"?×{r['unknown_facts']}")
                line = (f"  OK    {repo}  [{r['family']} · {r['layers']} layers"
                        + (f" · enc {r['encoders']}" if r["encoders"] else "") + "]")
                print(line)
                for f in flags:
                    print(f"        ⚠ {f}")
                    failures += 1
            else:
                print(f"  {status:5s} {repo}")
                if r.get("error"):
                    print(f"        {r['error']}")
                if status in ("PARSE FAIL", "RENDER FAIL"):
                    failures += 1
    print(f"\n{'FAILURES: %d' % failures if failures else 'ALL CLEAN'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
