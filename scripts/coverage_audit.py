#!/usr/bin/env python3
"""Coverage audit: parse real HF configs and report what we don't yet handle.

For every model in the curated target list (mirrors ``toserve_model.md``) this:

1. Fetches the model's ``config.json`` from the HuggingFace CDN (no auth, no
   ``transformers`` needed — just the JSON).
2. Runs it through ``model_unfolder``'s parser, capturing the centralized debug
   diagnostics (unparsed config fields + partial-config reasons).
3. Aggregates the results into ``docs/coverage_audit.md``:
     * every config field no model-unfolder code reads, and which models have it
     * every partial-config reason, and which models triggered it
     * which models were inaccessible (gated / 404 / network).

Run from anywhere:  python3 scripts/coverage_audit.py
Gated repos (Llama, Gemma) 401 without an HF token and are listed as gated.
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path

# Make the package importable when run from the repo without install.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from model_unfolder.adapters.transformer.parser import parse  # noqa: E402

# --- target list (mirrors toserve_model.md) -------------------------------
MODELS: dict[str, list[str]] = {
    "Llama 2 / Code Llama": [
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-70b-hf",
        "codellama/CodeLlama-34b-hf",
    ],
    "Llama 3 / 3.1 / 3.2 / 4": [
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    ],
    "Mistral": [
        "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1", "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Pixtral-12B-2409", "mistralai/Ministral-8B-Instruct-2410",
        "mistralai/Magistral-Small-2506",
    ],
    "Qwen": [
        "Qwen/Qwen2.5-72B", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/QwQ-32B",
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    ],
    "Gemma": [
        "google/gemma-7b", "google/gemma-2-27b", "google/gemma-3-4b-it",
        "google/gemma-3-27b-it", "google/recurrentgemma-2b",
    ],
    "DeepSeek": [
        "deepseek-ai/deepseek-llm-67b-chat", "deepseek-ai/DeepSeek-V2",
        "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek-ai/DeepSeek-V3.2-Exp",
    ],
    "GPT-OSS": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    "Moonshot Kimi K2": ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Thinking"],
    "Zhipu GLM": ["zai-org/GLM-4.5", "zai-org/GLM-4.6"],
    "Phi": [
        "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3.5-MoE-instruct", "microsoft/phi-4",
        "microsoft/phi-4-multimodal-instruct",
    ],
}

_UNPARSED_RE = re.compile(r"config field\(s\) not parsed: (.+)$")
_PARTIAL_HDR = re.compile(r"partial config")
_REASON_RE = re.compile(r"^\s*⚠ (.+)$")


def fetch_config(repo: str, timeout: int = 30) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "model-unfolder-audit"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def audit_one(repo: str) -> dict:
    """Return {status, unparsed, reasons, model_type, error}."""
    try:
        cfg = fetch_config(repo)
    except urllib.error.HTTPError as e:
        status = "gated" if e.code in (401, 403) else f"http_{e.code}"
        return {"status": status, "unparsed": [], "reasons": [], "model_type": None, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "unparsed": [], "reasons": [], "model_type": None, "error": str(e)}

    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            parse(cfg)
    except Exception as e:  # noqa: BLE001
        return {"status": "parse_error", "unparsed": [], "reasons": [],
                "model_type": cfg.get("model_type"), "error": f"{type(e).__name__}: {e}"}

    unparsed: list[str] = []
    reasons: list[str] = []
    in_partial = False
    for line in buf.getvalue().splitlines():
        m = _UNPARSED_RE.search(line)
        if m:
            unparsed = [f.strip() for f in m.group(1).split(",") if f.strip()]
            in_partial = False
            continue
        if _PARTIAL_HDR.search(line):
            in_partial = True
            continue
        rm = _REASON_RE.match(line)
        if rm and in_partial:
            reasons.append(rm.group(1).strip())
    return {"status": "ok", "unparsed": unparsed, "reasons": reasons,
            "model_type": cfg.get("model_type"), "error": None}


def main() -> None:
    results: dict[str, dict] = {}
    field_to_models: dict[str, list[str]] = defaultdict(list)
    reason_to_models: dict[str, list[str]] = defaultdict(list)

    for family, repos in MODELS.items():
        for repo in repos:
            print(f"… {repo}", file=sys.__stderr__)
            r = audit_one(repo)
            r["family"] = family
            results[repo] = r
            for f in r["unparsed"]:
                field_to_models[f].append(repo)
            for reason in r["reasons"]:
                reason_to_models[reason].append(repo)

    out = _render_doc(results, field_to_models, reason_to_models)
    doc_path = _PKG_ROOT / "docs" / "coverage_audit.md"
    doc_path.write_text(out, encoding="utf-8")
    print(f"\nWrote {doc_path}", file=sys.__stderr__)


def _render_doc(results, field_to_models, reason_to_models) -> str:
    ok = [r for r in results.values() if r["status"] == "ok"]
    gated = [k for k, r in results.items() if r["status"] == "gated"]
    failed = {k: r for k, r in results.items() if r["status"] not in ("ok", "gated")}

    lines: list[str] = []
    lines.append("# Coverage audit")
    lines.append("")
    lines.append(f"_Generated {date.today().isoformat()} by `scripts/coverage_audit.py`._")
    lines.append("")
    lines.append(f"- Models attempted: **{len(results)}**")
    lines.append(f"- Parsed: **{len(ok)}**  ·  Gated/inaccessible: **{len(gated)}**  ·  Errored: **{len(failed)}**")
    lines.append(f"- Distinct unparsed config fields: **{len(field_to_models)}**")
    lines.append("")

    # --- unparsed fields, most-common first ---
    lines.append("## Config fields we don't parse")
    lines.append("")
    lines.append("Architectural-looking keys present in configs that no parser code reads. "
                 "Sorted by how many models carry them.")
    lines.append("")
    lines.append("| Field | # models | Models |")
    lines.append("| --- | --- | --- |")
    for field, models in sorted(field_to_models.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shown = ", ".join(_short(m) for m in models)
        lines.append(f"| `{field}` | {len(models)} | {shown} |")
    lines.append("")

    # --- partial reasons ---
    lines.append("## Partial-config reasons")
    lines.append("")
    if reason_to_models:
        lines.append("| Reason | Models |")
        lines.append("| --- | --- |")
        for reason, models in sorted(reason_to_models.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {reason} | {', '.join(_short(m) for m in models)} |")
    else:
        lines.append("_None — every parsed model produced a complete structure._")
    lines.append("")

    # --- per-model detail ---
    lines.append("## Per-model detail")
    lines.append("")
    lines.append("| Model | model_type | status | unparsed fields |")
    lines.append("| --- | --- | --- | --- |")
    for repo, r in results.items():
        fields = ", ".join(f"`{f}`" for f in r["unparsed"]) or "—"
        mt = r["model_type"] or "—"
        lines.append(f"| {repo} | {mt} | {r['status']} | {fields} |")
    lines.append("")

    # --- inaccessible ---
    if gated or failed:
        lines.append("## Not audited")
        lines.append("")
        for repo in gated:
            lines.append(f"- **{repo}** — gated (needs HF token)")
        for repo, r in failed.items():
            lines.append(f"- **{repo}** — {r['status']}: {r['error']}")
        lines.append("")

    return "\n".join(lines)


def _short(repo: str) -> str:
    return repo.split("/", 1)[-1]


if __name__ == "__main__":
    main()
