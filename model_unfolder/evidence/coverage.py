"""Closed validation for the published S4 support-set denominator.

``coverage.json`` is a receipt, not an architecture authority.  It records how
many owner-qualified facts the existing evidence system proved and how many
unresolved findings the product visibly disclosed.  A non-zero ``silent``
count always blocks.
"""
from __future__ import annotations


EXPECTED_COHORT_SIZES = {"corpus": 29, "unseen": 15}


def coverage_problems(document: dict) -> list[str]:
    """Return every closed-world/schema/no-silence violation in a manifest."""
    problems: list[str] = []
    if document.get("schema") != 1:
        problems.append("coverage schema must be 1")
    environment = document.get("source_environment")
    required_environment = {"transformers", "diffusers", "huggingface_hub"}
    if not isinstance(environment, dict) or set(environment) != required_environment:
        problems.append("coverage source_environment must pin transformers, "
                        "diffusers and huggingface_hub")
    elif any(not isinstance(environment[name], str) or not environment[name]
             for name in required_environment):
        problems.append("coverage source_environment versions may not be empty")
    rows = document.get("models")
    if not isinstance(rows, list):
        return problems + ["coverage models must be a list"]

    seen_models: set[str] = set()
    seen_inputs: set[str] = set()
    cohort_counts = {name: 0 for name in EXPECTED_COHORT_SIZES}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"models[{index}] must be an object")
            continue
        model = row.get("model")
        cohort = row.get("cohort")
        input_path = row.get("input")
        if not isinstance(model, str) or not model:
            problems.append(f"models[{index}] has no model id")
        elif model in seen_models:
            problems.append(f"duplicate model id: {model}")
        else:
            seen_models.add(model)
        if cohort not in EXPECTED_COHORT_SIZES:
            problems.append(f"{model or index}: invalid cohort {cohort!r}")
        else:
            cohort_counts[cohort] += 1
        if not isinstance(input_path, str) or not input_path:
            problems.append(f"{model or index}: missing frozen input path")
        elif input_path in seen_inputs:
            problems.append(f"duplicate frozen input path: {input_path}")
        else:
            seen_inputs.add(input_path)
        for field in ("proven", "flagged", "silent"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"{model or index}: {field} must be a non-negative int")
        if isinstance(row.get("silent"), int) and row["silent"]:
            problems.append(
                f"{model or index}: silent={row['silent']} — a crash or blocking "
                "finding is not visible on the drawing")

    for cohort, expected in EXPECTED_COHORT_SIZES.items():
        actual = cohort_counts[cohort]
        if actual != expected:
            problems.append(
                f"{cohort} support denominator changed: expected {expected}, got {actual}")
    return problems


__all__ = ["EXPECTED_COHORT_SIZES", "coverage_problems"]
