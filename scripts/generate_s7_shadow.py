#!/usr/bin/env python3
"""Generate/check the S7 29+10 shadow reconciliation artifacts.

The target list is an experiment denominator, never a production dispatch
table.  Mechanism meanings still come only from exact facts and resolved code.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_unfolder import config_to_ir
from model_unfolder.diagram import Diagram
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.config_access import bound_document, resolve
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reconciliation import (
    projection_claims_from_product, reconcile, relation_rows_from_evidence,
    static_claims_from_owner_graph, unresolved_axis_findings,
    unresolved_reason_class_counts,
)
from model_unfolder.evidence.relation_source import (
    prove_post_stack_collapse, prove_recurrent_state_mix,
)
from model_unfolder.ir import detect_layer_period, distinct_layer_groups
from physics.execution_observation import (
    ExecutionRecipe, ObservationResult, TensorArgument,
    observe_in_subprocess,
)
from physics.instance_inventory import (
    BuildRequest, Failure, InventoryResult, inventory_in_subprocess,
)
from physics.relation_observation import (
    RelationObservationResult, observe_relations_in_subprocess,
)


CORPUS = ROOT / "tests" / "sable_test_corpus"
UNSEEN = ROOT / "tests" / "unseen_model_configs"
OUTPUT = ROOT / "verification" / "s7"


_TORCH_LOG_PREFIX = re.compile(
    r"(?m)^([A-Z])\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ ")


def _stable_observation_payload(result: ObservationResult) -> dict[str, Any]:
    """Remove process-local metadata from captured diagnostics only.

    Torch's structured stderr prefix embeds month/day, wall time and PID.  It
    is not execution evidence, while the log level, source location, traceback
    and exception are.  Persisting the volatile prefix made two identical S7
    observations hash differently, so normalize exactly that prefix and no
    diagnostic content after it.
    """
    payload = result.to_dict()
    for field in ("stdout", "stderr"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = _TORCH_LOG_PREFIX.sub(
                r"\1<date> <time> <pid> ", value)
    return payload
GIB = 1024 ** 3

# Exact, frozen S7 out-of-corpus denominator.  Reusing seven S4 inputs avoids
# inventing checkpoint values; the three hard relation/schedule witnesses are
# separately snapshotted from their locally cached config.json once.
TO_SERVE = (
    ("command-a-03-2025", "CohereLabs/c4ai-command-a-03-2025",
     "tests/unseen_model_configs/command-a-03-2025.json"),
    ("deepseek-coder-v2-lite", "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
     "tests/unseen_model_configs/deepseek-coder-v2-lite.json"),
    ("deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash",
     "verification/s7/inputs/deepseek-v4-flash.json"),
    ("gemma-3n-e2b", "google/gemma-3n-E2B",
     "verification/s7/inputs/gemma-3n-e2b.json"),
    ("jamba-v0-1", "ai21labs/Jamba-v0.1",
     "tests/unseen_model_configs/jamba-v0-1.json"),
    ("lfm2-1-2b", "LiquidAI/LFM2-1.2B",
     "tests/unseen_model_configs/lfm2-1-2b.json"),
    ("nemotron-h-8b", "nvidia/Nemotron-H-8B-Base-8K",
     "verification/s7/inputs/nemotron-h-8b.json"),
    ("qwen3-5-27b", "Qwen/Qwen3.5-27B",
     "tests/unseen_model_configs/qwen3-5-27b-full.json"),
    ("qwen3-omni-30b", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
     "tests/unseen_model_configs/qwen3-omni-30b.json"),
    ("qwen3-vl-235b", "Qwen/Qwen3-VL-235B-A22B-Instruct",
     "tests/unseen_model_configs/qwen3-vl-235b.json"),
)

HARD_INPUTS = {
    "gemma-3n-e2b": (
        "models--google--gemma-3n-E2B",
        "53dbc746ab2a4d1496dca0fd4449be6880e3783bedca2d8be7e50f0a1fa9b21a"),
    "deepseek-v4-flash": (
        "models--deepseek-ai--DeepSeek-V4-Flash",
        "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"),
    "nemotron-h-8b": (
        "models--nvidia--Nemotron-H-8B-Base-8K",
        "81e822f85d6471312bcc0ccd34cc62789075376ecedca7fde0e451584943368b"),
}

RELATION_RECIPES = {
    "gemma-3n-e2b": {
        "recipe_id": "gemma3n-relations", "target_path": "model.language_model",
        "stack_path": "model.language_model.layers", "dtype": "float32",
    },
    "deepseek-v4-flash": {
        "recipe_id": "deepseek-v4-relations", "target_path": "",
        "stack_path": "model.layers", "dtype": "bfloat16",
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(_json_bytes(value))


def _read_payload(path: Path) -> tuple[str, dict[str, Any]]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if "config" in row:
        return str(row.get("model_id") or row.get("model") or path.stem), row["config"]
    return path.stem, row


def _targets() -> tuple[dict[str, Any], ...]:
    rows = []
    for path in sorted(CORPUS.glob("*.json")):
        model, _config = _read_payload(path)
        rows.append({"cohort": "corpus", "slug": path.stem, "model": model,
                     "input": path.relative_to(ROOT).as_posix()})
    if len(rows) != 29:
        raise ValueError(f"S7 corpus denominator changed: expected 29, got {len(rows)}")
    rows.extend({"cohort": "to_serve", "slug": slug, "model": model,
                 "input": source} for slug, model, source in TO_SERVE)
    if len(rows) != 39 or len({row["slug"] for row in rows}) != 39:
        raise ValueError("S7 target denominator must be exactly 29 + 10")
    return tuple(rows)


def seed_hard_inputs() -> None:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    destinations = {slug: ROOT / source for slug, _model, source in TO_SERVE
                    if slug in HARD_INPUTS}
    for slug, (directory, expected) in HARD_INPUTS.items():
        candidates = sorted((cache / directory / "snapshots").glob("*/config.json"))
        matches = [path for path in candidates if _sha256(path.read_bytes()) == expected]
        if not matches:
            raise FileNotFoundError(
                f"no cached {slug} config with reviewed hash {expected}")
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        _write_json(destinations[slug], payload)


def _request(config: Mapping[str, Any], slug: str) -> BuildRequest:
    diffusers_class = config.get("_class_name")
    if isinstance(diffusers_class, str) and diffusers_class:
        return BuildRequest(
            config, "diffusers", "diffusers", diffusers_class,
            factory_method="from_config", timeout_seconds=300,
            memory_limit_bytes=16 * GIB, label=slug)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 \
            or not isinstance(architectures[0], str):
        raise ValueError(f"{slug}: exact singleton architecture address required")
    return BuildRequest(
        config, "transformers", "transformers", architectures[0],
        config_module="transformers", config_qualname="AutoConfig",
        config_method="for_model", timeout_seconds=300,
        memory_limit_bytes=16 * GIB, label=slug)


def _relation_recipe(slug: str) -> tuple[ExecutionRecipe, str] | None:
    spec = RELATION_RECIPES.get(slug)
    if spec is None:
        return None
    recipe = ExecutionRecipe(
        spec["recipe_id"], "tokens", "eval", "disabled", "decoder", False,
        spec["dtype"], {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }, target_path=spec["target_path"],
        tensor_arguments=(TensorArgument("input_ids", (1, 8), "long"),),
        literal_arguments={"use_cache": False})
    return recipe, spec["stack_path"]


def _s6_observations(slug: str, config_hash: str) -> tuple[ObservationResult, ...]:
    directory = ROOT / "verification" / "s6" / "pilots" / slug
    if not directory.exists():
        return ()
    rows = []
    for path in sorted(directory.glob("observation-*.json")):
        result = ObservationResult.from_dict(json.loads(path.read_text()))
        if result.provenance and result.provenance.config_sha256 == config_hash:
            rows.append(result)
    return tuple(rows)


def _s6_inventory(slug: str, config_hash: str) -> InventoryResult | None:
    path = (ROOT / "verification" / "s6" / "pilots" / slug
            / "inventory.json")
    if not path.exists():
        return None
    result = InventoryResult.from_dict(json.loads(path.read_text()))
    if (result.status == "ok" and result.inventory is not None
            and result.inventory.provenance.config_sha256 == config_hash):
        return result
    return None


def _s6_request(slug: str, config_hash: str) -> BuildRequest | None:
    path = ROOT / "verification" / "s6" / "pilots" / slug / "request.json"
    if not path.exists():
        return None
    request = BuildRequest.from_dict(json.loads(path.read_text()))
    digest = _sha256(json.dumps(
        dict(request.config), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    return request if digest == config_hash else None


_INTEGER_INPUTS = frozenset({
    "input_ids", "decoder_input_ids", "attention_mask",
    "decoder_attention_mask", "encoder_attention_mask",
})
_FLOAT_SCALARS = frozenset({"timestep", "timesteps", "guidance"})
_KNOWN_INPUTS = _INTEGER_INPUTS | _FLOAT_SCALARS | frozenset({
    "sample", "hidden_states", "x", "encoder_hidden_states",
    "pooled_projections", "pixel_values", "input_values",
})
_KNOWN_GROUPED_DTYPE_ERRORS = (
    "expected inputs of bf16 type but got",
    "grouped_mm only supports",
)


@dataclasses.dataclass(frozen=True)
class RecipeResolution:
    """The recipe decision, separate from every execution attempt."""

    status: str
    checkpoint_dtype: Mapping[str, Any]
    execution_dtype: str
    recipe: ExecutionRecipe
    argument_sources: Mapping[str, Any]
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"ok", "failed"}:
            raise ValueError("recipe resolution status is closed")
        if self.status == "ok" and self.failure_detail:
            raise ValueError("a resolved recipe carries no failure")
        if self.status == "failed" and not self.failure_detail:
            raise ValueError("a failed recipe resolution names the unknown input")
        if self.execution_dtype != self.recipe.dtype:
            raise ValueError("recipe execution dtype must match its resolution")
        if self.status == "failed" and self.recipe.flags.get(
                "resolution_status") != "failed":
            raise ValueError("a failed recipe must remain failed in its audit flags")


@dataclasses.dataclass(frozen=True)
class RecipeAttemptBundle:
    """One resolution and one execution, plus at most one dtype retry."""

    resolution: RecipeResolution
    attempts: tuple[ObservationResult, ...]

    def __post_init__(self) -> None:
        if len(self.attempts) not in {1, 2}:
            raise ValueError("a recipe bundle has one attempt and at most one retry")
        if self.attempts[0].recipe != self.resolution.recipe:
            raise ValueError("first attempt must use the resolved recipe")
        if self.resolution.status == "failed":
            first = self.attempts[0]
            if (len(self.attempts) != 1 or first.status != "failed"
                    or first.failure is None
                    or first.failure.kind != "ConfigurationFailed"
                    or first.failure.stage != "recipe_resolution"):
                raise ValueError(
                    "an unresolved recipe has one typed resolution failure")
        if len(self.attempts) == 2:
            first, retry = self.attempts
            if not _known_dtype_failure(first):
                raise ValueError("a retry requires the closed known dtype error")
            if first.recipe is None or first.recipe.dtype == "bfloat16":
                raise ValueError("a bfloat16 recipe cannot receive a dtype retry")
            if retry.recipe != _bf16_retry(first.recipe):
                raise ValueError(
                    "the single retry may change only execution dtype and cite "
                    "its first attempt")

    @property
    def final(self) -> ObservationResult:
        return self.attempts[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "resolution": dataclasses.asdict(self.resolution),
            "attempts": [_stable_observation_payload(row) for row in self.attempts],
            "final_status": self.final.status,
            "retry_count": len(self.attempts) - 1,
        }


def _normalise_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower().removeprefix("torch.")
    return {
        "bf16": "bfloat16", "bfloat16": "bfloat16",
        "fp16": "float16", "float16": "float16", "half": "float16",
        "fp32": "float32", "float32": "float32", "float": "float32",
    }.get(text)


def _config_value(prepared: Any, canonical: str, aliases=()) -> tuple[Any, dict[str, Any]]:
    result = resolve(
        prepared.document, canonical, aliases, component="root",
        class_defaults=prepared.class_overlay)
    return result.value, {
        "state": result.state, "path": result.selected_path,
        "spelling": result.selected_alias, "provenance": result.provenance,
        "source_kind": result.source_kind,
    }


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _signature_recipe(index: Any, root: Any, inventory: Any,
                      config: Mapping[str, Any]) -> RecipeResolution:
    """One neutral forward attempt derived only from the resolved signature.

    Parameter spellings are callable addresses, not family identities.  The
    Shape meanings come from the parameter address and their dimensions from
    the supported config-resolution path.  Probe-size calculations are named
    in the recipe; no model/class identity participates.
    """
    symbol = root.graph.root.symbol if root.address_resolved else None
    forwards = tuple(
        row for row in (index.callables_of(symbol) if symbol is not None else ())
        if row.symbol.qualified_name == f"{symbol.qualified_name}.forward")
    parameters = tuple(
        row for row in (forwards[0].params if len(forwards) == 1 else ())
        if row.name != "self" and row.kind not in {"vararg", "kwarg"})
    callable_failure = (
        "resolved owner has no forward callable"
        if not forwards else
        "resolved owner has multiple forward callables"
        if len(forwards) > 1 else "")
    required = tuple(row for row in parameters if not row.has_default)
    selected = list(required)
    if not required:
        primary = next((row for row in parameters
                        if row.name in _KNOWN_INPUTS), None)
        if primary is not None:
            selected.append(primary)
    parameter_rows = tuple((row.name, row.kind, row.has_default)
                           for row in parameters)
    digest = _sha256(json.dumps(
        parameter_rows, sort_keys=True, separators=(",", ":")).encode())[:16]
    versions = {row.package: row.version
                for row in inventory.provenance.packages}
    literal_arguments = {
        row.name: False for row in parameters
        if row.name in {"use_cache", "return_dict", "output_attentions",
                        "output_hidden_states"}
    }
    prepared = prepare_document(dict(config), merge=False)
    if prepared.failure is not None:
        checkpoint_dtype = {"state": "failed", "detail": str(prepared.failure)}
        execution_dtype = "float32"
        values = {}
        sources = {}
        dtype_failure = "config document preparation failed"
    else:
        with bound_document(DocumentBinding("root", (), prepared)):
            dtype_value, dtype_source = _config_value(
                prepared, "dtype", ("torch_dtype",))
            values = {}
            sources = {}
            for key, aliases in (
                ("hidden_size", ("d_model", "model_dim")),
                ("in_channels", ("num_channels",)),
                ("joint_attention_dim", ("cross_attention_dim",
                                           "caption_projection_dim",
                                           "text_embed_dim", "cap_feat_dim",
                                           "context_in_dim", "text_dim")),
                ("pooled_projection_dim", ("projection_dim",)),
                ("patch_size", ("spatial_patch_size",)),
                ("patch_size_t", ("temporal_patch_size",)),
                ("max_position_embeddings", ("max_sequence_length",)),
            ):
                values[key], sources[key] = _config_value(prepared, key, aliases)
        normalised = _normalise_dtype(dtype_value)
        checkpoint_dtype = {**dtype_source, "value": dtype_value}
        # An absent/null deployment declaration does not become a checkpoint
        # fact.  float32 is only the execution probe's explicit default.
        execution_dtype = normalised or "float32"
        dtype_failure = (
            f"unknown declared dtype {dtype_value!r}"
            if dtype_source["state"] == "present" and dtype_value is not None
            and normalised is None else "")

    seq_capacity = _positive_int(values.get("max_position_embeddings"))
    sequence = min(seq_capacity, 2) if seq_capacity else 2
    hidden = _positive_int(values.get("hidden_size"))
    channels = _positive_int(values.get("in_channels"))
    context = _positive_int(values.get("joint_attention_dim"))
    pooled = _positive_int(values.get("pooled_projection_dim"))
    patch = values.get("patch_size")
    patch = (_positive_int(patch) or
             (_positive_int(patch[0]) if isinstance(patch, (list, tuple)) and patch else None))
    patch_t = _positive_int(values.get("patch_size_t"))
    spatial = max(2, patch or 1)
    temporal = max(1, patch_t or 1)
    argument_sources: dict[str, Any] = {}
    tensors = []
    unresolved = []
    for row in selected:
        name = row.name
        shape = None
        dtype = execution_dtype
        used = []
        if name in _INTEGER_INPUTS:
            shape, dtype, used = (1, sequence), "long", ["max_position_embeddings"]
        elif name in _FLOAT_SCALARS:
            shape, used = (1,), []
        elif name in {"sample", "hidden_states", "x"}:
            if channels:
                shape = ((1, channels, temporal, spatial, spatial)
                         if patch_t else (1, channels, spatial, spatial))
                used = ["in_channels", *( ["patch_size_t"] if patch_t else []),
                        "patch_size"]
            elif hidden:
                shape, used = (1, sequence, hidden), ["hidden_size"]
        elif name == "encoder_hidden_states" and context:
            shape, used = (1, sequence, context), ["joint_attention_dim"]
        elif name == "pooled_projections" and pooled:
            shape, used = (1, pooled), ["pooled_projection_dim"]
        elif name == "pixel_values" and channels:
            shape, used = (1, channels, spatial, spatial), ["in_channels", "patch_size"]
        elif name == "input_values":
            # Sequence length is a bounded recipe probe, not architecture.
            shape, used = (1, sequence), ["max_position_embeddings"]
        if shape is None:
            unresolved.append(name)
            continue
        tensors.append(TensorArgument(name, shape, dtype))
        argument_sources[name] = {
            "shape": list(shape), "dtype": dtype,
            "config_inputs": {key: sources.get(key) for key in used},
            "calculation": "bounded probe dimensions from resolved values",
        }

    resolution_status = (
        "ok" if not unresolved and not dtype_failure and not callable_failure
        else "failed")
    failures = []
    if callable_failure:
        failures.append(callable_failure)
    if dtype_failure:
        failures.append(dtype_failure)
    if unresolved:
        failures.append("unknown input meaning/dimensions: " + ", ".join(unresolved))
    failure_detail = "; ".join(failures)
    conditioning_inputs = {
        "encoder_hidden_states", "pooled_projections", "pixel_values",
        "input_values", "decoder_input_ids",
    }
    conditioning_present = any(
        row.name in conditioning_inputs for row in tensors)
    recipe = ExecutionRecipe(
        f"signature-{digest}", "callable_signature", "eval", "disabled",
        "unspecified", conditioning_present, execution_dtype, versions,
        tensor_arguments=tuple(tensors),
        literal_arguments=literal_arguments,
        flags={
            "source": "resolved_callable_signature",
            "resolution": ("exact" if len(forwards) == 1
                           else "absent" if not forwards else "ambiguous"),
            "callable": (forwards[0].symbol.qualified_name
                         if len(forwards) == 1 else "unresolved"),
            "parameters": [list(row) for row in parameter_rows],
            "checkpoint_dtype": checkpoint_dtype,
            "execution_dtype": execution_dtype,
            "argument_sources": argument_sources,
            "resolution_status": resolution_status,
        },
    )
    return RecipeResolution(
        resolution_status, checkpoint_dtype, execution_dtype, recipe,
        argument_sources, failure_detail)


def _known_dtype_failure(result: ObservationResult) -> bool:
    if result.status != "failed" or result.failure is None \
            or result.failure.kind != "ExecutionFailed":
        return False
    detail = result.failure.detail.lower()
    return any(marker in detail for marker in _KNOWN_GROUPED_DTYPE_ERRORS)


def _bf16_retry(recipe: ExecutionRecipe) -> ExecutionRecipe:
    tensors = tuple(dataclasses.replace(
        row, dtype="bfloat16" if row.dtype in {"float16", "float32"} else row.dtype)
        for row in recipe.tensor_arguments)
    flags = dict(recipe.flags)
    flags.update({"retry_of": recipe.recipe_id, "execution_dtype": "bfloat16",
                  "retry_reason": "known_grouped_mm_dtype_error"})
    return dataclasses.replace(
        recipe, recipe_id=f"{recipe.recipe_id}-bf16-retry",
        dtype="bfloat16", tensor_arguments=tensors, flags=flags)


def _run_signature_recipe(request: BuildRequest,
                          resolution: RecipeResolution) -> RecipeAttemptBundle:
    if resolution.status == "failed":
        first = ObservationResult(
            "failed", recipe=resolution.recipe,
            failure=Failure("ConfigurationFailed", "recipe_resolution",
                            resolution.failure_detail))
        return RecipeAttemptBundle(resolution, (first,))
    first = observe_in_subprocess(request, resolution.recipe)
    attempts = [first]
    if _known_dtype_failure(first) and resolution.execution_dtype != "bfloat16":
        attempts.append(observe_in_subprocess(request, _bf16_retry(resolution.recipe)))
    return RecipeAttemptBundle(resolution, tuple(attempts))


def _source_inputs(config: dict[str, Any], inventory: Any):
    context = ParseContext.build(config)
    bundle = context.source_bundle
    runtime_name = inventory.provenance.resolved_class.qualname
    architectures = dict(bundle.component_architectures)
    architectures["root"] = runtime_name
    bundle = dataclasses.replace(bundle, component_architectures=architectures)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    claims = static_claims_from_owner_graph(root.graph) if root.address_resolved else ()
    context._program_index = index
    ir = config_to_ir(config, parse_context=context)
    return context, index, root, claims, ir


def _static_relation_proofs(index: Any, inventory: Any,
                            relation_result: RelationObservationResult | None):
    if relation_result is None or relation_result.status != "ok":
        return ()
    observation = relation_result.observation
    assert observation is not None
    module_by_path = {row.path: row for row in inventory.modules}
    hashes = tuple(row.source_id.content_fingerprint for row in index.source_nodes)
    proofs = []
    layer_classes = {
        (module_by_path[row.path].class_ref.module,
         module_by_path[row.path].class_ref.qualname): module_by_path[row.path].class_ref
        for row in observation.boundaries if row.path in module_by_path
    }
    if len(layer_classes) == 1:
        proof = prove_recurrent_state_mix(
            index, next(iter(layer_classes.values())), hashes)
        if proof is not None:
            proofs.append(proof)

    last_order = max(row.call_order for row in observation.boundaries)
    final_shape = tuple(observation.boundaries[-1].outputs[0].shape)
    candidates = tuple(row for row in observation.sibling_calls
                       if row.call_order > last_order and row.inputs and row.outputs
                       and tuple(row.inputs[0].shape) == final_shape
                       and len(row.outputs[0].shape) < len(final_shape))
    if len(candidates) == 1:
        parent_path, _, stack_field = observation.stack_path.rpartition(".")
        parent = module_by_path.get(parent_path)
        prefix = f"{parent_path}." if parent_path else ""
        sibling_path = candidates[0].path
        if parent is not None and sibling_path.startswith(prefix):
            head_field = sibling_path[len(prefix):]
            if "." not in head_field:
                proof = prove_post_stack_collapse(
                    index, parent.class_ref, hashes,
                    stack_field=stack_field, head_field=head_field)
                if proof is not None:
                    proofs.append(proof)
    return tuple(proofs)


def _layer_schedule(ir: Any) -> dict[str, Any]:
    layers = tuple(ir.layers)
    groups = distinct_layer_groups(layers)
    signatures = [layer.signature() for layer in layers]
    return {
        "layer_count": len(layers),
        "period": detect_layer_period(signatures),
        "groups_in_encounter_order": [list(group["indices"]) for group in groups],
    }


def _construction_schedules(inventory: Any) -> list[dict[str, Any]]:
    """All exact numeric-child container schedules, without selecting a role."""
    modules = {row.path: row for row in inventory.modules}
    rows = []
    for parent in inventory.modules:
        if len(parent.children) < 2 or not all(name.isdigit() for name in parent.children):
            continue
        child_paths = tuple(
            f"{parent.path}.{name}".lstrip(".") for name in parent.children)
        signatures = []
        for path in child_paths:
            child = modules[path]
            payload = {
                "class": [child.class_ref.module, child.class_ref.qualname],
                "children": [
                    [name, modules[f"{path}.{name}"].class_ref.module,
                     modules[f"{path}.{name}"].class_ref.qualname]
                    for name in child.children
                ],
                "parameters": [
                    [item.name, list(item.shape), item.dtype]
                    for item in child.parameters
                ],
            }
            signatures.append(_sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")).encode()))
        group_order = []
        for signature in signatures:
            if signature not in group_order:
                group_order.append(signature)
        rows.append({
            "container_path": parent.path,
            "member_paths": list(child_paths),
            "signature_sequence": signatures,
            "groups_in_encounter_order": [
                [index for index, value in enumerate(signatures)
                 if value == signature]
                for signature in group_order
            ],
            "period": detect_layer_period(signatures),
        })
    return rows


def _summary_schedules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    largest = max(len(row["member_paths"]) for row in rows)
    return [{
        "container_path": row["container_path"],
        "member_count": len(row["member_paths"]),
        "groups_in_encounter_order": row["groups_in_encounter_order"],
        "period": row["period"],
    } for row in rows if len(row["member_paths"]) == largest]


def _readme(matrix: Mapping[str, Any]) -> str:
    lines = [
        "# S7 shadow disagreement matrix",
        "",
        "This is a 29-corpus + 10-TO_SERVE observation denominator. It is not "
        "a production dispatch table. Runtime names are addresses only; custom "
        "mechanism meanings require exact resolved source and existing facts.",
        "",
        "Every occurrence has construction, execution and projection axes. "
        "`no_recipe_attempted` identifies our missing probe; "
        "`unobserved_no_static_proof` identifies an attempted recipe that did "
        "not prove this occurrence. Under v2.6, every unresolved value is "
        "classified as `investigation_missing`, `structure_unaccounted`, or "
        "`mechanism_unresolved`; the last is legal only with its typed "
        "investigation receipt and concrete reason. S7 does not relabel an "
        "execution observation as a known mechanism. Full per-occurrence "
        "tables are the deterministic gzip JSON files under `models/`.",
        "",
        "Recipe status is reported per target. Checkpoint dtype (including "
        "absence/null) is kept separate from the execution dtype; a recorded "
        "bf16 retry never rewrites deployment evidence.",
        "",
        "| cohort | model | recipe | checkpoint dtype | execution dtype | retry | occurrences | construction conflicts | no recipe | "
        "attempted-unobserved | rendered | grouped | containers | projection "
        "unresolved | investigation missing | structure unaccounted | mechanism "
        "unresolved | relations |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in matrix["models"]:
        relations = ", ".join(row["relation_kinds"]) or "none"
        checkpoint = row["checkpoint_dtype"]
        checkpoint_text = f"{checkpoint.get('value')!s} ({checkpoint.get('state')})"
        lines.append(
            f"| {row['cohort']} | {row['model']} | "
            f"{row['recipe_resolution']}→{row['recipe_status']} | "
            f"{checkpoint_text} | {row['recipe_execution_dtype']} | "
            f"{row['recipe_retry_count']} | {row['occurrences']} | "
            f"{row['construction_conflicts']} | {row['no_recipe_attempted']} | "
            f"{row['unobserved_no_static_proof']} | {row['rendered']} | "
            f"{row['grouped']} | {row['non_architectural_container']} | "
            f"{row['projection_unresolved']} | {row['investigation_missing']} | "
            f"{row['structure_unaccounted']} | {row['mechanism_unresolved']} | "
            f"{relations} |")
    return "\n".join(lines) + "\n"


def _one(target: Mapping[str, Any], *, write_relations: bool,
         output: Path = OUTPUT) -> dict[str, Any]:
    input_path = ROOT / target["input"]
    _label, config = _read_payload(input_path)
    config_hash = _sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    request = (_s6_request(target["slug"], config_hash)
               or _request(config, target["slug"]))
    inventory_result = (_s6_inventory(target["slug"], config_hash)
                        or inventory_in_subprocess(request))
    if inventory_result.status != "ok":
        raise RuntimeError(
            f"{target['slug']}: inventory failed: {inventory_result.failure}")
    inventory = inventory_result.inventory
    assert inventory is not None
    context, index, root, static_claims, ir = _source_inputs(config, inventory)

    # Every target receives one neutral, signature-derived attempt.  A failed
    # invocation is still a typed attempt and is materially different from the
    # generator never trying the model at all.
    recipe_resolution = _signature_recipe(index, root, inventory, config)
    signature_bundle = _run_signature_recipe(request, recipe_resolution)
    signature_result = signature_bundle.final
    if write_relations:
        _write_gzip_json(
            output / "observations" / f"{target['slug']}.json.gz",
            signature_bundle.to_dict())

    relation_result = None
    recipe_spec = _relation_recipe(target["slug"])
    if recipe_spec is not None:
        recipe, stack_path = recipe_spec
        relation_result = observe_relations_in_subprocess(request, recipe, stack_path)
        if relation_result.status != "ok":
            raise RuntimeError(
                f"{target['slug']}: relation observation failed: "
                f"{relation_result.failure}")
        if write_relations:
            _write_gzip_json(
                output / "relations" / f"{target['slug']}.json.gz",
                relation_result.to_dict())

    execution_rows = _s6_observations(
        target["slug"], inventory.provenance.config_sha256)
    all_observations = (*execution_rows, *signature_bundle.attempts,
                        *((relation_result,) if relation_result else ()))
    facts = context.facts.typed_records()
    diagram = Diagram(ir)
    diagram.to_html(standalone=True)
    projection_claims = projection_claims_from_product(
        index=index, inventory=inventory, static_claims=static_claims,
        ir=ir, facts=facts, render_events=diagram.render_events())
    proofs = _static_relation_proofs(index, inventory, relation_result)
    relation_rows = relation_rows_from_evidence(
        inventory=inventory,
        relation_observations=(relation_result,) if relation_result else (),
        facts=facts, static_proofs=proofs)
    table = reconcile(
        model=target["model"], inventory=inventory,
        observations=all_observations,
        config_document=prepare_document(config, merge=False),
        static_claims=static_claims, projection_claims=projection_claims,
        relation_rows=relation_rows)
    findings = unresolved_axis_findings(table)
    return {
        "schema_version": 1,
        "target": dict(target),
        "inventory_provenance": dataclasses.asdict(inventory.provenance),
        "root_resolution": root.status,
        "source_index_fingerprint": index.fingerprint,
        "fact_keys_consumed": sorted({key for relation in relation_rows
                                      for key in relation.fact_keys}),
        "product_layer_schedule": _layer_schedule(ir),
        "construction_schedules": _construction_schedules(inventory),
        "blocking_findings": findings,
        "signature_recipe": {
            "status": recipe_resolution.status,
            "checkpoint_dtype": dict(recipe_resolution.checkpoint_dtype),
            "execution_dtype": signature_bundle.final.recipe.dtype,
            "retry_count": len(signature_bundle.attempts) - 1,
            "final_status": signature_bundle.final.status,
            "failure": (dataclasses.asdict(signature_bundle.final.failure)
                        if signature_bundle.final.failure else None),
        },
        "table": table.to_dict(),
    }


def generate(*, output: Path = OUTPUT, ci_shadow: bool = False) -> dict[str, Any]:
    targets = _targets()
    rows = []
    artifact_hashes = {}
    observation_hashes = {}
    for target in targets:
        print(f"S7 shadow {target['cohort']} {target['slug']}", file=sys.stderr,
              flush=True)
        artifact = _one(target, write_relations=not ci_shadow, output=output)
        table = artifact["table"]
        reason_counts = unresolved_reason_class_counts(table)
        summary = {
            **dict(target),
            "occurrences": len(table["occurrences"]),
            "relations": len(table["relations"]),
            "construction_conflicts": sum(
                row["construction"]["kind"] == "construction_conflict"
                for row in table["occurrences"]),
            "execution_unresolved": sum(
                row["execution"]["kind"] == "execution_unresolved"
                for row in table["occurrences"]),
            "no_recipe_attempted": sum(
                row["execution"]["kind"] == "execution_unresolved"
                and row["execution"]["reason"] == "no_recipe_attempted"
                for row in table["occurrences"]),
            "unobserved_no_static_proof": sum(
                row["execution"]["kind"] == "execution_unresolved"
                and row["execution"]["reason"] == "unobserved_no_static_proof"
                for row in table["occurrences"]),
            "rendered": sum(row["projection"]["kind"] == "rendered"
                            for row in table["occurrences"]),
            "grouped": sum(row["projection"]["kind"] == "grouped"
                           for row in table["occurrences"]),
            "non_architectural_container": sum(
                row["projection"]["kind"] == "non_architectural"
                and row["projection"]["reason"] == "container"
                for row in table["occurrences"]),
            "projection_unresolved": sum(
                row["projection"]["kind"] == "projection_unresolved"
                for row in table["occurrences"]),
            "investigation_missing": reason_counts["investigation_missing"],
            "structure_unaccounted": reason_counts["structure_unaccounted"],
            "mechanism_unresolved": reason_counts["mechanism_unresolved"],
            "relation_kinds": sorted({row["kind"] for row in table["relations"]}),
            "blocking_findings": len(artifact["blocking_findings"]),
            "recipe_resolution": artifact["signature_recipe"]["status"],
            "recipe_status": artifact["signature_recipe"]["final_status"],
            "recipe_retry_count": artifact["signature_recipe"]["retry_count"],
            "checkpoint_dtype": artifact["signature_recipe"]["checkpoint_dtype"],
            "recipe_execution_dtype": artifact["signature_recipe"]["execution_dtype"],
            "root_resolution": artifact["root_resolution"],
            "product_layer_schedule": artifact["product_layer_schedule"],
            "construction_schedules": _summary_schedules(
                artifact["construction_schedules"]),
        }
        rows.append(summary)
        if not ci_shadow:
            path = output / "models" / f"{target['slug']}.json.gz"
            _write_gzip_json(path, artifact)
            artifact_hashes[path.relative_to(output).as_posix()] = _sha256(
                path.read_bytes())
            observation_path = (
                output / "observations" / f"{target['slug']}.json.gz")
            observation_hashes[
                observation_path.relative_to(output).as_posix()] = _sha256(
                    observation_path.read_bytes())
    result = {
        "schema_version": 1,
        "denominator": {"corpus": 29, "to_serve": 10},
        "sources": {
            path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes())
            for path in (
                Path(__file__).resolve(),
                ROOT / "model_unfolder/evidence/reconciliation.py",
                ROOT / "model_unfolder/evidence/relation_source.py",
                ROOT / "physics/relation_observation.py",
            )
        },
        "models": rows,
        "artifacts": artifact_hashes,
        "observation_artifacts": observation_hashes,
    }
    if not ci_shadow:
        _write_json(output / "targets.json", {"models": list(targets)})
        _write_json(output / "matrix.json", result)
        (output / "README.md").write_text(_readme(result), encoding="utf-8")
    return result


def check(output: Path = OUTPUT) -> None:
    matrix_path = output / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["denominator"] != {"corpus": 29, "to_serve": 10}:
        raise ValueError("S7 matrix denominator is not 29 + 10")
    expected_targets = {row["slug"] for row in _targets()}
    if {row["slug"] for row in matrix["models"]} != expected_targets:
        raise ValueError("S7 matrix target set drifted")
    for relative, digest in matrix["sources"].items():
        if _sha256((ROOT / relative).read_bytes()) != digest:
            raise ValueError(f"S7 matrix is stale for source {relative}")
    for relative, digest in matrix["artifacts"].items():
        path = output / relative
        if _sha256(path.read_bytes()) != digest:
            raise ValueError(f"S7 artifact hash mismatch: {relative}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            artifact = json.load(stream)
        table = artifact.get("table") or {}
        if not table.get("occurrences"):
            raise ValueError(f"S7 artifact has an empty denominator: {relative}")
        summary = next(
            row for row in matrix["models"]
            if row["slug"] == artifact["target"]["slug"])
        if len(table["occurrences"]) != summary["occurrences"]:
            raise ValueError(f"S7 artifact silently dropped occurrences: {relative}")
        reason_counts = unresolved_reason_class_counts(table)
        if any(summary.get(key) != value
               for key, value in reason_counts.items()):
            raise ValueError(
                f"S7 unresolved reason-class counts drifted: {relative}")
        if artifact.get("blocking_findings") != unresolved_axis_findings(table):
            raise ValueError(
                f"S7 blocking findings drifted from reason classes: {relative}")
        if not artifact.get("blocking_findings"):
            raise ValueError(
                f"S7 unresolved-axis gate vacuously green in shadow: {relative}")
    if len(matrix.get("observation_artifacts") or {}) != 39:
        raise ValueError("S7 matrix needs one typed signature attempt per target")
    for relative, digest in matrix["observation_artifacts"].items():
        path = output / relative
        if _sha256(path.read_bytes()) != digest:
            raise ValueError(f"S7 observation hash mismatch: {relative}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version") != 2:
            raise ValueError(f"S7 observation bundle schema drifted: {relative}")
        attempts = tuple(ObservationResult.from_dict(row)
                         for row in payload.get("attempts") or ())
        if len(attempts) not in {1, 2}:
            raise ValueError(f"S7 observation retry count is invalid: {relative}")
        observation = attempts[-1]
        if attempts[0].recipe is None \
                or attempts[0].recipe.flags.get("source") != \
                "resolved_callable_signature":
            raise ValueError(
                f"S7 observation is not a signature-derived attempt: {relative}")
        if payload.get("retry_count") != len(attempts) - 1:
            raise ValueError(f"S7 observation retry record drifted: {relative}")
        if len(attempts) == 2 and not _known_dtype_failure(attempts[0]):
            raise ValueError(f"S7 observation has an unlawful retry: {relative}")
        slug = Path(relative).name.removesuffix(".json.gz")
        summary = next(row for row in matrix["models"] if row["slug"] == slug)
        resolution = payload.get("resolution") or {}
        if (summary.get("recipe_resolution") != resolution.get("status")
                or summary.get("recipe_status") != observation.status
                or summary.get("recipe_retry_count") != len(attempts) - 1
                or summary.get("checkpoint_dtype") != resolution.get("checkpoint_dtype")
                or summary.get("recipe_execution_dtype") != observation.recipe.dtype):
            raise ValueError(f"S7 recipe summary drifted: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-hard-inputs", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ci-shadow", action="store_true")
    args = parser.parse_args()
    if args.seed_hard_inputs:
        seed_hard_inputs()
    if args.check:
        check()
    elif args.ci_shadow:
        result = generate(ci_shadow=True)
        if len(result["models"]) != 39:
            raise ValueError("Linux S7 shadow did not execute all 39 targets")
    else:
        generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
