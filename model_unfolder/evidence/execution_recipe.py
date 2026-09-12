"""Shared signature-derived probe recipes, independent of any model denominator.

Recipe dimensions are probe values with provenance, never deployment facts.
Execution stays in the existing bounded child process and is positive-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Mapping

from .document import DocumentBinding, prepare_document
from .config_access import bound_document, resolve
from physics.execution_observation import (ExecutionRecipe, ObservationResult,
                                            TensorArgument, observe_in_subprocess)
from physics.instance_inventory import BuildRequest, Failure


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


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


def _validate_checkpoint_dtype(row: Mapping[str, Any]) -> None:
    """Validate the deployment-dtype channel without borrowing probe defaults."""
    state = row.get("state")
    if state == "failed":
        if (set(row) != {"state", "detail"}
                or not isinstance(row.get("detail"), str)
                or not row["detail"]):
            raise ValueError("failed checkpoint dtype needs one typed detail")
        return
    if set(row) != {
            "state", "path", "spelling", "provenance", "source_kind",
            "value"}:
        raise ValueError("checkpoint dtype schema is closed by state")
    if state == "absent":
        if (row["path"] is not None or row["spelling"] is not None
                or row["provenance"] != ""
                or row["source_kind"] != "checkpoint"
                or row["value"] is not None):
            raise ValueError("absent checkpoint dtype carries no deployment value")
    elif state == "ambiguous":
        if (row["path"] is not None or row["spelling"] is not None
                or row["provenance"] != ""
                or row["source_kind"] != "checkpoint"
                or row["value"] is not None):
            raise ValueError("ambiguous checkpoint dtype carries no selected value")
    elif state == "present":
        if (not isinstance(row["path"], str) or not row["path"]
                or not isinstance(row["spelling"], str) or not row["spelling"]
                or row["provenance"] != "checkpoint_declared"
                or row["source_kind"] != "checkpoint"):
            raise ValueError(
                "present checkpoint dtype needs its exact checkpoint occurrence")
    else:
        raise ValueError("checkpoint dtype state is closed")


def _validate_execution_dtype_source(
    row: Mapping[str, Any], execution_dtype: str,
    checkpoint_dtype: Mapping[str, Any],
) -> None:
    """Validate the independent execution-dtype source and its authority join."""
    kind = row.get("kind")
    if kind in {"checkpoint_declared", "class_default"}:
        if set(row) != {"kind", "value", "path", "spelling"}:
            raise ValueError("resolved execution dtype source schema is closed")
    elif kind == "probe_default":
        if (set(row) != {"kind", "value", "reason"}
                or not isinstance(row.get("reason"), str)
                or not row["reason"]):
            raise ValueError("probe-default dtype needs one exact reason")
    else:
        raise ValueError("execution dtype source is closed")
    if row.get("value") != execution_dtype:
        raise ValueError("execution dtype needs its own exact source record")
    if kind == "checkpoint_declared":
        if (checkpoint_dtype.get("state") != "present"
                or row["path"] != checkpoint_dtype.get("path")
                or row["spelling"] != checkpoint_dtype.get("spelling")
                or _normalise_dtype(checkpoint_dtype.get("value"))
                != execution_dtype):
            raise ValueError(
                "checkpoint-derived execution dtype must cite its exact declaration")
    elif kind == "class_default":
        if (checkpoint_dtype.get("state") != "absent"
                or row["path"] is not None or row["spelling"] is not None):
            raise ValueError(
                "class-default execution dtype cannot become checkpoint evidence")


@dataclasses.dataclass(frozen=True)
class RecipeResolution:
    """The recipe decision, separate from every execution attempt."""

    status: str
    checkpoint_dtype: Mapping[str, Any]
    execution_dtype: str
    recipe: ExecutionRecipe
    argument_sources: Mapping[str, Any]
    failure_detail: str = ""
    execution_dtype_source: Mapping[str, Any] = dataclasses.field(
        default_factory=dict)

    def __post_init__(self) -> None:
        if (not isinstance(self.status, str)
                or not isinstance(self.execution_dtype, str)
                or not isinstance(self.failure_detail, str)):
            raise TypeError("recipe resolution scalar fields retain native types")
        if (not isinstance(self.recipe, ExecutionRecipe)
                or not isinstance(self.checkpoint_dtype, Mapping)
                or not isinstance(self.argument_sources, Mapping)
                or not isinstance(self.execution_dtype_source, Mapping)):
            raise TypeError("recipe resolution carries typed records")
        if self.status not in {"ok", "failed"}:
            raise ValueError("recipe resolution status is closed")
        if self.status == "ok" and self.failure_detail:
            raise ValueError("a resolved recipe carries no failure")
        if self.status == "failed" and not self.failure_detail:
            raise ValueError("a failed recipe resolution names the unknown input")
        if self.execution_dtype != self.recipe.dtype:
            raise ValueError("recipe execution dtype must match its resolution")
        _validate_checkpoint_dtype(self.checkpoint_dtype)
        _validate_execution_dtype_source(
            self.execution_dtype_source, self.execution_dtype,
            self.checkpoint_dtype)
        required_flags = {
            "source": "resolved_callable_signature",
            "checkpoint_dtype": dict(self.checkpoint_dtype),
            "execution_dtype": self.execution_dtype,
            "execution_dtype_source": dict(self.execution_dtype_source),
            "argument_sources": dict(self.argument_sources),
            "resolution_status": self.status,
        }
        if any(self.recipe.flags.get(key) != value
               for key, value in required_flags.items()):
            raise ValueError(
                "recipe flags must equal the complete recipe resolution")

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RecipeResolution":
        if not isinstance(row, Mapping) or set(row) != {
                "status", "checkpoint_dtype", "execution_dtype", "recipe",
                "argument_sources", "failure_detail",
                "execution_dtype_source"}:
            raise ValueError("recipe resolution schema is closed")
        if (not isinstance(row["checkpoint_dtype"], Mapping)
                or not isinstance(row["argument_sources"], Mapping)
                or not isinstance(row["execution_dtype_source"], Mapping)
                or not isinstance(row["recipe"], Mapping)):
            raise TypeError("recipe resolution carries typed mapping fields")
        if (not isinstance(row["status"], str)
                or not isinstance(row["execution_dtype"], str)
                or not isinstance(row["failure_detail"], str)):
            raise TypeError("recipe resolution scalar fields retain native types")
        return cls(
            status=row["status"],
            checkpoint_dtype=dict(row["checkpoint_dtype"]),
            execution_dtype=row["execution_dtype"],
            recipe=ExecutionRecipe.from_dict(row["recipe"]),
            argument_sources=dict(row["argument_sources"]),
            failure_detail=row["failure_detail"],
            execution_dtype_source=dict(row["execution_dtype_source"]),
        )


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
        if dict(self.attempts[0].recipe.flags.get("checkpoint_dtype") or {}) \
                != dict(self.resolution.checkpoint_dtype):
            raise ValueError(
                "the first attempt must preserve the resolved checkpoint dtype")
        if dict(self.attempts[0].recipe.flags.get(
                "execution_dtype_source") or {}) != dict(
                    self.resolution.execution_dtype_source):
            raise ValueError(
                "the first attempt must preserve its execution dtype source")
        if self.resolution.status == "failed":
            first = self.attempts[0]
            if (len(self.attempts) != 1 or first.status != "failed"
                    or first.failure is None
                    or first.failure.kind != "ConfigurationFailed"
                    or first.failure.stage != "recipe_resolution"):
                raise ValueError(
                    "an unresolved recipe has one typed resolution failure")
        elif (self.attempts[0].status == "failed"
              and self.attempts[0].failure is not None
              and self.attempts[0].failure.kind == "ConfigurationFailed"
              and self.attempts[0].failure.stage == "recipe_resolution"):
            raise ValueError(
                "a resolved recipe cannot carry a recipe-resolution failure")
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

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RecipeAttemptBundle":
        if not isinstance(row, Mapping) or set(row) != {
                "schema_version", "resolution", "attempts", "final_status",
                "retry_count"}:
            raise ValueError("recipe attempt bundle schema is closed")
        if (not isinstance(row["schema_version"], int)
                or isinstance(row["schema_version"], bool)
                or row["schema_version"] != 2):
            raise ValueError("recipe attempt bundle schema version is closed")
        attempts = row["attempts"]
        if (not isinstance(row["resolution"], Mapping)
                or not isinstance(attempts, list)
                or any(not isinstance(attempt, Mapping) for attempt in attempts)):
            raise TypeError("recipe attempt bundle carries a result list")
        if (not isinstance(row["final_status"], str)
                or not isinstance(row["retry_count"], int)
                or isinstance(row["retry_count"], bool)):
            raise TypeError("recipe attempt summary retains native scalar types")
        bundle = cls(
            RecipeResolution.from_dict(row["resolution"]),
            tuple(ObservationResult.from_dict(attempt) for attempt in attempts),
        )
        if row["final_status"] != bundle.final.status:
            raise ValueError("recipe attempt final status drifted")
        if row["retry_count"] != len(bundle.attempts) - 1:
            raise ValueError("recipe attempt retry count drifted")
        if _json_bytes(bundle.to_dict()) != _json_bytes(dict(row)):
            raise ValueError("recipe attempt bundle is not canonical typed JSON")
        return bundle


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
        execution_dtype_source = {
            "kind": "probe_default", "value": execution_dtype,
            "reason": "config document preparation failed",
        }
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
        checkpoint_declared = (
            dtype_source["state"] == "ambiguous"
            or (dtype_source["state"] == "present"
                and dtype_source.get("source_kind") == "checkpoint"
                and dtype_source.get("provenance") == "checkpoint_declared")
        )
        if checkpoint_declared:
            checkpoint_dtype = {**dtype_source, "value": dtype_value}
        else:
            # The installed class may offer a useful execution default, but it
            # is not a checkpoint deployment declaration.  Keep the channels
            # separate instead of storing the class's value under a checkpoint
            # heading.
            checkpoint_dtype = {
                "state": "absent", "path": None, "spelling": None,
                "provenance": "", "source_kind": "checkpoint", "value": None,
            }
        execution_dtype = normalised or "float32"
        if normalised and dtype_source.get("source_kind") == "class_default":
            execution_dtype_source = {
                "kind": "class_default", "value": execution_dtype,
                "path": dtype_source.get("path"),
                "spelling": dtype_source.get("spelling"),
            }
        elif normalised and checkpoint_declared:
            execution_dtype_source = {
                "kind": "checkpoint_declared", "value": execution_dtype,
                "path": dtype_source.get("path"),
                "spelling": dtype_source.get("spelling"),
            }
        else:
            if dtype_source["state"] == "ambiguous":
                default_reason = "checkpoint dtype declarations are ambiguous"
            elif dtype_source["state"] == "present" and dtype_value is None:
                default_reason = "checkpoint dtype is explicit null"
            elif dtype_source["state"] == "present":
                default_reason = "checkpoint dtype is unsupported by the probe"
            else:
                default_reason = "checkpoint dtype is absent"
            execution_dtype_source = {
                "kind": "probe_default", "value": execution_dtype,
                "reason": default_reason,
            }
        if dtype_source["state"] == "ambiguous":
            dtype_failure = "ambiguous checkpoint dtype declarations"
        elif dtype_value is not None and normalised is None:
            dtype_failure = f"unknown resolved execution dtype {dtype_value!r}"
        else:
            dtype_failure = ""

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
        sequence_formula = {
            "operation": "min_positive_or_fallback",
            "operands": {"max_position_embeddings": seq_capacity,
                         "upper_bound": 2},
            "fallback": 2, "result": sequence,
        }
        calculations = {
            **{key: sequence_formula for key in _INTEGER_INPUTS},
            "timestep": {"operation": "constant_probe_shape", "result": [1]},
            "timesteps": {"operation": "constant_probe_shape", "result": [1]},
            "guidance": {"operation": "constant_probe_shape", "result": [1]},
            "sample": {"operation": "resolved_latent_shape",
                       "operands": {"in_channels": channels,
                                    "patch_size_t": patch_t,
                                    "patch_size": patch},
                       "result": list(shape)},
            "hidden_states": {"operation": "resolved_hidden_or_latent_shape",
                              "operands": {"hidden_size": hidden,
                                           "in_channels": channels,
                                           "sequence": sequence,
                                           "patch_size_t": patch_t,
                                           "patch_size": patch},
                              "result": list(shape)},
            "x": {"operation": "resolved_hidden_or_latent_shape",
                  "operands": {"hidden_size": hidden,
                               "in_channels": channels,
                               "sequence": sequence,
                               "patch_size_t": patch_t,
                               "patch_size": patch},
                  "result": list(shape)},
            "encoder_hidden_states": {
                "operation": "resolved_context_shape",
                "operands": {"sequence": sequence,
                             "joint_attention_dim": context},
                "result": list(shape)},
            "pooled_projections": {
                "operation": "resolved_pooled_shape",
                "operands": {"pooled_projection_dim": pooled},
                "result": list(shape)},
            "pixel_values": {"operation": "resolved_image_shape",
                             "operands": {"in_channels": channels,
                                          "patch_size": patch},
                             "result": list(shape)},
            "input_values": sequence_formula,
        }
        argument_sources[name] = {
            "shape": list(shape), "dtype": dtype,
            "config_inputs": {key: sources.get(key) for key in used},
            "calculation": calculations[name],
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
            "execution_dtype_source": execution_dtype_source,
            "argument_sources": argument_sources,
            "resolution_status": resolution_status,
        },
    )
    return RecipeResolution(
        resolution_status, checkpoint_dtype, execution_dtype, recipe,
        argument_sources, failure_detail, execution_dtype_source)


def _known_dtype_failure(result: ObservationResult) -> bool:
    if result.status != "failed" or result.failure is None \
            or result.failure.kind != "ExecutionFailed":
        return False
    # The operation and dtype complaint must occur in the same causal exception
    # line.  A helper name in an earlier traceback frame cannot authorize a
    # retry for a later, unrelated generic dtype error.
    for line in result.failure.detail.lower().splitlines():
        if (re.search(
                r"(?<![a-z0-9_])(?:aten\.)?_?grouped_mm(?![a-z0-9_])",
                line)
                and any(marker in line
                        for marker in _KNOWN_GROUPED_DTYPE_ERRORS)):
            return True
    # Current Torch FakeTensor reports the exact failing operator in the first
    # diagnostic line and the dtype assertion at the end of that same captured
    # traceback.  Treat that closed pair as one causal exception; neither a
    # helper name nor an earlier unrelated frame can authorize the retry.
    stderr_lines = [line.strip().lower()
                    for line in result.stderr.splitlines() if line.strip()]
    if not stderr_lines:
        return False
    header = re.search(
        r"\] failed while attempting to run meta for "
        r"aten\._grouped_mm\.default$",
        stderr_lines[0])
    final = stderr_lines[-1]
    return bool(header and "] runtimeerror:" in final and any(
        marker in final for marker in _KNOWN_GROUPED_DTYPE_ERRORS))


def _bf16_retry(recipe: ExecutionRecipe) -> ExecutionRecipe:
    tensors = tuple(dataclasses.replace(
        row, dtype="bfloat16" if row.dtype in {"float16", "float32"} else row.dtype)
        for row in recipe.tensor_arguments)
    flags = dict(recipe.flags)
    flags.update({"retry_of": recipe.recipe_id, "execution_dtype": "bfloat16",
                  "retry_reason": "known_grouped_mm_dtype_error",
                  "execution_dtype_source": {
                      "kind": "known_grouped_mm_retry", "value": "bfloat16",
                      "from": recipe.dtype,
                  }})
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




def with_optional_concat_probe(resolution, bindings):
    """Try a source-addressed optional dictionary lane, without claiming activity.

    This is a degenerate stimulus: one dictionary value fills a constructed
    affine input width; a second supplies an empty sequence through an explicit
    flatten/project/reshape lane. Runtime success, not this recipe, establishes
    which calls execute. No deployment dimensions are inferred.
    """
    from .diffusion_stream import local_lineage_at_callable
    from .framework_operations import functional_operation_protocol_for_call
    from .program_index import SymbolId
    index = bindings.index
    root = bindings.symbol_at("")
    if root is None or resolution.status != "ok":
        return resolution
    forward = index.callable_by_symbol(SymbolId(root.source, root.qualified_name + ".forward"))
    if forward is None:
        return resolution
    root_formals = {p.name for p in forward.params if p.name != "self"}
    candidates = []
    def member(expr):
        if expr is not None and expr.kind == "attribute" and len(expr.children) == 1:
            base = expr.children[0]
            if base is not None and base.kind == "name" and base.name == "self":
                return expr.name
        return None
    for invocation in index.calls_in(forward.symbol):
        helper_name = member(invocation.callee)
        if helper_name is None:
            continue
        helper = index.callable_by_symbol(SymbolId(root.source, root.qualified_name + "." + helper_name))
        if helper is None:
            continue
        parameters = [p for p in helper.params if p.name != "self"]
        actuals = dict(zip((p.name for p in parameters), invocation.args))
        actuals.update(dict(invocation.kwargs))
        forwarded = {name: actual.name for name, actual in actuals.items()
                     if actual.kind == "name" and actual.name in root_formals}
        calls = {c.span: c for c in index.calls_in(helper.symbol)}
        lineage = local_lineage_at_callable(index, helper)
        def resolve_value(expr, cutoff, guard, seen=()):
            while expr is not None and expr.kind == "name":
                if expr.name in seen:
                    return None
                seen = (*seen, expr.name)
                expr, unresolved = lineage.definition(expr.name, cutoff, guard)
                if unresolved:
                    return None
            return expr
        def dictionary_value(expr, cutoff, guard):
            expr = resolve_value(expr, cutoff, guard)
            call = calls.get(expr.span) if expr is not None else None
            if call is None or call.callee.kind != "attribute" or call.callee.name != "get" \
                    or len(call.args) != 1 or call.args[0].kind != "constant" \
                    or not isinstance(call.args[0].const_value, str):
                return None
            receiver = call.receiver
            if receiver is None or receiver.kind != "name" or receiver.name not in forwarded:
                return None
            key = call.args[0].const_value
            if not key or "." in key:
                return None
            return forwarded[receiver.name] + "." + key
        for concat in calls.values():
            protocol = functional_operation_protocol_for_call(index, concat)
            if protocol is None or protocol.kind != "concat" or not concat.args:
                continue
            values = concat.args[0]
            if values.kind not in {"list", "tuple"} or len(values.children) != 2:
                continue
            direct = dictionary_value(values.children[0], concat.span, concat.guard)
            shaped = resolve_value(values.children[1], concat.span, concat.guard)
            reshape = calls.get(shaped.span) if shaped is not None else None
            if not direct or reshape is None or reshape.callee.name != "reshape":
                continue
            projected = resolve_value(reshape.receiver, reshape.span, reshape.guard)
            projection = calls.get(projected.span) if projected is not None else None
            projection_path = member(projection.callee) if projection is not None else None
            if projection_path is None or len(projection.args) != 1 \
                    or bindings.symbol_at(projection_path) is None \
                    or not bindings.route_forwards_unmodified(projection_path):
                continue
            flatten = calls.get(projection.args[0].span)
            if flatten is None or flatten.callee.name != "flatten" or flatten.args or flatten.kwargs:
                continue
            empty = dictionary_value(flatten.receiver, flatten.span, flatten.guard)
            if empty is None or empty.split(".")[0] != direct.split(".")[0]:
                continue
            for consumer in calls.values():
                target = member(consumer.callee)
                if target is None or not consumer.args:
                    continue
                value = resolve_value(consumer.args[0], consumer.span, consumer.guard)
                cast = calls.get(value.span) if value is not None else None
                if cast is not None and cast.callee.kind == "attribute" and cast.callee.name == "to":
                    value = resolve_value(cast.receiver, cast.span, cast.guard)
                if value is None or value.span != concat.span:
                    continue
                target_symbol = bindings.symbol_at(target)
                if target_symbol is None or not bindings.route_forwards_unmodified(target):
                    continue
                target_forward = index.callable_by_symbol(SymbolId(target_symbol.source, target_symbol.qualified_name + ".forward"))
                if target_forward is None:
                    continue
                formal = next((p.name for p in target_forward.params if p.name != "self"), None)
                affines = [c for c in index.calls_in(target_forward.symbol)
                           if member(c.callee) is not None and c.args
                           and c.args[0].kind == "name" and c.args[0].name == formal
                           and bindings.primitive_at(target + "." + member(c.callee)) == "linear"]
                if not affines:
                    continue
                affine = min(affines, key=lambda c: (c.span.line, c.span.col))
                affine_path = target + "." + member(affine.callee)
                weights = [p for p in bindings._modules[affine_path].parameters if p.name == "weight" and len(p.shape) == 2]
                if len(weights) != 1 or weights[0].shape[1] <= 0:
                    continue
                width = weights[0].shape[1]
                spans = (invocation.span, concat.span, reshape.span, projection.span,
                         flatten.span, consumer.span, affine.span)
                evidence = {"kind": "degenerate_source_port_probe", "branch_guard": "unresolved",
                            "source_refs": sorted({f"sha256:{s.source.content_fingerprint}:{s.line}:{s.col}" for s in spans}),
                            "affine_parameter": affine_path + ".weight", "affine_shape": list(weights[0].shape),
                            "empty_sequence_length": 0, "direct_feature_width": width,
                            "meaning": "attempted input ports only; no deployment shape or active-branch claim"}
                candidates.append((direct, empty, width, evidence))
    unique = {(a, b, width): evidence for a, b, width, evidence in candidates}
    if len(unique) != 1:
        return resolution
    (direct, empty, width), evidence = next(iter(unique.items()))
    additions = (TensorArgument(direct, (1, width), resolution.execution_dtype),
                 TensorArgument(empty, (1, 0), resolution.execution_dtype))
    sources = {**resolution.argument_sources, **{row.name: {
        "shape": list(row.shape), "dtype": row.dtype, "calculation": evidence,
        "config_inputs": {}} for row in additions}}
    flags = {**resolution.recipe.flags, "argument_sources": sources}
    recipe = dataclasses.replace(resolution.recipe,
        tensor_arguments=resolution.recipe.tensor_arguments + additions, flags=flags)
    return dataclasses.replace(resolution, recipe=recipe, argument_sources=sources)
