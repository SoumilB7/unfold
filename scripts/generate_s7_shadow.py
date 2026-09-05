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
from model_unfolder.evidence.claim_evidence import qualify_config_value_fact
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reconciliation import (
    projection_claims_from_product, reconcile, relation_rows_from_evidence,
    static_claims_from_owner_graph, unresolved_axis_findings,
    unresolved_reason_class_counts,
)
from model_unfolder.evidence.relation_source import prove_post_stack_collapse
from model_unfolder.evidence.relation_probe import (
    RelationProbePlanReceipt, resolve_relation_probes,
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

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n").encode("utf-8")


_SEMANTIC_ENVIRONMENT_PATHS = frozenset({
    ("inventory_provenance", "environment"),
    ("attempts", "*", "observation", "provenance", "environment"),
    ("attempts", "*", "provenance", "environment"),
    ("results", "*", "observation", "provenance", "environment"),
    ("results", "*", "provenance", "environment"),
})
_SEMANTIC_DIAGNOSTIC_PATHS = frozenset({
    ("attempts", "*", "stdout"),
    ("attempts", "*", "stderr"),
    ("results", "*", "stdout"),
    ("results", "*", "stderr"),
})


def _semantic_payload(value: Any, _path: tuple[str, ...] = ()) -> Any:
    """Normalize only the closed, schema-addressed host metadata surface.

    Field names are not authority.  An architectural payload may itself have
    a field called ``environment`` or ``stderr``; recursively erasing every
    such spelling would let a real evidence change pass the Linux equality
    gate.  The wildcard here denotes only a list position in one of the two
    persisted S7 bundle schemas.
    """
    if _path in _SEMANTIC_DIAGNOSTIC_PATHS:
        return "<diagnostic-capture>"
    if isinstance(value, Mapping):
        result = {
            key: _semantic_payload(child, (*_path, str(key)))
            for key, child in value.items()
        }
        if _path in _SEMANTIC_ENVIRONMENT_PATHS:
            environment = dict(result)
            for key in ("platform", "network", "python"):
                if key in environment:
                    environment[key] = f"<{key}>"
            return environment
        return result
    if isinstance(value, (list, tuple)):
        return [_semantic_payload(child, (*_path, "*")) for child in value]
    return value


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


def _generation_sources(targets: tuple[dict[str, Any], ...]) -> tuple[Path, ...]:
    """The closed input/code dependency surface of the persisted S7 result.

    S7 invokes the real parser, source index and physics workers.  Maintaining a
    hand-picked import list made freshness self-referential: a dependency could
    change without invalidating the matrix.  Hash the complete production
    Python surface plus every exact denominator input instead.  This is a
    conservative invalidation boundary, never an architectural classifier.
    """
    paths = {
        Path(__file__).resolve(),
        *(ROOT / row["input"] for row in targets),
        *(ROOT / "model_unfolder").rglob("*.py"),
        *(ROOT / "model_unfolder").rglob("*.yaml"),
        *(ROOT / "model_unfolder").rglob("*.yml"),
        *(ROOT / "physics").rglob("*.py"),
        *(ROOT / "verification" / "s6" / "pilots").rglob("*.json"),
    }
    return tuple(sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda path: path.relative_to(ROOT).as_posix()))


def _source_hashes(targets: tuple[dict[str, Any], ...]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes())
        for path in _generation_sources(targets)
    }


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
         output: Path = OUTPUT) -> tuple[
             dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    if write_relations:
        _write_gzip_json(
            output / "observations" / f"{target['slug']}.json.gz",
            signature_bundle.to_dict())

    probe_resolution = resolve_relation_probes(
        index, inventory, signature_bundle.final)
    relation_results = tuple(
        observe_relations_in_subprocess(
            request, plan.execution_recipe(), plan.stack_path)
        for plan in probe_resolution.plans)
    relation_payload = {
        "schema_version": 3,
        "probe_resolution": probe_resolution.to_dict(),
        "results": [row.to_dict() for row in relation_results],
    }
    if write_relations:
        _write_gzip_json(
            output / "relations" / f"{target['slug']}.json.gz",
            relation_payload)

    execution_rows = _s6_observations(
        target["slug"], inventory.provenance.config_sha256)
    all_observations = (*execution_rows, *signature_bundle.attempts,
                        *relation_results)
    raw_facts = context.facts.typed_records()
    # S7 may qualify only the semantic VALUE strength that is already proven by
    # this fact's exact reader-authored consumption events.  No fact-key table,
    # span presence, or status label is evidence.  Other semantic kinds remain
    # visible projection disagreements until their readers emit typed proofs in
    # S9.
    root_binding = context.prepared_documents.get("root")
    if not isinstance(root_binding, DocumentBinding):
        raise ValueError("S7 source parse did not retain its prepared root document")
    prepared = root_binding.prepared
    facts = {
        key: qualify_config_value_fact(
            fact, context.config_access.events, prepared)
        for key, fact in raw_facts.items()
    }
    diagram = Diagram(ir)
    diagram.to_html(standalone=True)
    projection_claims = projection_claims_from_product(
        index=index, inventory=inventory, static_claims=static_claims,
        ir=ir, facts=facts, render_events=diagram.render_events())
    proofs = set()
    for relation_result in relation_results:
        proofs.update(_static_relation_proofs(
            index, inventory, relation_result))
    relation_rows = relation_rows_from_evidence(
        inventory=inventory,
        relation_observations=relation_results,
        facts=facts, static_proofs=tuple(proofs))
    table = reconcile(
        model=target["model"], inventory=inventory,
        observations=all_observations,
        config_document=prepared,
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
        "relation_probe_resolution": probe_resolution.to_dict(),
        "blocking_findings": findings,
        "signature_recipe": {
            "status": recipe_resolution.status,
            "checkpoint_dtype": dict(recipe_resolution.checkpoint_dtype),
            "execution_dtype": signature_bundle.final.recipe.dtype,
            "execution_dtype_source": dict(signature_bundle.final.recipe.flags[
                "execution_dtype_source"]),
            "initial_execution_dtype_source": dict(
                recipe_resolution.execution_dtype_source),
            "retry_count": len(signature_bundle.attempts) - 1,
            "final_status": signature_bundle.final.status,
            "failure": (dataclasses.asdict(signature_bundle.final.failure)
                        if signature_bundle.final.failure else None),
        },
        "table": table.to_dict(),
    }, signature_bundle.to_dict(), relation_payload


def _assert_model_summary_matches(actual: Mapping[str, Any],
                                  expected: Mapping[str, Any]) -> None:
    """Fail at the first exact model field that differs across environments."""
    slug = actual.get("slug")
    if slug != expected.get("slug"):
        raise ValueError(
            f"live Linux S7 shadow expected slug {expected.get('slug')!r}, "
            f"got {slug!r}")
    keys = sorted(set(actual) | set(expected))
    changed = [key for key in keys if actual.get(key) != expected.get(key)]
    if changed:
        detail = "; ".join(
            f"{key}: committed={expected.get(key)!r}, live={actual.get(key)!r}"
            for key in changed)
        raise ValueError(
            f"live Linux S7 shadow model {slug!r} disagrees: {detail}")


def generate(*, output: Path = OUTPUT, ci_shadow: bool = False,
             expected_models: Mapping[str, Mapping[str, Any]] | None = None,
             ) -> dict[str, Any]:
    targets = _targets()
    rows = []
    artifact_hashes = {}
    observation_hashes = {}
    logical_artifact_hashes = {}
    logical_observation_hashes = {}
    relation_artifact_hashes = {}
    logical_relation_hashes = {}
    for target in targets:
        print(f"S7 shadow {target['cohort']} {target['slug']}", file=sys.stderr,
              flush=True)
        artifact, observation_payload, relation_payload = _one(
            target, write_relations=not ci_shadow, output=output)
        logical_artifact_hashes[f"models/{target['slug']}.json.gz"] = _sha256(
            _json_bytes(_semantic_payload(artifact)))
        logical_observation_hashes[
            f"observations/{target['slug']}.json.gz"] = _sha256(
                _json_bytes(_semantic_payload(observation_payload)))
        logical_relation_hashes[
            f"relations/{target['slug']}.json.gz"] = _sha256(
                _json_bytes(_semantic_payload(relation_payload)))
        # The complete typed attempt is persisted separately.  Re-open it on a
        # write run; on a CI shadow the generator retained the same bundle only
        # inside _one, so the model artifact's recipe summary is the exact
        # deterministic comparison surface.  Detailed observation freshness is
        # additionally closed by source/input hashes and check() below.
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
            "qualified_fact_citations": sum(
                len(row["projection"].get("fact_claim_proofs") or ())
                for row in table["occurrences"]),
            "unqualified_fact_citations": sum(
                len(row["projection"].get("unqualified_fact_keys") or ())
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
            "recipe_execution_dtype_source": artifact[
                "signature_recipe"]["execution_dtype_source"],
            "relation_probe_status": artifact[
                "relation_probe_resolution"]["status"],
            "relation_recipe_attempts": len(relation_payload["results"]),
            "relation_recipe_ok": sum(
                row.get("status") == "ok" for row in relation_payload["results"]),
            "relation_recipe_failed": sum(
                row.get("status") == "failed"
                for row in relation_payload["results"]),
            "root_resolution": artifact["root_resolution"],
            "product_layer_schedule": artifact["product_layer_schedule"],
            "construction_schedules": _summary_schedules(
                artifact["construction_schedules"]),
        }
        rows.append(summary)
        if ci_shadow and expected_models is not None:
            expected = expected_models.get(target["slug"])
            if expected is None:
                raise ValueError(
                    f"live Linux S7 shadow has no committed model row for "
                    f"{target['slug']!r}")
            _assert_model_summary_matches(summary, expected)
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
            with gzip.open(observation_path, "rt", encoding="utf-8") as stream:
                logical_observation_hashes[
                    observation_path.relative_to(output).as_posix()] = _sha256(
                        _json_bytes(_semantic_payload(json.load(stream))))
            relation_path = output / "relations" / f"{target['slug']}.json.gz"
            relation_artifact_hashes[
                relation_path.relative_to(output).as_posix()] = _sha256(
                    relation_path.read_bytes())
            with gzip.open(relation_path, "rt", encoding="utf-8") as stream:
                logical_relation_hashes[
                    relation_path.relative_to(output).as_posix()] = _sha256(
                        _json_bytes(_semantic_payload(json.load(stream))))
    result = {
        "schema_version": 1,
        "denominator": {"corpus": 29, "to_serve": 10},
        "sources": _source_hashes(targets),
        "models": rows,
        "artifacts": artifact_hashes,
        "observation_artifacts": observation_hashes,
        "logical_artifacts": logical_artifact_hashes,
        "logical_observation_artifacts": logical_observation_hashes,
        "relation_artifacts": relation_artifact_hashes,
        "logical_relation_artifacts": logical_relation_hashes,
    }
    if not ci_shadow:
        _write_json(output / "targets.json", {"models": list(targets)})
        _write_json(output / "matrix.json", result)
        (output / "README.md").write_text(_readme(result), encoding="utf-8")
    return result


def _validate_relation_payload(
    payload: Mapping[str, Any], relative: str,
    expected_base_recipe: ExecutionRecipe,
) -> tuple[RelationObservationResult, ...]:
    """Close the persisted plan -> execution-result join.

    Hash validity is not type validity.  Reconstruct every runtime result, then
    require a one-for-one partition with the exact planned recipe and stack.
    This prevents a stale, duplicated, reordered, or cross-stack result from
    satisfying a relation probe merely because the surrounding JSON hashes.
    """
    if payload.get("schema_version") != 3:
        raise ValueError(f"S7 relation resolution is malformed: {relative}")
    resolution = payload.get("probe_resolution")
    raw_results = payload.get("results")
    if not isinstance(resolution, Mapping) or not isinstance(raw_results, list):
        raise ValueError(f"S7 relation resolution is malformed: {relative}")
    plans = resolution.get("plans")
    issues = resolution.get("issues")
    status = resolution.get("status")
    if not isinstance(plans, list) or not isinstance(issues, list) or status not in {
            "resolved", "partial", "unresolved", "absent", "failed"}:
        raise ValueError(f"S7 relation probe resolution is malformed: {relative}")
    failure_kind = resolution.get("failure_kind")
    failure_detail = resolution.get("failure_detail")
    if resolution.get("semantic_negative") is not False:
        raise ValueError(f"S7 relation probe claimed a semantic negative: {relative}")
    if status == "failed":
        if (failure_kind not in {
                "base_execution_failed", "provenance_mismatch"}
                or not isinstance(failure_detail, str) or not failure_detail):
            raise ValueError(
                f"S7 relation probe failure payload is invalid: {relative}")
    elif failure_kind or failure_detail:
        raise ValueError(
            f"S7 nonfailed relation probe carries a failure: {relative}")
    if ((status == "resolved" and (not plans or issues))
            or (status == "partial" and (not plans or not issues))
            or (status == "unresolved" and (plans or not issues))
            or (status in {"absent", "failed"} and (plans or issues))):
        raise ValueError(f"S7 relation probe status payload is invalid: {relative}")

    plan_rows: list[tuple[str, str, ExecutionRecipe]] = []
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise ValueError(f"S7 relation plan is malformed: {relative}")
        stack = plan.get("stack_path")
        recipe_row = plan.get("recipe")
        if not isinstance(stack, str) or not stack or not isinstance(
                recipe_row, Mapping):
            raise ValueError(f"S7 relation plan lacks identity: {relative}")
        try:
            recipe = ExecutionRecipe.from_dict(recipe_row)
            base_recipe = ExecutionRecipe.from_dict(plan.get("base_recipe"))
            persisted_receipt = RelationProbePlanReceipt.from_dict(
                plan.get("receipt"))
            computed_receipt = RelationProbePlanReceipt.from_payload(dict(plan))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"S7 relation plan recipe is malformed: {relative}") from exc
        if recipe.flags.get("relation_stack_path") != stack:
            raise ValueError(f"S7 relation plan recipe/stack drifted: {relative}")
        if (recipe.flags.get("base_recipe_id") != base_recipe.recipe_id
                or persisted_receipt.base_recipe_id != base_recipe.recipe_id):
            raise ValueError(
                f"S7 relation plan base-recipe identity drifted: {relative}")
        if base_recipe != expected_base_recipe:
            raise ValueError(
                f"S7 relation plan cites the wrong observation recipe: {relative}")
        if persisted_receipt != computed_receipt:
            raise ValueError(f"S7 relation plan receipt drifted: {relative}")
        if (persisted_receipt.recipe_id != recipe.recipe_id
                or persisted_receipt.stack_path != stack):
            raise ValueError(f"S7 relation plan identity drifted: {relative}")
        plan_rows.append((recipe.recipe_id, stack, recipe))
    plan_keys = tuple((recipe_id, stack)
                      for recipe_id, stack, _recipe in plan_rows)
    if len(plan_keys) != len(set(plan_keys)):
        raise ValueError(f"S7 relation plans are not occurrence-exact: {relative}")

    try:
        results = tuple(RelationObservationResult.from_dict(row)
                        for row in raw_results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"S7 relation result is malformed: {relative}") from exc
    result_rows: list[tuple[str, str, ExecutionRecipe]] = []
    for result in results:
        if result.recipe is None:
            raise ValueError(f"S7 relation result lacks recipe identity: {relative}")
        stack = result.recipe.flags.get("relation_stack_path")
        if not isinstance(stack, str) or not stack:
            raise ValueError(f"S7 relation result lacks stack identity: {relative}")
        if result.observation is not None and result.observation.stack_path != stack:
            raise ValueError(
                f"S7 relation result observation/stack drifted: {relative}")
        result_rows.append((result.recipe.recipe_id, stack, result.recipe))
    result_keys = tuple((recipe_id, stack)
                        for recipe_id, stack, _recipe in result_rows)
    if result_keys != plan_keys:
        raise ValueError(
            f"S7 relation plan/result partition drifted: {relative}")
    if any(result_recipe != plan_recipe for (
            _result_id, _result_stack, result_recipe), (
            _plan_id, _plan_stack, plan_recipe) in zip(result_rows, plan_rows)):
        raise ValueError(f"S7 relation plan/result recipe drifted: {relative}")
    return results


def _validate_target_metadata(
    matrix: Mapping[str, Any], targets_payload: Mapping[str, Any],
    target_rows: tuple[dict[str, Any], ...],
) -> None:
    expected = [dict(row) for row in target_rows]
    if targets_payload != {"models": expected}:
        raise ValueError("S7 targets artifact does not equal the exact denominator")
    models = matrix.get("models")
    if not isinstance(models, list):
        raise ValueError("S7 matrix target set drifted")
    actual = [
        {key: row.get(key) for key in ("cohort", "slug", "model", "input")}
        for row in models if isinstance(row, Mapping)
    ]
    if len(actual) != len(models) or actual != expected:
        raise ValueError("S7 matrix target set drifted")


def _require_schema_version(
    payload: Mapping[str, Any], expected: int, label: str,
) -> None:
    version = payload.get("schema_version")
    if (not isinstance(version, int) or isinstance(version, bool)
            or version != expected):
        raise ValueError(f"{label} schema version drifted")


def _validate_relation_cross_file(
    *, payload: Mapping[str, Any], results: tuple[RelationObservationResult, ...],
    artifact: Mapping[str, Any], summary: Mapping[str, Any], relative: str,
) -> None:
    resolution = payload["probe_resolution"]
    if (artifact.get("relation_probe_resolution") != resolution
            or summary.get("relation_probe_status") != resolution["status"]
            or summary.get("relation_recipe_attempts") != len(results)
            or summary.get("relation_recipe_ok") != sum(
                row.status == "ok" for row in results)
            or summary.get("relation_recipe_failed") != sum(
                row.status == "failed" for row in results)):
        raise ValueError(f"S7 relation summary drifted: {relative}")


def check(output: Path = OUTPUT) -> None:
    matrix_path = output / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    _require_schema_version(matrix, 1, "S7 matrix")
    if matrix["denominator"] != {"corpus": 29, "to_serve": 10}:
        raise ValueError("S7 matrix denominator is not 29 + 10")
    target_rows = _targets()
    expected_targets = {row["slug"] for row in target_rows}
    expected_by_slug = {row["slug"]: dict(row) for row in target_rows}
    targets_payload = json.loads(
        (output / "targets.json").read_text(encoding="utf-8"))
    _validate_target_metadata(matrix, targets_payload, target_rows)
    if (len(matrix["models"]) != len(expected_targets)
            or {row["slug"] for row in matrix["models"]} != expected_targets):
        raise ValueError("S7 matrix target set drifted")
    if matrix.get("sources") != _source_hashes(target_rows):
        raise ValueError("S7 matrix dependency surface is stale")
    for relative, digest in matrix["sources"].items():
        if _sha256((ROOT / relative).read_bytes()) != digest:
            raise ValueError(f"S7 matrix is stale for source {relative}")
    artifacts_by_slug: dict[str, Mapping[str, Any]] = {}
    summaries_by_slug = {row["slug"]: row for row in matrix["models"]}
    for relative, digest in matrix["artifacts"].items():
        path = output / relative
        if _sha256(path.read_bytes()) != digest:
            raise ValueError(f"S7 artifact hash mismatch: {relative}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            artifact = json.load(stream)
        try:
            _require_schema_version(artifact, 1, "S7 model artifact")
        except ValueError as exc:
            raise ValueError(
                f"S7 model artifact schema version drifted: {relative}") from exc
        if matrix.get("logical_artifacts", {}).get(relative) != _sha256(
                _json_bytes(_semantic_payload(artifact))):
            raise ValueError(f"S7 logical artifact hash mismatch: {relative}")
        table = artifact.get("table") or {}
        if not table.get("occurrences"):
            raise ValueError(f"S7 artifact has an empty denominator: {relative}")
        slug = Path(relative).name.removesuffix(".json.gz")
        if (artifact.get("target") != expected_by_slug.get(slug)
                or artifact["target"]["slug"] != slug):
            raise ValueError(f"S7 model artifact target drifted: {relative}")
        artifacts_by_slug[slug] = artifact
        summary = summaries_by_slug[slug]
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
    model_keys = {f"models/{slug}.json.gz" for slug in expected_targets}
    observation_keys = {
        f"observations/{slug}.json.gz" for slug in expected_targets}
    relation_keys = {f"relations/{slug}.json.gz" for slug in expected_targets}
    if (set(matrix.get("artifacts") or {}) != model_keys
            or set(matrix.get("logical_artifacts") or {}) != model_keys
            or set(matrix.get("observation_artifacts") or {})
            != observation_keys
            or set(matrix.get("logical_observation_artifacts") or {})
            != observation_keys
            or set(matrix.get("relation_artifacts") or {}) != relation_keys
            or set(matrix.get("logical_relation_artifacts") or {})
            != relation_keys):
        raise ValueError("S7 artifact keysets do not equal the denominator")
    bundles_by_slug: dict[str, RecipeAttemptBundle] = {}
    for relative, digest in matrix["observation_artifacts"].items():
        path = output / relative
        if _sha256(path.read_bytes()) != digest:
            raise ValueError(f"S7 observation hash mismatch: {relative}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if matrix.get("logical_observation_artifacts", {}).get(relative) != \
                _sha256(_json_bytes(_semantic_payload(payload))):
            raise ValueError(
                f"S7 logical observation hash mismatch: {relative}")
        if payload.get("schema_version") != 2:
            raise ValueError(f"S7 observation bundle schema drifted: {relative}")
        try:
            bundle = RecipeAttemptBundle.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"S7 observation bundle is malformed: {relative}") from exc
        attempts = bundle.attempts
        observation = bundle.final
        if attempts[0].recipe is None \
                or attempts[0].recipe.flags.get("source") != \
                "resolved_callable_signature":
            raise ValueError(
                f"S7 observation is not a signature-derived attempt: {relative}")
        slug = Path(relative).name.removesuffix(".json.gz")
        bundles_by_slug[slug] = bundle
        summary = summaries_by_slug[slug]
        resolution = bundle.resolution
        model_recipe = artifacts_by_slug[slug].get("signature_recipe") or {}
        if (summary.get("recipe_resolution") != resolution.status
                or summary.get("recipe_status") != observation.status
                or summary.get("recipe_retry_count") != len(attempts) - 1
                or summary.get("checkpoint_dtype") != dict(
                    resolution.checkpoint_dtype)
                or summary.get("recipe_execution_dtype") != observation.recipe.dtype
                or summary.get("recipe_execution_dtype_source") !=
                observation.recipe.flags.get("execution_dtype_source")
                or model_recipe.get("status") != resolution.status
                or model_recipe.get("checkpoint_dtype") != dict(
                    resolution.checkpoint_dtype)
                or model_recipe.get("initial_execution_dtype_source") != dict(
                    resolution.execution_dtype_source)
                or model_recipe.get("execution_dtype") != observation.recipe.dtype
                or model_recipe.get("execution_dtype_source") !=
                observation.recipe.flags.get("execution_dtype_source")
                or model_recipe.get("retry_count") != len(attempts) - 1
                or model_recipe.get("final_status") != observation.status
                or model_recipe.get("failure") != (
                    dataclasses.asdict(observation.failure)
                    if observation.failure is not None else None)):
            raise ValueError(f"S7 recipe summary drifted: {relative}")
    for relative, digest in matrix["relation_artifacts"].items():
        path = output / relative
        if _sha256(path.read_bytes()) != digest:
            raise ValueError(f"S7 relation artifact hash mismatch: {relative}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if matrix["logical_relation_artifacts"].get(relative) != _sha256(
                _json_bytes(_semantic_payload(payload))):
            raise ValueError(f"S7 logical relation hash mismatch: {relative}")
        slug = Path(relative).name.removesuffix(".json.gz")
        expected_base_recipe = bundles_by_slug[slug].final.recipe
        if expected_base_recipe is None:  # closed above; keep file joins defensive
            raise ValueError(
                f"S7 observation lacks a relation base recipe: {relative}")
        results = _validate_relation_payload(
            payload, relative, expected_base_recipe)
        summary = summaries_by_slug[slug]
        _validate_relation_cross_file(
            payload=payload, results=results, artifact=artifacts_by_slug[slug],
            summary=summary, relative=relative)


def _assert_live_shadow_matches(live: Mapping[str, Any],
                                committed: Mapping[str, Any]) -> None:
    """The Linux rerun must reproduce committed semantic artifacts exactly."""
    for field in (
            "denominator", "sources", "models", "logical_artifacts",
            "logical_observation_artifacts", "logical_relation_artifacts"):
        if live.get(field) != committed.get(field):
            raise ValueError(
                f"live Linux S7 shadow disagrees with committed {field}")


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
        committed = json.loads(
            (OUTPUT / "matrix.json").read_text(encoding="utf-8"))
        expected_rows = committed.get("models")
        if not isinstance(expected_rows, list):
            raise ValueError("committed S7 matrix has no model rows")
        expected_models = {
            row.get("slug"): row for row in expected_rows
            if isinstance(row, Mapping) and isinstance(row.get("slug"), str)}
        if len(expected_models) != 39:
            raise ValueError("committed S7 matrix does not contain 39 unique slugs")
        result = generate(
            ci_shadow=True, expected_models=expected_models)
        if len(result["models"]) != 39:
            raise ValueError("Linux S7 shadow did not execute all 39 targets")
        _assert_live_shadow_matches(result, committed)
    else:
        generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
