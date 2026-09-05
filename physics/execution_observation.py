"""Recipe-qualified FakeTensor execution observation for S6.

This module records positive runtime evidence only: module calls, selected
functional tensor operations, and modules that appeared lazily during the
named recipe.  Absence from a trace is never serialized as a negative.  A
data-dependent branch or any unsupported execution becomes a typed failure.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .instance_inventory import (
    BuildRequest, Failure, NetworkRefused, PackageVersion, Provenance,
    ResolvedClass, SourceFile,
    _CAPTURE_LIMIT, _construct,
    _authorize_network_worker, _communicate_bounded, _install_network_guard,
    _network_isolated_command,
    _prepare_network_attestation,
    _write_network_attestation, inventory_model,
)


_EXEC_FAILURE_KINDS = frozenset({
    "NetworkRefused", "TimeoutExpired", "MemoryLimitExceeded", "ImportFailed",
    "ConfigurationFailed", "ConstructionFailed", "SerializationFailed",
    "WorkerFailed", "ExecutionFailed", "ExecutionUnresolved",
})
_OPS = frozenset({
    "scaled_dot_product_attention", "silu", "gelu", "chunk", "cat",
    "layer_norm", "add", "mul",
})


@dataclasses.dataclass(frozen=True)
class TensorArgument:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fill: str = "zeros"
    values: Any = None
    device: str = "meta"

    def __post_init__(self) -> None:
        if (not self.name or ".." in self.name
                or any(not isinstance(x, int) or x < 0 for x in self.shape)
                or not self.dtype):
            raise ValueError("tensor argument needs a name and non-negative shape")
        if self.fill not in {"zeros", "ones", "values"}:
            raise ValueError("tensor fill must be zeros, ones, or values")
        if (self.fill == "values") != (self.values is not None):
            raise ValueError("explicit tensor values require fill=values")
        if self.device not in {"meta", "cpu"}:
            raise ValueError("recipe tensor device must be meta or cpu")
        if self.values is not None:
            json.dumps(self.values, sort_keys=True)


@dataclasses.dataclass(frozen=True)
class ExecutionRecipe:
    recipe_id: str
    input_modality: str
    train_eval: str
    cache_state: str
    encoder_decoder_mode: str
    conditioning_present: bool
    dtype: str
    library_versions: Mapping[str, str]
    target_path: str = ""
    tensor_arguments: tuple[TensorArgument, ...] = ()
    literal_arguments: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    flags: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.recipe_id, self.input_modality, self.cache_state,
                    self.encoder_decoder_mode, self.dtype)):
            raise ValueError("recipe identity and axes are required")
        if self.train_eval not in {"train", "eval"}:
            raise ValueError("train_eval must be train or eval")
        if (not self.library_versions
                or any(not key or not value
                       for key, value in self.library_versions.items())):
            raise ValueError("recipe library versions are required")
        names = [arg.name for arg in self.tensor_arguments]
        if len(names) != len(set(names)) or set(names) & set(self.literal_arguments):
            raise ValueError("recipe argument names must be unique")
        if not isinstance(self.target_path, str):
            raise ValueError("recipe target path must be textual")
        json.dumps(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["tensor_arguments"] = [dataclasses.asdict(x) for x in self.tensor_arguments]
        row["literal_arguments"] = dict(self.literal_arguments)
        row["flags"] = dict(self.flags)
        row["library_versions"] = dict(self.library_versions)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ExecutionRecipe":
        values = dict(row)
        values["tensor_arguments"] = tuple(TensorArgument(
            name=x["name"], shape=tuple(x["shape"]), dtype=x["dtype"],
            fill=x.get("fill", "zeros"), values=x.get("values"),
            device=x.get("device", "meta"))
            for x in values.get("tensor_arguments", ()))
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class ModuleCall:
    index: int
    path: str
    class_ref: ResolvedClass

    def __post_init__(self) -> None:
        if self.index < 0 or not isinstance(self.path, str):
            raise ValueError("module-call record is invalid")


@dataclasses.dataclass(frozen=True)
class FunctionalOp:
    index: int
    op: str
    callable: str

    def __post_init__(self) -> None:
        if self.index < 0 or self.op not in _OPS or not self.callable:
            raise ValueError("functional op is outside the closed S6 vocabulary")


@dataclasses.dataclass(frozen=True)
class LazyObserved:
    path: str
    class_ref: ResolvedClass

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("a lazy module needs its exact non-root path")


@dataclasses.dataclass(frozen=True)
class ExecutionObservation:
    schema_version: int
    provenance: Provenance
    recipe: ExecutionRecipe
    module_calls: tuple[ModuleCall, ...]
    functional_ops: tuple[FunctionalOp, ...]
    lazy_observed: tuple[LazyObserved, ...]
    outcome: str = "observed"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.outcome != "observed":
            raise ValueError("successful S6 observation is positive-only")
        if tuple(x.index for x in self.module_calls) != tuple(range(len(self.module_calls))):
            raise ValueError("module call order must be dense and exact")
        if tuple(x.index for x in self.functional_ops) != tuple(range(len(self.functional_ops))):
            raise ValueError("functional op order must be dense and exact")
        lazy_paths = tuple(row.path for row in self.lazy_observed)
        if len(lazy_paths) != len(set(lazy_paths)):
            raise ValueError("lazy module paths must be unique")


@dataclasses.dataclass(frozen=True)
class ObservationResult:
    status: str
    recipe: ExecutionRecipe | None = None
    observation: ExecutionObservation | None = None
    provenance: Provenance | None = None
    failure: Failure | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("captured output must be text")
        if self.failure is not None and self.failure.kind not in _EXEC_FAILURE_KINDS:
            raise ValueError("unknown execution failure kind")
        if self.status == "ok":
            if (self.observation is None or self.failure is not None
                    or self.provenance != self.observation.provenance
                    or self.recipe != self.observation.recipe):
                raise ValueError("ok result requires only an observation")
        elif self.status == "failed":
            if self.failure is None or self.observation is not None:
                raise ValueError("failed result requires only a failure")
            if self.recipe is None and not (
                    self.failure.kind == "ConfigurationFailed"
                    and self.failure.stage == "request"):
                raise ValueError("an execution failure must retain its recipe")
        else:
            raise ValueError("result status must be ok or failed")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ObservationResult":
        observation = row.get("observation")
        failure = row.get("failure")
        return cls(
            status=str(row["status"]),
            recipe=ExecutionRecipe.from_dict(row["recipe"])
            if row.get("recipe") else None,
            observation=_observation_from_dict(observation) if observation else None,
            provenance=_provenance_from_dict(row.get("provenance")),
            failure=Failure(**failure) if failure else None,
            stdout=str(row.get("stdout", "")), stderr=str(row.get("stderr", "")),
        )


def _observation_from_dict(row: Mapping[str, Any]) -> ExecutionObservation:
    provenance = _provenance_from_dict(row["provenance"])
    if provenance is None:  # defensive closure for untyped external JSON
        raise ValueError("successful observation has no provenance")
    return ExecutionObservation(
        schema_version=row["schema_version"], provenance=provenance,
        recipe=ExecutionRecipe.from_dict(row["recipe"]),
        module_calls=tuple(ModuleCall(x["index"], x["path"],
                                      ResolvedClass(**x["class_ref"]))
                           for x in row["module_calls"]),
        functional_ops=tuple(FunctionalOp(**x) for x in row["functional_ops"]),
        lazy_observed=tuple(LazyObserved(x["path"], ResolvedClass(**x["class_ref"]))
                            for x in row["lazy_observed"]),
        outcome=row.get("outcome", "observed"),
    )


def _provenance_from_dict(row: Mapping[str, Any] | None) -> Provenance | None:
    if row is None:
        return None
    return Provenance(
        packages=tuple(PackageVersion(**x) for x in row["packages"]),
        source_files=tuple(SourceFile(**x) for x in row["source_files"]),
        config_sha256=row["config_sha256"],
        resolved_class=ResolvedClass(**row["resolved_class"]),
        requested_factory=row["requested_factory"],
        constructor_used=row["constructor_used"],
        build_flags=row["build_flags"], environment=row["environment"],
    )


def _normalise_op(function: Any) -> tuple[str, str] | None:
    name = str(getattr(function, "__name__", ""))
    qualified = f"{getattr(function, '__module__', '')}.{name}".strip(".")
    lowered = name.lower()
    if "scaled_dot_product_attention" in lowered:
        return "scaled_dot_product_attention", qualified
    if lowered in _OPS:
        return lowered, qualified
    if lowered in {"__add__", "add_", "radd"}:
        return "add", qualified
    if lowered in {"__mul__", "mul_", "rmul"}:
        return "mul", qualified
    return None


def _tensor_arguments(recipe: ExecutionRecipe) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = dict(recipe.literal_arguments)

    def assign(name: str, value: Any) -> None:
        parts = name.split(".")
        target = result
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"recipe argument path {name!r} collides with a literal")
            target = existing
        if parts[-1] in target:
            raise ValueError(f"duplicate recipe argument path {name!r}")
        target[parts[-1]] = value

    for spec in recipe.tensor_arguments:
        try:
            dtype = getattr(torch, spec.dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype {spec.dtype!r}") from exc
        if spec.fill == "values":
            tensor = torch.tensor(spec.values, dtype=dtype, device=spec.device)
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(f"explicit values for {spec.name!r} have wrong shape")
            assign(spec.name, tensor)
        else:
            maker = torch.zeros if spec.fill == "zeros" else torch.ones
            assign(spec.name, maker(spec.shape, dtype=dtype, device=spec.device))
    def collapse(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        mapped = {key: collapse(item) for key, item in value.items()}
        numeric = tuple(str(i) for i in range(len(mapped)))
        if set(mapped) == set(numeric):
            return tuple(mapped[key] for key in numeric)
        return mapped

    collapsed = collapse(result)
    if not isinstance(collapsed, dict):
        raise ValueError("top-level recipe arguments must be named")
    return collapsed


def observe_model(model: Any, request: BuildRequest, constructor_used: str,
                  recipe: ExecutionRecipe,
                  provenance: Provenance | None = None) -> ExecutionObservation:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.overrides import TorchFunctionMode

    try:
        execution_dtype = getattr(torch, recipe.dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown recipe dtype {recipe.dtype!r}") from exc
    if getattr(execution_dtype, "is_floating_point", False):
        # Dtype is an explicit recipe axis, not a fact inferred from the model.
        # Some real kernels (for example grouped GEMM) only admit bf16/fp16.
        model.to(dtype=execution_dtype)
    if provenance is None:
        provenance = inventory_model(model, request, constructor_used).provenance
    installed = {row.package: row.version for row in provenance.packages}
    mismatches = {name: (version, installed.get(name))
                  for name, version in recipe.library_versions.items()
                  if installed.get(name) != version}
    if mismatches:
        raise ValueError(f"recipe library version mismatch: {mismatches}")
    before = {name: type(module) for name, module in model.named_modules(remove_duplicate=False)}
    paths: dict[int, tuple[str, ...]] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        paths.setdefault(id(module), tuple())
        paths[id(module)] = (*paths[id(module)], path)
    calls: list[ModuleCall] = []
    ops: list[FunctionalOp] = []

    def pre_hook(module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        module_paths = paths.get(id(module), ("<lazy-unaddressed>",))
        # One object can have aliases. The call proves the object, not which alias
        # led to it, so retain all aliases as one canonical joined address.
        calls.append(ModuleCall(len(calls), " | ".join(module_paths),
                                ResolvedClass(type(module).__module__,
                                              type(module).__qualname__)))

    class OpMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            row = _normalise_op(function)
            result = function(*args, **(kwargs or {}))
            if row is not None:
                ops.append(FunctionalOp(len(ops), row[0], row[1]))
            return result

    hooks = [module.register_forward_pre_hook(pre_hook, with_kwargs=True)
             for _, module in model.named_modules(remove_duplicate=True)]
    target = model.get_submodule(recipe.target_path) if recipe.target_path else model
    model.train(recipe.train_eval == "train")
    try:
        with torch.device("meta"):
            arguments = _tensor_arguments(recipe)
        with FakeTensorMode(allow_non_fake_inputs=True), OpMode():
            target(**arguments)
    finally:
        for hook in hooks:
            hook.remove()
    after = {name: type(module) for name, module in model.named_modules(remove_duplicate=False)}
    lazy = tuple(LazyObserved(path, ResolvedClass(cls.__module__, cls.__qualname__))
                 for path, cls in sorted(after.items()) if path not in before)
    return ExecutionObservation(1, provenance, recipe, tuple(calls),
                                tuple(ops), lazy)


def _execution_failure(kind: str, stage: str, exc: BaseException,
                       provenance: Provenance | None = None,
                       recipe: ExecutionRecipe | None = None) -> ObservationResult:
    return ObservationResult("failed", recipe=recipe, provenance=provenance,
                             failure=Failure(
        kind, stage, f"{type(exc).__name__}: {str(exc)[:2000]}"))


def _is_data_dependent(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return ("datadependent" in name or "guardondata" in name
            or "dynamicoutputshape" in name
            or "data-dependent" in text or "could not guard on data" in text
            or "aten._local_scalar_dense" in text)


def _worker(request_path: Path, recipe_path: Path, result_path: Path) -> int:
    _write_network_attestation()
    _install_network_guard()
    try:
        request = BuildRequest.from_dict(json.loads(request_path.read_text()))
        recipe = ExecutionRecipe.from_dict(json.loads(recipe_path.read_text()))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result = _execution_failure("ConfigurationFailed", "request", exc)
    else:
        for path in reversed(request.import_paths):
            sys.path.insert(0, path)
        try:
            model, used = _construct(request)
        except NetworkRefused as exc:
            result = _execution_failure("NetworkRefused", "construct", exc,
                                        recipe=recipe)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            result = _execution_failure("ImportFailed", "construct", exc,
                                        recipe=recipe)
        except MemoryError as exc:
            result = _execution_failure("MemoryLimitExceeded", "construct", exc,
                                        recipe=recipe)
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            result = _execution_failure("ConstructionFailed", "construct", exc,
                                        recipe=recipe)
        except BaseException as exc:  # child boundary: arbitrary library code
            result = _execution_failure("WorkerFailed", "construct", exc,
                                        recipe=recipe)
        else:
            try:
                provenance = inventory_model(model, request, used).provenance
                observation = observe_model(model, request, used, recipe, provenance)
                result = ObservationResult("ok", recipe=recipe,
                                           observation=observation,
                                           provenance=provenance)
            except NetworkRefused as exc:
                result = _execution_failure("NetworkRefused", "execute", exc,
                                            locals().get("provenance"), recipe)
            except MemoryError as exc:
                result = _execution_failure("MemoryLimitExceeded", "execute", exc,
                                            locals().get("provenance"), recipe)
            except BaseException as exc:  # child boundary: arbitrary model code
                kind = ("ExecutionUnresolved" if _is_data_dependent(exc)
                        else "ExecutionFailed")
                result = _execution_failure(kind, "execute", exc,
                                            locals().get("provenance"), recipe)
    try:
        result_path.write_text(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        fallback = _execution_failure("SerializationFailed", "serialize", exc,
                                      recipe=locals().get("recipe"))
        try:
            result_path.write_text(json.dumps(fallback.to_dict(), sort_keys=True))
        except OSError:
            pass
        return 2


def observe_in_subprocess(request: BuildRequest,
                          recipe: ExecutionRecipe) -> ObservationResult:
    """Execute one named recipe within the same bounded isolation as inventory."""
    with tempfile.TemporaryDirectory(prefix="unfold-s6-exec-") as tmp:
        root = Path(tmp)
        request_path = root / "request.json"
        recipe_path = root / "recipe.json"
        result_path = root / "result.json"
        request_path.write_text(json.dumps(request.to_dict(), sort_keys=True))
        recipe_path.write_text(json.dumps(recipe.to_dict(), sort_keys=True))
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false"})
        attestation = _prepare_network_attestation(root, env)
        command = [sys.executable, "-m", "physics.execution_observation", "--worker",
                   str(request_path), str(recipe_path), str(result_path)]
        command = _network_isolated_command(command, env)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
            env=env, start_new_session=True)
        _authorize_network_worker(process, attestation)
        stdout, stderr, termination = _communicate_bounded(
            process, timeout=request.timeout_seconds,
            memory_limit=request.memory_limit_bytes)
        if termination == "timeout":
            return ObservationResult("failed", failure=Failure(
                "TimeoutExpired", "execute",
                f"exceeded {request.timeout_seconds:g}s wall timeout"),
                recipe=recipe,
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if termination and termination.startswith("memory:"):
            peak = int(termination.split(":", 1)[1])
            return ObservationResult("failed", failure=Failure(
                "MemoryLimitExceeded", "execute",
                f"process-tree RSS {peak} exceeded {request.memory_limit_bytes} bytes"),
                recipe=recipe,
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if termination:
            return ObservationResult("failed", failure=Failure(
                "ConfigurationFailed", "memory_monitor", termination),
                recipe=recipe,
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if result_path.exists():
            try:
                result = ObservationResult.from_dict(json.loads(result_path.read_text()))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result = _execution_failure("SerializationFailed", "parent_decode", exc,
                                            recipe=recipe)
        else:
            detail = stderr[-2000:] or f"worker exited {process.returncode} without a result"
            result = ObservationResult("failed", recipe=recipe,
                                       failure=_worker_exit_failure(detail))
        if result.recipe is None:
            result = dataclasses.replace(result, recipe=recipe)
        return dataclasses.replace(result, stdout=stdout[-_CAPTURE_LIMIT:],
                                   stderr=stderr[-_CAPTURE_LIMIT:])


def _worker_exit_failure(detail: str) -> Failure:
    """A missing result is never evidence that the memory cap fired.

    Only the supervisor's measured ``memory:<rss>`` termination may author a
    MemoryLimitExceeded result.
    """
    return Failure("WorkerFailed", "worker_exit", detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3, metavar=("REQUEST", "RECIPE", "RESULT"))
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("execution_observation is an internal worker; use observe_in_subprocess")
    return _worker(*(Path(x) for x in args.worker))


if __name__ == "__main__":
    raise SystemExit(main())
