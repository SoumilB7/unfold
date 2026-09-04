"""Neutral layer-boundary observations for S7 relation reconciliation.

This is still positive, recipe-qualified physics.  It records shapes and tensor
lineage at an explicitly addressed repeated container; it does not name a
residual stream, KV sharing, or a side head.  Those meanings require the S7
authority join with exact static source evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .execution_observation import (
    ExecutionRecipe, _is_data_dependent, _tensor_arguments,
)
from .instance_inventory import (
    BuildRequest, Failure, NetworkRefused, Provenance,
    _CAPTURE_LIMIT, _communicate_bounded, _construct,
    _install_network_guard, _network_isolated_command, inventory_model,
)


@dataclasses.dataclass(frozen=True)
class TensorShape:
    argument: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.argument or not self.dtype or any(
                not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("tensor shape evidence is malformed")


@dataclasses.dataclass(frozen=True)
class LayerBoundaryObservation:
    index: int
    path: str
    call_order: int
    inputs: tuple[TensorShape, ...]
    outputs: tuple[TensorShape, ...]

    def __post_init__(self) -> None:
        if self.index < 0 or self.call_order < 0 or not self.path:
            raise ValueError("a layer boundary needs index, path and call order")
        if not self.inputs or not self.outputs:
            raise ValueError("a layer boundary needs positive input/output shapes")


@dataclasses.dataclass(frozen=True)
class CrossLayerTensorUse:
    producer_index: int
    producer_path: str
    consumer_index: int
    consumer_path: str
    consumer_argument: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (self.producer_index < 0 or self.consumer_index <= self.producer_index
                or not self.producer_path or not self.consumer_path
                or not self.consumer_argument):
            raise ValueError("cross-layer use must move forward between exact layers")


@dataclasses.dataclass(frozen=True)
class SiblingCallObservation:
    path: str
    call_order: int
    inputs: tuple[TensorShape, ...]
    outputs: tuple[TensorShape, ...]

    def __post_init__(self) -> None:
        if not self.path or self.call_order < 0:
            raise ValueError("sibling call needs an exact path and order")


@dataclasses.dataclass(frozen=True)
class RelationObservation:
    schema_version: int
    provenance: Provenance
    recipe: ExecutionRecipe
    stack_path: str
    boundaries: tuple[LayerBoundaryObservation, ...]
    cross_layer_uses: tuple[CrossLayerTensorUse, ...]
    sibling_calls: tuple[SiblingCallObservation, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.stack_path or not self.boundaries:
            raise ValueError("relation observation needs an addressed nonempty stack")
        if tuple(row.index for row in self.boundaries) != tuple(
                range(len(self.boundaries))):
            raise ValueError("layer boundaries must be dense and in stack order")
        paths = tuple(row.path for row in self.boundaries)
        if len(paths) != len(set(paths)):
            raise ValueError("each layer boundary path occurs once")
        if any(row.producer_path != paths[row.producer_index]
               or row.consumer_path != paths[row.consumer_index]
               for row in self.cross_layer_uses):
            raise ValueError("cross-layer use must round-trip to its boundaries")


@dataclasses.dataclass(frozen=True)
class RelationObservationResult:
    status: str
    recipe: ExecutionRecipe | None = None
    observation: RelationObservation | None = None
    provenance: Provenance | None = None
    failure: Failure | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if self.status == "ok":
            if (self.observation is None or self.failure is not None
                    or self.recipe != self.observation.recipe
                    or self.provenance != self.observation.provenance):
                raise ValueError("ok relation result requires one observation")
        elif self.status == "failed":
            if self.failure is None or self.observation is not None:
                raise ValueError("failed relation result requires one typed failure")
        else:
            raise ValueError("relation result status is closed")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RelationObservationResult":
        from .execution_observation import _provenance_from_dict
        recipe = ExecutionRecipe.from_dict(row["recipe"]) if row.get("recipe") else None
        provenance = _provenance_from_dict(row.get("provenance"))
        observed = row.get("observation")
        observation = None
        if observed:
            obs_provenance = _provenance_from_dict(observed.get("provenance"))
            if obs_provenance is None:
                raise ValueError("relation observation lacks provenance")
            observation = RelationObservation(
                int(observed["schema_version"]), obs_provenance,
                ExecutionRecipe.from_dict(observed["recipe"]),
                str(observed["stack_path"]),
                tuple(LayerBoundaryObservation(
                    int(item["index"]), str(item["path"]), int(item["call_order"]),
                    tuple(TensorShape(x["argument"], tuple(x["shape"]), x["dtype"])
                          for x in item["inputs"]),
                    tuple(TensorShape(x["argument"], tuple(x["shape"]), x["dtype"])
                          for x in item["outputs"]),
                ) for item in observed["boundaries"]),
                tuple(CrossLayerTensorUse(
                    int(item["producer_index"]), str(item["producer_path"]),
                    int(item["consumer_index"]), str(item["consumer_path"]),
                    str(item["consumer_argument"]), tuple(item["shape"]),
                ) for item in observed.get("cross_layer_uses", ())),
                tuple(SiblingCallObservation(
                    str(item["path"]), int(item["call_order"]),
                    tuple(TensorShape(x["argument"], tuple(x["shape"]), x["dtype"])
                          for x in item["inputs"]),
                    tuple(TensorShape(x["argument"], tuple(x["shape"]), x["dtype"])
                          for x in item["outputs"]),
                ) for item in observed.get("sibling_calls", ())),
            )
        return cls(
            str(row["status"]), recipe, observation, provenance,
            Failure(**row["failure"]) if row.get("failure") else None,
            str(row.get("stdout", "")), str(row.get("stderr", "")),
        )


def _tree_shapes(value: Any, prefix: str) -> tuple[TensorShape, ...]:
    import torch
    rows: list[TensorShape] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            rows.append(TensorShape(path, tuple(int(x) for x in item.shape),
                                    str(item.dtype)))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(child, f"{path}.{index}")
        elif isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(child, f"{path}.{key}")
        elif dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                walk(getattr(item, field.name), f"{path}.{field.name}")

    walk(value, prefix)
    return tuple(rows)


def _named_inputs(module: Any, args: tuple[Any, ...],
                  kwargs: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Bind hook inputs to forward-parameter names without interpreting them."""
    try:
        parameters = tuple(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        parameters = ()
    rows = [
        (parameters[index] if index < len(parameters) else f"args.{index}", value)
        for index, value in enumerate(args)
    ]
    rows.extend((str(name), value) for name, value in kwargs.items())
    return tuple(rows)


def _mapping_tensor_ids(value: Any) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """Exact tensor identities currently stored in a mutable mapping input."""
    import torch
    rows: list[tuple[str, int, tuple[int, ...]]] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            rows.append((path, id(item), tuple(int(x) for x in item.shape)))
        elif isinstance(item, Mapping):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(child, f"{path}.{index}" if path else str(index))

    if isinstance(value, Mapping):
        walk(value, "")
    return tuple(rows)


def observe_relations(model: Any, request: BuildRequest, constructor_used: str,
                      recipe: ExecutionRecipe, stack_path: str,
                      provenance: Provenance | None = None) -> RelationObservation:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.overrides import TorchFunctionMode

    stack = model.get_submodule(stack_path)
    if not isinstance(stack, (torch.nn.ModuleList, torch.nn.Sequential)):
        raise ValueError("relation stack path must address an ordered module container")
    if not len(stack):
        raise ValueError("relation stack must be nonempty")
    if provenance is None:
        provenance = inventory_model(model, request, constructor_used).provenance
    installed = {item.package: item.version for item in provenance.packages}
    mismatch = {key: (value, installed.get(key))
                for key, value in recipe.library_versions.items()
                if installed.get(key) != value}
    if mismatch:
        raise ValueError(f"recipe library version mismatch: {mismatch}")
    dtype = getattr(torch, recipe.dtype)
    if getattr(dtype, "is_floating_point", False):
        model.to(dtype=dtype)

    paths = {id(module): path for path, module in
             model.named_modules(remove_duplicate=True)}
    boundary_inputs: dict[int, tuple[TensorShape, ...]] = {}
    boundary_outputs: dict[int, tuple[TensorShape, ...]] = {}
    boundary_order: dict[int, int] = {}
    # Direct returned tensors and tensors inserted into a mutable mapping are
    # separate provenance channels.  Arbitrary functional intermediates are
    # deliberately not promoted to a cross-layer relation: a no-op/view can
    # preserve object identity and would otherwise fabricate producer edges.
    output_origin: dict[int, int] = {}
    mapping_origin: dict[int, tuple[int, str, tuple[int, ...]]] = {}
    mapping_before: dict[int, tuple[tuple[str, Any, frozenset[int]], ...]] = {}
    cross: dict[tuple, CrossLayerTensorUse] = {}
    sibling_inputs: dict[str, tuple[TensorShape, ...]] = {}
    sibling_outputs: dict[str, tuple[TensorShape, ...]] = {}
    sibling_order: dict[str, int] = {}
    current = {"layer": None, "order": 0}

    def tensors(value: Any):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from tensors(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                yield from tensors(getattr(value, field.name))

    def layer_pre(index: int, path: str):
        def hook(module, args, kwargs):
            order = current["order"]
            current["order"] += 1
            boundary_order[index] = order
            named = _named_inputs(module, args, kwargs)
            # Retain exact forward-parameter spellings without classifying
            # them.  Uninspectable positional inputs retain ``args.N``.
            boundary_inputs[index] = tuple(
                shape
                for name, value in named
                for shape in _tree_shapes(value, name)
            )
            # The primary hidden state is the residual chain.  Cross-layer
            # relations concern additional tensors, so exclude exactly the
            # first tensor argument (positional or hidden_states kwarg).
            primary_value = next((value for name, value in named
                                  if name == "hidden_states"), None)
            primary = next(tensors(primary_value), None)
            if primary is None:
                primary = next(tensors(args), None)
            tracked_mappings: list[tuple[str, Any, frozenset[int]]] = []
            for arg_name, value in named:
                for tensor in tensors(value):
                    if tensor is primary:
                        continue
                    producer = output_origin.get(id(tensor))
                    if producer is None or producer >= index:
                        continue
                    key = (producer, index, arg_name, tuple(tensor.shape))
                    cross[key] = CrossLayerTensorUse(
                        producer, f"{stack_path}.{producer}", index, path,
                        arg_name, tuple(int(x) for x in tensor.shape))
                entries = _mapping_tensor_ids(value)
                # Empty mutable state is still a provenance boundary: the
                # current layer may populate it for a later layer.  Tracking
                # only non-empty mappings loses the first producer.
                if isinstance(value, Mapping):
                    tracked_mappings.append((
                        arg_name, value, frozenset(identity for _, identity, _ in entries)))
            mapping_before[index] = tuple(tracked_mappings)
            current["layer"] = index
        return hook

    def layer_post(index: int):
        def hook(_module, _args, _kwargs, output):
            boundary_outputs[index] = _tree_shapes(output, "output")
            for tensor in tensors(output):
                output_origin.setdefault(id(tensor), index)
            for name, mapping, before in mapping_before.pop(index, ()):
                for entry_path, identity, shape in _mapping_tensor_ids(mapping):
                    if identity not in before:
                        mapping_origin.setdefault(
                            identity,
                            (index, f"{name}.{entry_path}".rstrip("."), shape),
                        )
            current["layer"] = None
        return hook

    class TagMode(TorchFunctionMode):
        """Keep the relation recipe under the same functional interception.

        S6's functional log is the operation authority.  This observer only
        needs the mode's FakeTensor compatibility; it intentionally does not
        infer lineage from arbitrary functional outputs.
        """

        def __torch_function__(self, function, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            index = current["layer"]
            if index is not None:
                for tensor in tensors((args, kwargs)):
                    origin = mapping_origin.get(id(tensor))
                    if origin is None or origin[0] >= index:
                        continue
                    producer, argument, shape = origin
                    key = (producer, index, argument, shape)
                    cross[key] = CrossLayerTensorUse(
                        producer, f"{stack_path}.{producer}", index,
                        paths.get(id(stack[index]), f"{stack_path}.{index}"),
                        argument, shape)
            return function(*args, **kwargs)

    stack_parent_path, _, stack_field = stack_path.rpartition(".")
    stack_parent = model.get_submodule(stack_parent_path) if stack_parent_path else model
    siblings = {
        f"{stack_parent_path}.{name}".strip("."): child
        for name, child in stack_parent.named_children() if name != stack_field
    }

    def sibling_pre(path: str):
        def hook(_module, args, kwargs):
            sibling_order[path] = current["order"]
            current["order"] += 1
            sibling_inputs[path] = _tree_shapes(args, "args") + _tree_shapes(kwargs, "kwargs")
        return hook

    def sibling_post(path: str):
        def hook(_module, _args, _kwargs, output):
            sibling_outputs[path] = _tree_shapes(output, "output")
        return hook

    handles = []
    for index, layer in enumerate(stack):
        path = paths.get(id(layer), f"{stack_path}.{index}")
        handles.append(layer.register_forward_pre_hook(
            layer_pre(index, path), with_kwargs=True))
        handles.append(layer.register_forward_hook(
            layer_post(index), with_kwargs=True))
    for path, module in siblings.items():
        handles.append(module.register_forward_pre_hook(
            sibling_pre(path), with_kwargs=True))
        handles.append(module.register_forward_hook(
            sibling_post(path), with_kwargs=True))

    target = model.get_submodule(recipe.target_path) if recipe.target_path else model
    model.train(recipe.train_eval == "train")
    try:
        with torch.device("meta"):
            arguments = _tensor_arguments(recipe)
        with FakeTensorMode(allow_non_fake_inputs=True), TagMode():
            target(**arguments)
    finally:
        for handle in handles:
            handle.remove()

    boundaries = tuple(LayerBoundaryObservation(
        index, paths.get(id(stack[index]), f"{stack_path}.{index}"),
        boundary_order[index], boundary_inputs[index], boundary_outputs[index])
        for index in range(len(stack)))
    sibling_rows = tuple(SiblingCallObservation(
        path, sibling_order[path], sibling_inputs.get(path, ()),
        sibling_outputs.get(path, ())) for path in sorted(sibling_order))
    return RelationObservation(
        1, provenance, recipe, stack_path, boundaries,
        tuple(sorted(cross.values(), key=lambda row: (
            row.producer_index, row.consumer_index, row.consumer_argument,
            row.shape))), sibling_rows)


def _failed(kind: str, stage: str, exc: BaseException,
            recipe: ExecutionRecipe | None = None,
            provenance: Provenance | None = None) -> RelationObservationResult:
    return RelationObservationResult(
        "failed", recipe=recipe, provenance=provenance,
        failure=Failure(kind, stage, f"{type(exc).__name__}: {str(exc)[:2000]}"))


def _worker(request_path: Path, recipe_path: Path, stack_path: str,
            result_path: Path) -> int:
    _install_network_guard()
    try:
        request = BuildRequest.from_dict(json.loads(request_path.read_text()))
        recipe = ExecutionRecipe.from_dict(json.loads(recipe_path.read_text()))
        model, used = _construct(request)
        provenance = inventory_model(model, request, used).provenance
        observation = observe_relations(
            model, request, used, recipe, stack_path, provenance)
        result = RelationObservationResult(
            "ok", recipe, observation, provenance)
    except NetworkRefused as exc:
        result = _failed("NetworkRefused", "relation_observe", exc,
                         locals().get("recipe"), locals().get("provenance"))
    except MemoryError as exc:
        result = _failed("MemoryLimitExceeded", "relation_observe", exc,
                         locals().get("recipe"), locals().get("provenance"))
    except BaseException as exc:  # isolated arbitrary library boundary
        kind = "ExecutionUnresolved" if _is_data_dependent(exc) else "ExecutionFailed"
        result = _failed(kind, "relation_observe", exc,
                         locals().get("recipe"), locals().get("provenance"))
    try:
        result_path.write_text(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError):
        return 2


def observe_relations_in_subprocess(
    request: BuildRequest, recipe: ExecutionRecipe, stack_path: str,
) -> RelationObservationResult:
    if not stack_path:
        raise ValueError("relation observation requires a stack path")
    with tempfile.TemporaryDirectory(prefix="unfold-s7-relation-") as tmp:
        root = Path(tmp)
        request_path = root / "request.json"
        recipe_path = root / "recipe.json"
        result_path = root / "result.json"
        request_path.write_text(json.dumps(request.to_dict(), sort_keys=True))
        recipe_path.write_text(json.dumps(recipe.to_dict(), sort_keys=True))
        env = {**__import__("os").environ,
               "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        command = _network_isolated_command(
            [sys.executable, "-m", "physics.relation_observation", "--worker",
             str(request_path), str(recipe_path), stack_path, str(result_path)], env)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", env=env, start_new_session=True)
        stdout, stderr, termination = _communicate_bounded(
            process, timeout=request.timeout_seconds,
            memory_limit=request.memory_limit_bytes)
        if termination:
            kind = ("TimeoutExpired" if termination == "timeout" else
                    "MemoryLimitExceeded" if termination.startswith("memory:") else
                    "ConfigurationFailed")
            result = _failed(kind, "relation_observe", RuntimeError(termination), recipe)
        elif result_path.exists():
            try:
                result = RelationObservationResult.from_dict(
                    json.loads(result_path.read_text()))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result = _failed("SerializationFailed", "parent_decode", exc, recipe)
        else:
            result = _failed("WorkerFailed", "worker_exit", RuntimeError(
                f"worker exited {process.returncode}"), recipe)
        return dataclasses.replace(result, stdout=stdout[-_CAPTURE_LIMIT:],
                                   stderr=stderr[-_CAPTURE_LIMIT:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=4,
                        metavar=("REQUEST", "RECIPE", "STACK", "RESULT"))
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("relation_observation is an internal worker")
    request, recipe, stack, result = args.worker
    return _worker(Path(request), Path(recipe), stack, Path(result))


if __name__ == "__main__":
    raise SystemExit(main())
