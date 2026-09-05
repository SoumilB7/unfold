"""Neutral layer-boundary observations for S7 relation reconciliation.

This is still positive, recipe-qualified physics.  It records shapes and tensor
lineage at an explicitly addressed repeated container; it does not name a
residual stream, KV sharing, or a side head.  Those meanings require the S7
authority join with exact static source evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
    _CAPTURE_LIMIT, _authorize_network_worker, _communicate_bounded, _construct,
    _install_network_guard, _network_isolated_command,
    _prepare_network_attestation,
    _write_network_attestation, inventory_model,
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
    # Exact recipe tensor arguments whose runtime dataflow reaches the primary
    # layer input.  Empty means the observer could not prove lineage; it never
    # means that no lineage exists.
    primary_input_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.index < 0 or self.call_order < 0 or not self.path:
            raise ValueError("a layer boundary needs index, path and call order")
        if not self.inputs or not self.outputs:
            raise ValueError("a layer boundary needs positive input/output shapes")
        if (tuple(sorted(set(self.primary_input_origins)))
                != self.primary_input_origins):
            raise ValueError("primary-input origins are canonical positive evidence")


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
class MatrixContractionObservation:
    """One executed matrix contraction of an exact layer-input axis.

    This is deliberately neutral physics: it records neither a residual nor a
    model mechanism.  ``state_operand`` identifies which matmul operand carries
    lineage from the exact boundary primary input; ``input_axis`` is the axis
    contracted by matmul, while ``output_axis`` is the equally-sized axis
    supplied by the other operand.  The row is emitted only when its result has
    positive lineage to that same layer's returned output.
    """

    layer_index: int
    layer_path: str
    call_order: int
    op: str
    state_operand: int
    input_axis: int
    output_axis: int
    extent: int
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    source_fingerprint: str
    source_line: int

    def __post_init__(self) -> None:
        if (self.layer_index < 0 or not self.layer_path or self.call_order < 0
                or self.op != "matmul" or self.state_operand not in {0, 1}
                or self.extent <= 1 or self.source_line <= 0):
            raise ValueError("matrix-contraction evidence is malformed")
        if len(self.input_shapes) != 2 or not self.output_shape:
            raise ValueError("matrix contraction needs exact input/output shapes")
        state_shape = self.input_shapes[self.state_operand]
        if (not 0 <= self.input_axis < len(state_shape)
                or not 0 <= self.output_axis < len(self.output_shape)
                or state_shape[self.input_axis] != self.extent
                or self.output_shape[self.output_axis] != self.extent):
            raise ValueError("matrix contraction axes must round-trip to their extent")
        if (len(self.source_fingerprint) != 64 or any(
                ch not in "0123456789abcdef" for ch in self.source_fingerprint)):
            raise ValueError("matrix contraction needs an exact source fingerprint")


@dataclasses.dataclass(frozen=True)
class RelationObservation:
    schema_version: int
    provenance: Provenance
    recipe: ExecutionRecipe
    stack_path: str
    boundaries: tuple[LayerBoundaryObservation, ...]
    cross_layer_uses: tuple[CrossLayerTensorUse, ...]
    sibling_calls: tuple[SiblingCallObservation, ...]
    matrix_contractions: tuple[MatrixContractionObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 3 or not self.stack_path or not self.boundaries:
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
        allowed_hashes = {row.sha256 for row in self.provenance.source_files}
        for row in self.matrix_contractions:
            if (row.layer_index >= len(paths)
                    or row.layer_path != paths[row.layer_index]
                    or row.source_fingerprint not in allowed_hashes):
                raise ValueError(
                    "matrix contraction must round-trip to layer and source")
        orders = tuple(row.call_order for row in self.matrix_contractions)
        if orders != tuple(sorted(set(orders))):
            raise ValueError("matrix-contraction order is exact and unique")


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
                    tuple(item.get("primary_input_origins", ())),
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
                tuple(MatrixContractionObservation(
                    int(item["layer_index"]), str(item["layer_path"]),
                    int(item["call_order"]), str(item["op"]),
                    int(item["state_operand"]), int(item["input_axis"]),
                    int(item["output_axis"]), int(item["extent"]),
                    tuple(tuple(shape) for shape in item["input_shapes"]),
                    tuple(item["output_shape"]),
                    str(item["source_fingerprint"]), int(item["source_line"]),
                ) for item in observed.get("matrix_contractions", ())),
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


def _mapping_tensors(value: Any) -> tuple[tuple[str, Any, tuple[int, ...]], ...]:
    """Exact tensor objects currently stored in a mutable mapping input."""
    import torch
    rows: list[tuple[str, Any, tuple[int, ...]]] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            rows.append((path, item, tuple(int(x) for x in item.shape)))
        elif isinstance(item, Mapping):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(child, f"{path}.{index}" if path else str(index))

    if isinstance(value, Mapping):
        walk(value, "")
    return tuple(rows)


def _identity_get(rows: Mapping[int, tuple[Any, Any]], tensor: Any) -> Any:
    """Read an identity-keyed value only when the retained object is exact."""
    row = rows.get(id(tensor))
    return row[1] if row is not None and row[0] is tensor else None


def _identity_set(rows: dict[int, tuple[Any, Any]], tensor: Any,
                  value: Any) -> None:
    """Retain the object so CPython cannot recycle its id into evidence."""
    rows[id(tensor)] = (tensor, value)


def _identity_delete(rows: dict[int, tuple[Any, Any]], tensor: Any) -> None:
    """Forget evidence only when the stored identity is this exact object."""
    row = rows.get(id(tensor))
    if row is not None and row[0] is tensor:
        rows.pop(id(tensor), None)


def _same_tensor_view(left: Any, right: Any) -> bool:
    """Whether two wrappers address the exact same tensor view.

    PyTorch may return a fresh Python wrapper for the same FakeTensor view.
    Object identity alone then loses real lineage, while storage aliasing alone
    is too broad (disjoint slices alias).  Storage, offset, shape, stride,
    dtype, and device together form the exact view address used here.
    """
    import torch
    try:
        return (torch._C._is_alias_of(left, right)
                and int(left.storage_offset()) == int(right.storage_offset())
                and tuple(left.shape) == tuple(right.shape)
                and tuple(left.stride()) == tuple(right.stride())
                and left.dtype == right.dtype and left.device == right.device)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _mix_get(rows: Mapping[int, tuple[Any, Any]], tensor: Any) -> Any:
    exact = _identity_get(rows, tensor)
    if exact is not None:
        return exact
    matches = [value for stored, value in rows.values()
               if _same_tensor_view(stored, tensor)]
    if not matches:
        return None
    first = matches[0]
    return first if all(value == first for value in matches[1:]) else None


def _mix_set(rows: dict[int, tuple[Any, Any]], tensor: Any,
             value: Any) -> None:
    for key, (stored, _old) in tuple(rows.items()):
        if _same_tensor_view(stored, tensor):
            rows.pop(key, None)
    _identity_set(rows, tensor, value)


def _mix_delete(rows: dict[int, tuple[Any, Any]], tensor: Any) -> None:
    for key, (stored, _old) in tuple(rows.items()):
        if stored is tensor or _same_tensor_view(stored, tensor):
            rows.pop(key, None)


def _unique_extra_axis(shape: tuple[int, ...], recipe_shape: tuple[int, ...]) \
        -> tuple[int, int] | None:
    """Return the one axis not explained by the recipe prefix, if unique.

    This is address/shape evidence only.  Repeated equal dimensions that admit
    two alignments are deliberately unresolved.
    """
    if len(shape) != 4 or len(recipe_shape) < 2:
        return None
    prefix = shape[:-1]
    needle = recipe_shape[:2]
    alignments: list[tuple[int, ...]] = []

    def align(start: int, chosen: tuple[int, ...]) -> None:
        if len(chosen) == len(needle):
            alignments.append(chosen)
            return
        value = needle[len(chosen)]
        for index in range(start, len(prefix)):
            if prefix[index] == value:
                align(index + 1, (*chosen, index))

    align(0, ())
    remainders = {
        tuple((axis, value) for axis, value in enumerate(prefix)
              if axis not in selected)
        for selected in alignments
    }
    if len(remainders) != 1:
        return None
    remaining = next(iter(remainders))
    if len(remaining) != 1 or remaining[0][1] <= 1:
        return None
    return remaining[0]


def _normalised_function_name(function: Any) -> str:
    name = str(getattr(function, "__name__", "")).lower()
    return {
        "__matmul__": "matmul", "__getitem__": "getitem",
        "__add__": "add", "__radd__": "add", "add_": "add",
        "__sub__": "sub", "__rsub__": "sub", "sub_": "sub",
        "__mul__": "mul", "__rmul__": "mul", "mul_": "mul",
    }.get(name, name)


def _source_site(allowed_hashes: frozenset[str],
                 cache: dict[str, str | None]) -> tuple[str, int] | None:
    """Find the nearest executed frame whose bytes match resolved provenance."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = frame.f_code.co_filename
            fingerprint = cache.get(filename)
            if filename not in cache:
                try:
                    fingerprint = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                except OSError:
                    fingerprint = None
                cache[filename] = fingerprint
            if fingerprint in allowed_hashes:
                return str(fingerprint), int(frame.f_lineno)
            frame = frame.f_back
    finally:
        del frame
    return None


def _matmul_contraction(
    left: Any,
    right: Any,
    result: Any,
    axes: Mapping[int, tuple[Any, tuple[int, int]]],
) -> tuple[int, int, int, int] | None:
    """Return (state operand, input axis, output axis, extent) if exact.

    The state axis must be the contracted matmul dimension and the other
    operand must provide a same-sized output axis.  Merely carrying a rank-4
    tensor or performing elementwise arithmetic can never satisfy this rule.
    """
    try:
        left_shape = tuple(int(x) for x in left.shape)
        right_shape = tuple(int(x) for x in right.shape)
        output_shape = tuple(int(x) for x in result.shape)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(left_shape) < 2 or len(right_shape) < 2 or len(output_shape) < 2:
        return None
    candidates: list[tuple[int, int, int, int]] = []
    left_axis = _identity_get(axes, left)
    if (left_axis is not None and left_axis[0] == len(left_shape) - 1
            and left_shape[-1] == right_shape[-2] == right_shape[-1]
            and output_shape[-1] == left_axis[1]):
        candidates.append((0, left_axis[0], len(output_shape) - 1, left_axis[1]))
    right_axis = _identity_get(axes, right)
    if (right_axis is not None and right_axis[0] == len(right_shape) - 2
            and right_shape[-2] == left_shape[-1] == left_shape[-2]
            and output_shape[-2] == right_axis[1]):
        candidates.append((1, right_axis[0], len(output_shape) - 2, right_axis[1]))
    return candidates[0] if len(candidates) == 1 else None


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
    # The container's registered child keys are the occurrence addresses.
    # Enumerating a named Sequential and formatting ``.0``, ``.1`` would invent
    # paths that do not exist.  This reads the closed torch container protocol
    # directly and refuses shared child objects because hooks cannot distinguish
    # two registrations of the same object as separate occurrences.
    stack_members = tuple(stack._modules.items())
    if (len(stack_members) != len(stack)
            or any(module is None for _name, module in stack_members)
            or len({id(module) for _name, module in stack_members})
            != len(stack_members)):
        raise ValueError(
            "relation stack members must have unique exact registered addresses")
    member_paths = tuple(
        f"{stack_path}.{name}" for name, _module in stack_members)
    if any(model.get_submodule(path) is not module
           for path, (_name, module) in zip(member_paths, stack_members)):
        raise ValueError("relation stack member address does not round-trip")
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

    boundary_inputs: dict[int, tuple[TensorShape, ...]] = {}
    boundary_outputs: dict[int, tuple[TensorShape, ...]] = {}
    boundary_order: dict[int, int] = {}
    boundary_primary_origins: dict[int, tuple[str, ...]] = {}
    # Direct returned tensors and tensors inserted into a mutable mapping are
    # separate provenance channels.  Arbitrary functional intermediates are
    # deliberately not promoted to a cross-layer relation: a no-op/view can
    # preserve object identity and would otherwise fabricate producer edges.
    output_origin: dict[int, tuple[Any, int]] = {}
    mapping_origin: dict[int, tuple[Any, tuple[int, str, tuple[int, ...]]]] = {}
    mapping_before: dict[int, tuple[tuple[str, Any, frozenset[int]], ...]] = {}
    cross: dict[tuple, CrossLayerTensorUse] = {}
    sibling_inputs: dict[str, tuple[TensorShape, ...]] = {}
    sibling_outputs: dict[str, tuple[TensorShape, ...]] = {}
    sibling_order: dict[str, int] = {}
    current = {"layer": None, "order": 0}
    tensor_origins: dict[int, tuple[Any, frozenset[str]]] = {}
    stream_axes: dict[int, tuple[Any, tuple[int, int]]] = {}
    # Signed coefficients prove exact cancellation through add/sub/neg.  This
    # remains lineage, not value interpretation: multiplication by an
    # arbitrary tensor conservatively retains a nonzero dependency.
    # ``None`` is a structurally present but numerically unknown multiplier.
    # Only exact numeric opposites may cancel; a learned/tensor multiplier can
    # never be silently treated as coefficient one.
    mix_lineage: dict[int, tuple[Any, dict[int, float | None]]] = {}
    pending_contractions: list[MatrixContractionObservation] = []
    output_contractions: set[int] = set()
    source_hash_cache: dict[str, str | None] = {}
    allowed_source_hashes = frozenset(
        row.sha256 for row in provenance.source_files)

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
            boundary_primary_origins[index] = tuple(sorted(
                _identity_get(tensor_origins, primary) or frozenset()
                if primary is not None else ()))
            origins = boundary_primary_origins[index]
            if primary is not None and len(origins) == 1:
                specs = tuple(row for row in recipe.tensor_arguments
                              if row.name == origins[0])
                if len(specs) == 1:
                    extra = _unique_extra_axis(
                        tuple(int(x) for x in primary.shape),
                        tuple(specs[0].shape))
                    if extra is not None:
                        _identity_set(stream_axes, primary, extra)
            tracked_mappings: list[tuple[str, Any, frozenset[int]]] = []
            for arg_name, value in named:
                for tensor in tensors(value):
                    if tensor is primary:
                        continue
                    producer = _identity_get(output_origin, tensor)
                    if producer is None or producer >= index:
                        continue
                    key = (producer, index, arg_name, tuple(tensor.shape))
                    cross[key] = CrossLayerTensorUse(
                        producer, member_paths[producer], index, path,
                        arg_name, tuple(int(x) for x in tensor.shape))
                entries = _mapping_tensors(value)
                # Empty mutable state is still a provenance boundary: the
                # current layer may populate it for a later layer.  Tracking
                # only non-empty mappings loses the first producer.
                if isinstance(value, Mapping):
                    tracked_mappings.append((
                        arg_name, value,
                        frozenset(id(tensor) for _, tensor, _ in entries)))
            mapping_before[index] = tuple(tracked_mappings)
            current["layer"] = index
        return hook

    def layer_post(index: int):
        def hook(_module, _args, _kwargs, output):
            boundary_outputs[index] = _tree_shapes(output, "output")
            for tensor in tensors(output):
                if _identity_get(output_origin, tensor) is None:
                    _identity_set(output_origin, tensor, index)
                output_contractions.update(
                    marker for marker, coefficient in
                    (_mix_get(mix_lineage, tensor) or {}).items()
                    if coefficient != 0)
            for name, mapping, before in mapping_before.pop(index, ()):
                for entry_path, tensor, shape in _mapping_tensors(mapping):
                    if (id(tensor) not in before
                            and _identity_get(mapping_origin, tensor) is None):
                        _identity_set(
                            mapping_origin, tensor,
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
                    origin = _identity_get(mapping_origin, tensor)
                    if origin is None or origin[0] >= index:
                        continue
                    producer, argument, shape = origin
                    key = (producer, index, argument, shape)
                    cross[key] = CrossLayerTensorUse(
                        producer, member_paths[producer], index,
                        member_paths[index],
                        argument, shape)
            inherited = frozenset(
                origin
                for tensor in tensors((args, kwargs))
                for origin in (_identity_get(tensor_origins, tensor)
                               or frozenset()))
            result = function(*args, **kwargs)
            if inherited:
                for tensor in tensors(result):
                    _identity_set(tensor_origins, tensor, inherited)
            name = _normalised_function_name(function)
            input_tensors = tuple(tensors((args, kwargs)))
            output_tensors = tuple(tensors(result))

            # Preserve exact stream-axis position only through closed shape
            # transforms.  Unknown functions erase this proof rather than
            # borrowing a plausible axis from equal dimensions.
            if input_tensors and len(output_tensors) == 1:
                source_axis = _identity_get(stream_axes, input_tensors[0])
                output_tensor = output_tensors[0]
                if source_axis is not None:
                    axis, extent = source_axis
                    if name == "permute":
                        dims = tuple(int(x) for x in args[1:])
                        if len(dims) == len(input_tensors[0].shape) \
                                and sorted(dims) == list(range(len(dims))):
                            _identity_set(
                                stream_axes, output_tensor,
                                (dims.index(axis), extent))
                    elif name == "transpose" and len(args) >= 3:
                        dim0, dim1 = int(args[1]), int(args[2])
                        rank = len(input_tensors[0].shape)
                        dim0 %= rank
                        dim1 %= rank
                        new_axis = dim1 if axis == dim0 else dim0 if axis == dim1 else axis
                        _identity_set(
                            stream_axes, output_tensor, (new_axis, extent))
                    elif (name in {"clone", "contiguous", "detach", "to", "type_as"}
                          and tuple(output_tensor.shape) == tuple(input_tensors[0].shape)):
                        _identity_set(stream_axes, output_tensor, source_axis)
                if name in {"add", "mul", "sub"}:
                    agreeing = {
                        _identity_get(stream_axes, tensor)
                        for tensor in input_tensors
                        if _identity_get(stream_axes, tensor) is not None
                        and tuple(tensor.shape) == tuple(output_tensor.shape)
                    }
                    if len(agreeing) == 1:
                        _identity_set(
                            stream_axes, output_tensor, next(iter(agreeing)))

            if (name == "matmul" and index is not None and len(input_tensors) >= 2
                    and len(output_tensors) == 1):
                contraction = _matmul_contraction(
                    input_tensors[0], input_tensors[1], output_tensors[0],
                    stream_axes)
                site = _source_site(allowed_source_hashes, source_hash_cache)
                if contraction is not None and site is not None:
                    state_operand, input_axis, output_axis, extent = contraction
                    evidence_index = len(pending_contractions)
                    pending_contractions.append(MatrixContractionObservation(
                        index, member_paths[index], evidence_index, "matmul",
                        state_operand, input_axis, output_axis, extent,
                        (tuple(int(x) for x in input_tensors[0].shape),
                         tuple(int(x) for x in input_tensors[1].shape)),
                        tuple(int(x) for x in output_tensors[0].shape),
                        site[0], site[1]))
                    _mix_set(
                        mix_lineage, output_tensors[0], {evidence_index: 1.0})
                    _identity_set(
                        stream_axes, output_tensors[0], (output_axis, extent))

            # Once a matrix contraction exists, retain its positive dataflow
            # through a deliberately closed set of tensor value transforms.
            inputs_mix = [
                _mix_get(mix_lineage, tensor) or {}
                for tensor in input_tensors]
            inherited_mix: dict[int, float | None] = {}
            if name in {"add", "sub"}:
                alpha = kwargs.get("alpha", 1)
                for position, lineage in enumerate(inputs_mix):
                    factor: float | None = 1.0
                    if position > 0:
                        if isinstance(alpha, (int, float)) \
                                and not isinstance(alpha, bool):
                            factor = float(alpha) * (-1.0 if name == "sub" else 1.0)
                        else:
                            factor = None
                    for marker, coefficient in lineage.items():
                        accumulated = inherited_mix.get(marker, 0.0)
                        if (accumulated is None or coefficient is None
                                or factor is None):
                            inherited_mix[marker] = None
                        else:
                            inherited_mix[marker] = (
                                accumulated + factor * coefficient)
            elif name == "neg" and inputs_mix:
                inherited_mix = {
                    marker: (-coefficient if coefficient is not None else None)
                    for marker, coefficient in inputs_mix[0].items()}
            else:
                for lineage in inputs_mix:
                    inherited_mix.update(lineage)
            if name == "mul":
                scalars = [
                    value for value in args
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)]
                if any(value == 0 for value in scalars):
                    inherited_mix = {}
                elif len(scalars) == 1:
                    inherited_mix = {
                        marker: (coefficient * float(scalars[0])
                                 if coefficient is not None else None)
                        for marker, coefficient in inherited_mix.items()}
                else:
                    inherited_mix = {
                        marker: None for marker in inherited_mix}
            inherited_mix = {
                marker: coefficient for marker, coefficient in inherited_mix.items()
                if coefficient != 0}
            mix_transforms = {
                "add", "clone", "contiguous", "detach", "getitem", "mul",
                "neg", "permute", "repeat", "reshape", "sub", "to",
                "transpose", "type_as", "unsqueeze", "view"}
            if name in mix_transforms:
                for tensor in output_tensors:
                    if inherited_mix:
                        _mix_set(mix_lineage, tensor, dict(inherited_mix))
                    else:
                        _mix_delete(mix_lineage, tensor)
            else:
                # An unsupported in-place transform must not inherit the old
                # object's evidence merely because it returned that object.
                for tensor in output_tensors:
                    if any(tensor is source for source in input_tensors):
                        _mix_delete(mix_lineage, tensor)
            return result

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
    for index, ((_name, layer), path) in enumerate(zip(
            stack_members, member_paths)):
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
        with FakeTensorMode(allow_non_fake_inputs=True), TagMode():
            # Create and tag recipe inputs inside the same FakeTensor boundary
            # that executes the model.  Tagging a meta tensor before FakeTensor
            # wraps it binds evidence to a different Python object and invites
            # id-reuse false positives; exact identity must survive end to end.
            with torch.device("meta"):
                arguments = _tensor_arguments(recipe)
            for argument_name, value in arguments.items():
                for tensor in tensors(value):
                    _identity_set(
                        tensor_origins, tensor,
                        frozenset({str(argument_name)}))
            target(**arguments)
    finally:
        for handle in handles:
            handle.remove()

    boundaries = tuple(LayerBoundaryObservation(
        index, member_paths[index],
        boundary_order[index], boundary_inputs[index], boundary_outputs[index],
        boundary_primary_origins.get(index, ()))
        for index in range(len(stack_members)))
    sibling_rows = tuple(SiblingCallObservation(
        path, sibling_order[path], sibling_inputs.get(path, ()),
        sibling_outputs.get(path, ())) for path in sorted(sibling_order))
    contractions = tuple(
        dataclasses.replace(row, call_order=order)
        for order, row in enumerate(
            (row for index, row in enumerate(pending_contractions)
             if index in output_contractions))
    )
    return RelationObservation(
        3, provenance, recipe, stack_path, boundaries,
        tuple(sorted(cross.values(), key=lambda row: (
            row.producer_index, row.consumer_index, row.consumer_argument,
            row.shape))), sibling_rows, contractions)


def _failed(kind: str, stage: str, exc: BaseException,
            recipe: ExecutionRecipe | None = None,
            provenance: Provenance | None = None) -> RelationObservationResult:
    return RelationObservationResult(
        "failed", recipe=recipe, provenance=provenance,
        failure=Failure(kind, stage, f"{type(exc).__name__}: {str(exc)[:2000]}"))


def _worker(request_path: Path, recipe_path: Path, stack_path: str,
            result_path: Path) -> int:
    _write_network_attestation()
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
        attestation = _prepare_network_attestation(root, env)
        command = _network_isolated_command(
            [sys.executable, "-m", "physics.relation_observation", "--worker",
             str(request_path), str(recipe_path), stack_path, str(result_path)], env)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", env=env, start_new_session=True)
        _authorize_network_worker(process, attestation)
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
