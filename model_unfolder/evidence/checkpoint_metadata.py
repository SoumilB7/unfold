"""Independent stored tensor shapes, read from exact safetensors header bytes.

This channel never infers a layer or mechanism from a tensor name. Only an
exact instantiated parameter address may be compared. Unmatched addresses
remain limited; no prefix stripping or model-specific name mapping is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path


MAX_HEADER_BYTES = 16 * 1024 * 1024
_DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
    "I32": 32, "U32": 32, "I64": 64, "U64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8,
}


class CheckpointMetadataError(ValueError):
    """The supplied header cannot establish stored tensor metadata."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointMetadataError("duplicate safetensors header key")
        result[key] = value
    return result


@dataclass(frozen=True)
class StoredTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offsets: tuple[int, int]


@dataclass(frozen=True)
class CheckpointHeader:
    """The bytes are the authority; derived tensor rows cannot be supplied."""

    header_bytes: bytes = field(repr=False)
    tensors: tuple[StoredTensor, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self):
        raw = self.header_bytes
        if type(raw) is not bytes or not 2 <= len(raw) <= MAX_HEADER_BYTES:
            raise CheckpointMetadataError("safetensors header size is out of bounds")
        try:
            document = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointMetadataError("invalid safetensors header JSON") from exc
        if not isinstance(document, dict):
            raise CheckpointMetadataError("safetensors header must be an object")
        tensors = []
        for name, row in document.items():
            if name == "__metadata__":
                if not isinstance(row, dict) or any(
                        not isinstance(k, str) or not isinstance(v, str)
                        for k, v in row.items()):
                    raise CheckpointMetadataError("header metadata must contain strings")
                continue
            if not name or not isinstance(row, dict) or set(row) != {
                    "shape", "dtype", "data_offsets"}:
                raise CheckpointMetadataError("invalid tensor metadata row")
            shape, dtype, offsets = row["shape"], row["dtype"], row["data_offsets"]
            if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise CheckpointMetadataError("tensor dimensions must be nonnegative integers")
            if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
                raise CheckpointMetadataError("unsupported stored tensor dtype")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(
                    type(x) is not int or x < 0 for x in offsets):
                raise CheckpointMetadataError("invalid tensor data offsets")
            if offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BITS[dtype] // 8:
                raise CheckpointMetadataError("tensor shape and byte extent disagree")
            tensors.append(StoredTensor(name, tuple(shape), dtype, tuple(offsets)))
        end = 0
        for tensor in sorted(tensors, key=lambda item: (item.offsets, item.name)):
            if tensor.offsets[0] != end:
                raise CheckpointMetadataError("tensor byte extents contain a gap or overlap")
            end = tensor.offsets[1]
        object.__setattr__(self, "tensors", tuple(sorted(tensors, key=lambda item: item.name)))
        object.__setattr__(self, "sha256", hashlib.sha256(raw).hexdigest())

    @classmethod
    def read(cls, path: str | Path):
        """Read the length prefix and header only; never read weight payloads."""
        with Path(path).open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CheckpointMetadataError("truncated safetensors length prefix")
            size = int.from_bytes(prefix, "little")
            if not 2 <= size <= MAX_HEADER_BYTES:
                raise CheckpointMetadataError("safetensors header size is out of bounds")
            raw = handle.read(size)
            if len(raw) != size:
                raise CheckpointMetadataError("truncated safetensors header")
        return cls(raw)


def compare_checkpoint_shapes(inventory, headers, *, config):
    """Compare independent stored shapes against this config's exact inventory."""
    from physics.instance_inventory import InstanceInventory

    if not isinstance(inventory, InstanceInventory):
        raise TypeError("shape consistency requires an actual instance inventory")
    config_hash = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if inventory.provenance.config_sha256 != config_hash:
        raise ValueError("consistency inventory belongs to a different config")
    if not isinstance(headers, tuple) or not headers or any(
            not isinstance(header, CheckpointHeader) for header in headers):
        raise TypeError("shape consistency requires exact checkpoint headers")
    stored = {}
    for header in headers:
        for tensor in header.tensors:
            if tensor.name in stored:
                raise CheckpointMetadataError("tensor address appears in multiple headers")
            stored[tensor.name] = (tensor, header.sha256)
    constructed = {
        f"{module.path}.{parameter.name}".lstrip("."): parameter.shape
        for module in inventory.modules for parameter in module.parameters
    }
    shared = sorted(constructed.keys() & stored.keys())
    unbound = sorted(stored.keys() - constructed.keys())
    missing = sorted(constructed.keys() - stored.keys())
    conflicts = [{
        "reason": "construction_conflict", "parameter": name,
        "constructed_shape": list(constructed[name]),
        "checkpoint_shape": list(stored[name][0].shape),
        "checkpoint_header_sha256": stored[name][1],
    } for name in shared if constructed[name] != stored[name][0].shape]
    return {
        "inventory_config_sha256": config_hash,
        "checkpoint_header_sha256": sorted(header.sha256 for header in headers),
        "compared_parameters": len(shared), "conflicts": conflicts,
        "unbound_checkpoint_parameters": unbound,
        "parameters_without_checkpoint_metadata": missing,
        "unresolved_binding_reason": ({
            "reason_class": "investigation_missing",
            "concrete_reason": "checkpoint_parameter_binding_incomplete",
            "investigation": None,
        } if unbound or missing else None),
    }
