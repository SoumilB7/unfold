"""Independent tensor metadata cannot be replaced by same-config construction."""
from dataclasses import replace
import io
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.checkpoint_metadata import (
    CheckpointHeader, CheckpointMetadataError, MAX_HEADER_BYTES,
    compare_checkpoint_shapes,
)


def _header(name="weight", shape=(2, 3)):
    import math
    return json.dumps({name: {"shape": list(shape), "dtype": "F32",
                             "data_offsets": [0, 4 * math.prod(shape)]}}).encode()


def test_reader_never_reads_the_weight_payload(monkeypatch):
    raw = _header()
    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            assert size >= 0 and self.tell() + size <= 8 + len(raw)
            return super().read(size)
    stream = HeaderOnly(len(raw).to_bytes(8, "little") + raw + b"UNREAD WEIGHTS")
    monkeypatch.setattr(Path, "open", lambda *a, **k: stream)
    header = CheckpointHeader.read("checkpoint.safetensors")
    assert header.tensors[0].shape == (2, 3)


@pytest.mark.parametrize("raw", [
    b'[]', b'{"weight":{},"weight":{}}',
    b'{"x":{"shape":[true],"dtype":"F32","data_offsets":[0,4]}}',
    b'{"x":{"shape":[2],"dtype":"F32","data_offsets":[0,4]}}',
    b'{"x":{"shape":[1],"dtype":"F32","data_offsets":[4,8]}}',
    b'{"x":{"shape":[1],"dtype":"UNKNOWN","data_offsets":[0,4]}}',
])
def test_malformed_headers_do_not_establish_values(raw):
    with pytest.raises(CheckpointMetadataError):
        CheckpointHeader(raw)


def test_oversize_header_is_rejected_before_reading_it(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
    with pytest.raises(CheckpointMetadataError, match="size"):
        CheckpointHeader.read(path)


@pytest.fixture
def inventory_and_config():
    from physics.instance_inventory import InventoryResult
    pilot = Path(__file__).resolve().parents[1] / "verification/s6/pilots/llama-7b"
    result = InventoryResult.from_dict(json.loads((pilot / "inventory.json").read_text()))
    config = json.loads((pilot / "request.json").read_text())["config"]
    assert result.status == "ok"
    return result.inventory, config


def test_changed_constructed_dimension_conflicts_with_unchanged_stored_header(inventory_and_config):
    inventory, config = inventory_and_config
    module = next(item for item in inventory.modules if item.parameters)
    parameter = module.parameters[0]
    name = f"{module.path}.{parameter.name}".lstrip(".")
    header = CheckpointHeader(_header(name, parameter.shape))
    before = compare_checkpoint_shapes(inventory, (header,), config=config)
    assert before["compared_parameters"] == 1 and before["conflicts"] == []
    changed_parameter = replace(parameter, shape=(parameter.shape[0] + 1, *parameter.shape[1:]))
    changed_module = replace(module, parameters=(changed_parameter, *module.parameters[1:]))
    changed = replace(inventory, modules=tuple(
        changed_module if item is module else item for item in inventory.modules))
    after = compare_checkpoint_shapes(changed, (header,), config=config)
    assert after["conflicts"] == [{
        "reason": "construction_conflict", "parameter": name,
        "constructed_shape": list(changed_parameter.shape),
        "checkpoint_shape": list(parameter.shape),
        "checkpoint_header_sha256": header.sha256,
    }]


def test_another_config_inventory_is_rejected(inventory_and_config):
    inventory, config = inventory_and_config
    with pytest.raises(ValueError, match="different config"):
        compare_checkpoint_shapes(inventory, (CheckpointHeader(_header()),),
                                  config={**config, "num_key_value_heads": -1})


def test_unmatched_parameter_never_receives_a_guessed_name_binding(inventory_and_config):
    inventory, config = inventory_and_config
    result = compare_checkpoint_shapes(
        inventory, (CheckpointHeader(_header("another_owner.weight")),), config=config)
    assert result["compared_parameters"] == 0 and result["conflicts"] == []
    assert result["unbound_checkpoint_parameters"] == ["another_owner.weight"]


def test_duplicate_shard_address_is_not_silently_overwritten(inventory_and_config):
    inventory, config = inventory_and_config
    header = CheckpointHeader(_header())
    with pytest.raises(CheckpointMetadataError, match="multiple headers"):
        compare_checkpoint_shapes(inventory, (header, header), config=config)
