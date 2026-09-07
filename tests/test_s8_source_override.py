"""The scratch demonstration must construct and inspect the same bytes."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import pytest

from physics.instance_inventory import BuildRequest, inventory_in_subprocess
from physics.source_override import SourceOverride, source_overrides
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


ROOT = Path(__file__).resolve().parents[1]
MODULE = "test_support.s6_models"


def _request(override=()):
    return BuildRequest(
        config={"width": 4}, framework="custom", factory_module=MODULE,
        factory_qualname="InventoryFixture", source_overrides=override,
        timeout_seconds=45, memory_limit_bytes=8 * 1024**3)


def _scratch(tmp_path, *, changed=False):
    original = ROOT / "test_support" / "s6_models.py"
    source = original.read_text()
    if changed:
        source = source.replace("for _ in range(2)", "for _ in range(3)")
    path = tmp_path / original.name
    path.write_text(source)
    return SourceOverride(MODULE, str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def test_empty_override_preserves_serialized_request():
    request = _request()
    assert "source_overrides" not in request.to_dict()
    assert BuildRequest.from_dict(request.to_dict()).to_dict() == request.to_dict()


def test_unchanged_scratch_has_identical_inventory(tmp_path):
    original = inventory_in_subprocess(_request())
    override = _scratch(tmp_path)
    copied = inventory_in_subprocess(_request((override,)))
    assert original.status == copied.status == "ok", (original.failure, copied.failure)
    assert asdict(original.inventory) == asdict(copied.inventory)


def test_changed_source_changes_construction_and_both_hashes_agree(tmp_path):
    override = _scratch(tmp_path, changed=True)
    request = _request((override,))
    assert BuildRequest.from_dict(request.to_dict()).to_dict() == request.to_dict()
    result = inventory_in_subprocess(request)
    assert result.status == "ok", result.failure
    inventory = result.inventory
    assert "layers.2" in {row.path for row in inventory.modules}
    runtime = next(row for row in inventory.provenance.source_files if row.module == MODULE)
    bundle = SourceBundle(source="path", files=(override.path,),
                          component_files={"root": (override.path,)})
    index = build_program_index(bundle)
    symbol = next(row.symbol for row in index.classes
                  if row.symbol.qualified_name == "InventoryFixture")
    assert runtime.sha256 == symbol.source.content_fingerprint == override.sha256


def test_wrong_hash_cannot_fall_back_to_installed_source(tmp_path):
    override = replace(_scratch(tmp_path), sha256="0" * 64)
    result = inventory_in_subprocess(_request((override,)))
    assert result.status == "failed"
    assert result.failure.kind == "ConstructionFailed"
    assert "fingerprint differs" in result.failure.detail


def test_unused_override_cannot_claim_the_builder_examined_it(tmp_path):
    override = replace(_scratch(tmp_path), module="test_support.not_imported")
    result = inventory_in_subprocess(_request((override,)))
    assert result.status == "failed"
    assert "not examined" in result.failure.detail


def test_preloaded_module_cannot_bypass_substitution(tmp_path):
    override = replace(_scratch(tmp_path), module="physics.source_override")
    with pytest.raises(ValueError, match="already imported"):
        with source_overrides((override,)):
            pytest.fail("stale source must not run")


def test_duplicate_import_address_is_rejected(tmp_path):
    override = _scratch(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        _request((override, override))
