"""S6 isolation, inventory, and positive-execution poison matrix."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import pytest

from physics.execution_observation import (
    ExecutionRecipe, FunctionalOp, ModuleCall, ObservationResult, TensorArgument,
    observe_in_subprocess,
)
from physics.instance_inventory import (
    BuildRequest, Failure, InventoryResult, PackageVersion, ResolvedClass,
    SourceFile, _network_isolated_command, inventory_in_subprocess,
)


ROOT = Path(__file__).resolve().parents[1]
PILOTS = ROOT / "verification" / "s6" / "pilots"
PILOT_SLUGS = {
    "llama-7b", "deepseek-v3", "qwen2-vl-7b-instruct",
    "stable-diffusion-3-5-large", "pixart-sigma-xl-2-1024-ms",
    "stable-diffusion-xl-base-1-0", "musicgen-small", "dbrx-base",
}


def _request(name: str, config=None, **kwargs) -> BuildRequest:
    return BuildRequest(
        config=config or {}, framework="custom",
        factory_module="test_support.s6_models", factory_qualname=name,
        timeout_seconds=kwargs.pop("timeout_seconds", 30),
        memory_limit_bytes=kwargs.pop("memory_limit_bytes", 8 * 1024**3),
        label=name, **kwargs)


def _recipe(name="synthetic", width=4) -> ExecutionRecipe:
    return ExecutionRecipe(
        recipe_id=name, input_modality="hidden_states", train_eval="eval",
        cache_state="disabled", encoder_decoder_mode="decoder",
        conditioning_present=False, dtype="float32",
        library_versions={"torch": importlib.metadata.version("torch")},
        tensor_arguments=(TensorArgument("x", (1, 2, width), "float32"),),
    )


def test_inventory_is_exact_typed_and_versioned():
    result = inventory_in_subprocess(_request("InventoryFixture", {"width": 4}))
    assert result.status == "ok", result.failure
    inv = result.inventory
    assert inv is not None
    assert inv.provenance.config_sha256 == __import__("hashlib").sha256(
        b'{"width":4}').hexdigest()
    assert inv.provenance.resolved_class.qualname == "InventoryFixture"
    assert {x.package for x in inv.provenance.packages} >= {"torch"}
    assert all(len(x.sha256) == 64 for x in inv.provenance.source_files)
    assert any(group.members == ("layers.0", "layers.1")
               for group in inv.repetition_groups)
    assert any(group.names == ("embed.weight", "head.weight")
               for group in inv.parameter_aliases)
    root = inv.modules[0]
    assert root.init_attributes["width"] == 4
    optional = next(row for row in root.guarded_none_children
                    if row["path"] == "optional")
    assert optional["value"] is None and optional["guards"]
    assert not any(row["path"] == "unconditional_none"
                   for row in root.guarded_none_children)


def test_constructor_cannot_rewrite_the_checkpoint_config_hash():
    config = {"nested": {"value": 7}}
    result = inventory_in_subprocess(_request("MutatingConfigModel", config))
    assert result.status == "ok", result.failure
    assert result.inventory is not None
    expected = hashlib.sha256(b'{"nested":{"value":7}}').hexdigest()
    assert result.inventory.provenance.config_sha256 == expected
    assert config == {"nested": {"value": 7}}


def test_constructor_network_attempt_is_typed_network_refused():
    result = inventory_in_subprocess(_request("NetworkModel"))
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "NetworkRefused"
    assert "example.com" not in result.stdout  # no secret/request echo channel

    escaped = inventory_in_subprocess(_request("SubprocessNetworkModel"))
    assert escaped.status == "failed"
    assert escaped.failure and escaped.failure.kind == "NetworkRefused"
    assert "network client process refused" in escaped.failure.detail


def test_linux_os_sandbox_command_is_explicit_never_audit_hook_laundered(
        monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("physics.instance_inventory.shutil.which",
                        lambda name: f"/usr/bin/{name}")
    env = {"UNFOLD_LINUX_NETWORK_SANDBOX": "sudo-unshare"}
    command = _network_isolated_command(["python", "worker.py"], env)
    assert command == [
        "/usr/bin/sudo", "-n", "/usr/bin/unshare", "--net", "--",
        "python", "worker.py"]
    assert env["UNFOLD_NETWORK_SANDBOX"] == (
        "linux-sudo-unshare-net+python-audit-hook")

    # No opt-in is honestly recorded as audit-only, never as OS-isolated.
    env = {}
    assert _network_isolated_command(["python"], env) == ["python"]
    assert env["UNFOLD_NETWORK_SANDBOX"] == "python-audit-hook-only"


def test_constructor_timeout_is_typed_and_child_is_killed():
    result = inventory_in_subprocess(_request(
        "SlowModel", {"seconds": 5}, timeout_seconds=0.2))
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "TimeoutExpired"


def test_constructor_output_is_drained_without_deadlock_and_capture_is_bounded():
    result = inventory_in_subprocess(_request(
        "NoisyModel", {"characters": 1024 * 1024}, timeout_seconds=10))
    assert result.status == "ok", result.failure
    assert 60_000 < len(result.stdout) <= 65_536


def test_constructor_memory_cap_is_typed():
    result = inventory_in_subprocess(_request(
        "MemoryModel", {"bytes": 3 * 1024**3, "chunk_bytes": 64 * 1024**2},
        memory_limit_bytes=1024**3, timeout_seconds=30))
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "MemoryLimitExceeded"


def test_execution_records_module_order_function_ops_and_lazy_module():
    lazy = observe_in_subprocess(_request("LazyModel"), _recipe("lazy"))
    assert lazy.status == "ok", lazy.failure
    assert lazy.observation and [x.path for x in lazy.observation.module_calls][0] == ""
    assert [x.path for x in lazy.observation.lazy_observed] == ["cache"]
    assert {x.op for x in lazy.observation.functional_ops} >= {"add", "mul"}

    functional = observe_in_subprocess(_request("FunctionalModel"), _recipe())
    assert functional.status == "ok", functional.failure
    assert functional.observation
    assert {x.op for x in functional.observation.functional_ops} >= {
        "chunk", "cat", "layer_norm", "silu", "gelu", "add", "mul"}
    assert functional.observation.recipe.dtype == "float32"
    assert functional.observation.provenance.packages


def test_data_dependent_control_flow_is_unresolved_not_guessed():
    result = observe_in_subprocess(_request("DataDependentModel"), _recipe("dynamic"))
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "ExecutionUnresolved"
    assert result.provenance is not None
    assert result.recipe == _recipe("dynamic")


def test_recipe_preserves_nested_tuple_and_mapping_arguments():
    recipe = ExecutionRecipe(
        recipe_id="nested", input_modality="vision", train_eval="eval",
        cache_state="disabled", encoder_decoder_mode="encoder",
        conditioning_present=True, dtype="float32",
        library_versions={"torch": importlib.metadata.version("torch")},
        tensor_arguments=(
            TensorArgument("pair.0", (1, 2), "float32"),
            TensorArgument("pair.1", (1, 2), "float32"),
            TensorArgument("nested.value", (1, 2), "float32"),
        ),
    )
    result = observe_in_subprocess(_request("TupleInputModel"), recipe)
    assert result.status == "ok", result.failure
    assert result.observation is not None
    assert [op.op for op in result.observation.functional_ops] == ["add", "add"]


def test_recipe_with_false_library_version_is_rejected():
    recipe = dataclasses.replace(_recipe("wrong-version"),
                                 library_versions={"torch": "0.invalid"})
    result = observe_in_subprocess(_request("FunctionalModel"), recipe)
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "ExecutionFailed"
    assert "library version mismatch" in result.failure.detail


def test_result_dtos_are_closed():
    failure = Failure("WorkerFailed", "x", "y")
    with pytest.raises(ValueError):
        InventoryResult("ok", failure=failure)
    with pytest.raises(ValueError):
        ObservationResult("failed")
    with pytest.raises(ValueError):
        dataclasses.replace(_recipe(), train_eval="sometimes")
    with pytest.raises(ValueError):
        BuildRequest({}, "custom", "x", "Y", build_flags={"trust_remote_code": True})
    with pytest.raises(ValueError):
        PackageVersion("", "1")
    with pytest.raises(ValueError):
        SourceFile("m", "m.py", "not-a-hash")
    with pytest.raises(ValueError):
        ModuleCall(-1, "", ResolvedClass("m", "C"))
    with pytest.raises(ValueError):
        FunctionalOp(0, "conventional_attention", "torch.guess")


def test_physics_has_no_production_consumer_and_no_model_identity_branch():
    consumers = [*ROOT.glob("model_unfolder/adapters/**/*.py"),
                 *ROOT.glob("model_unfolder/renderers/**/*.py")]
    assert not [(path, line) for path in consumers
                for line in path.read_text().splitlines() if "physics" in line]
    production = list((ROOT / "model_unfolder").rglob("*.py"))
    assert not [(path, line) for path in production
                for line in path.read_text().splitlines()
                if line.lstrip().startswith(("import physics", "from physics"))]
    source = "\n".join((ROOT / "physics" / name).read_text()
                       for name in ("instance_inventory.py", "execution_observation.py"))
    for identity in ("llama", "deepseek", "qwen", "pixart", "sdxl", "musicgen", "dbrx"):
        assert identity not in source.lower()


def test_persisted_result_is_json_roundtrippable():
    result = inventory_in_subprocess(_request("InventoryFixture"))
    rebuilt = InventoryResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert rebuilt == result


def _artifact(slug: str, name: str):
    return json.loads((PILOTS / slug / name).read_text(encoding="utf-8"))


def test_all_eight_versioned_pilots_have_inventory_and_named_observations():
    assert {path.name for path in PILOTS.iterdir() if path.is_dir()} == PILOT_SLUGS
    for slug in PILOT_SLUGS:
        directory = PILOTS / slug
        request = _artifact(slug, "request.json")
        result = _artifact(slug, "inventory.json")
        rebuilt_inventory = InventoryResult.from_dict(result)
        assert json.loads(json.dumps(rebuilt_inventory.to_dict())) == result
        assert request["label"] == slug
        assert result["status"] == "ok", (slug, result.get("failure"))
        provenance = result["inventory"]["provenance"]
        assert len(provenance["config_sha256"]) == 64
        config_bytes = json.dumps(
            request["config"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")
        assert provenance["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
        assert provenance["packages"] and provenance["source_files"]
        assert provenance["environment"]["network"] == (
            "macos-sandbox-exec+python-audit-hook")
        assert all(len(row["sha256"]) == 64 for row in provenance["source_files"])
        for source in provenance["source_files"]:
            installed = Path(importlib.import_module(source["module"]).__file__)
            assert installed.name == source["path"]
            assert hashlib.sha256(installed.read_bytes()).hexdigest() == source["sha256"]
        observations = sorted(directory.glob("observation-*.json"))
        assert observations, slug
        for path in observations:
            row = json.loads(path.read_text(encoding="utf-8"))
            rebuilt_observation = ObservationResult.from_dict(row)
            assert json.loads(json.dumps(rebuilt_observation.to_dict())) == row
            # Construction succeeded for every frozen pilot, so even a typed
            # execution failure retains its exact §1c construction provenance.
            assert row["provenance"] == provenance
            assert row["recipe"]
            if row.get("observation"):
                assert row["recipe"] == row["observation"]["recipe"]


def test_sd35_first_block_pins_the_measured_functional_relation_counts():
    row = _artifact(
        "stable-diffusion-3-5-large",
        "observation-transformer-block-0-bf16.json")
    assert row["status"] == "ok"
    counts: dict[str, int] = {}
    for operation in row["observation"]["functional_ops"]:
        counts[operation["op"]] = counts.get(operation["op"], 0) + 1
    assert counts["scaled_dot_product_attention"] == 1
    assert counts["add"] == 16
    assert counts["mul"] == 16


def test_pilot_outliers_remain_explicit_and_positive_subrecipes_stay_useful():
    qwen_full = _artifact(
        "qwen2-vl-7b-instruct", "observation-vision-tower-eval.json")
    qwen_local = _artifact(
        "qwen2-vl-7b-instruct", "observation-vision-mlp-block-0.json")
    assert qwen_full["status"] == "failed"
    assert qwen_full["failure"]["kind"] == "ExecutionUnresolved"
    assert qwen_local["status"] == "ok"
    assert qwen_local["observation"]["module_calls"]

    dbrx_full = _artifact("dbrx-base", "observation-text-eval-no-cache.json")
    dbrx_attention = _artifact(
        "dbrx-base", "observation-attention-block-0-bf16.json")
    assert dbrx_full["status"] == "failed"
    assert dbrx_full["failure"]["kind"] == "ExecutionFailed"
    assert dbrx_attention["status"] == "ok"
    assert any(row["op"] == "scaled_dot_product_attention"
               for row in dbrx_attention["observation"]["functional_ops"])


def test_inventory_records_runtime_factory_remap_and_guarded_none_child():
    pixart = _artifact("pixart-sigma-xl-2-1024-ms", "inventory.json")["inventory"]
    assert pixart["provenance"]["requested_factory"].endswith(".Transformer2DModel")
    assert pixart["provenance"]["resolved_class"]["qualname"] == (
        "PixArtTransformer2DModel")

    sd35 = _artifact("stable-diffusion-3-5-large", "inventory.json")["inventory"]
    block0 = next(row for row in sd35["modules"]
                  if row["path"] == "transformer_blocks.0")
    assert any(row["path"] == "transformer_blocks.0.attn2"
               and row["value"] is None
               and row["guards"]
               for row in block0["guarded_none_children"])


def test_pilot_manifest_is_exact_non_vacuous_and_hashes_every_artifact():
    manifest = json.loads((PILOTS / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["pilots"]) == PILOT_SLUGS
    actual = {
        path.relative_to(PILOTS).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PILOTS.glob("*/*.json")
    }
    assert manifest["artifacts"] == actual
    assert len(actual) == 27
