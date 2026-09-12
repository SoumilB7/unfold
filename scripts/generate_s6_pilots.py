#!/usr/bin/env python3
"""Generate the frozen S6 instance/trace pilot artifacts.

Model identifiers in this file select the deliberately enumerated experiment;
they never enter the generic physics substrate or a production consumer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.execution_observation import (
    ExecutionRecipe, TensorArgument, observe_in_subprocess,
)
from physics.instance_inventory import BuildRequest, inventory_in_subprocess


CORPUS = ROOT / "tests" / "sable_test_corpus"
DEFAULT_OUTPUT = ROOT / "verification" / "s6" / "pilots"
GIB = 1024**3


def _versions(*packages: str) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in packages}


def _config(slug: str) -> dict[str, Any]:
    return json.loads((CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]


def _transformers(slug: str, module: str, model: str,
                  config_module: str, config_class: str) -> BuildRequest:
    return BuildRequest(
        config=_config(slug), framework="transformers", factory_module=module,
        factory_qualname=model, config_module=config_module,
        config_qualname=config_class, timeout_seconds=240,
        memory_limit_bytes=16 * GIB, label=slug,
    )


def _diffusers(slug: str, factory: str) -> BuildRequest:
    return BuildRequest(
        config=_config(slug), framework="diffusers", factory_module="diffusers",
        factory_qualname=factory, factory_method="from_config",
        timeout_seconds=240, memory_limit_bytes=16 * GIB, label=slug,
    )


def _text(recipe_id: str, *, dtype: str = "float32") -> ExecutionRecipe:
    return ExecutionRecipe(
        recipe_id=recipe_id, input_modality="tokens", train_eval="eval",
        cache_state="disabled", encoder_decoder_mode="decoder",
        conditioning_present=False, dtype=dtype,
        library_versions=_versions("torch", "transformers"),
        tensor_arguments=(TensorArgument("input_ids", (1, 8), "long"),),
        literal_arguments={"use_cache": False},
    )


def pilot_specs() -> dict[str, tuple[BuildRequest, tuple[ExecutionRecipe, ...]]]:
    llama = _transformers(
        "llama-7b", "transformers.models.llama.modeling_llama",
        "LlamaForCausalLM", "transformers.models.llama.configuration_llama",
        "LlamaConfig")
    deepseek = _transformers(
        "deepseek-v3", "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3ForCausalLM",
        "transformers.models.deepseek_v3.configuration_deepseek_v3",
        "DeepseekV3Config")
    qwen = _transformers(
        "qwen2-vl-7b-instruct",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "Qwen2VLForConditionalGeneration",
        "transformers.models.qwen2_vl.configuration_qwen2_vl",
        "Qwen2VLConfig")
    musicgen = _transformers(
        "musicgen-small", "transformers.models.musicgen.modeling_musicgen",
        "MusicgenForConditionalGeneration",
        "transformers.models.musicgen.configuration_musicgen", "MusicgenConfig")
    dbrx = _transformers(
        "dbrx-base", "transformers.models.dbrx.modeling_dbrx",
        "DbrxForCausalLM", "transformers.models.dbrx.configuration_dbrx",
        "DbrxConfig")
    return {
        "llama-7b": (llama, (_text("text-eval-no-cache"),)),
        "deepseek-v3": (deepseek, (_text("text-eval-no-cache", dtype="bfloat16"),)),
        "qwen2-vl-7b-instruct": (qwen, (
            _text("text-eval-no-cache"),
            ExecutionRecipe(
                recipe_id="vision-tower-eval", input_modality="vision",
                train_eval="eval", cache_state="disabled",
                encoder_decoder_mode="encoder", conditioning_present=False,
                dtype="float32", library_versions=_versions("torch", "transformers"),
                target_path="model.visual",
                tensor_arguments=(
                    TensorArgument("hidden_states", (4, 1176), "float32"),
                    TensorArgument("grid_thw", (1, 3), "long", "values",
                                   [[1, 2, 2]], "cpu"),
                ),
            ),
            ExecutionRecipe(
                recipe_id="vision-mlp-block-0", input_modality="vision",
                train_eval="eval", cache_state="disabled",
                encoder_decoder_mode="encoder", conditioning_present=False,
                dtype="float32", library_versions=_versions("torch", "transformers"),
                target_path="model.visual.blocks.0.mlp",
                tensor_arguments=(
                    TensorArgument("x", (4, 1280), "float32"),
                ),
            ),
        )),
        "stable-diffusion-3-5-large": (
            _diffusers("stable-diffusion-3-5-large", "SD3Transformer2DModel"),
            (ExecutionRecipe(
                recipe_id="transformer-block-0-bf16", input_modality="latent",
                train_eval="eval", cache_state="disabled",
                encoder_decoder_mode="joint", conditioning_present=True,
                dtype="bfloat16", library_versions=_versions("torch", "diffusers"),
                target_path="transformer_blocks.0",
                tensor_arguments=(
                    TensorArgument("hidden_states", (1, 4, 2432), "bfloat16"),
                    TensorArgument("encoder_hidden_states", (1, 4, 2432),
                                   "bfloat16"),
                    TensorArgument("temb", (1, 2432), "bfloat16"),
                ),
            ),),
        ),
        "pixart-sigma-xl-2-1024-ms": (
            # The legacy factory name is intentional: the result must record
            # the runtime remap to PixArtTransformer2DModel.
            _diffusers("pixart-sigma-xl-2-1024-ms", "Transformer2DModel"),
            (ExecutionRecipe(
                recipe_id="latent-eval-conditioned", input_modality="latent",
                train_eval="eval", cache_state="disabled",
                encoder_decoder_mode="denoiser", conditioning_present=True,
                dtype="float32", library_versions=_versions("torch", "diffusers"),
                tensor_arguments=(
                    TensorArgument("hidden_states", (1, 4, 8, 8), "float32"),
                    TensorArgument("encoder_hidden_states", (1, 4, 4096),
                                   "float32"),
                    TensorArgument("timestep", (1,), "long"),
                ), literal_arguments={"return_dict": False},
            ),),
        ),
        "stable-diffusion-xl-base-1-0": (
            _diffusers("stable-diffusion-xl-base-1-0", "UNet2DConditionModel"),
            (ExecutionRecipe(
                recipe_id="unet-eval-conditioned", input_modality="latent",
                train_eval="eval", cache_state="disabled",
                encoder_decoder_mode="denoiser", conditioning_present=True,
                dtype="float32", library_versions=_versions("torch", "diffusers"),
                tensor_arguments=(
                    TensorArgument("sample", (1, 4, 8, 8), "float32"),
                    TensorArgument("timestep", (1,), "long"),
                    TensorArgument("encoder_hidden_states", (1, 4, 2048),
                                   "float32"),
                    TensorArgument("added_cond_kwargs.text_embeds", (1, 1280),
                                   "float32"),
                    TensorArgument("added_cond_kwargs.time_ids", (1, 6),
                                   "float32"),
                ), literal_arguments={"return_dict": False},
            ),),
        ),
        "musicgen-small": (musicgen, (ExecutionRecipe(
            recipe_id="text-audio-eval-no-cache", input_modality="text+audio",
            train_eval="eval", cache_state="disabled",
            encoder_decoder_mode="encoder-decoder", conditioning_present=True,
            dtype="float32", library_versions=_versions("torch", "transformers"),
            tensor_arguments=(
                TensorArgument("input_ids", (1, 4), "long"),
                TensorArgument("decoder_input_ids", (4, 4), "long"),
            ), literal_arguments={"use_cache": False},
        ),)),
        "dbrx-base": (dbrx, (
            # The whole-model recipe intentionally retains the typed failure
            # produced by the installed Transformers implementation/config pair.
            _text("text-eval-no-cache", dtype="bfloat16"),
            ExecutionRecipe(
                recipe_id="attention-block-0-bf16",
                input_modality="hidden_states", train_eval="eval",
                cache_state="disabled", encoder_decoder_mode="decoder",
                conditioning_present=False, dtype="bfloat16",
                library_versions=_versions("torch", "transformers"),
                target_path="transformer.blocks.0.norm_attn_norm.attn",
                tensor_arguments=(
                    TensorArgument("hidden_states", (1, 8, 6144), "bfloat16"),
                    TensorArgument("position_embeddings.0", (1, 8, 128),
                                   "bfloat16"),
                    TensorArgument("position_embeddings.1", (1, 8, 128),
                                   "bfloat16"),
                ),
            ),
        )),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def generate(output: Path, only: set[str] | None = None, *,
             inventories_only: bool = False) -> None:
    specs = pilot_specs()
    unknown = (only or set()) - set(specs)
    if unknown:
        raise ValueError(f"unknown pilot(s): {sorted(unknown)}")
    for slug, (request, recipes) in specs.items():
        if only and slug not in only:
            continue
        target = output / slug
        _write_json(target / "request.json", request.to_dict())
        inventory = inventory_in_subprocess(request)
        _write_json(target / "inventory.json", inventory.to_dict())
        if inventories_only:
            continue
        for recipe in recipes:
            observation = observe_in_subprocess(request, recipe)
            _write_json(target / f"observation-{recipe.recipe_id}.json",
                        observation.to_dict())
    write_manifest(output)


def write_manifest(output: Path) -> None:
    artifacts: dict[str, str] = {}
    pilots: dict[str, Any] = {}
    for directory in sorted(path for path in output.iterdir() if path.is_dir()):
        inventory = json.loads((directory / "inventory.json").read_text())
        provenance = (inventory.get("inventory") or {}).get("provenance") or {}
        pilots[directory.name] = {
            "inventory_status": inventory["status"],
            "resolved_class": provenance.get("resolved_class"),
            "packages": provenance.get("packages", []),
            "observations": {
                path.stem.removeprefix("observation-"):
                    json.loads(path.read_text())["status"]
                for path in sorted(directory.glob("observation-*.json"))
            },
        }
        for path in sorted(directory.glob("*.json")):
            relative = path.relative_to(output).as_posix()
            artifacts[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(output / "manifest.json", {
        "schema_version": 1, "pilots": pilots, "artifacts": artifacts,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--inventories-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        write_manifest(args.output)
    else:
        generate(args.output, set(args.only) or None,
                 inventories_only=args.inventories_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
