"""The everchanging config-vocabulary layer — domain folders + loaders.

Vocabulary (aliases, ignore lists, type maps, approved block stages) is data in
``everchanging/<domain>/*.yaml``, loaded via PyYAML when present and a built-in
flow/block reader otherwise.  These tests pin both paths and the taxonomies.
"""
import builtins

import pytest

from model_unfolder import everchanging as ec
from model_unfolder.block_schema import DIFFUSION_PART_KINDS, DIFFUSION_STAGES, TRANSFORMER_STAGES


def test_transformer_vocab_loads():
    assert "num_hidden_layers" in ec.load_aliases()
    assert ec.load_ignored_fields()["keys"]
    assert ec.load_transformer_typing()["stages"]


def test_layer_type_labels_externalized_and_cover_modern_spellings():
    labels = ec.load_layer_type_labels()
    # All four mask groups present, loaded from YAML (not hardcoded in the parser).
    assert set(labels) == {"full", "sliding", "compressed_sparse", "heavily_compressed"}
    # The spellings whose absence used to produce false "treated as causal" warnings.
    assert "attention" in labels["full"]                  # Nemotron-H / hybrid stacks
    assert "deepseek_sparse_attention" in labels["full"]  # DeepSeek-V3.2 DSA
    assert "" in labels["full"] and "sliding_attention" in labels["sliding"]


def test_diffusor_vocab_loads():
    assert "num_layers" in ec.load_diffusion_aliases()
    typing = ec.load_diffusion_typing()
    assert typing["stages"] and typing["block_ids"] and typing["part_kinds"] and typing["dit_class_markers"]
    assert ec.load_diffusion_text_encoders().get("CLIPTextModel") == "CLIP"


def test_transformer_stage_taxonomy_is_exhaustive():
    # A representative slice across families must be blessed.
    for stage in (
        "token_embedding", "lm_head", "mtp_head", "norm", "attention", "residual",
        "feed_forward", "moe", "router", "expert", "q_proj", "k_proj", "v_proj",
        "o_proj", "rope", "qk_norm", "kv_cache", "q_lora", "kv_lora",
        "gate_proj", "up_proj", "down_proj", "activation", "ssm", "recurrent",
        "rwkv", "linear_attention", "cross_attention", "per_layer_embedding",
        "vision_encoder", "patch_embedding", "multimodal_fusion",
    ):
        assert stage in TRANSFORMER_STAGES, stage


def test_diffusion_stage_taxonomy_is_exhaustive():
    for stage in (
        "noise_input", "timestep", "guidance", "prompt", "text_encoder",
        "controlnet", "ip_adapter", "vae_encode", "denoiser", "scheduler",
        "vae_decode", "image_output", "patchify", "unpatchify", "attention",
        "cross_attention", "resnet", "upsample", "decoder_block", "conv_out",
    ):
        assert stage in DIFFUSION_STAGES, stage


def test_diffusion_part_kind_taxonomy_is_exhaustive():
    for kind in (
        "conv_in", "conv_out", "down_stage", "mid_stage", "up_stage",
        "latent_stage", "image_stage", "output_head", "resnet_cell",
        "sample_step",
    ):
        assert kind in DIFFUSION_PART_KINDS, kind


def test_loaders_work_without_pyyaml(monkeypatch):
    """The built-in flow/block reader must parse every shipped file (no hard dep)."""
    real_import = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    assert ec.load_aliases()["num_hidden_layers"]
    assert len(ec.load_transformer_typing()["stages"]) == len(TRANSFORMER_STAGES)
    assert len(ec.load_diffusion_typing()["stages"]) == len(DIFFUSION_STAGES)
    assert ec.load_diffusion_text_encoders()["T5EncoderModel"] == "T5"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
