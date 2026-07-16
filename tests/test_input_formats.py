"""Input-format normalization must remain syntax-only and identity-free."""
from __future__ import annotations

import json

import pytest

from model_unfolder.errors import ConfigParseError
from model_unfolder.everchanging import load_aliases, load_input_format_aliases
from model_unfolder.input_formats import normalize_params_json
from model_unfolder.parser import config_to_ir


def _params(*, with_vision: bool = True) -> dict:
    data = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 14336,
        "norm_eps": 1e-5,
        "head_dim": 128,
        "rope_theta": 1_000_000,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
    }
    if with_vision:
        data["vision_encoder"] = {
            "num_channels": 3,
            "image_size": 1024,
            "patch_size": 16,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "rope_theta": 10000,
            "image_token_id": 10,
        }
    return data


def test_params_json_normalizes_through_scoped_aliases_without_identity():
    cfg = normalize_params_json(_params())
    assert cfg is not None
    assert cfg["hidden_size"] == 4096
    assert cfg["num_hidden_layers"] == 32
    assert cfg["num_key_value_heads"] == 8
    assert cfg["intermediate_size"] == 14336
    assert cfg["vision_config"]["hidden_size"] == 1024
    assert cfg["vision_config"]["image_token_id"] == 10
    assert "model_type" not in cfg
    assert "architectures" not in cfg


def test_params_aliases_are_scoped_and_cannot_leak_globally():
    global_aliases = load_aliases()
    scoped = load_input_format_aliases("layered_transformer_params")
    assert "dim" not in global_aliases["hidden_size"]
    assert scoped["text.hidden_size"] == ["dim"]
    assert not any(key.startswith("input_format.") for key in global_aliases)


def test_params_format_requires_a_transformer_shape_and_parses_without_identity():
    too_weak = {"dim": 4096, "n_layers": 32}
    assert normalize_params_json(too_weak) is None

    cfg = normalize_params_json(_params(with_vision=False))
    assert cfg is not None and "model_type" not in cfg
    ir = config_to_ir(cfg)
    assert len(ir.layers) == 32
    assert ir.hidden_size == 4096
    assert ir.layers[0].attention.num_heads == 32


def test_params_json_rejects_conflicting_format_local_aliases():
    params = _params(with_vision=False)
    params["hidden_size"] = 2048
    with pytest.raises(ConfigParseError, match="Conflicting input-format aliases"):
        normalize_params_json(params)


def test_raw_loader_uses_params_dialect_without_model_specific_stamp(
        monkeypatch, tmp_path):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(_params()), encoding="utf-8")

    import huggingface_hub
    import model_unfolder.parser as parser

    def fake_download(**kwargs):
        if kwargs["filename"] == "config.json":
            raise FileNotFoundError("no config.json")
        assert kwargs["filename"] == "params.json"
        return str(params_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cfg = parser._load_raw_config_json("org/checkpoint", None)
    assert cfg["hidden_size"] == 4096
    assert cfg["_repo_id"] == "org/checkpoint"
    assert cfg["_name_or_path"] == "org/checkpoint"
    assert "model_type" not in cfg
