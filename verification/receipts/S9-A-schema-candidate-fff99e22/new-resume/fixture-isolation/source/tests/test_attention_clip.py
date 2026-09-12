"""U6 exact Q/K/V clipping evidence."""

from test_support.s9_fixtures.attention_clip import _bundle, _read

from pathlib import Path







def test_fused_projection_clamp_reaching_attention_is_resolved(tmp_path):
    result = _read(_bundle(tmp_path, "qkv = qkv.clamp(max=self.limit)"))
    assert result.status == "resolved", result.failures
    assert result.value.config_path == ("clip_qkv",)
    assert result.value.clamp_call.callee.name == "clamp"


def test_same_config_field_without_clamp_cannot_author_clipping(tmp_path):
    result = _read(_bundle(tmp_path, "qkv = qkv"))
    assert result.status == "failed"


def test_unused_clamp_result_cannot_certify_the_live_qkv_lane(tmp_path):
    result = _read(_bundle(tmp_path, "qkv.clamp(max=self.limit)"))
    assert result.status == "failed"


def test_clamp_operand_path_is_exact_not_a_familiar_spelling(tmp_path):
    bundle = _bundle(tmp_path, "qkv = qkv.clamp(max=self.limit)")
    path = Path(bundle.files[0])
    path.write_text(
        path.read_text().replace(
            "self.limit = config.clip_qkv",
            "self.limit = config.unfamiliar_bound"),
        encoding="utf-8",
    )
    result = _read(bundle)
    assert result.status == "resolved", result.failures
    assert result.value.config_path == ("unfamiliar_bound",)
