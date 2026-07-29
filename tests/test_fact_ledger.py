"""U2 P0 — the FactLedger foundation (SURGICAL_PLAN_EVIDENCE.md).

Pure-bookkeeping gate: the ledger exists, carries the B5 asserted tags with
owner paths, serializes into ir.extras, and changes NOTHING about what is
drawn. The census target (zero asserted at zero evidence) is enforced by the
later P4 net; here we only prove the accounting rails.
"""
from model_unfolder.evidence.context import (
    FACT_STATUSES,
    FactLedger,
    ParseContext,
)
from model_unfolder.parser import config_to_ir


LLAMA_MINIMAL = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 128,
}


def test_ledger_records_and_serializes():
    ledger = FactLedger()
    ledger.record("layers[0].attention", "mask", "causal", "asserted")
    ledger.record("layers[0].ffn", "gated", True, "code_proven",
                  source="decoder_ffn_mechanism_for_path")
    assert ledger.asserted() == ("layers[0].attention.mask",)
    d = ledger.to_dict()
    assert d["layers[0].ffn.gated"]["status"] == "code_proven"
    assert d["layers[0].ffn.gated"]["source"] == \
        "decoder_ffn_mechanism_for_path"
    # unknown statuses are a programming error, loudly
    import pytest
    with pytest.raises(ValueError):
        ledger.record("x", "y", 1, "vibes")


def test_parse_folds_asserted_tags_into_extras():
    ir = config_to_ir(LLAMA_MINIMAL)
    prov = ir.extras.get("fact_provenance")
    assert prov is not None
    # every entry carries a valid status. P1 default-kill: mask is no longer
    # an asserted dataclass default. U2 P2d strengthening: llama's INSTALLED
    # source proves causality (unconditional create_causal_mask), which
    # outranks the config-decoderness declaration.
    assert all(rec["status"] in FACT_STATUSES for rec in prov.values())
    mask_rec = prov["decoder.attention.mask"]
    assert mask_rec["status"] == "code_proven"
    assert mask_rec["value"] == "causal"
    assert mask_rec.get("source") == "attention_causality_from_files"
    asserted = [k for k, rec in prov.items() if rec["status"] == "asserted"]
    assert not any(k.endswith(".mask") for k in asserted)
    # H3 Phase B activation (§11 step 4): the geometry/embedding family is
    # migrated to consume(), so a real parse now publishes a NON-empty consumed
    # census (never the misleading empty [] the earlier gate guarded against).
    assert "config_consumed" in ir.extras
    assert "hidden_size" in ir.extras["config_consumed"]


def test_ledger_is_call_local():
    ctx1 = ParseContext.build(LLAMA_MINIMAL, source="local")
    config_to_ir(LLAMA_MINIMAL, parse_context=ctx1)
    ctx2 = ParseContext.build(LLAMA_MINIMAL, source="local")
    assert ctx2.facts.records == {}  # a fresh context starts clean


def test_render_event_carries_facts_projected_default():
    from model_unfolder.renderers.html.render_context import RenderEvent
    ev = RenderEvent(view="v", block_path=(), component="root", variant="",
                     source_owner="", source_file="", source_line=None,
                     drawn_ops=frozenset(), node_ids=frozenset())
    assert ev.facts_projected == frozenset()

# ---------------------------------------------------------------------------
# U2 P1 — the default-kill, per fact family.
#
# ZERO_EVIDENCE is the census appendix's D-quadrant witness: numbers-only
# (no model_type, no architectures, no strings/bools) so NO channel — config,
# code, or class default — can back any structural fact.
# ---------------------------------------------------------------------------

ZERO_EVIDENCE = {
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
}


def _prov(ir):
    return ir.extras["fact_provenance"]


def test_zero_evidence_parse_has_typed_unknowns_not_defaults():
    """The P1 acceptance shape: at zero evidence every P1 family is a typed
    unknown; no asserted mechanism convention survives."""
    ir = config_to_ir(ZERO_EVIDENCE)
    layer = ir.layers[0]
    assert layer.attention.mask == "unknown"        # not "causal"
    assert layer.attention.bias is None             # not False
    assert layer.ffn.gated is None                  # not rmsnorm-derived True
    assert layer.ffn.activation is None             # not "silu"
    assert layer.norm_kind == "unknown"             # not "rmsnorm"
    assert layer.norm_placement == "unknown"        # not "pre"
    assert ir.tie_word_embeddings is None           # not False
    allowed_asserted_facts = set()
    for key, rec in _prov(ir).items():
        if rec["status"] == "asserted":
            assert key.rsplit(".", 1)[1] in allowed_asserted_facts, key
    # the banner states the unresolved families once, honestly
    assert any(w.startswith("Unresolved code-defined facts") for w in ir.warnings)


def test_evidence_backed_values_are_unchanged_and_status_real():
    """Control: an evidence-rich parse keeps every concrete value and records
    REAL statuses (this is where P0's ledger got its statuses)."""
    ir = config_to_ir(dict(LLAMA_MINIMAL, hidden_act="silu", rms_norm_eps=1e-5,
                           tie_word_embeddings=False))
    layer = ir.layers[0]
    assert layer.attention.mask == "causal"
    assert layer.ffn.gated is True
    assert layer.ffn.activation == "silu"
    assert layer.norm_kind == "rmsnorm"
    prov = _prov(ir)
    # U2 P2c strengthening: llama's installed source proves the
    # ACT2FN[config.hidden_act] dispatch, so a config-supplied activation is
    # CODE-AND-CONFIG (code proves the mechanism + names the field, config
    # supplies which) — stronger than a bare declaration.
    assert prov["decoder.ffn.activation"]["status"] == "code_and_config"
    assert prov["decoder.ffn.activation"]["source"] == \
        "decoder_ffn_mechanism_for_path:hidden_act"
    assert prov["model.tie_word_embeddings"]["status"] == "config_declared"
    # U2 P2d strengthening: installed llama source proves the causal mask.
    assert prov["decoder.attention.mask"]["status"] == "code_proven"
    # llama's installed source proves these two
    assert prov["decoder.ffn.gated"]["status"] == "code_proven"
    assert prov["decoder.layer.norm_placement"]["status"] == "code_proven"


def test_gated_heuristic_abstains_instead_of_reading_norm_kind():
    """The census cascade is dead: rmsnorm + non-gate activation + no code
    verdict must yield None (undeclared FFN), never gated=True.

    U2 P2c retirement (STRICT rule): the silu/swish/tanh-GELU family tier is
    GONE — plain elementwise spellings are used by dense and gated FFNs
    alike, so they were never proof.  The MoE expert-hop in
    decoder_ffn_mechanism_for_path code-proves the fixtures that tier was
    protecting (deepseek-v3 / glm-4-5 / gpt-oss), corpus-audited at zero
    derived reliance before retiring."""
    from model_unfolder.adapters.transformer.parser import _is_gated
    assert _is_gated("relu", "rmsnorm", None) is None      # was True (cascade)
    assert _is_gated("gelu", "layernorm", None) is None    # was False (cascade)
    assert _is_gated("silu", None, None) is None           # STRICT: not gate proof
    assert _is_gated("gelu_pytorch_tanh", None, None) is None  # STRICT
    assert _is_gated("swiglu", None, None) is True         # explicit GLU channel
    assert _is_gated("gelu", "rmsnorm", False) is False    # code always wins
    assert _is_gated(None, "rmsnorm", None) is None        # zero evidence


def test_direct_ffn_gate_declarations_cannot_replace_source_binding():
    """A gate declaration is an operand, not proof that this exact FFN uses it."""
    dense = config_to_ir(dict(
        ZERO_EVIDENCE, hidden_act="relu", is_gated_act=False,
        feed_forward_proj="relu",
    ))
    assert dense.layers[0].ffn.gated is None
    assert "decoder.ffn.gated" not in _prov(dense)

    gated = config_to_ir(dict(
        ZERO_EVIDENCE, hidden_act="gelu", feed_forward_proj="gated-gelu",
    ))
    assert gated.layers[0].ffn.gated is None

    # A bare silu proves NOTHING (STRICT rule, tier retired): at zero code /
    # config / class-default evidence the FFN stays a typed unknown — the
    # counterexample that prevents replacing one heuristic (RMSNorm) with
    # another (silu implies gated).
    ambiguous = config_to_ir(dict(ZERO_EVIDENCE, hidden_act="silu"))
    assert ambiguous.layers[0].ffn.gated is None
    assert "decoder.ffn.gated" not in _prov(ambiguous)


def test_tie_class_default_tier_fixes_absent_flag():
    """gpt2-shaped dict WITHOUT the tie flag: the installed config-class
    default (True) decides, recorded as class_default — the live wrong-value
    fix (was silently False)."""
    cfg = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
           "n_embd": 64, "n_layer": 2, "n_head": 4, "vocab_size": 128}
    ir = config_to_ir(cfg)
    assert ir.tie_word_embeddings is True
    rec = _prov(ir)["model.tie_word_embeddings"]
    assert rec["status"] == "class_default"


def test_mask_counterexample_classes():
    """Same numbers, different declarations — the mask follows the config's
    decoder-ness channel, never a dataclass default."""
    base = dict(ZERO_EVIDENCE)
    # (1) causal-LM architecture suffix → causal, config_declared
    ir = config_to_ir(dict(base, architectures=["FrobnitzForCausalLM"]))
    assert ir.layers[0].attention.mask == "causal"
    # (2) is_decoder flag → causal
    ir = config_to_ir(dict(base, is_decoder=True))
    assert ir.layers[0].attention.mask == "causal"
    # (3) enc-dec generation wrapper → NOT decoder-declared → unknown
    ir = config_to_ir(dict(base, architectures=["FrobnitzForConditionalGeneration"],
                           is_encoder_decoder=True))
    assert ir.layers[0].attention.mask == "unknown"
    # (4) decoder-only generation wrapper (VLM shape) → causal
    ir = config_to_ir(dict(base, architectures=["FrobnitzForConditionalGeneration"]))
    assert ir.layers[0].attention.mask == "causal"


def test_mask_code_channel_bert_bidirectional():
    """U2 P2d witness shape: a BERT config (installed source, is_decoder
    absent → PretrainedConfig base default False) yields a CODE-PROVEN
    bidirectional mask AND the label says so — the census's headline fixed
    at both the fact and the label tier."""
    from model_unfolder.labels import mask_short
    bert = {"architectures": ["BertForMaskedLM"], "model_type": "bert",
            "hidden_size": 768, "intermediate_size": 3072,
            "num_attention_heads": 12, "num_hidden_layers": 12,
            "vocab_size": 30522, "hidden_act": "gelu",
            "layer_norm_eps": 1e-12, "max_position_embeddings": 512}
    ir = config_to_ir(bert)
    assert ir.layers[0].attention.mask == "bidirectional"
    rec = _prov(ir)["decoder.attention.mask"]
    assert rec["status"] == "code_proven"
    assert rec["source"] == "attention_causality_from_files"
    assert mask_short({"mask": "bidirectional"}) == "bidirectional"

    # SAME source, decoder checkpoint: the config value flips the verdict
    ir2 = config_to_ir(dict(bert, is_decoder=True,
                            architectures=["BertLMHeadModel"]))
    assert ir2.layers[0].attention.mask == "causal"


def test_mask_code_causal_discarded_on_flat_encdec(tmp_path):
    """The Whisper shape: a flat enc-dec config draws the ENCODER half while
    the file also contains the (undrawn) decoder's causal machinery — a
    causal-only verdict must be discarded, the mask stays a typed unknown."""
    from model_unfolder.evidence.models import SourceBundle

    src = tmp_path / "modeling_fake.py"
    src.write_text(
        "class FakeModel:\n"
        "    def forward(self, x):\n"
        "        mask = create_causal_mask(config=self.config)\n"
        "        return mask\n"
    )
    ctx = ParseContext(source_bundle=SourceBundle(source="local",
                                                  files=(str(src),)))
    ir = config_to_ir(dict(ZERO_EVIDENCE, is_encoder_decoder=True),
                      parse_context=ctx)
    assert ir.layers[0].attention.mask == "unknown"
    # the same verdict on a plain (non enc-dec) config IS the stack's fact
    ctx2 = ParseContext(source_bundle=SourceBundle(source="local",
                                                   files=(str(src),)))
    ir2 = config_to_ir(dict(ZERO_EVIDENCE), parse_context=ctx2)
    assert ir2.layers[0].attention.mask == "causal"
    assert _prov(ir2)["decoder.attention.mask"]["status"] == "code_proven"


def test_t5_declarations_stay_unknown_until_the_exact_encoder_ffn_is_bound():
    """Class/checkpoint defaults cannot by themselves author T5's FFN graph."""
    t5_shape = {"model_type": "t5",
                "architectures": ["T5ForConditionalGeneration"],
                "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12,
                "vocab_size": 32128, "is_encoder_decoder": True,
                "layer_norm_epsilon": 1e-6}
    ir = config_to_ir(t5_shape)
    assert ir.layers[0].ffn.gated is None
    assert ir.layers[0].ffn.activation is None
    prov = _prov(ir)
    assert "decoder.ffn.gated" not in prov
    assert prov["decoder.ffn.activation"]["status"] == "ambiguous"

    flan = config_to_ir(dict(t5_shape, feed_forward_proj="gated-gelu"))
    assert flan.layers[0].ffn.gated is None
    assert flan.layers[0].ffn.activation is None
    prov2 = _prov(flan)
    assert "decoder.ffn.gated" not in prov2
    assert prov2["decoder.ffn.activation"]["value"] is None


def test_position_declaration_cannot_author_qk_rotation():
    """Theta/scaling values remain data; without source application evidence
    they cannot create a RoPE operation."""
    ir = config_to_ir(dict(ZERO_EVIDENCE, rope_theta=500000.0))
    a = ir.layers[0].attention
    assert a.rope is None and a.position_kind == "unknown"
    assert a.position_application == "none"
    assert a.position_declared is False
    assert "decoder.attention.position" not in _prov(ir)
    assert any("positional scheme remains unknown" in w for w in ir.warnings)

    # the modern nested spelling (rope_scaling/rope_parameters dict) counts
    ir2 = config_to_ir(dict(ZERO_EVIDENCE, rope_scaling={"rope_type": "linear",
                                                         "factor": 2.0}))
    a2 = ir2.layers[0].attention
    assert a2.rope is None and a2.position_declared is False
    assert "decoder.attention.position" not in _prov(ir2)

    # NEGATIVE: no declaration → typed unknown + the honest banner stays
    ir3 = config_to_ir(ZERO_EVIDENCE)
    a3 = ir3.layers[0].attention
    assert a3.rope is None and a3.position_kind == "unknown"
    assert a3.position_declared is False
    assert any("positional scheme remains unknown" in w for w in ir3.warnings)

    # CODE-PROVEN control (llama): the declared-tier marker never rides a
    # proven parse — serialized output stays byte-identical.
    ir4 = config_to_ir(dict(LLAMA_MINIMAL, rope_theta=10000.0))
    a4 = ir4.layers[0].attention
    assert a4.position_declared is False
    from model_unfolder.ir import _attention_to_dict
    assert "position_declared" not in _attention_to_dict(a4)
    assert "rope_theta_declared" not in _attention_to_dict(a4)


def test_position_declared_chip_text():
    """The θ chip states the tier on the attention card."""
    from model_unfolder.labels import attention_summary
    _, facts = attention_summary({
        "kind": "mha", "num_heads": 32, "num_kv_heads": 32, "head_dim": 128,
        "mask": "unknown", "rope": True, "position_kind": "rope",
        "position_application": "qk_rotation", "position_declared": True,
        "rope_theta_declared": 640000.0,
    })
    assert any("θ=640,000" in f and "config-declared" in f for f in facts)


def test_bias_declaration_without_constructor_binding_stays_unknown():
    ir = config_to_ir(dict(ZERO_EVIDENCE, attention_bias=True))
    assert ir.layers[0].attention.bias is None
    assert _prov(ir)["decoder.attention.bias"]["status"] == "oracle_missing"
    ir = config_to_ir(ZERO_EVIDENCE)
    assert ir.layers[0].attention.bias is None


def test_bias_alias_spellings_do_not_create_projection_bias():
    """Aliases normalize syntax; without a bound Linear constructor they do
    not prove the mechanism, including a declared False."""
    ir = config_to_ir(dict(ZERO_EVIDENCE, bias=True))
    assert ir.layers[0].attention.bias is None
    assert _prov(ir)["decoder.attention.bias"]["status"] == "oracle_missing"
    ir = config_to_ir(dict(ZERO_EVIDENCE, qkv_bias=False))
    assert ir.layers[0].attention.bias is None
    assert _prov(ir)["decoder.attention.bias"]["status"] == "oracle_missing"


def test_tie_code_channel_order(tmp_path):
    """U2 P2b channel order: config_declared → code_proven → class_default →
    unknown.  An unconditional manual tie in source proves True when the flag
    is absent; a declared flag still outranks the code idiom."""
    from model_unfolder.evidence.models import SourceBundle

    src = tmp_path / "modeling_fake.py"
    src.write_text(
        "from torch import nn\n"
        "\n"
        "class FakeBlock:\n"
        "    def forward(self, hidden):\n"
        "        return hidden\n"
        "\n"
        "class FakeModel:\n"
        "    def __init__(self):\n"
        "        self.embed_tokens = nn.Embedding(16, 4)\n"
        "        self.layers = nn.ModuleList(\n"
        "            [FakeBlock() for _ in range(2)])\n"
        "\n"
        "    def forward(self, token_ids):\n"
        "        hidden = self.embed_tokens(token_ids)\n"
        "        for layer in self.layers:\n"
        "            hidden = layer(hidden)\n"
        "        return hidden\n"
        "\n"
        "class FakeForCausalLM:\n"
        "    base_model_prefix = 'transformer'\n"
        "\n"
        "    def __init__(self):\n"
        "        self.transformer = FakeModel()\n"
        "        self.lm_head = nn.Linear(4, 16, bias=False)\n"
        "        self.lm_head.weight = self.transformer.embed_tokens.weight\n"
        "\n"
        "    def forward(self, token_ids):\n"
        "        return self.lm_head(self.transformer(token_ids))\n"
    )
    bundle = SourceBundle(
        source="local",
        files=(str(src),),
        component_files={"root": (str(src),)},
        component_architectures={"root": "FakeForCausalLM"},
    )
    ctx = ParseContext(source_bundle=bundle)
    # flag absent + no model_type (no class default) → the code idiom decides
    ir = config_to_ir(dict(ZERO_EVIDENCE), parse_context=ctx)
    assert ir.tie_word_embeddings is True
    rec = _prov(ir)["model.tie_word_embeddings"]
    assert rec["status"] == "code_proven"
    assert rec["source"] == "manual_weight_tying_for_path"

    # a DECLARED flag outranks the code idiom (spec order: config first)
    ctx2 = ParseContext(source_bundle=bundle)
    ir2 = config_to_ir(dict(ZERO_EVIDENCE, tie_word_embeddings=False),
                       parse_context=ctx2)
    assert ir2.tie_word_embeddings is False
    assert _prov(ir2)["model.tie_word_embeddings"]["status"] == "config_declared"


def test_param_estimate_annotates_unknowns_never_silently_branches():
    from model_unfolder.params import estimate_params
    ir = config_to_ir(ZERO_EVIDENCE)
    est = estimate_params(ir)
    notes = est.get("assumptions") or []
    assert any("tying unknown" in n for n in notes)
    assert any("FFN structure unknown" in n for n in notes)
    # evidence-rich control stays annotation-free (byte-stable)
    est2 = estimate_params(config_to_ir(dict(LLAMA_MINIMAL, tie_word_embeddings=False)))
    assert "assumptions" not in est2


def test_unknown_tying_survives_every_projection():
    """A typed tying unknown must not turn into an untied claim after the IR.

    This pins the model-block prose, card metadata, and expanded JSON — the
    three downstream bool() coercions that originally recreated the bug.
    """
    from model_unfolder.diagram import Diagram
    from model_unfolder.renderers.html.metadata import _make_info

    ir = config_to_ir(ZERO_EVIDENCE)
    raw = Diagram(ir).to_ir()
    blocks = (raw["extras"]["render"]["model_blocks"])
    embed = next(block for block in blocks if block["id"] == "embed")
    head = next(block for block in blocks if block["id"] == "lm_head")
    assert "unresolved" in embed["description"]
    assert "unresolved" in head["description"]

    expanded = Diagram(ir).to_json()
    assert "tied_to_token_embedding" in expanded["io"]["lm_head"]
    assert expanded["io"]["lm_head"]["tied_to_token_embedding"] is None

    metadata = _make_info(raw)["meta"]
    # Metadata values are (title, description, facts) tuples keyed by block id.
    assert "unresolved" in metadata["embed"][1]
    assert "unresolved" in metadata["lm_head"][1]


def test_unknown_placement_draws_pale_wiring_not_pre_norm():
    """B2: placement-unknown draws the declared sublayers plus ONE pale
    code-defined-wiring block — no fabricated rms1/rms2 pre-norm cells."""
    ir = config_to_ir(ZERO_EVIDENCE)
    ids = [b["id"] for b in ir.layers[0].blocks]
    assert "wiring_unresolved" in ids
    assert "rms1" not in ids and "rms2" not in ids
    wiring = next(b for b in ir.layers[0].blocks if b["id"] == "wiring_unresolved")
    assert wiring.get("resolved") is False
