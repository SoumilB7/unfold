"""F04 controls: G2 "executed = loaded" (INTENT.md) and the authority cache bound to the complete load hypothesis.

Each real control is a tiny randomly-initialised library model saved as a local checkpoint (no downloads). It runs the
worker's path: libload.Authority's library load (the load-config snapshot + the executed build's attention
implementation, as worker_v2 passes them), the executed build exactly as worker_v2._fresh() (zero storage, float32,
eager), an optional named MUTATION of the executed model (the negative controls), and the G2 comparison. Pure controls
exercise the comparator and the cache directly.

    python3 selftest_g2.py [--harness DIR] [name-substring ...]     # exit 0 = every control as expected
--harness runs the controls against another harness tree (e.g. HEAD-before-F04: there the G2 decision is the old
names/shapes comparison inside g1_assess, and the cache is keyed by class name) -- the negative controls must FAIL there.

want syntax: "OK" (no G2 failure, no G2 unknown) | "FAIL:<key>" (a G2 failure with that key) | "UNKNOWN:<text>" (a G2
unknown whose text contains it, no failure); optional "+named:<key>" (the G2 record names it, e.g. dequantization).
"""
import os, sys, json, copy, tempfile, shutil, collections, warnings
ARGS = sys.argv[1:]
HARNESS = os.path.dirname(os.path.abspath(__file__))
if "--harness" in ARGS:
    i = ARGS.index("--harness"); HARNESS = os.path.abspath(ARGS[i + 1]); del ARGS[i:i + 2]
SEL = ARGS
sys.path.insert(0, HARNESS)
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch, transformers
from safetensors.torch import save_file
import libload, lowcost, common
NEW = hasattr(libload, "g2_compare")                 # HEAD-before-F04 has no comparator: its G2 is g1_assess's names/shapes


def save(d, name, model, edit=None, cfg_edit=None):
    p = os.path.join(d, name); os.makedirs(p, exist_ok=True)
    model.config.save_pretrained(p)
    sd = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    if edit: sd = edit(sd)
    save_file(sd, os.path.join(p, "model.safetensors"))
    if cfg_edit:
        j = json.load(open(os.path.join(p, "config.json"))); cfg_edit(j); json.dump(j, open(os.path.join(p, "config.json"), "w"))
    return p


def fresh(cls, cfg):
    """worker_v2._fresh()"""
    with lowcost.zero_storage():
        try:
            mm = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32); attn = "eager"
        except (TypeError, ValueError):
            mm = cls._from_config(cfg, dtype=torch.float32); attn = None
    return lowcost.normalise_learned(lowcost.uniform_float32(mm.eval())), attn


def g2_of(path, cls, mutate=None, authority_kw=None):
    """The worker's G2 decision on one local checkpoint -> (outcome, detail)."""
    cfg = transformers.AutoConfig.from_pretrained(path)
    cfg_load = copy.deepcopy(cfg)                     # worker_v2 CFG_LOAD: snapshot before any build
    A = libload.Authority(path); A.mirror()
    aux = set(common.ckpt_split({k: v for h in A.headers.values() for k, v in h.items()})[1])
    try:
        m, attn = fresh(cls, cfg)
        if NEW:
            kw = {"attn_implementation": attn} if attn else {}
            if authority_kw is not None: kw = authority_kw
            LIB = A.report(cls, aux, config=cfg_load, **kw)
        else:
            LIB = A.report(cls, aux, config=cfg_load)
        if "error" in LIB:
            return "ERROR", {"report": LIB["error"]}
        if mutate: mutate(m)
        if NEW:
            G = libload.g2_compare(LIB.get("identity"), libload.model_identity(m), {"attn_implementation": attn})
            if G["fail"]:
                return "FAIL", {"fail": G["fail"], "named": G["named"]}
            if G["unknowns"]:
                return "UNKNOWN", {"unknowns": [u["what"] for u in G["unknowns"]], "named": G["named"]}
            return "OK", {"named": G["named"]}
        sd = m.state_dict(keep_vars=True)
        G = libload.g1_assess(LIB.get("provenance") or {}, set(sd), {k: list(t.shape) for k, t in sd.items()}, {})
        u2 = [u["what"] for u in G["unknowns"] if u["predicate"] == "G2"]
        return ("UNKNOWN", {"unknowns": u2}) if u2 else ("OK", {})
    finally:
        A.close()


LLAMA = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, max_position_embeddings=32)
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fp8_block(sd, scale_name="weight_scale_inv"):
    out = dict(sd)
    for k in list(sd):
        if k.endswith(".weight") and any(f".{p_}." in k for p_ in PROJ):
            o_, i_ = sd[k].shape
            out[k] = sd[k].to(torch.float8_e4m3fn); out[k[:-len("weight")] + scale_name] = torch.ones((o_ + 7) // 8, (i_ + 7) // 8)
    return out


def untie(m):
    m.lm_head.weight = torch.nn.Parameter(m.lm_head.weight.detach().clone())


def tie(m):
    m.lm_head.weight = m.model.embed_tokens.weight


class _Sub(torch.nn.Module):
    pass


def swap_attention_class(m):
    a = m.model.layers[0].self_attn
    a.__class__ = type("LlamaAttentionVariant", (type(a),), {})     # same parameters, another module class


def swap_activation(m):
    m.model.layers[0].mlp.act_fn = torch.nn.GELU()                  # parameter-free: names / shapes / state equal


def extra_module(m):
    m.model.layers[0].extra = torch.nn.Identity()


def share_module(m):
    m.model.layers[1].mlp = m.model.layers[0].mlp                   # one module object under two names (and its params)


def int_buffer_dtype(m):
    bn = m.resnet.embedder.embedder.normalization
    bn.num_batches_tracked = bn.num_batches_tracked.to(torch.int32)


def persistence_flip(m):
    bn = m.resnet.embedder.embedder.normalization
    v = bn.num_batches_tracked
    del bn._buffers["num_batches_tracked"]
    bn.register_buffer("num_batches_tracked", v, persistent=False)


def nonpersistent_change(m):
    r = m.model.rotary_emb
    r.register_buffer("inv_freq", r.inv_freq[:2].clone(), persistent=False)


def build(d):
    """name -> (dir, class, mutate, authority_kw override, want)"""
    torch.manual_seed(0)
    C = {}
    L = transformers.LlamaForCausalLM
    m = L(transformers.LlamaConfig(**LLAMA, tie_word_embeddings=False))
    mt = L(transformers.LlamaConfig(**LLAMA, tie_word_embeddings=True))
    p0 = save(d, "V1", m)
    pt = save(d, "V2", mt, lambda sd: {k: v for k, v in sd.items() if k != "lm_head.weight"})
    # ---- VALID: normal library behaviour stays certified
    C["V1 clean llama (float32 checkpoint)"] = (p0, L, None, None, "OK")
    C["V2 library tie (tie_word_embeddings, one tensor shipped)"] = (pt, L, None, None, "OK")
    mb = L(transformers.LlamaConfig(**LLAMA, tie_word_embeddings=False)).to(torch.bfloat16); mb.config.dtype = torch.bfloat16
    C["V3 bf16 checkpoint: float32 build named, not a failure"] = (save(d, "V3", mb), L, None, None, "OK+named:float32_conversion")
    m64 = L(transformers.LlamaConfig(**LLAMA, tie_word_embeddings=False)).to(torch.float64); m64.config.dtype = torch.float64
    C["N12 float64 checkpoint executed as float32 (values rounded): a G2 unknown (F04-2)"] = (save(d, "N12", m64), L, None, None, "UNKNOWN:float64")
    # cross-exam B1: layouts the HARNESS dequantizes are one unknown, never a structural failure (F04-3)
    def bnb8(sd):
        out = dict(sd)
        for k in list(sd):
            if k.endswith(".weight") and any(f".{p_}." in k for p_ in PROJ):
                w = sd[k]; out[k] = w.to(torch.int8); out[k[:-len("weight")] + "SCB"] = torch.ones(w.shape[0])
                out[k[:-len("weight")] + "weight_format"] = torch.tensor(0, dtype=torch.uint8)
        return out
    try:
        import bitsandbytes  # noqa: F401
        C["N13 bnb 8-bit checkpoint: the harness runs its codes as floats -> unknown, not PARTIAL"] = (
            save(d, "N13", m, bnb8, cfg_edit=lambda j: j.__setitem__("quantization_config", {"quant_method": "bitsandbytes", "load_in_8bit": True})),
            L, None, None, "UNKNOWN:quantized layout!named:dequantized_at_load_by_library")
    except ImportError:
        C["N13 bnb 8-bit checkpoint (NOT RUN: bitsandbytes is not installed)"] = (None, None, None, None, "NOT-RUN")
    def bitnet(sd):
        out = dict(sd)
        for k in list(sd):
            if k.endswith(".weight") and any(f".{p_}." in k for p_ in PROJ):
                o_, i_ = sd[k].shape
                out[k] = torch.zeros((o_ + 3) // 4, i_, dtype=torch.uint8); out[k[:-len("weight")] + "weight_scale"] = torch.ones(1)
        return out
    C["N14 BitNet checkpoint: packed codes run as floats -> unknown, not PARTIAL"] = (
        save(d, "N14", m, bitnet, cfg_edit=lambda j: j.__setitem__("quantization_config", {"quant_method": "bitnet", "linear_class": "bitlinear", "quantization_mode": "offline"})),
        L, None, None, "UNKNOWN:quantized layout")
    # cross-exam S2: config / module-setting differences are visible (unknown)
    C["N15 loaded under the library default attention, executed eager (same classes): config differs -> unknown"] = (p0, L, None, {}, "UNKNOWN:config fields differ")
    def _eps(mm):
        mm.config.rms_norm_eps = 0.5
        for l in mm.model.layers: l.input_layernorm.variance_epsilon = 0.5
    C["N16 executed norm eps differs (config + module setting) -> unknown"] = (p0, L, _eps, None, "UNKNOWN:config fields differ")
    def _sc(mm):
        mm.model.layers[0].self_attn.scaling = 123.0; mm.model.layers[0].self_attn.is_causal = False
    C["N17 an attention module's scaling / is_causal setting changed -> unknown"] = (p0, L, _sc, None, "UNKNOWN:module settings differ")
    C["V4 fine-grained FP8 dequantized by the library on CPU: named"] = (
        save(d, "V4", m, fp8_block, cfg_edit=lambda j: j.__setitem__("quantization_config", {"quant_method": "fp8", "weight_block_size": [8, 8], "activation_scheme": "dynamic"})),
        L, None, None, "OK+named:dequantized_at_load_by_library")
    u = transformers.UMT5ForConditionalGeneration(transformers.UMT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                                                                          num_heads=2, decoder_start_token_id=0))
    C["V5 UMT5 shared embeddings (library ties encoder/decoder/shared)"] = (save(d, "V5", u), transformers.UMT5ForConditionalGeneration, None, None, "OK")
    dab = transformers.DabDetrConfig(use_timm_backbone=False, use_pretrained_backbone=False,
                                     backbone_config=transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 16, 32], depths=[1, 1, 1, 1], out_features=["stage4"]),
                                     backbone=None, hidden_size=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                                     decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, num_queries=4, num_labels=3)
    C["V6 DAB-DETR module-pattern ties (bbox head shared with the decoder)"] = (
        save(d, "V6", transformers.DabDetrForObjectDetection(dab)), transformers.DabDetrForObjectDetection, None, None, "OK")
    r = transformers.ResNetForImageClassification(transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], num_labels=3))
    pr = save(d, "V7", r)
    C["V7 ResNet integer BN counters"] = (pr, transformers.ResNetForImageClassification, None, None, "OK")
    dl = transformers.DiffLlamaForCausalLM(transformers.DiffLlamaConfig(**LLAMA))
    pdl = save(d, "V8", dl)
    C["V8 DiffLlama: executed eager attention = the authority's load hypothesis"] = (pdl, transformers.DiffLlamaForCausalLM, None, None, "OK")
    enc = transformers.EncodecConfig(target_bandwidths=[1.5, 3.0, 6.0], num_filters=4, codebook_size=16, codebook_dim=8, hidden_size=8,
                                     upsampling_ratios=[2, 2], num_lstm_layers=1, sampling_rate=800)
    C["V9 EnCodec weight-norm parametrizations + codebook buffers"] = (save(d, "V9", transformers.EncodecModel(enc)), transformers.EncodecModel, None, None, "OK")
    clip = transformers.CLIPModel(transformers.CLIPConfig(
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=8),
        projection_dim=8))
    C["V10 CLIP composite (sub-configs)"] = (save(d, "V10", clip), transformers.CLIPModel, None, None, "OK")
    nx = transformers.GPTNeoXForCausalLM(transformers.GPTNeoXConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                         intermediate_size=32, max_position_embeddings=32, tie_word_embeddings=False))
    def neox_masks(sd):
        sd = dict(sd)
        for i in range(2):
            sd[f"gpt_neox.layers.{i}.attention.bias"] = torch.tril(torch.ones(1, 1, 32, 32, dtype=torch.uint8))
            sd[f"gpt_neox.layers.{i}.attention.masked_bias"] = torch.tensor(-1e9)
        return sd
    C["V11 GPT-NeoX with declared-ignored U8 mask tables"] = (save(d, "V11", nx, neox_masks), transformers.GPTNeoXForCausalLM, None, None, "OK")
    # ---- NEGATIVE: equal names and shapes, different model (must not certify)
    C["N1 executed tie broken (lm_head no longer embed_tokens)"] = (pt, L, untie, None, "FAIL:tie_groups_differ")
    C["N2 executed tie added (loaded untied)"] = (p0, L, tie, None, "FAIL:tie_groups_differ")
    C["N3 executed module class differs (same parameters)"] = (p0, L, swap_attention_class, None, "FAIL:module_classes_differ")
    C["N4 parameter-free module differs (activation)"] = (p0, L, swap_activation, None, "FAIL:module_classes_differ")
    C["N5 extra parameter-free module executed"] = (p0, L, extra_module, None, "FAIL:modules_differ")
    C["N6 one module object under two names (shared MLP)"] = (p0, L, share_module, None, "FAIL:module_aliasing_differs")
    C["N7 integer buffer dtype differs (int64 loaded, int32 executed)"] = (pr, transformers.ResNetForImageClassification, int_buffer_dtype, None, "FAIL:buffer_dtypes_differ")
    C["N8 buffer persistence flips"] = (pr, transformers.ResNetForImageClassification, persistence_flip, None, "FAIL:buffer_persistence_differs")
    C["N9 non-persistent buffer changed (rotary table)"] = (p0, L, nonpersistent_change, None, "FAIL:buffer_shapes_differ")
    C["N10 DiffLlama loaded under the library default attention, executed eager (old hypothesis)"] = (
        pdl, transformers.DiffLlamaForCausalLM, None, {}, "FAIL:module_classes_differ")
    C["N11 FP8 scales under a name the library's converter does not collect (codes cast, not dequantized)"] = (
        save(d, "N11", m, lambda sd: fp8_block(sd, "weight_scale_x"),
             cfg_edit=lambda j: j.__setitem__("quantization_config", {"quant_method": "fp8", "weight_block_size": [8, 8], "activation_scheme": "dynamic"})),
        L, None, None, "UNKNOWN:cast to floats")
    return C


def pure_controls():
    """(name, fn -> (ok, detail)) controls of the comparator and the cache that need no checkpoint."""
    out = []
    # P1 the audit's API control: same names / shapes, different alias relation
    def p1():
        if not NEW:
            G = libload.g1_assess({"nonloaded": {}, "state_shapes": {"a": [2, 2], "b": [2, 2]}}, {"a", "b"}, {"a": [2, 2], "b": [2, 2]}, {})
            return bool(G["fail"]) or any(u["predicate"] == "G2" for u in G["unknowns"]), G
        Lid = {"modules": {"(root)": "x.M"}, "params": {"a": [[2, 2], "float32"], "b": [[2, 2], "float32"]}, "buffers": {}, "ties": [["a", "b"]]}
        Eid = dict(Lid, ties=[])
        G = libload.g2_compare(Lid, Eid)
        return "tie_groups_differ" in G["fail"], G["fail"]
    out.append(("P1 same names/shapes, loaded tied vs executed separate (pure)", p1))
    # P2 a loaded layout that keeps quantized parameters, executed as floats: unknown (harness dequantized), named
    def p2():
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M", "l": "bitsandbytes.nn.modules.Linear4bit"}, "params": {"l.weight": [[64, 1], "uint8"]}, "buffers": {}, "ties": [],
               "quantizer": "Bnb4BitHfQuantizer", "quantized_codes": ["l.weight"]}
        Eid = {"modules": {"(root)": "x.M", "l": "torch.nn.modules.linear.Linear"}, "params": {"l.weight": [[8, 16], "float32"]}, "buffers": {}, "ties": []}
        G = libload.g2_compare(Lid, Eid)
        return (not G["fail"]) and any("quantized layout" in u["what"] for u in G["unknowns"]), G
    out.append(("P2 loaded keeps a quantized layout (Linear4bit), executed as a float nn.Linear -> unknown, no failure (pure)", p2))
    def p2b():
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M", "l": "gptqmodel.QuantLinear"}, "params": {"l.bias": [[8], "float16"]},
               "buffers": {"l.qweight": [[2, 8], "int32", True], "l.qzeros": [[1, 1], "int32", True], "l.scales": [[1, 8], "float16", True],
                           "l.g_idx": [[16], "int32", True]}, "ties": [], "quantizer": "GptqHfQuantizer", "quantized_codes": ["l.qweight", "l.qzeros"]}
        Eid = {"modules": {"(root)": "x.M", "l": "torch.nn.modules.linear.Linear"}, "params": {"l.weight": [[8, 16], "float32"], "l.bias": [[8], "float32"]},
               "buffers": {}, "ties": []}
        G = libload.g2_compare(Lid, Eid)
        ok1 = (not G["fail"]) and any("quantized layout" in u["what"] for u in G["unknowns"])
        G2_ = libload.g2_compare(Lid, dict(Eid, params={"l.weight": [[8, 16], "float32"], "l.bias": [[9], "float32"]}))
        return ok1 and "parameter_shapes_differ" in G2_["fail"], {"layout": G, "bias_shape_still_compared": G2_["fail"]}
    out.append(("P2b GPTQ-style codes held in BUFFERS of a QuantLinear, executed as nn.Linear -> unknown; its float bias is STILL compared (pure)", p2b))
    def p2g():                                   # cross-exam r2 S1: quantization elsewhere never excuses an ordinary module
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M", "bn": "torch.nn.BatchNorm1d", "q": "x.QLinear"},
               "params": {"q.weight": [[4, 4], "float8_e4m3fn"]},
               "buffers": {"bn.num_batches_tracked": [[], "int64", True], "bn.running_mean": [[4], "float32", True]},
               "ties": [], "quantizer": "FineGrainedFP8HfQuantizer", "quantized_codes": ["q.weight"]}
        Eid = {"modules": {"(root)": "x.M", "bn": "torch.nn.BatchNorm1d", "q": "torch.nn.modules.linear.Linear"},
               "params": {"q.weight": [[4, 4], "float32"], "bn.weight": [[4], "float32"]},
               "buffers": {"bn.num_batches_tracked": [[], "int32", True], "bn.running_mean": [[4], "float32", True]}, "ties": []}
        G = libload.g2_compare(Lid, Eid)
        return "buffer_dtypes_differ" in G["fail"] and "parameters_differ" in G["fail"], G["fail"]
    out.append(("P2g a quantizer elsewhere in the model never excuses an ordinary module's differences (pure)", p2g))
    def p2h():                                   # B1-r2: where the library loaded from is not a setting
        if not NEW:
            return False, "no comparator"
        a_, b_ = transformers.LlamaConfig(**LLAMA), transformers.LlamaConfig(**LLAMA)
        with torch.device("meta"):
            ma, mb_ = transformers.LlamaForCausalLM(a_), transformers.LlamaForCausalLM(b_)
        ma.name_or_path, mb_.name_or_path = "/tmp/libload_mirror_xyz", "org/the-repo"
        ma.config._name_or_path, mb_.config._name_or_path = "/tmp/libload_mirror_xyz", "org/the-repo"
        G = libload.g2_compare(libload.model_identity(ma), libload.model_identity(mb_))
        return not G["fail"] and not G["unknowns"], G
    out.append(("P2h a library load from its own mirror path (name_or_path) is no difference (pure)", p2h))
    def p2i():                                   # M15 through model_identity: qualified class names
        if not NEW:
            return False, "no comparator"
        T1 = type("Block", (torch.nn.Module,), {"__module__": "pkg_one"}); T2 = type("Block", (torch.nn.Module,), {"__module__": "pkg_two"})
        ma, mb_ = torch.nn.Module(), torch.nn.Module(); ma.a = T1(); mb_.a = T2()
        G = libload.g2_compare(libload.model_identity(ma), libload.model_identity(mb_))
        return "module_classes_differ" in G["fail"], G["fail"]
    out.append(("P2i model_identity keeps qualified class names (same bare name, other module -> failure)", p2i))
    def p2j():                                   # R5: sub-config private implementation choices are in the cache key
        if not NEW:
            return False, "no comparator"
        c1 = transformers.CLIPConfig(); c2 = transformers.CLIPConfig()
        c2.vision_config._attn_implementation_internal = "eager"
        k1 = libload.hypothesis("r", transformers.CLIPModel, (), c1, {})["key"]; k2 = libload.hypothesis("r", transformers.CLIPModel, (), c2, {})["key"]
        return k1 != k2, (k1, k2)
    out.append(("P2j a sub-config's private attention implementation is part of the load hypothesis", p2j))
    def p2k():                                   # M7 / M8: naming needs a library op; codes landing in floats are unknown
        if not NEW:
            return False, "no comparator"
        base = {"modules": {"(root)": "x.M"}, "params": {"w": [[4, 4], "float32"]}, "buffers": {}, "ties": [], "quantizer": "Q"}
        no_op = dict(base, conversions={"w": {"sources": ["w", "w_scale"], "aux_sources": ["w_scale"], "source_dtypes": ["F32", "F8_E4M3"], "ops": [], "quant_op": None}})
        g1 = libload.g2_compare(no_op, dict(base))
        g2_ = libload.g2_compare(dict(base, quantized_codes=["w"]), dict(base))
        return ("dequantized_at_load_by_library" not in g1["named"]
                and any("without dequantizing" in u["what"] for u in g2_["unknowns"])), {"no_op": g1["named"], "codes_in_floats": g2_["unknowns"]}
    out.append(("P2k no library operation -> never named dequantized; quantized codes landing in floats -> unknown", p2k))
    def p2c():
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M"}, "params": {"experts.gate_up": [[4, 8], "bfloat16"]}, "buffers": {}, "ties": [],
               "conversions": {"experts.gate_up": {"sources": ["experts.0.gate.weight", "experts.0.up.weight"], "aux_sources": [],
                                                   "source_dtypes": ["F8_E4M3"], "ops": ["MergeModulelist", "Concatenate"], "quant_op": None}},
               "quantizer": None, "quantized_codes": ["experts.gate_up"]}
        G = libload.g2_compare(Lid, dict(Lid, params={"experts.gate_up": [[4, 8], "float32"]}))
        return "dequantized_at_load_by_library" not in G["named"] and any("without dequantizing" in u["what"] for u in G["unknowns"]), G
    out.append(("P2c two FP8 codes merged with NO scale source are never named dequantized (an unknown) (pure)", p2c))
    def p2d():
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M"}, "params": {"t": [[4], "int64"]}, "buffers": {}, "ties": []}
        G = libload.g2_compare(Lid, dict(Lid, params={"t": [[4], "int32"]}))
        return "parameter_dtypes_differ" in G["fail"], G["fail"]
    out.append(("P2d a non-floating parameter dtype difference (no quantizer) is a failure (pure)", p2d))
    def p2e():
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M", "a": "pkg_one.Block"}, "params": {}, "buffers": {}, "ties": []}
        G = libload.g2_compare(Lid, dict(Lid, modules={"(root)": "x.M", "a": "pkg_two.Block"}))
        return "module_classes_differ" in G["fail"], G["fail"]
    out.append(("P2e two module classes with the same bare name in different modules differ (pure)", p2e))
    def p2f():
        if not NEW:
            return False, "no comparator"
        A = libload.Authority("/unused/f04/selftest")
        h = libload.hypothesis(A.repo, transformers.LlamaForCausalLM, (), None, {})
        A.cache[h["key"]] = {"hypothesis": dict(h, key="another-hypothesis")}
        try:
            A.report(transformers.LlamaForCausalLM, ())
            return False, "served"
        except RuntimeError as e:
            return "another load hypothesis" in str(e), str(e)
    out.append(("P2f the cache never serves a report of another hypothesis (the invariant raises)", p2f))
    def p7():                                    # the worker's whole decision, behaviourally
        if not hasattr(libload, "g2_decide"):
            return False, "no g2_decide"
        I = {"modules": {"(root)": "x.M"}, "params": {"a": [[2], "float32"]}, "buffers": {}, "ties": []}
        J = dict(I, params={"a": [[3], "float32"]})
        u0, f0, _ = libload.g2_decide(I, [I, I], I, None)
        u1, f1, _ = libload.g2_decide(I, [J], J, None)                 # executed differs from loaded -> failure items
        u2, f2, _ = libload.g2_decide(I, [I, J], I, None)              # builds differ -> invariant unknown
        u3, f3, _ = libload.g2_decide(I, [I], J, None)                 # the graded model changed -> invariant unknown
        u4, f4, _ = libload.g2_decide(I, [{"modules": None}], I, None) # comparator error -> unknown, never raised
        ok = (not u0 and not f0 and f1 and not f2 and any("between its builds" in u["what"] for u in u2)
              and any("between its builds" in u["what"] for u in u3) and any("comparator error" in u["what"] for u in u4))
        return ok, {"same": (u0, f0), "differs": f1, "builds": u2, "changed": u3, "error": u4}
    out.append(("P7 g2_decide: equal -> nothing; different -> T1 failure items; builds differ / model changed -> unknown; error -> unknown", p7))
    def p8():                                    # R11: dict-valued config fields are compared (rope_parameters.rope_theta)
        if not NEW:
            return False, "no comparator"
        a_, b_ = transformers.LlamaConfig(**LLAMA), transformers.LlamaConfig(**LLAMA)
        rp = dict(getattr(b_, "rope_parameters", None) or {"rope_type": "default", "rope_theta": 10000.0}); rp["rope_theta"] = 77.0
        b_.rope_parameters = rp
        with torch.device("meta"):
            ma, mb_ = transformers.LlamaForCausalLM(a_), transformers.LlamaForCausalLM(b_)
        G = libload.g2_compare(libload.model_identity(ma), libload.model_identity(mb_))
        return any("config fields differ" in u["what"] and any("rope_theta" in i for i in u["items"]) for u in G["unknowns"]), G["unknowns"]
    out.append(("P8 a nested config value (rope_parameters.rope_theta) that differs is visible", p8))
    def p9():                                    # R12: tuple settings are compared (conv stride)
        if not NEW:
            return False, "no comparator"
        ma, mb_ = torch.nn.Module(), torch.nn.Module()
        ma.c = torch.nn.Conv2d(2, 2, 3, stride=(1, 1)); mb_.c = torch.nn.Conv2d(2, 2, 3, stride=(2, 2))
        G = libload.g2_compare(libload.model_identity(ma), libload.model_identity(mb_))
        return any("module settings differ" in u["what"] and any("stride" in i for i in u["items"]) for u in G["unknowns"]), G["unknowns"]
    out.append(("P9 a tuple setting (conv stride) that differs is visible", p9))
    def p10():                                   # R6: a layout needs the executed module to run FLOATING parameters
        if not NEW:
            return False, "no comparator"
        Lid = {"modules": {"(root)": "x.M", "l": "x.Q"}, "params": {"l.weight": [[4, 4], "int8"]}, "buffers": {}, "ties": [],
               "quantizer": "Q", "quantized_codes": ["l.weight"]}
        G = libload.g2_compare(Lid, dict(Lid, params={"l.weight": [[4, 4], "int32"]}))
        return "parameter_dtypes_differ" in G["fail"], G
    out.append(("P10 codes executed as other integer codes (no floating parameters) are a failure, not a layout", p10))
    def p11():                                   # F04-2: settings computed in the checkpoint's half precision
        if not NEW:
            return False, "no comparator"
        base = {"modules": {"(root)": "x.M", "d": "x.DropPath"}, "params": {"w": [[2], "float16"]}, "buffers": {}, "ties": []}
        L_ = dict(base, module_attrs={"d.drop_prob": repr(0.004344940185546875)})
        E_ = dict(base, params={"w": [[2], "float32"]}, module_attrs={"d.drop_prob": repr(0.004347826354205608)})
        G = libload.g2_compare(L_, E_)
        E2 = dict(E_, module_attrs={"d.drop_prob": repr(0.0049)})            # a real difference stays an unknown
        G2_ = libload.g2_compare(L_, E2)
        return (not G["unknowns"] and "settings_computed_in_loaded_precision" in G["named"]
                and any("module settings differ" in u["what"] for u in G2_["unknowns"])), {"half": G, "real": G2_["unknowns"]}
    out.append(("P11 a float setting computed in the loaded half precision is named, a real difference stays an unknown", p11))
    # P3 no loaded identity -> unknown, never silently equal
    def p3():
        if not NEW:
            return False, "no comparator"
        G = libload.g2_compare(None, {"modules": {}, "params": {}, "buffers": {}, "ties": []})
        return (not G["fail"]) and bool(G["unknowns"]), G["unknowns"]
    out.append(("P3 no loaded identity -> G2 unknown (pure)", p3))
    # P4 the cache: same class, different config / aux / keywords / same-__name__ class -> separate loads
    def p4():
        calls = []
        orig = libload._report
        libload._report = lambda A, cls, aux=(), **kw: (calls.append((cls, tuple(sorted(aux)), kw.get("config"), kw.get("attn_implementation"))) or {"n": len(calls)})
        try:
            A = libload.Authority("/unused/f04/selftest")
            c1 = transformers.LlamaConfig(**LLAMA); c2 = transformers.LlamaConfig(**LLAMA, tie_word_embeddings=False, hidden_act="gelu")
            class LlamaForCausalLM(transformers.LlamaForCausalLM):      # same __name__, another class
                pass
            rs = [A.report(transformers.LlamaForCausalLM, (), config=c1), A.report(transformers.LlamaForCausalLM, (), config=c2),
                  A.report(transformers.LlamaForCausalLM, ("x.scale",), config=c1), A.report(LlamaForCausalLM, (), config=c1)]
            if NEW:
                rs.append(A.report(transformers.LlamaForCausalLM, (), config=c1, attn_implementation="eager"))
            again = A.report(transformers.LlamaForCausalLM, (), config=c1)
        finally:
            libload._report = orig
        distinct = len({id(r) for r in rs}) == len(rs)
        return distinct and again is rs[0] and len(calls) == len(rs), {"loads": len(calls), "requests": len(rs) + 1, "distinct": distinct}
    out.append(("P4 cache: different config / aux / class object / keywords never share a report; same hypothesis reuses", p4))
    # P5 the worker sequence: selection report, then the executed build mutates the caller's config, then the final call
    def p5():
        calls = []
        orig = libload._report
        libload._report = lambda A, cls, aux=(), **kw: (calls.append(1) or {"cfg_attn": getattr(kw.get("config"), "_attn_implementation_internal", None),
                                                                            "cfg_dtype": str(getattr(kw.get("config"), "dtype", None))})
        try:
            A = libload.Authority("/unused/f04/selftest")
            cfg = transformers.DiffLlamaConfig(**LLAMA)
            snap = copy.deepcopy(cfg)                           # worker_v2 CFG_LOAD
            kw = {"attn_implementation": "eager"} if NEW else {}
            r1 = A.report(transformers.DiffLlamaForCausalLM, (), config=snap if NEW else cfg, **kw)
            with torch.device("meta"):
                transformers.DiffLlamaForCausalLM._from_config(cfg, attn_implementation="eager", dtype=torch.float32)   # mutates cfg
            r2 = A.report(transformers.DiffLlamaForCausalLM, (), config=snap if NEW else cfg, **kw)
            A2 = libload.Authority("/unused/f04/selftest")      # a run where class selection did not happen
            r3 = A2.report(transformers.DiffLlamaForCausalLM, (), config=snap if NEW else cfg, **kw)
        finally:
            libload._report = orig
        same = (r1["cfg_attn"], r1["cfg_dtype"]) == (r3["cfg_attn"], r3["cfg_dtype"])
        return same and r1 is r2, {"selection_path": r1, "no_selection_path": r3}
    out.append(("P5 the load hypothesis does not depend on whether class selection ran first", p5))
    # P6 a real harness producer: the one-float32 conversion (lowcost.uniform_float32) splits a buffer object shared by
    # two modules when it is not float32 -> the executed model loses a tie the loaded model has: must be a G2 failure
    def p6():
        class A_(torch.nn.Module):
            def __init__(s, t): super().__init__(); s.register_buffer("tab", t, persistent=False)
        class M_(torch.nn.Module):
            def __init__(s):
                super().__init__(); t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3); s.a, s.b = A_(t), A_(t)
        m = M_()
        if not NEW:
            sd = m.state_dict(keep_vars=True); shp = {k: list(t.shape) for k, t in sd.items()}
            lowcost.uniform_float32(m)
            G = libload.g1_assess({"nonloaded": {}, "state_shapes": shp}, set(), {k: list(t.shape) for k, t in m.state_dict(keep_vars=True).items()}, {})
            return bool(G["fail"]) or any(u["predicate"] == "G2" for u in G["unknowns"]), G
        Lid = libload.model_identity(m)
        lowcost.uniform_float32(m)
        G = libload.g2_compare(Lid, libload.model_identity(m))
        return "tie_groups_differ" in G["fail"], G["fail"]
    out.append(("P6 the float32 build splits a shared bf16 buffer (lowcost.uniform_float32) -> G2 failure", p6))
    return out


def _ctx_ok(name_node, tree, allowed):
    """A use of a name is a read in an allowed role: a keyword argument named `config`, or the argument of deepcopy."""
    import ast
    for p_ in ast.walk(tree):
        for ch in ast.iter_child_nodes(p_):
            if ch is name_node:
                if isinstance(p_, ast.keyword) and p_.arg in allowed:
                    return True
                if isinstance(p_, ast.Call) and name_node in p_.args and getattr(p_.func, "attr", getattr(p_.func, "id", "")) in allowed:
                    return True
                return False
    return False


def wiring():
    """The worker's F04 call sites, pinned by their syntax tree (a statement changed in any way, e.g. `and False`, an
    emptied list, the loaded identity compared with itself, CFG_LOAD aliased to cfg, fails here). The decision itself
    (libload.g2_decide) is tested behaviourally (P7)."""
    import ast
    W = open(os.path.join(HARNESS, "worker_v2.py")).read()
    T = ast.parse(W)
    stmts = [n for n in ast.walk(T) if isinstance(n, ast.stmt)]
    dump = lambda src: ast.dump(ast.parse(src).body[0])
    has = lambda src: any(ast.dump(n) == dump(src) for n in stmts)
    names = lambda nm: [n for n in ast.walk(T) if isinstance(n, ast.Name) and n.id == nm]
    def assigned(nm):
        def flat(t):
            return [x for e in t.elts for x in flat(e)] if isinstance(t, (ast.Tuple, ast.List)) else [t]
        return [n for n in stmts if isinstance(n, (ast.Assign, ast.AugAssign)) and any(
            isinstance(t, ast.Name) and t.id == nm for tt in (n.targets if isinstance(n, ast.Assign) else [n.target]) for t in flat(tt))]
    cfg_load = assigned("CFG_LOAD")
    fresh = next((n for n in T.body if isinstance(n, ast.FunctionDef) and n.name == "_fresh"), None)
    return {
        "CFG_LOAD is assigned once, a deep copy of cfg, before _fresh is defined":
            len(cfg_load) == 1 and ast.dump(cfg_load[0].value) == ast.dump(ast.parse("_copy.deepcopy(cfg)").body[0].value)
            and fresh is not None and cfg_load[0].lineno < fresh.lineno,
        "selection authority call: CFG_LOAD + the executed attention":
            any(ast.dump(n) == ast.dump(ast.parse("AUTH.report(klass, _A_aux, config=CFG_LOAD, **_lkw(klass))").body[0].value) for n in ast.walk(T) if isinstance(n, ast.Call)),
        "final authority call: CFG_LOAD + the executed attention":
            any(ast.dump(n) == ast.dump(ast.parse("AUTH.report(cls, set(A), config=CFG_LOAD, **_lkw(cls))").body[0].value) for n in ast.walk(T) if isinstance(n, ast.Call)),
        "every executed build records its identity": has("BUILD_IDS.append(libload.model_identity(mm))"),
        "the G2 decision compares the LOADED identity with the executed builds":
            has('_g2u, _g2f, _g2r = libload.g2_decide(LIB.get("identity"), BUILD_IDS, _EID, {"attn_implementation": BUILD_ATTN, "dtype": "float32", "zero_storage": True})'),
        "G2 unknowns join the row's unknowns, untouched in between": has("UNKNOWNS.extend(_g2u)") and len(names("_g2u")) == 2,
        "a G2 failure fails T1, unconditionally on its items":
            any(isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(ast.parse("_g2f").body[0].value)
                and any(ast.dump(b) == dump('_g1f["executed_differs_from_loaded"] = _g2f') for b in n.body) for n in stmts)
            and len(assigned("_g2f")) == 2,
        # nothing else may touch the F04 state between the pinned statements (a statement ADDED around them)
        "CFG_LOAD is only read: passed as config=, or deep-copied (never mutated, never re-pointed)":
            all(_ctx_ok(n, T, ("config", "deepcopy", "forget", "lookup")) for n in names("CFG_LOAD")[1:]),
        "BUILD_IDS is only created, appended to in _fresh, and handed to g2_decide":
            len(assigned("BUILD_IDS")) == 1
            and all(n.attr == "append" for n in ast.walk(T) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "BUILD_IDS")
            and sum(1 for n in ast.walk(fresh) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "BUILD_IDS") == 2
            and len(names("BUILD_IDS")) == 4,
        "UNKNOWNS is only ever appended to / extended (never filtered, popped, cleared or re-pointed)":
            len(assigned("UNKNOWNS")) == 1 and all(n.attr in ("append", "extend") for n in ast.walk(T) if isinstance(n, ast.Attribute)
                                                     and isinstance(n.value, ast.Name) and n.value.id == "UNKNOWNS")
            and not any(isinstance(n, (ast.Delete, ast.AugAssign)) and "UNKNOWNS" in ast.dump(n) for n in stmts)
            and not any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "UNKNOWNS"
                        and isinstance(n.ctx, ast.Store) for n in ast.walk(T)),
        "the T1 failure key is written once and never removed":
            W.count('"executed_differs_from_loaded"') == 1 and not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                                                                       and n.value.id == "_g1f" and n.attr in ("pop", "clear", "popitem")
                                                                       for n in ast.walk(T))
            and not any(isinstance(n, ast.Delete) and "_g1f" in ast.dump(n) for n in stmts),
        "the attention choice never caches an unexpected error (e.g. a stage time limit)":
            any(isinstance(n, ast.ExceptHandler) and n.type is not None and ast.dump(n.type) == ast.dump(ast.parse("(TypeError, ValueError)").body[0].value)
                for f in [x for x in T.body if isinstance(x, ast.FunctionDef) and x.name == "_attn_choice"] for n in ast.walk(f))
            and not any(isinstance(n, ast.ExceptHandler) and (n.type is None or getattr(n.type, "id", "") in ("Exception", "BaseException"))
                        for f in [x for x in T.body if isinstance(x, ast.FunctionDef) and x.name == "_attn_choice"] for n in ast.walk(f)),
    }


def main():
    d = tempfile.mkdtemp(prefix="selftest_g2_")
    bad = 0
    try:
        for name, (path, cls, mut, akw, want) in build(d).items():
            if SEL and not any(s.lower() in name.lower() for s in SEL): continue
            try:
                got, why = g2_of(path, cls, mut, akw)
            except Exception as e:
                got, why = "ERROR", {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if want == "NOT-RUN":
                bad += 1; print(f"FAIL {name}", flush=True); continue
            want, _, forbid = want.partition("!named:")
            w, _, need = want.partition("+named:")
            wv, _, wk = w.partition(":")
            ok = got == wv
            if ok and wv == "FAIL": ok = wk in (why.get("fail") or {})
            if ok and wv == "UNKNOWN": ok = any(wk in u for u in why.get("unknowns") or [])
            if ok and need: ok = need in (why.get("named") or {})
            if ok and forbid: ok = forbid not in (why.get("named") or {})
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})")
                  + f"  {json.dumps(why, default=str)[:300]}", flush=True)
        if NEW:
            for name, ok in wiring().items():
                if SEL and not any(s_.lower() in name.lower() for s_ in SEL): continue
                bad += not ok
                print(f"{'ok  ' if ok else 'FAIL'} W {name}: {ok}", flush=True)
        for name, fn in pure_controls():
            if SEL and not any(s.lower() in name.lower() for s in SEL): continue
            try:
                ok, det = fn()
            except Exception as e:
                ok, det = False, f"{type(e).__name__}: {str(e)[:200]}"
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}  {json.dumps(det, default=str)[:300]}", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{bad} unexpected ({'F04 comparator' if NEW else 'HEAD-before-F04: names/shapes only, cache by class name'})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
