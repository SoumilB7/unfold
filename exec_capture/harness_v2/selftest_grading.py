"""Negative controls for the G1/G2 grade predicates (INTENT.md; decisions in z-docs/alternative-approach/
G1_LOAD_PROVENANCE_2026-10-09.md). Each control is a tiny randomly-initialised library model saved as a local
checkpoint with one named defect (no downloads). It runs through the SAME code the worker uses: libload.Authority's
library load + provenance, the DagRecorder execution (zero-storage build, as the worker), and libload.g1_assess. Every
control must give its intended verdict.

    python3 selftest_grading.py [name-substring ...]          # exit 0 = every control as expected
Controls from the 2026-10-09 research (model-benchmark/_reruns/g1_research), validated there against real CPU loads.
"""
import os, sys, json, tempfile, shutil, collections, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch, transformers
from safetensors.torch import save_file
import libload, lowcost
from dag import DagRecorder, math_attention

SEL = sys.argv[1:]


def save(d, name, model, edit=None, cfg_edit=None, extra_files=None):
    p = os.path.join(d, name); os.makedirs(p, exist_ok=True)
    model.config.save_pretrained(p)
    sd = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    if edit: sd = edit(sd)
    save_file(sd, os.path.join(p, "model.safetensors"))
    if cfg_edit:
        j = json.load(open(os.path.join(p, "config.json"))); cfg_edit(j); json.dump(j, open(os.path.join(p, "config.json"), "w"))
    for f, tens in (extra_files or {}).items():
        os.makedirs(os.path.dirname(os.path.join(p, f)), exist_ok=True)
        save_file(tens, os.path.join(p, f))
    return p


def build(d):
    """name -> (dir, class, inputs, intended verdict)"""
    torch.manual_seed(0)
    C = {}
    ids = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    llama = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                                     tie_word_embeddings=False)
    m = transformers.LlamaForCausalLM(llama)
    L = "LlamaForCausalLM"
    C["C0 clean llama"] = (save(d, "C0", m), L, ids, "FULL")
    C["C1 lm_head not shipped (random, executed)"] = (save(d, "C1", m, lambda sd: {k: v for k, v in sd.items() if k != "lm_head.weight"}), L, ids, "PARTIAL:random_init_executed")
    C["C2 final norm not shipped (ones)"] = (save(d, "C2", m, lambda sd: {k: v for k, v in sd.items() if k != "model.norm.weight"}), L, ids, "FULL")
    C["C3 extra tensor shipped"] = (save(d, "C3", m, lambda sd: dict(sd, **{"model.extra_head.weight": torch.zeros(4, 16)})), L, ids, "PARTIAL")
    C["C15 extra weight file, not a duplicate"] = (save(d, "C15", m, extra_files={"extra.safetensors": {"x.weight": torch.zeros(7, 3)}}), L, ids, "FULL_UNVERIFIED")
    C["C16 extra weight file, shape-duplicate"] = (save(d, "C16", m, extra_files={"copy.safetensors": {"lm_head.weight": torch.zeros(64, 16)}}), L, ids, "FULL")
    C["C18 subfolder layer, same shape as a read tensor but its own (LaBSE 2_Dense)"] = (
        save(d, "C18", m, extra_files={"2_Dense/model.safetensors": {"linear.weight": torch.zeros(16, 16)}}), L, ids, "FULL_UNVERIFIED")
    mt = transformers.LlamaForCausalLM(transformers.LlamaConfig(**{**llama.to_dict(), "tie_word_embeddings": True}))
    C["C4 tied, one shipped"] = (save(d, "C4", mt, lambda sd: {k: v for k, v in sd.items() if k != "lm_head.weight"}), L, ids, "FULL")
    C["C5 tied, both shipped (equal)"] = (save(d, "C5", mt), L, ids, "FULL_UNVERIFIED")
    C["C6 tied, both shipped (different)"] = (save(d, "C6", mt, lambda sd: dict(sd, **{"lm_head.weight": sd["lm_head.weight"] + 1.0})), L, ids, "FULL_UNVERIFIED")
    r = transformers.ResNetForImageClassification(transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], num_labels=3))
    px = {"pixel_values": torch.randn(1, 3, 32, 32)}
    C["C7 BN counter not shipped"] = (save(d, "C7", r, lambda sd: {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}), "ResNetForImageClassification", px, "FULL")
    C["C8 BN running stats not shipped (deterministic)"] = (save(d, "C8", r, lambda sd: {k: v for k, v in sd.items() if not k.endswith(("running_mean", "running_var"))}), "ResNetForImageClassification", px, "FULL")
    b = transformers.BertForPreTraining(transformers.BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                                                intermediate_size=32, max_position_embeddings=32))
    C["C9 BERT NSP head not shipped (random, executed)"] = (save(d, "C9", b, lambda sd: {k: v for k, v in sd.items() if not k.startswith("cls.seq_relationship")}), "BertForPreTraining", ids, "PARTIAL")
    enc = transformers.EncodecConfig(target_bandwidths=[1.5, 3.0, 6.0], num_filters=4, codebook_size=16, codebook_dim=8, hidden_size=8,
                                     upsampling_ratios=[2, 2], num_lstm_layers=1, sampling_rate=800)
    e = transformers.EncodecModel(enc)
    wav = {"input_values": torch.randn(1, 1, 800)}
    C["C11 EnCodec default bandwidth (codebooks unread while siblings read)"] = (save(d, "C11", e), "EncodecModel", wav, "PARTIAL")
    C["C11b EnCodec full bandwidth (only dead EMA state unread)"] = (save(d, "C11b", e), "EncodecModel", dict(wav, bandwidth=6.0), "FULL_LEFTOVERS")
    u = transformers.UMT5ForConditionalGeneration(transformers.UMT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                                                                          num_heads=2, decoder_start_token_id=0))
    C["C12 UMT5 raw tie false, both shipped"] = (save(d, "C12", u, cfg_edit=lambda j: j.__setitem__("tie_word_embeddings", False)),
                                                "UMT5ForConditionalGeneration", dict(ids, decoder_input_ids=torch.tensor([[0, 1]])), "FULL_UNVERIFIED")
    qn = transformers.Qwen3NextConfig(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                      num_key_value_heads=1, head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1,
                                      linear_key_head_dim=8, linear_value_head_dim=8, num_experts=2, num_experts_per_tok=1,
                                      moe_intermediate_size=16, shared_expert_intermediate_size=16, layer_types=["linear_attention"],
                                      max_position_embeddings=32)
    q = transformers.Qwen3NextForCausalLM(qn)
    C["C13 Qwen3-Next A_log not shipped (random, executed)"] = (save(d, "C13", q, lambda sd: {k: v for k, v in sd.items() if not k.endswith("A_log")}),
                                                                "Qwen3NextForCausalLM", dict(ids, use_cache=False), "PARTIAL")
    pv = transformers.PvtV2ForImageClassification(transformers.PvtV2Config(hidden_sizes=[8, 16, 16, 32], depths=[1, 1, 1, 1],
                                                                          num_attention_heads=[1, 1, 1, 1], num_labels=3))
    C["C17 classifier not shipped, library inits it with trunc_normal_ (random, executed)"] = (
        save(d, "C17", pv, lambda sd: {k: v for k, v in sd.items() if not k.startswith("classifier.")}),
        "PvtV2ForImageClassification", {"pixel_values": torch.randn(1, 3, 64, 64)}, "PARTIAL:random_init_executed")
    dab = transformers.DabDetrConfig(use_timm_backbone=False, use_pretrained_backbone=False,
                                     backbone_config=transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 16, 32], depths=[1, 1, 1, 1], out_features=["stage4"]),
                                     backbone=None, hidden_size=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                                     decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, num_queries=4, num_labels=3)
    dm = transformers.DabDetrForObjectDetection(dab)
    C["C14 DAB-DETR head shipped under the shared alias (library re-initialises it)"] = (
        save(d, "C14", dm, lambda sd: {k: v for k, v in sd.items() if not k.startswith("bbox_predictor.")}),
        "DabDetrForObjectDetection", {"pixel_values": torch.randn(1, 3, 64, 64)}, "PARTIAL:random_init_executed")
    return C


def grade(path, cls_name, kw):
    """The worker's G1/G2 path on one local checkpoint: library load + provenance, zero-storage execution, g1_assess."""
    cls = getattr(transformers, cls_name)
    cfg = transformers.AutoConfig.from_pretrained(path)
    A = libload.Authority(path)
    LIB = A.report(cls, config=cfg)
    with lowcost.zero_storage():
        m = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()
    rec = DagRecorder(m, kw)
    try:
        with torch.no_grad(), math_attention(), rec:
            m(**kw)
    finally:
        rec.remove_hooks()
    used = set(rec.used_params) | set(rec.used_buffers)
    sd = m.state_dict(keep_vars=True)
    alias = collections.defaultdict(list)
    for k, v in sd.items(): alias[id(v)].append(k)
    for names in alias.values():
        if any(n in used for n in names): used |= set(names)
    pn = {n for n, _ in m.named_parameters(remove_duplicate=False)}
    bg = {k: (type(m.get_submodule(k.rpartition(".")[0]) if "." in k else m).__name__, k.rpartition(".")[2])
          for k, t in sd.items() if k not in pn and t.is_floating_point()}
    G = libload.g1_assess(LIB["provenance"], used, {k: list(t.shape) for k, t in sd.items()}, bg)
    # the worker's other T1 inputs, as the library reports them: unexpected / mismatched keys, loaded params never read
    unexp = [k for k in LIB["unexpected"] if not k.endswith("num_batches_tracked")]
    loaded = {n for n, _ in m.named_parameters()} - set(LIB["missing"])
    unused = sorted(p for p in loaded if p not in used)
    t1_old = not unexp and not LIB["mismatched"] and not unused
    if G["fail"] or not t1_old:
        v = "PARTIAL"
    elif G["leftovers"]:
        v = "FULL_LEFTOVERS"
    else:
        v = "FULL"
    if v in ("FULL", "FULL_LEFTOVERS") and G["unknowns"]:
        v = "FULL_UNVERIFIED"
    why = {"fail": G["fail"], "unexpected": unexp, "unused": unused[:4], "leftovers": G["leftovers"],
           "unknowns": [u["what"][:70] for u in G["unknowns"]]}
    return v, why


def main():
    d = tempfile.mkdtemp(prefix="selftest_grading_")
    bad = 0
    try:
        for name, (path, cls_name, kw, want) in build(d).items():
            if SEL and not any(s.lower() in name.lower() for s in SEL): continue
            try:
                got, why = grade(path, cls_name, kw)
            except Exception as e:
                got, why = "ERROR", f"{type(e).__name__}: {str(e)[:200]}"
            want_v, _, want_why = want.partition(":")       # "PARTIAL:<reason>": the verdict AND that failure reason
            ok = got == want_v and (not want_why or want_why in (why.get("fail") or {} if isinstance(why, dict) else {}))
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})")
                  + f"  {json.dumps({k: v for k, v in why.items() if v}, default=str)[:260] if isinstance(why, dict) else why}", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{bad} unexpected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
