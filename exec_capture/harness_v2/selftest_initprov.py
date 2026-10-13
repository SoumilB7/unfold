"""F03 controls: initialization provenance (EXECUTION_PLAN P2a.2; dossier model-benchmark/_reruns/p2a2_f03; rulings
z-docs/alternative-approach/DECISIONS_2026-10-13_F03.md). Every control runs through the worker's real G1 path:
a tiny local checkpoint -> libload.Authority library load on meta (InitTracer during _finalize_model_loading) ->
libload._provenance -> zero-storage DagRecorder execution -> libload.g1_assess (selftest_grading.grade).
Rule: a value's provenance is known only if every element's final value follows from known sources; allocation is not
initialization; a later FULL overwrite re-establishes provenance; a partial unknown write invalidates; a loaded tensor
the library changes after loading is no longer the checkpoint's value.

    python3 selftest_initprov.py        # exit 0 = every control as expected (pre-F03 loadtrace / libload fail it)
The 416-case perturbation-oracle matrix (model-benchmark/_reruns/p2a2_f03/matrix.py) is a separate receipt."""
import os, sys, json, tempfile, shutil, warnings, time
warnings.filterwarnings("ignore"); os.environ["TRANSFORMERS_VERBOSITY"] = "error"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import numpy as np
import torch, transformers as T
from transformers import initialization as init
import libload
import selftest_grading as SG

SH = (4, 6)
LL = T.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                   num_key_value_heads=1, max_position_embeddings=32, tie_word_embeddings=False)


def probe_class(seq, tgt):
    """A Llama whose _init_weights runs `seq` on one extra tensor (param or persistent buffer) that its forward reads."""
    class Probe(T.LlamaForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            self.probe_p = torch.nn.Parameter(torch.zeros(SH))
            self.register_buffer("probe_b", torch.zeros(SH))
            self.src_p = torch.nn.Parameter(torch.zeros(SH))   # always shipped: a loaded source for F03-2 copies
            self._ext = torch.zeros(SH) + 0.25                # a tensor of unknown provenance to the tracer
            self.post_init()

        def _init_weights(self, module):
            super()._init_weights(module)
            if module is self and seq is not None and hasattr(self, "_ext"):
                with torch.no_grad():
                    seq(self, getattr(self, tgt))

        def forward(self, input_ids=None, **kw):
            out = super().forward(input_ids=input_ids, **kw)
            out.logits = out.logits + (self.probe_p.sum() + self.probe_b.sum() + self.src_p.sum())
            return out
    Probe.__name__ = Probe.__qualname__ = f"Probe_{tgt}_{abs(hash(seq)) % 10 ** 6}"
    return Probe


def table():
    return (torch.arange(24, dtype=torch.float32).view(SH) * 0.1).sin()


def _stale(m, t):
    init.normal_(t); c = t.clone(); t.zero_(); t.copy_(c)


def _tmp_cover(m, t):
    x = torch.empty(SH); x[:, 0::2] = table()[:, 0::2]; x[:, 1::2] = table()[:, 1::2]; init.copy_(t, x)


def _tmp_half(m, t):
    x = torch.empty(SH); x[:, 0::2] = table()[:, 0::2]; init.copy_(t, x)


def _tmp_ones_last(m, t):
    x = torch.ones(SH); x[-1] = 0.1; init.copy_(t, x)


def _inv_freq(m, t):
    init.copy_(t, (1.0 / (10000 ** (torch.arange(0, 48, 2, dtype=torch.float32) / 48))).view(SH))


# want: "VERDICT[:fail key][+unknown text]" as in selftest_grading; class_want: the probe tensor's provenance class
PROBES = {
    # ---- negatives: the F03 counterexamples (Astra runtime_controls.json) and their neighbours ----
    "A1 in-place mul_ on an unfilled target": (lambda m, t: t.mul_(2), False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "A2 zero_ then a PARTIAL copy of unknown data": (lambda m, t: (t.zero_(), t[:1].copy_(m._ext[:1])), False,
                                                    "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "A3 copy_ of an empty allocation": (lambda m, t: t.copy_(torch.empty(SH)), False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "A4 computed table, then a partial unknown overwrite": (lambda m, t: (init.copy_(t, table()), t[:1].copy_(m._ext[:1])), False,
                                                           "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "N1 masked_fill_ of an unfilled target": (lambda m, t: t.masked_fill_(torch.arange(24).view(SH) % 2 == 0, 0.0), False,
                                             "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "N2 empty temporary half covered, init.copy_": (_tmp_half, False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "N3 requires_grad_ is not a value write": (lambda m, t: t.requires_grad_(False), False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "N4 /= on an unfilled target": (lambda m, t: t.__itruediv__(3), False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    "N5 loaded, then /= sqrt(N) (Mamba rescale_prenorm_residual)": (lambda m, t: t.__itruediv__(2 ** 0.5), True,
                                                                   "FULL_UNVERIFIED+changed after loading", "LOADED_MODIFIED"),
    "N6 loaded, then init.zeros_(t.data[1]) (Youtu / PP-DocLayout padding row)": (lambda m, t: init.zeros_(t.data[1]), True,
                                                                               "FULL_UNVERIFIED+changed after loading", "LOADED_MODIFIED"),
    "N7 loaded, then a full copy of unknown data": (lambda m, t: t.copy_(m._ext), True, "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    # ---- F03-2 (Soumil 2026-10-13): a tensor not shipped, filled by copying a LOADED tensor ----
    "D1 whole copy of an unmodified loaded tensor: checkpoint-derived, certified": (lambda m, t: init.copy_(t, m.src_p), False, "FULL", "INIT_CKPT_DERIVED"),
    "D2 copy of a loaded tensor the library modified first: unknown": (lambda m, t: (m.src_p.mul_(2), init.copy_(t, m.src_p)), False,
                                                                     "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "D3 partial copy of a loaded tensor (rest never filled)": (lambda m, t: t[:1].copy_(m.src_p[:1]), False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
    # ---- v2 (F03 cross-exam p2a2_xexam): certification exactly per F03-2; computed / constant sources recognised ----
    "X1 copy of a loaded tensor x 2 (derived, not a copy): unknown": (lambda m, t: init.copy_(t, m.src_p * 2), False,
                                                                     "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "X2 zero_ then a partial copy of a loaded tensor: unknown": (lambda m, t: (t.zero_(), t[:1].copy_(m.src_p[:1])), False,
                                                               "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "X3 copy of a loaded tensor whose row 1 was zeroed first: unknown": (lambda m, t: (init.zeros_(m.src_p.data[1]), init.copy_(t, m.src_p)), False,
                                                                       "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "X4 random, clone, zero_, copy the clone back (stale identity): random": (_stale, False, "PARTIAL:random_init_executed", "INIT_RANDOM"),
    "X5 copy of torch.tensor(numpy random): unknown": (lambda m, t: init.copy_(t, torch.tensor(np.random.randn(*SH), dtype=torch.float32)), False,
                                                     "FULL_UNVERIFIED+copy of unknown provenance", "UNPROVEN"),
    "X6 repeat_interleave computed table (Sam3-style): computed": (lambda m, t: init.copy_(t, torch.arange(12.).repeat_interleave(2).view(SH).cos()), False,
                                                                  "FULL", "INIT_COMPUTED"),
    "X7 legacy constructor torch.FloatTensor(list) (Levit-style): constant": (lambda m, t: init.copy_(t, torch.FloatTensor(list(range(24))).view(SH)), False,
                                                                            "FULL", "INIT_*"),
    # ---- nearby valid controls: must stay certified / random must stay rejected ----
    "V1 init.constant_": (lambda m, t: init.constant_(t, 0.5), False, "FULL", "INIT_DETERMINISTIC"),
    "V2 init.ones_ (LayerNorm weight)": (lambda m, t: init.ones_(t), False, "FULL", "INIT_DETERMINISTIC"),
    "V3 sinusoid table: empty temporary, both halves, init.copy_ (Florence-2)": (_tmp_cover, False, "FULL", "INIT_COMPUTED"),
    "V4 inv_freq table, init.copy_": (_inv_freq, False, "FULL", "INIT_COMPUTED"),
    "V5 ones temporary, [-1] = c, init.copy_ (MaskFormer empty_weight)": (_tmp_ones_last, False, "FULL", "INIT_*"),
    "V6 random, then a FULL zero_ (re-establishes)": (lambda m, t: (init.normal_(t), t.zero_()), False, "FULL", "INIT_DETERMINISTIC"),
    "V7 zero_, then init.normal_ (random still rejected)": (lambda m, t: (t.zero_(), init.normal_(t)), False, "PARTIAL:random_init_executed", "INIT_RANDOM"),
    "V8 init.trunc_normal_": (lambda m, t: init.trunc_normal_(t), False, "PARTIAL:random_init_executed", "INIT_RANDOM"),
    "V9 loaded, untouched": (None, True, "FULL", None),
    "V10 loaded; the init.normal_ wrapper skips it (_is_hf_initialized)": (lambda m, t: init.normal_(t), True, "FULL", None),
    "V11 not shipped, never initialised": (None, False, "PARTIAL:uninitialized_executed", "UNINITIALIZED"),
}


def mamba(r):
    return T.MambaForCausalLM(T.MambaConfig(vocab_size=64, hidden_size=16, state_size=4, num_hidden_layers=2, rescale_prenorm_residual=r,
                                            tie_word_embeddings=False))


def youtu(pad):
    return T.YoutuForCausalLM(T.YoutuConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                             num_key_value_heads=2, pad_token_id=pad, max_position_embeddings=32, bos_token_id=1,
                                             eos_token_id=2, tie_word_embeddings=False))


def _pad_row(sd):
    sd = dict(sd); sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"].clone(); sd["model.embed_tokens.weight"][3] = 0.777
    return sd


ids = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
# real library classes: (factory, edit, inputs, want, {tensor: class_want})
class NPBuf(T.LlamaForCausalLM):
    """A non-persistent buffer read by the forward: garbage (never filled) or computed (filled by _init_weights)."""
    FILL = False
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("np_tab", torch.empty(SH), persistent=False)
        self.post_init()
    def _init_weights(self, module):
        super()._init_weights(module)
        if module is self and type(self).FILL and hasattr(self, "np_tab"):
            with torch.no_grad():
                init.copy_(self.np_tab, table())
    def forward(self, input_ids=None, **kw):
        out = super().forward(input_ids=input_ids, **kw)
        out.logits = out.logits + self.np_tab.sum(); return out
class NPBufFilled(NPBuf):
    FILL = True


REAL = {
    "R7 Levit: attention_bias_idxs from torch.LongTensor (CPU load refills it identically)": (
        lambda: T.LevitModel(T.LevitConfig(image_size=32, patch_size=16, hidden_sizes=[16, 32, 48], num_attention_heads=[1, 2, 3],
                                           depths=[1, 1, 1], key_dim=[8, 8, 8])), None, {"pixel_values": torch.randn(1, 3, 32, 32)}, "FULL", {}),
    "R8 Sam3ViT: rotary tables built with repeat_interleave (CPU load refills them identically)": (
        lambda: T.Sam3ViTModel(T.Sam3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56,
                                               patch_size=14, window_size=2, global_attn_indexes=[1], pretrain_image_size=28)), None,
        {"pixel_values": torch.randn(1, 3, 56, 56)}, "FULL", {}),
    "F5a non-persistent buffer never filled (allocation garbage), read by the forward": (
        lambda: NPBuf(LL), None, ids, "PARTIAL:uninitialized_executed", {}),
    "F5b non-persistent buffer filled with a computed table": (lambda: NPBufFilled(LL), None, ids, "FULL", {}),
    "R1 Mamba rescale_prenorm_residual=True: the library divides the LOADED out_proj": (
        lambda: mamba(True), None, ids, "FULL_UNVERIFIED+changed after loading",
        {"backbone.layers.0.mixer.out_proj.weight": "LOADED_MODIFIED"}),
    "R2 Mamba rescale_prenorm_residual=False (every real Mamba config in the catalog)": (
        lambda: mamba(False), None, ids, "FULL", {}),
    "R3 Youtu pad_token_id set: the library zeroes the LOADED padding row": (
        lambda: youtu(3), _pad_row, ids, "FULL_UNVERIFIED+changed after loading", {"model.embed_tokens.weight": "LOADED_MODIFIED"}),
    "R4 XLM tied head: _adjust_bias rewrites a LOADED bias with itself (identity, stays loaded)": (
        lambda: T.XLMWithLMHeadModel(T.XLMConfig(vocab_size=64, emb_dim=16, n_layers=1, n_heads=2, max_position_embeddings=32)),
        None, ids, None, {"pred_layer.proj.bias": None}),
    "R5 (selftest_grading C2) final norm not shipped: ones": (
        lambda: T.LlamaForCausalLM(LL), lambda sd: {k: v for k, v in sd.items() if k != "model.norm.weight"}, ids, "FULL",
        {"model.norm.weight": "INIT_DETERMINISTIC"}),
    "R6 (selftest_grading C8) BN running stats not shipped: zeros / ones": (
        lambda: T.ResNetForImageClassification(T.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], num_labels=3)),
        lambda sd: {k: v for k, v in sd.items() if not k.endswith(("running_mean", "running_var"))},
        {"pixel_values": torch.randn(1, 3, 32, 32)}, "FULL", {"resnet.embedder.embedder.normalization.running_mean": "INIT_DETERMINISTIC",
                                                              "resnet.embedder.embedder.normalization.running_var": "INIT_DETERMINISTIC"}),
}


def check(v, why, want):
    if want is None:
        return True
    want, _, need = want.partition("+")
    wv, _, wf = want.partition(":")
    return v == wv and (not wf or wf in (why.get("fail") or {})) and (not need or any(need in u for u in why.get("unknowns") or []))


def cls_ok(got, want):
    if want is None:
        return got is None
    return got is not None and (got == want or (want.endswith("*") and got.startswith(want[:-1])))


def main():
    t0 = time.time(); d = tempfile.mkdtemp(prefix="initprov_"); bad = 0
    try:
        for name, (seq, shipped, want, cw) in PROBES.items():
            for tgt in ("probe_p", "probe_b"):
                cls = probe_class(seq, tgt)
                torch.manual_seed(0)
                p = SG.save(d, f"p{bad}_{abs(hash(name + tgt)) % 10 ** 8}", cls(LL), lambda sd, tgt=tgt: {k: v for k, v in sd.items() if k != tgt or shipped})
                A = libload.Authority(p); A.mirror()
                nl = (A.report(cls, config=T.AutoConfig.from_pretrained(p)).get("provenance") or {}).get("nonloaded") or {}
                c = (nl.get(tgt) or {}).get("class")
                v, why = SG.grade(p, cls, ids)
                ok = check(v, why, want) and cls_ok(c, cw)
                bad += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {name} [{tgt}]: {v} class={c}" + ("" if ok else f"  (want {want}, class {cw})"), flush=True)
        for name, (mk, edit, kw, want, classes) in REAL.items():
            torch.manual_seed(0)
            m = mk(); p = SG.save(d, f"r{abs(hash(name)) % 10 ** 8}", m, edit)
            A = libload.Authority(p); A.mirror()
            nl = (A.report(type(m), config=T.AutoConfig.from_pretrained(p)).get("provenance") or {}).get("nonloaded") or {}
            v, why = SG.grade(p, type(m), kw)
            ok = check(v, why, want) and all(cls_ok((nl.get(k) or {}).get("class"), c) for k, c in classes.items())
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {v} " + json.dumps({k: (nl.get(k) or {}).get("class") for k in classes}), flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{bad} unexpected ({time.time() - t0:.0f}s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
