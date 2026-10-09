"""Negative controls for exec_to_ir: tiny random Llamas, each recorded once with one deliberate change to what it
computes. The unchanged control must be drawn; EVERY mutant must be refused (NotDrawn). A drawing gate is trusted only
while this passes. First mutants from the independent Codex audit (model-benchmark/_reruns/audit_codex/probes.py).

    python3 selftest_drawing.py [--unfold-pkg PATH]       # exit 0 = all as expected
"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
pkg = (sys.argv[sys.argv.index("--unfold-pkg") + 1] if "--unfold-pkg" in sys.argv
       else os.path.join(HERE, "..", "renderer_snapshot"))
sys.path.insert(0, os.path.abspath(pkg))
import torch, transformers
from transformers.models.llama import modeling_llama as LL
import capture, exec_to_ir

torch.manual_seed(0)
ORIG_ROT, ORIG_NORM, ORIG_HALF = LL.apply_rotary_pos_emb, LL.LlamaRMSNorm.forward, LL.rotate_half


def bundle_of(patch=None, rope=None):
    LL.apply_rotary_pos_emb, LL.LlamaRMSNorm.forward, LL.rotate_half = ORIG_ROT, ORIG_NORM, ORIG_HALF
    c = transformers.LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                                 **({"rope_parameters": rope} if rope else {}))
    m = transformers.LlamaForCausalLM._from_config(c, attn_implementation="eager", dtype=torch.float32).eval()
    if patch: patch(m)
    kw = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "use_cache": False}
    r = capture.CaptureRecorder(m, kw, skip=False)
    try:
        with torch.no_grad(), r:
            m(**kw)
    finally:
        r.remove_hooks()
        LL.apply_rotary_pos_emb, LL.LlamaRMSNorm.forward, LL.rotate_half = ORIG_ROT, ORIG_NORM, ORIG_HALF
    b = {"format": capture.FORMAT, "repo": "selftest/llama", "class": type(m).__name__, "config": c.to_dict(),
         "model": capture.model_facts(m),
         "passes": {"text": {"graph": capture.graph(r), "squares": capture.squares(r), "inputs": capture.input_shapes(kw)}},
         "pass_order": ["text"], "verdict": "FULL", "checks": {}, "max_position_embeddings": 128}
    b["model"]["tie_groups_shipped"] = []
    return b


def rot(fn):                                    # replace apply_rotary_pos_emb with fn(q, k, cos, sin)
    def patch(m):
        LL.apply_rotary_pos_emb = lambda q, k, cos, sin, position_ids=None, unsqueeze_dim=1: fn(
            q, k, cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim))
    return patch


def half(x): return x.shape[-1] // 2
MUTANTS = {
    "swap cos/sin": rot(lambda q, k, c, s: (q * s + ORIG_HALF(q) * c, k * s + ORIG_HALF(k) * c)),
    "negate first half": rot(lambda q, k, c, s: tuple(
        x * c + torch.cat((-x[..., :half(x)], x[..., half(x):]), -1) * s for x in (q, k))),
    "halves not swapped": rot(lambda q, k, c, s: tuple(
        x * c + torch.cat((x[..., :half(x)], -x[..., half(x):]), -1) * s for x in (q, k))),
    "subtract the rotation": rot(lambda q, k, c, s: (q * c - ORIG_HALF(q) * s, k * c - ORIG_HALF(k) * s)),
    "rotation table scaled x2": rot(lambda q, k, c, s: (q * (c * 2) + ORIG_HALF(q) * s, k * (c * 2) + ORIG_HALF(k) * s)),
    "last frequency x7": lambda m: m.model.rotary_emb.inv_freq.__setitem__(-1, m.model.rotary_emb.inv_freq[-1] * 7),
    "middle frequency x1.01": lambda m: m.model.rotary_emb.inv_freq.__setitem__(3, m.model.rotary_emb.inv_freq[3] * 1.01),
    "norm over tokens": lambda m: setattr(LL.LlamaRMSNorm, "forward", lambda self, h: h * torch.rsqrt(
        h.pow(2).mean(-2, keepdim=True) + self.variance_epsilon) * self.weight),
    "norm of mean squared": lambda m: setattr(LL.LlamaRMSNorm, "forward", lambda self, h: h * torch.rsqrt(
        h.mean(-1, keepdim=True).pow(2) + self.variance_epsilon) * self.weight),
    "norm statistic of another value": lambda m: setattr(LL.LlamaRMSNorm, "forward", lambda self, h: (h * 2) * torch.rsqrt(
        h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) * self.weight),
}


def wrong_residual(layer_idx):
    def patch(m):
        def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                    position_embeddings=None, **kw):
            hidden_states = self.input_layernorm(hidden_states)
            residual = hidden_states                       # the residual taken AFTER the norm
            h, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                  past_key_values=past_key_values, use_cache=use_cache,
                                  position_embeddings=position_embeddings, **kw)
            hidden_states = residual + h
            residual = hidden_states
            return residual + self.mlp(self.post_attention_layernorm(hidden_states))
        lay = m.model.layers[layer_idx]; lay.forward = types.MethodType(forward, lay)
    return patch
MUTANTS["residual from the normed input (layer 1)"] = wrong_residual(1)
MUTANTS["residual from the normed input (every layer)"] = lambda m: [wrong_residual(i)(m) for i in range(3)]


def skip_layer_input(m):                              # layer 2 reads layer 0's output instead of layer 1's
    saved = {}
    l0, l2 = m.model.layers[0], m.model.layers[2]
    f0, f2 = l0.forward, l2.forward
    def g0(self, hidden_states, *a, **kw):
        out = f0(hidden_states, *a, **kw); saved["h"] = out; return out
    def g2(self, hidden_states, *a, **kw):
        return f2(saved["h"], *a, **kw)
    l0.forward = types.MethodType(g0, l0); l2.forward = types.MethodType(g2, l2)
MUTANTS["layer 2 reads layer 0's output"] = skip_layer_input


def mlp_down_reads_gate(m):                           # up runs, but the product is skipped: down(act(gate))
    def fwd(self, x):
        self.up_proj(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)))
    for lay in m.model.layers:
        lay.mlp.forward = types.MethodType(fwd, lay.mlp)
MUTANTS["MLP down reads act(gate) only"] = mlp_down_reads_gate


def main():
    fails = []
    def drawn(b):
        try:
            exec_to_ir.recognize(b); return True, ""
        except exec_to_ir.NotDrawn as e:
            return False, str(e)
    for name, kw in (("control", {}), ("control (llama3 scaling)", {"rope": {
            "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 64}})):
        ok, why = drawn(bundle_of(**kw))
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'drawn' if ok else 'REFUSED ' + why[:200]}")
        if not ok: fails.append(name)
    for name, patch in MUTANTS.items():
        ok, why = drawn(bundle_of(patch))
        print(f"{'FAIL' if ok else 'ok  '} {name}: {'DRAWN (a false drawing)' if ok else 'refused: ' + why[:150]}")
        if ok: fails.append(name)
    print(f"\n{len(MUTANTS) + 2 - len(fails)}/{len(MUTANTS) + 2} as expected")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
