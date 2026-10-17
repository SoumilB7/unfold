"""Adversarial mutants for exec_to_ir (from the 2026-10-09 cross-examination). Each computes something different from a
standard pre-norm Llama / Qwen3 layer. Every mutant must be REFUSED except:
  BENIGN     - the drawing is still true (the change is a renaming or a value the drawing does not show);
  KNOWN_OPEN - a hole the recognizer cannot close by matching (values reordered / broadcast through reshapes); it is
               closed by the functional-equivalence oracle (z-docs/alternative-approach/METHOD_2026-10-09_why_we_miss.md).
A KNOWN_OPEN mutant that becomes refused is reported so it can be moved out of the list; any other drawn mutant fails.
    python3 selftest_mutants.py [name-substring ...] [--unfold-pkg PATH]      # exit 0 = as expected
"""
import os, sys, types, math
CODE = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else \
    "/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-exec-capture/exec_capture/harness_v2"
SEL = [a for a in sys.argv[1:] if not os.path.isdir(a)]
sys.path.insert(0, CODE); sys.path.insert(1, "/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-exec-capture/exec_capture/harness_v2")
sys.path.insert(0, "/Users/soumil/Code/Projects/Understand/llmvisualizer/model-benchmark/renderer_snapshot")
import torch, transformers
from torch import nn
from transformers.models.llama import modeling_llama as LL
from transformers.models.qwen3 import modeling_qwen3 as QQ
import capture, exec_to_ir

SAVE = {M: dict(vars(M)) for M in (LL, QQ)}
SAVE_CLS = {(c, "forward"): c.forward for c in (LL.LlamaRMSNorm, QQ.Qwen3RMSNorm, LL.LlamaRotaryEmbedding,
                                                 QQ.Qwen3RotaryEmbedding, LL.LlamaAttention, QQ.Qwen3Attention,
                                                 LL.LlamaMLP, QQ.Qwen3MLP)}


def restore():
    for M, d in SAVE.items():
        for k in ("apply_rotary_pos_emb", "rotate_half", "eager_attention_forward", "repeat_kv"):
            if k in d: setattr(M, k, d[k])
    for (c, a), f in SAVE_CLS.items(): setattr(c, a, f)


def bundle_of(patch=None, fam="llama", S=5, heads=4, kv=2, hd=16, layers=3, rope=None, ids=None):
    restore()
    torch.manual_seed(0)
    kw_ = dict(vocab_size=32, hidden_size=64, intermediate_size=96, num_hidden_layers=layers, num_attention_heads=heads,
               num_key_value_heads=kv, head_dim=hd, max_position_embeddings=128, **({"rope_parameters": rope} if rope else {}))
    if fam == "llama":
        c = transformers.LlamaConfig(**kw_); cls = transformers.LlamaForCausalLM
    else:
        c = transformers.Qwen3Config(**kw_); cls = transformers.Qwen3ForCausalLM
    m = cls._from_config(c, attn_implementation="eager", dtype=torch.float32).eval()
    if patch: patch(m)
    kw = {"input_ids": torch.tensor([ids or list(range(1, S + 1))]), "use_cache": False}
    r = capture.CaptureRecorder(m, kw, skip=False)
    try:
        with torch.no_grad(), r:
            m(**kw)
    finally:
        r.remove_hooks(); restore()
    b = {"format": capture.FORMAT, "repo": f"selftest/{fam}", "class": type(m).__name__, "config": c.to_dict(),
         "model": capture.model_facts(m),
         "passes": {"text": {"graph": capture.graph(r), "squares": capture.squares(r), "inputs": capture.input_shapes(kw)}},
         "pass_order": ["text"], "verdict": "FULL", "checks": {}, "max_position_embeddings": 128}
    b["model"]["tie_groups_shipped"] = []
    return b


class Wrap(nn.Module):
    def __init__(self, inner, pre=None, post=None):
        super().__init__(); self.inner = inner; self.pre = pre; self.post = post
    def forward(self, x):
        y = self.inner(self.pre(x) if self.pre else x)
        return self.post(y) if self.post else y
def H(x): return x.shape[-1] // 2
def rot_half(x): return torch.cat((-x[..., H(x):], x[..., :H(x)]), -1)
MUT = {}
def mutant(name, **bkw):
    def deco(f): MUT[name] = (f, bkw); return f
    return deco


def set_rot(fn):
    def p(m):
        for M in (LL, QQ):
            M.apply_rotary_pos_emb = lambda q, k, cos, sin, position_ids=None, unsqueeze_dim=1: fn(
                q, k, cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim))
    return p


def set_eager(fn):
    def p(m):
        for M in (LL, QQ): M.eager_attention_forward = fn
    return p


def eager(transform_scores=None, softmax_dim=-1, weights_T=False, kfirst=False, rep=None, scale_mul=1.0, v_tf=None):
    def f(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        rk = rep or LL.repeat_kv
        k_ = rk(key, module.num_key_value_groups); v_ = rk(value, module.num_key_value_groups)
        if v_tf: v_ = v_tf(v_)
        if kfirst: w = torch.matmul(k_, query.transpose(2, 3)) * scaling
        else: w = torch.matmul(query, k_.transpose(2, 3)) * (scaling * scale_mul)
        if attention_mask is not None: w = w + attention_mask
        w = nn.functional.softmax(w, dim=softmax_dim, dtype=torch.float32).to(query.dtype)
        if weights_T: w = w.transpose(-1, -2)
        out = torch.matmul(w, v_).transpose(1, 2).contiguous()
        return out, w
    return f


# ---------------- controls
mutant("CONTROL llama")(None)
mutant("CONTROL qwen3", fam="qwen3")(None)
mutant("CONTROL llama MHA S=16", kv=4, S=16)(None)

# ---------------- attention core
mutant("softmax over queries (dim=-2)")(set_eager(eager(softmax_dim=-2)))
mutant("probabilities transposed before @V (S=8)", S=8)(set_eager(eager(weights_T=True)))
mutant("scores = K @ Q^T (MHA)", kv=4)(set_eager(eager(kfirst=True)))
mutant("score scale x 1/sqrt(d) twice (scale drawn as fact?)")(set_eager(eager(scale_mul=0.25)))
def rep_tile(h, n):
    b, k, s, d = h.shape
    if n == 1: return h
    return h[:, None].expand(b, n, k, s, d).reshape(b, k * n, s, d)
mutant("GQA repeat tiled instead of interleaved")(set_eager(eager(rep=rep_tile)))
mutant("V of token 0 for every token (slice+expand)")(set_eager(eager(v_tf=lambda v: v[:, :, :1].expand_as(v))))

# ---------------- RoPE
mutant("positions arange(1): one position for all tokens")(lambda m: [setattr(type(m.model.rotary_emb), "forward",
    (lambda orig: lambda self, x, position_ids: orig(self, x, torch.arange(1)[None]))(type(m.model.rotary_emb).forward))])
mutant("positions offset +3")(lambda m: setattr(type(m.model.rotary_emb), "forward",
    (lambda orig: lambda self, x, position_ids: orig(self, x, position_ids + 3))(type(m.model.rotary_emb).forward)))
def angle_doubled(m):
    def fwd(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 2)
        pos = position_ids[:, None, :].float().expand(-1, 2, -1)
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)
    type(m.model.rotary_emb).forward = fwd
mutant("angle doubled via matmul contraction width 2")(angle_doubled)
def table_T(m):
    def fwd(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        freqs = (inv @ position_ids[:, None, :].float())        # [1, d/2, S], NOT transposed back
        emb = torch.cat((freqs, freqs), dim=-1)                 # requires S == d/2 to broadcast
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)
    type(m.model.rotary_emb).forward = fwd
mutant("angle table not transposed (pos/freq axes swapped, S=d/2=8)", S=8)(table_T)
mutant("rotate_half on a head/token-scrambled copy of q,k", S=8)(set_rot(lambda q, k, c, s: tuple(
    x * c + rot_half(x.transpose(1, 2).reshape(x.shape)) * s for x in (q, k))))
mutant("K-only: sin/cos swapped on K")(set_rot(lambda q, k, c, s: (q * c + rot_half(q) * s, k * s + rot_half(k) * c)))
mutant("K-only: rotation skipped on K")(set_rot(lambda q, k, c, s: (q * c + rot_half(q) * s, k)))
mutant("Q of token 0 broadcast before rope")(set_rot(lambda q, k, c, s: (lambda q2: (q2 * c + rot_half(q2) * s,
                                                                               k * c + rot_half(k) * s))(q[:, :, :1].expand_as(q))))
mutant("llama3 buffer, config says other params", rope={"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
       "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64})(
    lambda m: m.model.rotary_emb.inv_freq.copy_(transformers.modeling_rope_utils._compute_llama3_parameters(
        transformers.LlamaConfig(head_dim=16, num_attention_heads=4, hidden_size=64, rope_parameters={
            "rope_type": "llama3", "rope_theta": 500000.0, "factor": 4.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 64}))[0]))

# ---------------- norms
def set_norm(fn, fam_cls=(LL.LlamaRMSNorm, QQ.Qwen3RMSNorm)):
    def p(m):
        for c in fam_cls: c.forward = fn
    return p
mutant("norm eps = 1.0")(set_norm(lambda self, h: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1.0) * self.weight))
mutant("norm weight applied before scaling")(set_norm(lambda self, h: (h * self.weight) * torch.rsqrt(
    h.pow(2).mean(-1, keepdim=True) + 1e-6)))
mutant("norm scales token-0 copy of its input")(set_norm(lambda self, h: h[:, :1].expand_as(h) * torch.rsqrt(
    h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight))
mutant("norm statistic of token 0 only")(set_norm(lambda self, h: h * torch.rsqrt(
    h[:, :1].expand_as(h).pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight))
mutant("norm: rsqrt of mean of |x| (abs instead of pow 2)")(set_norm(lambda self, h: h * torch.rsqrt(
    h.abs().mean(-1, keepdim=True) + 1e-6) * self.weight))
mutant("norm: x * sqrt(stat) (mul with sqrt)")(set_norm(lambda self, h: h * torch.sqrt(
    h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight))
mutant("norm: pow 2 then pow 0.5? x / sqrt(mean x^2) (legit RMS, sanity)")(set_norm(lambda self, h: h / torch.sqrt(
    h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight))
mutant("norm: weight is 2*w")(set_norm(lambda self, h: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * (2 * self.weight)))

# ---------------- QK-norm (qwen3)
def qwen_attn(order="kafter", junk=None, k_ln=False):
    def p(m):
        if k_ln:
            for lay in m.model.layers: lay.self_attn.k_norm = nn.LayerNorm(16, eps=1e-6)
            return
        def fwd(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
            ish = hidden_states.shape[:-1]; hs = (*ish, -1, self.head_dim)
            q = self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
            cos, sin = position_embeddings
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            q = q * cos + rot_half(q) * sin
            k = k * cos + rot_half(k) * sin
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
            if junk: k = junk(k)
            o, w = QQ.eager_attention_forward(self, q, k, v, attention_mask, scaling=self.scaling, dropout=0.0, **kw)
            return self.o_proj(o.reshape(*ish, -1).contiguous()), w
        for lay in m.model.layers: lay.self_attn.forward = types.MethodType(fwd, lay.self_attn)
    return p
mutant("qwen3: K-norm after RoPE", fam="qwen3")(qwen_attn())
mutant("qwen3: K-norm after RoPE + tanh(k)", fam="qwen3")(qwen_attn(junk=torch.tanh))
mutant("qwen3: K-norm is LayerNorm (Q is RMSNorm)", fam="qwen3")(qwen_attn(k_ln=True))

# ---------------- wiring / layers
def share_q(m):
    for lay in m.model.layers[1:]: lay.self_attn.q_proj.weight = m.model.layers[0].self_attn.q_proj.weight
mutant("layers 1,2 share layer 0's q_proj weight")(share_q)
def resid_tok0(m):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                position_embeddings=None, **kw):
        residual = hidden_states[:, :1].expand_as(hidden_states)
        h, _ = self.self_attn(hidden_states=self.input_layernorm(hidden_states), attention_mask=attention_mask,
                              position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                              position_embeddings=position_embeddings, **kw)
        hidden_states = residual + h
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    for lay in m.model.layers: lay.forward = types.MethodType(forward, lay)
mutant("residual reads token 0 of layer input (slice+expand)")(resid_tok0)
def o_tok0(m):
    for lay in m.model.layers:
        op = lay.self_attn.o_proj
        lay.self_attn.o_proj = Wrap(op, pre=lambda x: x[:, :1].expand_as(x))
mutant("o_proj fed token 0 of weighted sum (wrapper module)")(o_tok0)
def mlp_rev(m):
    def fwd(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x.flip(1)))
    for lay in m.model.layers: lay.mlp.forward = types.MethodType(fwd, lay.mlp)
mutant("MLP up reads token-reversed input (flip)")(mlp_rev)
def mlp_perm(m):
    def fwd(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x.transpose(1, 2).reshape(x.shape)))
    for lay in m.model.layers: lay.mlp.forward = types.MethodType(fwd, lay.mlp)
mutant("MLP up reads scrambled input (transpose+reshape)")(mlp_perm)
def mlp_swap_act(m):
    def fwd(self, x):
        return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))
    for lay in m.model.layers: lay.mlp.forward = types.MethodType(fwd, lay.mlp)
mutant("MLP act on up instead of gate (naming only, expect drawn=ok)")(mlp_swap_act)
def head_tok0(m):
    lm = m.lm_head
    m.lm_head = Wrap(lm, pre=lambda x: x[:, :1].expand_as(x))
mutant("lm_head reads token 0 of final norm output")(head_tok0)
def emb_scale(m):
    e = m.model.embed_tokens
    m.model.embed_tokens = Wrap(e, post=lambda t: t.flip(1))
mutant("embeddings reversed in token order (flip)")(emb_scale)
def emb_perm(m):
    e = m.model.embed_tokens
    m.model.embed_tokens = Wrap(e, post=lambda t: t.transpose(1, 2).reshape(t.shape))
mutant("embeddings scrambled (transpose+reshape)")(emb_perm)


BENIGN = {"scores = K @ Q^T (MHA)", "score scale x 1/sqrt(d) twice (scale drawn as fact?)",
          "norm: pow 2 then pow 0.5? x / sqrt(mean x^2) (legit RMS, sanity)", "norm: weight is 2*w",
          "MLP act on up instead of gate (naming only, expect drawn=ok)",
          "o_proj/down_proj as addmm with bias, alpha=1 (expect drawn: bias)"}
KNOWN_OPEN = {"probabilities transposed before @V (S=8)", "GQA repeat tiled instead of interleaved",
              "angle doubled via matmul contraction width 2", "angle table not transposed (pos/freq axes swapped, S=d/2=8)",
              "rotate_half on a head/token-scrambled copy of q,k", "MLP up reads scrambled input (transpose+reshape)",
              "embeddings scrambled (transpose+reshape)", "cos/sin broadcast over HEADS not positions (unsqueeze_dim=2, S=heads=4)"}


def main():
    fails, still_open, closed = [], [], []
    for name, (patch, bkw) in MUT.items():
        if SEL and not any(s_.lower() in name.lower() for s_ in SEL): continue
        try:
            b = bundle_of(patch, **bkw)
        except Exception as e:
            print(f"FAIL BUILD {name}: {type(e).__name__} {str(e)[:150]}"); fails.append(name); continue
        try:
            exec_to_ir.recognize(b); drawn, why = True, ""
        except exec_to_ir.NotDrawn as e:
            drawn, why = False, str(e)[:150]
        if name.startswith("CONTROL") or name in BENIGN:
            tag = "ok  " if drawn else "FAIL"
            if not drawn: fails.append(name)
        elif name in KNOWN_OPEN:
            tag = "OPEN" if drawn else "CLOSED (move out of KNOWN_OPEN)"
            (still_open if drawn else closed).append(name)
        else:
            tag = "FAIL" if drawn else "ok  "
            if drawn: fails.append(name)
        print(f"{tag} {'drawn  ' if drawn else 'refused'} {name}{': ' + why if why else ''}", flush=True)
    print(f"\n{len(fails)} unexpected, {len(still_open)} known-open holes still drawn, {len(closed)} newly closed")
    sys.exit(1 if fails else 0)

mutant("cos/sin broadcast over HEADS not positions (unsqueeze_dim=2, S=heads=4)", S=4, heads=4, kv=4)(
    lambda m: [setattr(M, "apply_rotary_pos_emb", lambda q, k, cos, sin, position_ids=None, unsqueeze_dim=1: (
        q * cos.unsqueeze(2) + rot_half(q) * sin.unsqueeze(2), k * cos.unsqueeze(2) + rot_half(k) * sin.unsqueeze(2)))
     for M in (LL, QQ)])
mutant("softmax dim=-2 + probabilities transposed (S=8)", S=8)(set_eager(eager(softmax_dim=-2, weights_T=True)))

def eager_alpha0(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k_ = LL.repeat_kv(key, module.num_key_value_groups); v_ = LL.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k_.transpose(2, 3)) * scaling
    w = torch.add(w, attention_mask, alpha=0.0)          # the mask is multiplied by 0: unmasked attention
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(w, v_).transpose(1, 2).contiguous(), w
mutant("mask added with alpha=0 (unmasked attention)")(set_eager(eager_alpha0))
class AddmmLinear(nn.Module):
    def __init__(self, lin, alpha):
        super().__init__(); self.lin = lin; self.alpha = alpha
        if lin.bias is None: lin.bias = nn.Parameter(torch.zeros(lin.out_features))
    def forward(self, x):
        y = torch.addmm(self.lin.bias, x.reshape(-1, x.shape[-1]), self.lin.weight.t(), alpha=self.alpha)
        return y.view(*x.shape[:-1], -1)
def addmm_all(alpha):
    def p(m):
        for lay in m.model.layers:
            for mod, nm_ in ((lay.self_attn, "o_proj"), (lay.mlp, "down_proj")):
                setattr(mod, nm_, AddmmLinear(getattr(mod, nm_), alpha))
    return p
mutant("o_proj/down_proj as addmm with bias, alpha=1 (expect drawn: bias)")(addmm_all(1.0))
mutant("o_proj/down_proj as addmm with bias, alpha=3")(addmm_all(3.0))

def mlp_both_tok0(m):
    def fwd(self, x):
        x = x[:, :1].expand_as(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    for lay in m.model.layers: lay.mlp.forward = types.MethodType(fwd, lay.mlp)
mutant("MLP gate AND up read token 0 of the normed input")(mlp_both_tok0)
def attn_in_tok0(m):
    for lay in m.model.layers:
        a = lay.self_attn; f0 = a.forward
        a.forward = (lambda f0_: lambda hidden_states, *aa, **kk: f0_(hidden_states[:, :1].expand_as(hidden_states), *aa, **kk))(f0)
mutant("Q,K,V all read token 0 of the normed input")(attn_in_tok0)

if __name__ == "__main__":
    main()
