"""The vLLM witness recorder's own helpers (F14, Soumil 2026-10-13): the checkpoint split, the library versions, the
witness stamp, the per-run inputs and their fingerprints. Every file the recorder runs is in WITNESS_FILES; a record made
by any other version of them is stale. The grading rules that read witness records live in common.py (not stamped).
common.py re-exports these names, so the main harness uses the very same ckpt_split / runtime."""
import re

AUX = re.compile(r"(\.biases|inv_freq|position_ids|num_batches_tracked|_scales|weight_global_scale|input_global_scale|_scale_inv|weight_scale(_2)?|input_scale|output_scale|k_scale|v_scale|zero_point|\.g_idx|\.qzeros|\.scales|\.SCB|absmax|quant_map|quant_state|nested_|_blocks_scale)$")
INT_DTYPES = {"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"}


def ckpt_split(T):
    """T: {name: (shape, dtype)} -> (weights, aux, packed_flag). Packed-quantized names are
    normalized to the plain weight name (qweight/weight_packed/_blocks -> weight)."""
    weights, aux = {}, {}
    packed = any(d in INT_DTYPES for k, (s, d) in T.items()
                 if not AUX.search(k) and re.search(r"(weight|qweight|weight_packed|_blocks)$", k))
    def quant_scale(k):
        if not k.endswith(".scale"):
            return False
        sib = T.get(k[: -len(".scale")] + ".weight")
        return sib is not None and (sib[1] in INT_DTYPES or str(sib[1]).startswith("F8"))
    def packed_bookkeeping(k):
        """compressed-tensors' pack-quantized schema: <m>.weight_shape (1-D integer, length = the packed tensor's rank)
        and <m>.weight_g_idx sit next to an integer <m>.weight_packed (P0.6 B3); a same-named tensor elsewhere is a weight."""
        for suf in (".weight_shape", ".weight_g_idx"):
            if k.endswith(suf):
                sib = T.get(k[: -len(suf)] + ".weight_packed")
                s_, d_ = T[k]
                return (sib is not None and sib[1] in INT_DTYPES and d_ in INT_DTYPES
                        and (suf != ".weight_shape" or (len(s_) == 1 and s_[0] == len(sib[0]))))
        return False
    for k, (s, d) in T.items():
        if AUX.search(k) or quant_scale(k) or packed_bookkeeping(k):
            aux[k] = (tuple(s), d); continue
        if packed:
            k = re.sub(r"\.(qweight|weight_packed)$", ".weight", k)
            k = re.sub(r"_blocks$", "", k)
        weights[k] = (tuple(s), d)
    return weights, aux, packed


def runtime():
    """Library versions and run settings that can change a verdict (recorded beside grading_id)."""
    import sys, os, platform, importlib
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "transformers", "diffusers", "huggingface_hub", "safetensors", "numpy", "tokenizers", "accelerate",
              "compressed_tensors", "bitsandbytes", "gptqmodel", "awq", "gguf", "torchvision", "sentencepiece", "PIL",
              "mistral_common", "vllm", "scipy"):
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception:
            pass
    v["settings"] = {k: os.environ[k] for k in sorted(os.environ)
                     if k.startswith("BENCH_") or k in ("HF_ENDPOINT", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_XET", "TRANSFORMERS_OFFLINE")}
    return v


# ---- F14: witness identity and inputs ----
WITNESS_SCHEMA = 2
# the recorder's own code: a record made by any other version of these files is stale (re-record). The grading rules
# (common.py, worker_v2, aggregate, ...) are not in it (Soumil 2026-10-13, F14-2).
WITNESS_FILES = ("mtp_witness_run.py", "vllm_witness.py", "witness_io.py", "dag.py", "opclass.py", "lowcost.py", "noweights.py")
WITNESS_STRUCTURAL_INPUTS = frozenset({"positions"})       # functions of the token count only: equal in both runs


def witness_producer_id():
    """Content hash of the witness recorder's code (WITNESS_FILES)."""
    import hashlib, os
    here = os.path.dirname(os.path.abspath(__file__)); h = hashlib.sha256()
    for f in WITNESS_FILES:
        p = os.path.join(here, f)
        h.update(f.encode()); h.update(open(p, "rb").read() if os.path.exists(p) else b"<missing>")
    return h.hexdigest()[:12]


def vllm_source_id(vllm_module):
    """Content hash of the vLLM source tree the witness imported: every file under the package (bytecode excluded)."""
    import hashlib, os
    root = os.path.dirname(os.path.abspath(vllm_module.__file__)); h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, root).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def witness_inputs(n_tokens, hidden, run_index):
    """Run `run_index` of the witness: its own seed (= the run index), so the two stability runs never share inputs."""
    import torch
    g = torch.Generator().manual_seed(int(run_index))
    return {"input_ids": torch.randint(0, 1000, (n_tokens,), generator=g), "positions": torch.arange(n_tokens),
            "hidden_states": torch.randn(n_tokens, hidden, generator=g) * 0.02}


def input_fingerprints(inputs):
    """{name: sha256(dtype, shape, bytes)[:16]} of the tensors a forward received."""
    import hashlib
    out = {}
    for k in sorted(inputs):
        t = inputs[k]; h = hashlib.sha256()
        h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().contiguous().cpu().numpy().tobytes()); out[k] = h.hexdigest()[:16]
    return out
