"""Execution witness for MTP blocks no transformers version runs: vLLM's own MTP module (from source, no compiled
extension, pure-PyTorch 'native' ops), built from the repo's config with zero-storage weights, run under our recorder.
Results are labelled 'executed by vLLM <version>'. Run in .venv_vllm with the vLLM source on PYTHONPATH."""
import os, sys, json, time, socket
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noweights; noweights.install()


def _native_activation(name):
    """Out-variant ops (out, x) mapped to vLLM's OWN pure-PyTorch definitions (forward_native) of the same activation."""
    from vllm.model_executor.layers import activation as A
    table = {"silu_and_mul": lambda x: A.SiluAndMul.forward_native(x),
             "mul_and_silu": (lambda x: A.MulAndSilu.forward_native(x)) if hasattr(A, "MulAndSilu") else None}
    f = table.get(name)
    if f is None:
        return None
    def out_variant(out, x, *a, **k):
        out.copy_(f(x))
    return out_variant


class _NoCompiledOps:
    """vLLM looks some compiled kernels up at layer construction even on CPU paths that never call them (forward_cpu ->
    forward_native). Only those construction-time names resolve, to a stub that RAISES if called; every other name is
    missing, exactly as without the extension (so vLLM's hasattr-guarded registrations stay skipped)."""
    CONSTRUCTION_LOOKUPS = {"silu_and_mul", "mul_and_silu", "gelu_and_mul", "gelu_tanh_and_mul", "fatrelu_and_mul",
                            "swigluoai_and_mul", "gelu_new", "gelu_fast", "gelu_quick"}
    def __getattr__(self, name):
        if name not in self.CONSTRUCTION_LOOKUPS:
            raise AttributeError(name)
        native = _native_activation(name)
        if native is not None:
            return native
        def missing(*a, **k):
            raise RuntimeError(f"vLLM compiled op _C.{name} was called (no compiled extension in this witness)")
        return missing


try:
    torch.ops._C.silu_and_mul
except Exception:
    torch.ops.__dict__["_C"] = _NoCompiledOps()


def _register_cache_write():
    """vLLM writes new keys/values into the paged KV cache with a compiled kernel (_C_cache_ops.reshape_and_cache_flash).
    The witness registers the same op name with a pure-PyTorch scatter (each token's K/V into its slot). This is cache
    plumbing, not model structure; results record it as 'KV-cache write provided by the witness'."""
    import torch.library as L
    try:
        torch.ops._C_cache_ops.reshape_and_cache_flash
        return
    except Exception:
        pass
    lib = L.Library("_C_cache_ops", "DEF")
    lib.define("reshape_and_cache_flash(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale) -> ()")
    def impl(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
        n = slot_mapping.numel()
        kc = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
        vc = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])
        kc.index_copy_(0, slot_mapping.long(), key[:n].to(kc.dtype))
        vc.index_copy_(0, slot_mapping.long(), value[:n].to(vc.dtype))
    lib.impl("reshape_and_cache_flash", impl, "CPU")
    _register_cache_write.lib = lib                       # keep the library alive


def _prefer_flex_attention():
    """vLLM's CPU platform always picks its compiled CPU_ATTN backend for standard attention. The witness uses vLLM's own
    pure-PyTorch FlexAttention backend instead (MLA keeps vLLM's CPU reference backend). Recorded in the result."""
    from vllm.platforms.cpu import CpuPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    orig = CpuPlatform.get_attn_backend_cls.__func__
    def pick(cls, selected_backend, attn_selector_config, num_heads=None):
        if getattr(attn_selector_config, "use_mla", False) or getattr(attn_selector_config, "use_sparse", False):
            return orig(cls, selected_backend, attn_selector_config, num_heads)
        return AttentionBackendEnum.FLEX_ATTENTION.get_path()
    CpuPlatform.get_attn_backend_cls = classmethod(pick)


def _reference_moe():
    """vLLM's CPU MoE kernels are compiled grouped GEMMs. The witness assembles vLLM's own reference MoE pair instead:
    BatchedPrepareAndFinalize + NaiveBatchedExperts (plain PyTorch, documented by vLLM as its reference experts)."""
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as UM
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import BatchedPrepareAndFinalize
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import NaiveBatchedExperts
    def make_ref(quant_config, moe_config, backend=None, experts_cls=None, routing_tables=None, **kw):
        pf = BatchedPrepareAndFinalize(max_num_tokens=64, num_local_experts=moe_config.num_local_experts,
                                       num_dispatchers=1, rank=0)
        ex = NaiveBatchedExperts(moe_config=moe_config, quant_config=quant_config,
                                 max_num_tokens=pf.max_num_tokens_per_rank(), num_dispatchers=1)
        return mk.FusedMoEKernel(pf, ex)
    UM.make_unquantized_moe_kernel = make_ref


def build(repo):
    _prefer_flex_attention()
    _register_cache_write()
    _reference_moe()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    from vllm.config import set_current_vllm_config
    # quantization is storage, not architecture: build unquantized (as the main harness does; scale tensors are
    # auxiliary there). vLLM's documented hf_overrides drops the quantization config.
    def _drop_quant(cfg):
        for c in (cfg, getattr(cfg, "text_config", None)):
            if c is not None and getattr(c, "quantization_config", None) is not None:
                try: delattr(c, "quantization_config")
                except Exception: c.quantization_config = None
        return cfg
    args = EngineArgs(model=repo, skip_tokenizer_init=True, enforce_eager=True, load_format="dummy", hf_overrides=_drop_quant,
                      dtype="float32", trust_remote_code=False,
                      speculative_config={"method": "mtp", "num_speculative_tokens": 1},
                      compilation_config={"custom_ops": ["none"], "mode": 0},   # every custom op on its pure-PyTorch path
                      attention_config={"backend": "FLEX_ATTENTION"})          # vLLM's pure-PyTorch attention backend
    vcfg = args.create_engine_config()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    with set_current_vllm_config(vcfg):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
        ensure_model_parallel_initialized(1, 1)
    draft = vcfg.speculative_config.draft_model_config
    from vllm.model_executor.models.registry import ModelRegistry
    arch = draft.hf_config.architectures[0]
    cls, _ = ModelRegistry.resolve_model_cls([arch], draft)
    import copy
    dcfg = copy.copy(vcfg); dcfg.model_config = draft
    from lowcost import zero_storage
    with set_current_vllm_config(dcfg), zero_storage():
        model = cls(vllm_config=dcfg, prefix="")
    # vLLM's own post-load step (its loader always runs it): picks each layer's CPU routine; without the compiled
    # extension that is vLLM's documented fallback, torch.nn.functional.linear
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    with set_current_vllm_config(dcfg):
        process_weights_after_loading(model, draft, torch.device("cpu"))
    return model.eval(), vcfg, dcfg, arch


if __name__ == "__main__":
    repo = sys.argv[1]
    t = time.time()
    model, vcfg, dcfg, arch = build(repo)
    print("built", arch, type(model).__name__, sum(p.numel() for p in model.parameters()), "params in", round(time.time() - t, 1), "s")
    print([n for n, _ in model.named_parameters()][:12])


def attention_metadata(model, dcfg, n_tokens):
    """What vLLM's engine would hand the attention layers for one request of n_tokens: a KV cache allocated from each
    layer's own spec and bound the way vLLM's runner binds it, and FlexAttention metadata from vLLM's own builder."""
    import vllm.v1.attention.backends.flex_attention as FA
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FA.flex_attention_compiled = flex_attention          # same function, eager: the recorder sees its steps
    if hasattr(FA, "create_block_mask_compiled"): FA.create_block_mask_compiled = create_block_mask
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.config import set_current_vllm_config
    layers = {n: m for n, m in model.named_modules() if hasattr(m, "impl") and hasattr(m, "get_kv_cache_spec")}
    md = {}
    with set_current_vllm_config(dcfg):
        for name, layer in layers.items():
            spec = layer.get_kv_cache_spec(dcfg)
            backend = layer.get_attn_backend()
            nblk = max(1, -(-n_tokens // spec.block_size))
            dcfg.cache_config.num_gpu_blocks = nblk          # what the engine sets after memory profiling
            if "MLA" in type(spec).__name__:
                # MLA's paged layout: one latent vector per token: (num_blocks, block_size, latent + rope size)
                layer.kv_cache = torch.zeros(nblk, spec.block_size, spec.head_size, dtype=spec.dtype)
            else:
                # FlexAttention's paged layout: (num_blocks, num_kv_heads, block_size, 2 * head_size)
                layer.kv_cache = torch.zeros(nblk, spec.num_kv_heads, spec.block_size, 2 * spec.head_size, dtype=spec.dtype)
            dcfg.compilation_config.static_forward_context[layer.layer_name] = layer
            common = CommonAttentionMetadata(
                query_start_loc=torch.tensor([0, n_tokens], dtype=torch.int32), query_start_loc_cpu=torch.tensor([0, n_tokens], dtype=torch.int32),
                seq_lens=torch.tensor([n_tokens], dtype=torch.int32), num_reqs=1, num_actual_tokens=n_tokens, max_query_len=n_tokens,
                max_seq_len=n_tokens, block_table_tensor=torch.arange(nblk, dtype=torch.int32).view(1, -1),
                slot_mapping=torch.arange(n_tokens, dtype=torch.int64), causal=True,
                **({"seq_lens_cpu_upper_bound": torch.tensor([n_tokens], dtype=torch.int32)}
                   if "seq_lens_cpu_upper_bound" in CommonAttentionMetadata.__dataclass_fields__ else {}))
            builder = backend.get_builder_cls()(spec, [layer.layer_name], dcfg, torch.device("cpu"))
            md[layer.layer_name] = builder.build(0, common)
    return md


def _flex_visible_to_recorder():
    """FlexAttention is a higher-order op; under a recording mode torch needs a rule for it. The rule runs torch's own
    eager reference implementation (sdpa_dense) with the recorder active, so vLLM's mask functions and every attention
    step are recorded. torch's documented switch keeps flex_attention out of the compiler."""
    import torch.nn.attention.flex_attention as FAmod
    import importlib
    FH = importlib.import_module("torch._higher_order_ops.flex_attention")   # the module (the name is also the op)
    from dag import DagRecorder
    FAmod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    if getattr(_flex_visible_to_recorder, "done", False):
        return
    @FH.flex_attention.py_impl(DagRecorder)
    def _rule(mode, query, key, value, score_mod, block_mask, scale, kernel_options,
              score_mod_other_buffers=(), mask_mod_other_buffers=()):
        with mode:
            return FH.sdpa_dense(query, key, value, score_mod, block_mask, scale, kernel_options,
                                 score_mod_other_buffers, mask_mod_other_buffers)
    _flex_visible_to_recorder.done = True


def run_forward(model, dcfg, n_tokens=8, attn_metadata=None):
    from vllm.forward_context import set_forward_context
    from vllm.config import set_current_vllm_config
    from dag import DagRecorder, analyse
    hf = dcfg.model_config.hf_config
    H = getattr(hf, "hidden_size", None) or hf.get_text_config().hidden_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 1000, (n_tokens,), generator=g)
    pos = torch.arange(n_tokens)
    hid = torch.randn(n_tokens, H, generator=g) * 0.02          # the target model's last hidden states (values only)
    inputs = {"input_ids": ids, "positions": pos, "hidden_states": hid}
    _flex_visible_to_recorder()
    for m in model.modules():                            # vLLM's direct-call path: attention is not one opaque op
        if hasattr(m, "use_direct_call"):
            m.use_direct_call = True
    rec = DagRecorder(model, inputs)
    slots = {name: torch.arange(n_tokens, dtype=torch.int64) for name in (attn_metadata or {})}
    with set_current_vllm_config(dcfg), set_forward_context(attn_metadata, dcfg, num_tokens=n_tokens, slot_mapping=slots or None), \
            torch.no_grad(), rec:
        out = model(**inputs)
        if callable(getattr(model, "compute_logits", None)):   # the MTP's own output head (vLLM applies it separately)
            model.compute_logits(out[0] if isinstance(out, (tuple, list)) else out)
    rec.remove_hooks()
    return rec, out


def checkpoint_mapping(model, dcfg, header):
    """Which vLLM parameter each checkpoint tensor loads into, by vLLM's own load_weights: every parameter's
    weight_loader is replaced by a recorder (no data copied); checkpoint tensors are fed as shape-only meta tensors."""
    import vllm.model_executor.model_loader.weight_utils as WU
    from vllm.config import set_current_vllm_config
    names = {id(p): n for n, p in model.named_parameters(remove_duplicate=False)}
    hits = {}                                   # checkpoint name -> vLLM param name
    current = {"ckpt": None}
    def recorder(param, *a, **k):
        hits.setdefault(current["ckpt"], set()).add(names.get(id(param), "?"))
    saved = {}
    for n, p in model.named_parameters():
        saved[n] = getattr(p, "weight_loader", None)
        p.weight_loader = recorder
    orig_default = WU.default_weight_loader
    WU.default_weight_loader = recorder
    def feed():
        for k, (shape, dt) in header.items():
            current["ckpt"] = k
            yield k, torch.empty(tuple(shape), device="meta")
    try:
        with set_current_vllm_config(dcfg):
            model.load_weights(feed())
    except Exception as e:
        hits["__error__"] = {f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        WU.default_weight_loader = orig_default
        for n, p in model.named_parameters():
            if saved.get(n) is not None: p.weight_loader = saved[n]
    return {k: sorted(v) for k, v in hits.items()}
