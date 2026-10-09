"""Run the vLLM execution witness for MTP blocks and write results_v2/_witness/vllm/<repo>.json.
Usage (in .venv_vllm, vLLM source on PYTHONPATH): python mtp_witness_run.py <repo> [<repo> ...]"""
import os, sys, json, re, time, resource, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_witness as W
from lowcost import fetch_headers

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v2", "_witness", "vllm")


def norm(nodes):
    return [(re.sub(r"\.\d+", ".N", n.module), n.func) for n in nodes]


STAGE = {"now": "headers"}
REG = {}


def one(repo):
    R = {"repo": repo, "executed_by": "vLLM (source, no compiled extension; custom ops on their pure-PyTorch path)", "t0": time.time()}
    import vllm
    R["vllm_version"] = getattr(vllm, "__version__", "0.30.0-source")
    R["notes"] = ["standard attention: vLLM FlexAttention backend, computed by torch's reference flex implementation (sdpa_dense)",
                  "KV-cache write: pure-PyTorch scatter registered as _C_cache_ops.reshape_and_cache_flash (vLLM's kernel is not compiled)",
                  "MoE: vLLM's reference experts (BatchedPrepareAndFinalize + NaiveBatchedExperts) instead of the compiled CPU grouped GEMM"]
    # does an executing runtime exist? vLLM's own decision: its MTP config override on this model's config
    STAGE["now"] = "registry"
    try:
        import copy
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig
        hc = AutoConfig.from_pretrained(repo, trust_remote_code=False)
        before = list(getattr(hc, "architectures", None) or [])
        after = list(getattr(SpeculativeConfig.hf_config_override(copy.deepcopy(hc)), "architectures", None) or [])
        R["vllm_mtp_arch"] = next((a for a in after if "MTP" in a and a not in before), None); REG["arch"] = R["vllm_mtp_arch"]
    except Exception as e:
        R["vllm_mtp_arch"] = None; R["registry_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    STAGE["now"] = "headers"
    T, _ = fetch_headers(repo)
    mtp = {k: v for k, v in T.items() if k.startswith("mtp.") or ".mtp." in k}
    R["checkpoint_mtp_tensors"] = len(mtp)
    STAGE["now"] = "config"                     # vLLM decides here whether an MTP implementation exists for this model
    model, vcfg, dcfg, arch = W.build(repo)
    R["vllm_class"] = arch
    STAGE["now"] = "run"
    md = W.attention_metadata(model, dcfg, 8)
    rec_a, _ = W.run_forward(model, dcfg, 8, md)
    md = W.attention_metadata(model, dcfg, 8)
    rec_b, _ = W.run_forward(model, dcfg, 8, md)                 # T5: same shape, different values
    import torch
    R["ops"] = len(rec_a.nodes)
    R["attention_ops"] = sum(1 for n in rec_a.nodes if "attn" in n.module.rsplit(".", 1)[-1])
    R["stable_structure"] = norm(rec_a.nodes) == norm(rec_b.nodes)
    from common import ckpt_split
    Wl, Aux, _packed = ckpt_split(T)                              # learned tensors vs auxiliary (quant scales), as the main harness
    R["auxiliary_tensors_not_fed"] = len(Aux)
    mp = W.checkpoint_mapping(model, dcfg, Wl)                    # every learned checkpoint tensor through vLLM's own loader
    T = Wl
    R["mapping_error"] = sorted(mp.pop("__error__", []))
    P = {n: p for n, p in model.named_parameters(remove_duplicate=False)}
    alias = {}
    for n, p in P.items(): alias.setdefault(id(p), set()).add(n)
    used = set(rec_a.used_params)
    for names in alias.values():
        if names & used: used |= names
    R["executed_checkpoint_keys"] = sorted(k for k in T if mp.get(k) and all(t in used for t in mp[k]))
    R["loaded_not_used_keys"] = sorted(k for k in T if mp.get(k) and not all(t in used for t in mp[k]))
    R["mtp_not_loaded"] = sorted(k for k in mtp if not mp.get(k))
    R["mtp_loaded_not_used"] = sorted(k for k in mtp if mp.get(k) and not all(t in used for t in mp[k]))
    R["mtp_executed"] = sorted(k for k in mtp if mp.get(k) and all(t in used for t in mp[k]))
    # closure against the library's declared-unbuilt list is decided in the main environment (apply_witness.py)
    R["closed"] = bool(R["executed_checkpoint_keys"]) and not R["loaded_not_used_keys"] and R["stable_structure"] and not R["mapping_error"]
    R["op_kinds"] = sorted({n.func for n in rec_a.nodes})
    return R


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for repo in sys.argv[1:]:
        t = time.time(); REG.clear()
        try:
            R = one(repo)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            R = {"repo": repo, "closed": False, "failed_stage": STAGE["now"], "vllm_mtp_arch": REG.get("arch"), "error": f"{type(e).__name__}: {str(e)[:300]}",
                 "where": f"{tb[-1].filename.split('vllm-0.30.0/')[-1]}:{tb[-1].lineno}" if tb else ""}
        R["secs"] = round(time.time() - t, 1); R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        R.pop("t0", None)
        json.dump(R, open(os.path.join(OUT, repo.replace("/", "__") + ".json"), "w"), indent=1)
        print(f"{repo:44} closed={R.get('closed')} executed_keys={len(R.get('executed_checkpoint_keys', []))} loaded_unused={len(R.get('loaded_not_used_keys', []))} "
              f"ops={R.get('ops')} attn_ops={R.get('attention_ops')} stable={R.get('stable_structure')} {R.get('error', '')[:120]}", flush=True)
