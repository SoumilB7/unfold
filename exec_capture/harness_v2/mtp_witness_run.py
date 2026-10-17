"""Run the vLLM execution witness for MTP blocks and write results_v2/_witness/vllm/<repo>.json.
Usage (in .venv_vllm, vLLM source on PYTHONPATH): python mtp_witness_run.py <repo> [<repo> ...]"""
import os, sys, json, re, time, resource, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_witness as W
from lowcost import fetch_headers
import witness_io
PRODUCER_ID = witness_io.witness_producer_id()   # hashed once, when the recorder's code is loaded (F14)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v2", "_witness", "vllm")


def norm(nodes):
    return [(re.sub(r"\.\d+", ".N", n.module), n.func) for n in nodes]


STAGE = {"now": "headers"}
REG = {}
CUR = {}                                         # the record being built: an error keeps everything already shown (F14)


def one(repo):
    R = CUR["R"] = {"repo": repo, "executed_by": "vLLM (source, no compiled extension; custom ops on their pure-PyTorch path)", "t0": time.time()}
    import vllm, noweights, witness_io
    from huggingface_hub import HfApi
    # identity (G6, F14): what made this record; the grader uses no statement from a record without it
    R["schema"] = witness_io.WITNESS_SCHEMA
    R["producer_id"] = PRODUCER_ID
    R["vllm_version"] = getattr(vllm, "__version__", None)           # unknown stays unknown (an identity gap)
    R["vllm_source_id"] = witness_io.vllm_source_id(vllm)
    R["runtime"] = witness_io.runtime()
    R["settings"] = dict(W.SETTINGS, env={k: os.environ[k] for k in sorted(os.environ) if k.startswith("VLLM_")})
    R["revision"] = HfApi().model_info(repo).sha             # the witness is bound to one checkpoint revision (G6)
    noweights.pin(repo, R["revision"])
    R["notes"] = ["standard attention: vLLM FlexAttention backend, computed by torch's reference flex implementation (sdpa_dense)",
                  "KV-cache write: pure-PyTorch scatter registered as _C_cache_ops.reshape_and_cache_flash (vLLM's kernel is not compiled)",
                  "MoE: vLLM's reference experts (BatchedPrepareAndFinalize + NaiveBatchedExperts) instead of the compiled CPU grouped GEMM"]
    # does an executing runtime exist? vLLM's own decision: its MTP config override on this model's config
    STAGE["now"] = "registry"
    try:
        import copy
        from vllm.transformers_utils.config import get_config
        from vllm.config.speculative import SpeculativeConfig
        # the config exactly as vLLM's engine parses it (e.g. it fills a missing `architectures` from the model type)
        hc = get_config(repo, trust_remote_code=False, revision=R["revision"])
        snap = lambda c: {"architectures": list(getattr(c, "architectures", None) or []),
                          "model_type": getattr(c, "model_type", None), "n_predict": getattr(c, "n_predict", None)}
        before, after = snap(hc), snap(SpeculativeConfig.hf_config_override(copy.deepcopy(hc)))
        # vLLM's own evidence: its MTP override replaces the architecture with its draft module's (never a name test).
        # The whole before/after is recorded: an override that changes the config without adding an architecture is
        # not a "no MTP module" answer (the grader re-derives that)
        R["registry_override"] = {"before": before, "after": after}
        R["vllm_mtp_arch"] = next((a for a in after["architectures"] if a not in before["architectures"]), None)
        REG["arch"] = R["vllm_mtp_arch"]
        R["registry_checked"] = True              # vLLM's registry decision completed
    except Exception as e:
        R["vllm_mtp_arch"] = None; R["registry_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        _net = noweights.network_error(e)
        if _net:                                  # (F14 e) the registry was not answered because of the network
            R["registry_network_error"] = str(_net)[:200]
    STAGE["now"] = "headers"
    T, _ = fetch_headers(repo)
    mtp = {k: v for k, v in T.items() if k.startswith("mtp.") or ".mtp." in k}
    R["checkpoint_mtp_tensors"] = len(mtp)
    STAGE["now"] = "config"                     # vLLM decides here whether an MTP implementation exists for this model
    model, vcfg, dcfg, arch = W.build(repo)
    R["vllm_class"] = arch
    STAGE["now"] = "run"
    n_tok = W.SETTINGS["n_tokens"]; seed_a, seed_b = W.SETTINGS["run_seeds"]
    md = W.attention_metadata(model, dcfg, n_tok)
    rec_a, out_a = W.run_forward(model, dcfg, n_tok, md, seed_a)
    md = W.attention_metadata(model, dcfg, n_tok)
    rec_b, out_b = W.run_forward(model, dcfg, n_tok, md, seed_b)      # T5: same shape, different values (own seed)
    import torch
    R["inputs"] = {"a": {"seed": rec_a.seed, "n_tokens": n_tok, "fingerprints": rec_a.input_fingerprints},
                   "b": {"seed": rec_b.seed, "n_tokens": n_tok, "fingerprints": rec_b.input_fingerprints}}
    R["distinct_inputs"] = all(rec_a.input_fingerprints[k] != rec_b.input_fingerprints[k]
                               for k in rec_a.input_fingerprints if k not in witness_io.WITNESS_STRUCTURAL_INPUTS)
    R["ops"] = len(rec_a.nodes)
    R["attention_ops"] = sum(1 for n in rec_a.nodes if "attn" in n.module.rsplit(".", 1)[-1])
    R["stable_structure"] = norm(rec_a.nodes) == norm(rec_b.nodes)
    from witness_io import ckpt_split
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
    # the MTP tensors under the names fed to vLLM's loader (ckpt_split names: packed weights normalised, auxiliary
    # quantization tensors not fed and not listed)
    _norm = (lambda k: re.sub(r"_blocks$", "", re.sub(r"\.(qweight|weight_packed)$", ".weight", k))) if _packed else (lambda k: k)
    mtp_l = sorted({_norm(k) for k in mtp} & set(T))
    R["mtp_not_loaded"] = sorted(k for k in mtp_l if not mp.get(k))
    R["mtp_loaded_not_used"] = sorted(k for k in mtp_l if mp.get(k) and not all(t in used for t in mp[k]))
    R["mtp_executed"] = sorted(k for k in mtp_l if mp.get(k) and all(t in used for t in mp[k]))
    # closure against the library's declared-unbuilt list is decided by the grader (runtime_credit_needed)
    # (input distinctness is recorded separately and re-derived by the grader: an undistinct pair is a G5a unknown there,
    #  not a failed run)
    R["closed"] = bool(R["executed_checkpoint_keys"]) and not R["loaded_not_used_keys"] and R["stable_structure"] and not R["mapping_error"]
    R["op_kinds"] = sorted({n.func for n in rec_a.nodes})
    # transparency (INTENT.md G4): full op names, every op transparent by registry evidence, nothing from outside
    from dag import analyse
    R["op_full"] = sorted(set(rec_a.op_full))
    # F10: the one classifier, per recorded node of BOTH runs (dag.analyse), exactly as the main harness grades T4
    _a = analyse(rec_a, out_a)
    _b = analyse(rec_b, out_b)
    # (the reason the RECORDER gave, with the call's tensor types: a subclass-dispatched aten::mm is reported as such,
    #  never re-derived from the bare name as "core"; Fable N1)
    R["opaque_ops"] = {f: x["opaque_reasons"].get(f) or "opaque" for x in (_a, _b) for f in x["opaque_ops"]}
    R["opacity_unknown_ops"] = sorted(set(_a["opacity_unknown_ops"]) | set(_b["opacity_unknown_ops"]))
    R["decomposition_disagrees"] = sorted(set(_a["decomposition_disagrees"]) | set(_b["decomposition_disagrees"]))
    R["decomposed_on_record"] = dict(rec_a.decomposed)
    R["external_nonscalar"] = _a["external_nonscalar"]
    R["ir_torch_wrap"] = False
    import opclass
    # F10-24: rejected vocabulary entries (this venv's torch) are a NOTE on the record, never part of its transparency:
    # a dropped entry credits nothing, and every recorded op above was graded on its own evidence
    R["vocabulary_errors"] = list(opclass.VOCAB_ERRORS)
    from witness_io import witness_transparent
    R["transparent"] = witness_transparent(R)
    R["revision_pin"] = _pin_log()
    return R


def _pin_log():
    """What the hub reads of this run named (noweights.PIN_LOG): shows the record's revision, not only declares it."""
    try:
        import noweights
        return {"pinned": noweights.PIN_LOG["pinned"], "other_revision": list(noweights.PIN_LOG["other_revision"])}
    except Exception:
        return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for repo in sys.argv[1:]:
        t = time.time(); REG.clear(); CUR.clear(); STAGE["now"] = "identity"
        import noweights
        noweights.PIN_LOG.update(pinned=0, other_revision=[])     # each record's pin log is its own repo's
        try:
            R = one(repo)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            # the partial record keeps its identity and the registry result (registry_checked / registry_error), so
            # "the registry answered no MTP, then the build failed" stays distinct from "the registry never answered"
            R = CUR.get("R") or {"repo": repo}
            if "arch" in REG:                     # only an answer the registry actually gave (never a default None)
                R.setdefault("vllm_mtp_arch", REG["arch"])
            if "revision" in R:
                R["revision_pin"] = _pin_log()
            try:                                  # a network / hub failure is never evidence about the model (F14 e)
                import noweights as _nw
                _net = _nw.network_error(e)
            except Exception:
                _net = None
            if _net:
                R["network_error"] = str(_net)[:200]
            R.update({"closed": False, "failed_stage": STAGE["now"], "error": f"{type(e).__name__}: {str(e)[:300]}",
                      "where": f"{tb[-1].filename.split('vllm-0.30.0/')[-1]}:{tb[-1].lineno}" if tb else ""})
        R["secs"] = round(time.time() - t, 1); R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        R.pop("t0", None)
        json.dump(R, open(os.path.join(OUT, repo.replace("/", "__") + ".json"), "w"), indent=1)
        print(f"{repo:44} closed={R.get('closed')} executed_keys={len(R.get('executed_checkpoint_keys', []))} loaded_unused={len(R.get('loaded_not_used_keys', []))} "
              f"ops={R.get('ops')} attn_ops={R.get('attention_ops')} stable={R.get('stable_structure')} {R.get('error', '')[:120]}", flush=True)
