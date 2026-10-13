"""Negative controls for the small grading rules fixed in EXECUTION_PLAN_2026-10-12 P0.6 (dossier:
model-benchmark/_reruns/p06_research). Each control states its rule. `--legacy` loads the same modules as they are at git
HEAD (before the uncommitted fix) to show the controls catch the old behaviour. No network access is made.

    python3 selftest_rules.py [--legacy]      # exit 0 = every control as expected
"""
import os, sys, subprocess, types, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def module(name, legacy):
    """The harness module, or its committed HEAD version when legacy."""
    if not legacy:
        return importlib.import_module(name)
    src = subprocess.run(["git", "show", f"HEAD:exec_capture/harness_v2/{name}.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    m = types.ModuleType(name + "_head"); m.__file__ = os.path.join(HERE, name + ".py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def main():
    legacy = "--legacy" in sys.argv
    bad = 0
    def check(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}" + ("" if ok else f" (want {want!r})"))

    # ---- D1b-C8: a failure raised INSIDE the HTTP client is network (never evidence about a model); a server's answer
    #      (HTTPStatusError, e.g. 404) and a model error with the same text are not ----
    import httpx, torch
    NW = module("noweights", legacy)
    cases = {}
    c = httpx.Client(); c.close()
    try: c.get("https://huggingface.co/api/models/gpt2")       # raises before any request: the client is closed
    except Exception as e: cases["closed httpx client"] = (e, True)
    try:
        try: c.get("https://huggingface.co/x")
        except Exception as e0: raise OSError("Can't load tokenizer for 'x'") from e0
    except Exception as e: cases["OSError wrapping a closed-client error"] = (e, True)
    try: torch.zeros(2, 3) @ torch.zeros(2, 3)
    except Exception as e: cases["model RuntimeError (torch shape)"] = (e, False)
    try:
        req = httpx.Request("GET", "https://huggingface.co/x"); httpx.Response(404, request=req).raise_for_status()
    except Exception as e: cases["404 HTTPStatusError (a server answer)"] = (e, False)
    try:
        def model_code(): raise RuntimeError("Cannot send a request, as the client has been closed.")
        model_code()
    except Exception as e: cases["the same message raised by non-httpx code"] = (e, False)
    try: httpx.Client().get(12345)                            # wrong argument type: a caller bug raised inside httpx
    except Exception as e: cases["TypeError raised inside httpx (caller bug)"] = (e, False)
    for k, (e, want) in cases.items():
        check(f"D1b-C8 network: {k}", bool(NW.network_error(e)), want)

    # ---- P0.1-F9b: an exception AFTER a verdict was set (revision pin / unknown promotion never ran) never leaves a
    #      certified FULL*; an observed failure stays; before grading it is FAIL (both workers' _last_resort) ----
    CM = module("common", legacy)
    lr = getattr(CM, "after_grading_exception", None)
    def run(verdict, unknowns):
        R = {"verdict": verdict} if verdict else {}
        U = list(unknowns)
        if lr: lr(R, U, "PermissionError: [Errno 1] Operation not permitted")
        else: R["verdict"] = R.get("verdict") or "FAIL"               # HEAD: _last_resort keeps any verdict
        return R["verdict"], any(u.get("predicate") == "G6" for u in R.get("unknowns", []))
    check("F9b FULL then an exception -> FULL_UNVERIFIED with a G6 unknown", run("FULL", []), ("FULL_UNVERIFIED", True))
    check("F9b FULL_LEFTOVERS then an exception -> FULL_UNVERIFIED", run("FULL_LEFTOVERS", [])[0], "FULL_UNVERIFIED")
    check("F9b FULL with pending unknowns then an exception -> FULL_UNVERIFIED", run("FULL", [{"predicate": "G1"}])[0], "FULL_UNVERIFIED")
    check("F9b PARTIAL then an exception -> PARTIAL (observed failure stays)", run("PARTIAL", [])[0], "PARTIAL")
    check("F9b no verdict yet -> FAIL", run(None, [])[0], "FAIL")
    def run_ref(verdict):
        R = {"verdict": verdict, "library_refuses_checkpoint": True}
        if lr: lr(R, [], "PermissionError")
        return R["verdict"]
    check("R-REFUSE holds through a crash after grading (FULL_UNVERIFIED path never wins)", run_ref("FULL"), "FAIL")

    # ---- B3: compressed-tensors bookkeeping (`<m>.weight_shape`, `<m>.weight_g_idx`) is auxiliary ONLY next to an
    #      integer `<m>.weight_packed` (schema evidence), never a learned weight fed to a runtime ----
    H = {"a.weight_packed": ((8, 2), "I32"), "a.weight_shape": ((2,), "I64"), "a.weight_scale": ((8, 1), "F16"),
         "shape_head.weight_shape": ((4, 4), "F32"), "b.weight_shape": ((2,), "I64"), "b.weight": ((4, 4), "F32")}
    w_, aux_, _ = CM.ckpt_split(H)
    check("B3 weight_shape next to an int weight_packed is auxiliary", "a.weight_shape" in aux_, True)
    check("B3 a learned tensor named weight_shape (float 4x4) stays a weight", "shape_head.weight_shape" in w_, True)
    check("B3 weight_shape with no weight_packed sibling stays a weight", "b.weight_shape" in w_, True)

    # ---- B4 + C7: ONE stored-constant rule on every path. Quantized weights are never stored tables (raw packed names,
    #      or I8..U32 codes with a quantization-aux sibling); a 0-d float is stored only when the library declares the
    #      key ignorable (D-C7 "strict unless declared", Soumil 2026-10-12); float 1x1xNxN is a mask when N is a position
    #      length; integer/bool tables are tables ----
    raw = {"mtp.layers.0.mlp.down_proj.weight_packed": ((32, 4), "I32"), "mtp.layers.0.mlp.down_proj.weight_scale": ((32, 1), "F16"),
           "mtp.layers.0.ffn.w1.weight": ((32, 16), "I8"), "mtp.layers.0.ffn.w1.scale": ((32, 1), "F8_E8M0"),
           "mtp.layers.0.rel_index": ((16,), "I64"), "bn.num_batches_tracked": ((), "I64"),
           "text_model.logit_scale": ((), "F32"), "h.0.attn.masked_bias": ((), "F32"),
           "h.0.attn.bias": ((1, 1, 32, 32), "U8"), "h.0.attn.fmask": ((1, 1, 32, 32), "F32"), "x.proj": ((1, 1, 7, 7), "F32")}
    sc = getattr(CM, "stored_constant", None)
    qw = CM.quantized_weight_keys(raw) if hasattr(CM, "quantized_weight_keys") else set()
    def stored(k, declared=False):
        s_, d_ = raw[k]
        if sc is None:                                   # HEAD: two rules; the matcher's looks_like_stored_buffer
            return CM.looks_like_stored_buffer(s_, d_)
        return sc(k, s_, d_, quantized=qw, pos={32}, declared=declared)
    check("B4 packed int MTP weight (weight_packed) is not a stored table", stored("mtp.layers.0.mlp.down_proj.weight_packed"), False)
    check("B4 I8 codes with a quantization scale sibling are not a stored table", stored("mtp.layers.0.ffn.w1.weight"), False)
    check("B4 an I64 index table stays a stored table", stored("mtp.layers.0.rel_index"), True)
    check("B4 an I64 counter stays a stored table", stored("bn.num_batches_tracked"), True)
    check("C7 CLIP logit_scale (learned 0-d float, undeclared) is not stored", stored("text_model.logit_scale"), False)
    check("C7 undeclared 0-d masked_bias is not stored (strict unless declared)", stored("h.0.attn.masked_bias"), False)
    check("C7 0-d masked_bias the library declares ignorable is stored", stored("h.0.attn.masked_bias", declared=True), True)
    check("C7 U8 1x1xNxN causal mask is a stored table", stored("h.0.attn.bias"), True)
    check("C7 float 1x1xNxN with N a position length is a stored mask", stored("h.0.attn.fmask"), True)
    check("C7 float 1x1xNxN with N not a position length (undeclared) is not stored", stored("x.proj"), False)

    # ---- DB-3 reading B (Soumil 2026-10-12): runtime (witness) credit is required only for declared-unbuilt tensors a
    #      runtime is known to run (the keys vLLM's loader maps); a missing / errored record is an unknown, not a failure;
    #      a "no MTP module" record is not cited for tensors it says nothing about ----
    rc = getattr(CM, "runtime_credit_needed", None)
    def db3(unbuilt, wit):
        if rc:
            need, unk = rc(unbuilt, wit)
            return {"needs_credit": sorted(need), "unknown": bool(unk), "cites_runtime": bool(need)}
        no_rt = bool(wit) and not wit.get("vllm_mtp_arch") and not wit.get("registry_error") and "vllm_mtp_arch" in wit
        return {"needs_credit": [] if no_rt else sorted(unbuilt), "unknown": False, "cites_runtime": True}   # HEAD
    t5 = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    check("DB-3 non-MTP declared-unbuilt + 'no MTP module' record: no credit needed, vLLM not cited",
          db3(t5, {"vllm_mtp_arch": None}), {"needs_credit": [], "unknown": False, "cites_runtime": False})
    check("DB-3 declared-unbuilt, no witness record: an unknown, not a failure",
          db3(t5, {}), {"needs_credit": [], "unknown": True, "cites_runtime": False})
    mtp = ["mtp.fc.weight", "mtp.layers.0.input_layernorm.weight"]
    check("DB-3 MTP tensors vLLM maps (closed witness): credit needed for exactly those",
          db3(mtp + t5, {"vllm_mtp_arch": "Qwen3_5MTP", "closed": True, "executed_checkpoint_keys": mtp, "loaded_not_used_keys": []}),
          {"needs_credit": sorted(mtp), "unknown": False, "cites_runtime": True})
    check("DB-3 MTP arch but the witness never mapped (failed run): every declared-unbuilt tensor needs credit",
          db3(mtp, {"vllm_mtp_arch": "Qwen3_5MTP", "closed": False}), {"needs_credit": sorted(mtp), "unknown": False, "cites_runtime": True})

    # B-1 (P0.6 cross-exam): a failed witness run (not closed / mapping error) proves nothing about which tensors vLLM
    # runs: every declared-unbuilt tensor needs credit, even if an unrelated key (lm_head) got mapped
    cyan = ["mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight"]
    check("DB-3 failed witness (mapping error, only lm_head mapped): every declared-unbuilt tensor needs credit",
          db3(cyan, {"vllm_mtp_arch": "Qwen3_5MTP", "closed": False, "mapping_error": ["ValueError: no module"],
                     "executed_checkpoint_keys": ["lm_head.weight"], "loaded_not_used_keys": []})["needs_credit"], sorted(cyan))
    # vLLM records raw checkpoint names; a packed weight is listed by the harness under its ckpt_split name
    check("DB-3 a packed MTP weight vLLM maps under its raw name still needs credit",
          db3(["mtp.layers.0.mlp.down_proj.weight"], {"vllm_mtp_arch": "Qwen3_5MTP", "closed": True,
              "executed_checkpoint_keys": ["mtp.layers.0.mlp.down_proj.weight_packed"], "loaded_not_used_keys": []})["needs_credit"],
          ["mtp.layers.0.mlp.down_proj.weight"])

    # ---- P0.2 (Soumil 2026-10-13): the diffusion worker has no library-load authority yet, so every diffusion row
    #      carries a G1 unknown (never FULL) ----
    DRm = module("diffusion_rules", legacy)
    nl = getattr(DRm, "no_load_authority_unknown", None)
    u = nl(["unet", "vae"]) if nl else None
    check("P0.2 diffusion rows carry the G1 'no library load' unknown", (u or {}).get("predicate"), "G1")

    # ---- S-2 (P0.6 cross-exam): the fixed call sites really call the shared rules (a reverted site fails here) ----
    def src(name):
        if not legacy:
            return open(os.path.join(HERE, name)).read()
        return subprocess.run(["git", "show", f"HEAD:exec_capture/harness_v2/{name}"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout
    W2, WD = src("worker_v2.py"), src("worker_v2_diffusion.py")
    wiring = {
        "worker_v2: matcher class scoring uses the common stored-constant rule": "unc0 = [c for c in unc0 if not common.stored_constant(" in W2,
        "worker_v2: _stored_constant is the common rule": "return common.stored_constant(x, *T[x], quantized=QW, pos=_POS)" in W2,
        "worker_v2: both split_library_ignored calls pass the quantized keys": W2.count("split_library_ignored(") == 2 and "split_library_ignored(mm, W0, ren0, QW)" in W2 and "split_library_ignored(m, W, ren, QW)" in W2,
        "worker_v2: matcher masks use the common rule": "masks = [c for c in un_c if common.stored_constant(" in W2,
        "worker_v2: no call to the old looks_like_stored_buffer rule": "looks_like_stored_buffer(*" not in W2,
        "worker_v2: leftover eligibility uses runtime_credit_needed": "common.runtime_credit_needed(unbuilt, WIT)" in W2,
        "worker_v2: class-selection candidate errors consult network_error": W2.count("if network_error(e):") >= 2,
        "worker_v2: _last_resort applies after_grading_exception": "common.after_grading_exception(R, UNKNOWNS" in W2,
        "worker_v2: T1 loaded set uses loaded_params": "libload.loaded_params(" in W2,
        "worker_v2: both class-selection paths add selection_incomplete": W2.count("class_rules.selection_incomplete(") == 2,
        "worker_v2_diffusion: masks use the common rule": "if common.stored_constant(c, *W[c], quantized=QW, pos=POS)]" in WD,
        "worker_v2_diffusion: split_library_ignored passes the quantized keys": "split_library_ignored(m, W, ren, QW)" in WD,
        "worker_v2_diffusion: _last_resort applies after_grading_exception": "common.after_grading_exception(R, UNKNOWNS" in WD,
        "worker_v2_diffusion: _attempt records the traceback components": "diffusion_rules.components_on_traceback(e.__traceback__, models)" in WD,
        "worker_v2_diffusion: direct-run reason from evidence": "diffusion_rules.direct_run_reason(cn," in WD,
        "worker_v2_diffusion: the P0.2 unknown is added BEFORE the FULL -> FULL_UNVERIFIED promotion":
            0 <= WD.find("UNKNOWNS.append(diffusion_rules.no_load_authority_unknown(") < WD.find('if R["verdict"] == "FULL" and UNKNOWNS:'),
        "worker_v2: the R-REFUSE FAIL is set before the promotion": 0 <= W2.find('R["reason"] = (f"library refuses this checkpoint') < W2.find('if R["verdict"] in ("FULL", "FULL_LEFTOVERS") and UNKNOWNS:'),
        "worker_v2: R-REFUSE judges the declared (default) class": "_decl_rep = LIB if cls is DECLARED_CLS else" in W2,
        "worker_v2: a candidate the library refuses is never selected": "if libload.refuses(_rc):" in W2,
        "worker_v2: the executed build is one float32 build with learned params normalised (F07)": 'normalise_learned(_lc.uniform_float32(mm.eval()))' in W2,
        "worker_v2_diffusion: both component builds normalise learned params (F07-3)": WD.count("normalise_learned(") == 2,
        "worker_v2: zero-storage faithfulness unknowns (F07)": "zero_storage_findings(" in W2 and W2.find("zero_storage_findings(") < W2.find('if R["verdict"] in ("FULL", "FULL_LEFTOVERS") and UNKNOWNS:'),
        "worker_v2_diffusion: zero-storage faithfulness unknowns (F07)": "zero_storage_findings(" in WD and WD.find("zero_storage_findings(") < WD.find('if R["verdict"] == "FULL" and UNKNOWNS:'),
    }
    for k, v in wiring.items():
        check(f"S-2 wiring: {k}", v, True)

    print(f"\n{bad} unexpected" + (" (legacy: expected to fail)" if legacy else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
