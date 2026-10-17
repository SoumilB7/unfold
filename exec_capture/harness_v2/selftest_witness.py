"""F14 controls (Soumil 2026-10-13): a vLLM witness record is used only when bound to the graded revision and the current
witness code, with vLLM version + source and library versions named, and its two stability runs shown to have received
different inputs (every value input differs; positions may be equal). Dossier: model-benchmark/_reruns/p2a4_f14.
`--legacy` runs the same controls against the committed HEAD modules to show they catch the old behaviour. No network,
no vLLM: the producer is executed with stubbed vLLM / hub / transformers, writing into a temporary directory.

    python3 selftest_witness.py [--legacy]      # exit 0 = every control as expected
"""
import os, sys, json, shutil, subprocess, tempfile, types, importlib, inspect
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def source(name, legacy):
    if not legacy:
        return open(os.path.join(HERE, name)).read()
    return subprocess.run(["git", "show", f"HEAD:exec_capture/harness_v2/{name}"], cwd=HERE,
                          capture_output=True, text=True, check=True).stdout


def module(name, legacy, file_dir=HERE):
    m = types.ModuleType(name + ("_head" if legacy else "_cur")); m.__file__ = os.path.join(file_dir, name + ".py")
    try:
        src = source(name + ".py", legacy) if file_dir == HERE else open(m.__file__).read()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return m                                  # (not at HEAD: an empty module, so every control on it fails)
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
    CM = module("common", legacy)
    has = lambda f: getattr(CM, f, None)
    missing = lambda *a, **k: None

    # ---- 1. the input helper: each run its own seed; distinctness = every value input differs ----
    wi, fp, di = has("witness_inputs") or missing, has("input_fingerprints") or missing, has("distinct_inputs") or (lambda *a: None)
    a, b, a2 = wi(8, 16, 0), wi(8, 16, 1), wi(8, 16, 0)
    fa, fb, fa2 = (fp(x) if x is not None else None for x in (a, b, a2))
    check("NEG run 0 and run 1 inputs are distinct (every value input)", di(fa, fb), True)
    check("VALID the helper is deterministic per run index", fa is not None and fa == fa2, True)
    check("NEG two runs on the same seed are not distinct", di(fa, fa2), False)
    check("NEG only hidden_states differs (input_ids shared) is not distinct",
          di(fa, dict(fb, input_ids=fa["input_ids"])) if fa and fb else None, False)
    check("VALID equal positions (a function of the token count) do not block distinctness",
          bool(fa and fb and fa["positions"] == fb["positions"] and di(fa, fb)), True)
    check("NEG missing fingerprints are not distinct", di(None, fb), False)
    check("NEG fingerprints of different input names are not distinct", di({"x": "1"}, {"y": "2"}), False)
    check("NEG only structural inputs fingerprinted is not distinct", di({"positions": "1"}, {"positions": "2"}), False)

    # ---- 2. execution credit: proven only when bound, stable on distinct inputs, transparent ----
    REV, PID = "r" * 40, "pid123"
    good = {"closed": True, "transparent": True, "stable_structure": True, "schema": getattr(CM, "WITNESS_SCHEMA", 2),
            "revision": REV, "producer_id": PID, "vllm_version": "0.30.0", "vllm_source_id": "src1", "runtime": {"torch": "2.14.1"},
            "revision_pin": {"pinned": 3, "other_revision": []},
            "op_full": ["aten::mm"], "opaque_ops": {}, "opacity_unknown_ops": [], "decomposition_disagrees": [], "external_nonscalar": 0,
            "vocabulary_errors": [], "inputs": {"a": {"seed": 0, "fingerprints": fa}, "b": {"seed": 1, "fingerprints": fb}},
            "vllm_mtp_arch": "Qwen3_5MTP", "executed_checkpoint_keys": ["mtp.fc.weight"], "loaded_not_used_keys": []}
    ec = has("witness_execution_credit")
    def credit(w, rev=REV):
        if ec:
            return sorted({p for p, _ in ec(w, rev, PID)})
        # HEAD (worker_v2.py): proven iff transparent and the record's revision equals the graded one; else one G4 unknown
        return [] if (w.get("transparent") is True and rev is not None and w.get("revision") == rev) else ["G4"]
    drop = lambda k: {x: v for x, v in good.items() if x != k}
    check("VALID bound, distinct-input, transparent witness: proven", credit(good), [])
    check("NEG the same input in both runs: G5a unknown", credit(dict(good, inputs={"a": {"fingerprints": fa}, "b": {"fingerprints": fa2}})), ["G5a"])
    check("NEG no input fingerprints (the HEAD record shape): G5a unknown", credit(drop("inputs")), ["G5a"])
    check("NEG stable_structure not True: G5a unknown", credit(dict(good, stable_structure=None)), ["G5a"])
    check("NEG no producer id: G6 unknown", credit(drop("producer_id")), ["G6"])
    check("NEG made by another version of the witness code: G6 unknown", credit(dict(good, producer_id="old")), ["G6"])
    check("NEG no vLLM source id: G6 unknown", credit(drop("vllm_source_id")), ["G6"])
    check("NEG no library versions (runtime): G6 unknown", credit(drop("runtime")), ["G6"])
    check("NEG old schema: G6 unknown", credit(dict(good, schema=1)), ["G6"])
    # F10 (Soumil 2026-10-14, WITNESS_SCHEMA 3): a record made before the fail-closed opacity fields (schema 2) is
    # never credited as transparent; the current schema is 3
    check("NEG a schema-2 record (before F10's opacity fields): G6 unknown", credit(dict(good, schema=2)), ["G6"])
    check("VALID the witness schema is 3 (F10)", getattr(CM, "WITNESS_SCHEMA", None), 3)
    check("NEG revision missing on the record: G6 unknown", credit(drop("revision")), ["G6"])
    check("NEG recorded for another revision: G6 unknown", credit(dict(good, revision="x" * 40)), ["G6"])
    check("NEG graded revision unknown: G6 unknown", credit(good, None), ["G6"])
    # F10-24: transparency is re-derived from the record's evidence by the producer's own rule (common.witness_transparent);
    # rejected vocabulary entries are a note and never enter it
    check("NEG an opaque op in the recording: G4 unknown", credit(dict(good, opaque_ops={"f10ctl::secret": "opaque"})), ["G4"])
    check("NEG an op without transparency shown: G4 unknown", credit(dict(good, opacity_unknown_ops=["aten::x"])), ["G4"])
    check("NEG a disagreeing decomposition: G4 unknown", credit(dict(good, decomposition_disagrees=["aten::roll"])), ["G4"])
    check("NEG a tensor from outside the recording: G4 unknown", credit(dict(good, external_nonscalar=1)), ["G4"])
    check("NEG no full op names: G4 unknown", credit(dict(good, op_full=[])), ["G4"])
    check("NEG opacity fields missing: G4 unknown, never an exception", credit(drop("opaque_ops")), ["G4"])
    # (Opus R8d: each opacity field alone; external_nonscalar must be exactly 0, a missing / None count is not "nothing external")
    check("NEG external_nonscalar missing (op_full and empty opacity fields present): G4 unknown", credit(drop("external_nonscalar")), ["G4"])
    check("NEG external_nonscalar None: G4 unknown", credit(dict(good, external_nonscalar=None)), ["G4"])
    check("VALID F10-24 rejected vocabulary entries on a clean recording: a note, proven",
          credit(dict(good, vocabulary_errors=["'aten::_fft_c2r' needs no review (it is transparent: core)"], transparent=False)), [])
    check("NEG no revision pin log (revision declared, not shown): G6 unknown", credit(drop("revision_pin")), ["G6"])
    check("NEG pin log shows a read of another revision: G6 unknown",
          credit(dict(good, revision_pin={"pinned": 3, "other_revision": ["main2"]})), ["G6"])
    check("NEG library versions malformed (runtime not a version map): G6 unknown", credit(dict(good, runtime="junk")), ["G6"])
    def safe(f):
        try: return f()
        except Exception as e: return type(e).__name__
    check("NEG malformed inputs field: a G5a unknown, never an exception",
          safe(lambda: credit(dict(good, inputs={"a": "x", "b": ["y"]}))), ["G5a"])
    check("NEG empty fingerprint values are not distinct",
          safe(lambda: credit(dict(good, inputs={"a": {"fingerprints": {"input_ids": "", "positions": "p"}},
                                                  "b": {"fingerprints": {"input_ids": "", "positions": "p"}}}))), ["G5a"])

    # ---- 3. "no MTP module" (DB-3) statements count only from a bound record whose registry stage completed ----
    rc = has("runtime_credit_needed")
    takes_binding = rc is not None and len(inspect.signature(rc).parameters) >= 4
    def need(unbuilt, w, rev=REV):
        n, u = rc(unbuilt, w, rev, PID) if takes_binding else rc(unbuilt, w)
        pred = u[0] if isinstance(u, tuple) else ("G3" if u else None)
        return sorted(n), pred
    mtp = {"mtp.fc.weight"}
    same = {"architectures": ["FooForCausalLM"], "model_type": "foo", "n_predict": None}
    noarch = {"vllm_mtp_arch": None, "registry_checked": True, "closed": False, "failed_stage": "config",
              "registry_override": {"before": same, "after": dict(same)},
              **{k: good[k] for k in ("schema", "revision", "producer_id", "vllm_version", "vllm_source_id", "runtime", "revision_pin")}}
    check("VALID bound 'no MTP module' record: no credit needed, no unknown", need(mtp, noarch), ([], None))
    head_rec = {"repo": "x", "closed": False, "failed_stage": "config", "vllm_mtp_arch": None, "error": "NotImplementedError"}
    check("NEG identity-free 'no MTP module' record (HEAD error path): G6 unknown, never 'no runtime'", need(mtp, head_rec), ([], "G6"))
    check("NEG 'no MTP module' record for another revision: G6 unknown", need(mtp, dict(noarch, revision="x" * 40)), ([], "G6"))
    check("NEG registry stage not shown completed: G6 unknown", need(mtp, dict(noarch, registry_checked=False)), ([], "G6"))
    check("NEG 'no MTP module' made by another witness code: G6 unknown", need(mtp, dict(noarch, producer_id="old")), ([], "G6"))
    check("VALID registry error stays the G3 unknown", need(mtp, dict(noarch, registry_error="OSError: x")), ([], "G3"))
    check("NEG vLLM's override changed the config (model_type, n_predict) but added no architecture: G3 unknown, not 'no MTP'",
          need(mtp, dict(noarch, registry_override={"before": same, "after": dict(same, model_type="deepseek_mtp", n_predict=1)})), ([], "G3"))
    check("NEG 'no MTP' record without its registry evidence (before/after): G6 unknown",
          need(mtp, {k: v for k, v in noarch.items() if k != "registry_override"}), ([], "G6"))
    check("NEG pin log with no pinned read (revision declared, not shown): G6 unknown",
          need(mtp, dict(noarch, revision_pin={"pinned": 0, "other_revision": []})), ([], "G6"))
    check("VALID no record stays the G3 unknown", need(mtp, None), ([], "G3"))
    check("VALID closed MTP witness: credit needed for exactly what vLLM maps", need(mtp | {"t5.rel"}, good), (sorted(mtp), None))
    check("VALID failed MTP witness: every declared-unbuilt tensor needs credit", need(mtp, dict(good, closed=False)), (sorted(mtp), None))
    check("NEG (F14 e) an MTP witness cut short by a network failure: G6 unknown, never 'needs credit for all' (PARTIAL)",
          need(mtp, dict(good, closed=False, failed_stage="headers", network_error="ConnectTimeout: handshake timed out")), ([], "G6"))
    check("VALID (F14 e) a bound 'no MTP' answer given before a later network failure is still the registry's answer",
          need(mtp, dict(noarch, failed_stage="headers", network_error="ConnectTimeout")), ([], None))
    unbound = lambda w: {k: v for k, v in w.items() if k not in ("producer_id", "vllm_source_id", "schema", "revision_pin")}
    check("NEG UNBOUND failed MTP witness: G6 unknown, no credit demand (never PARTIAL from an unbound record)",
          need(mtp, unbound(dict(good, closed=False))), ([], "G6"))
    check("NEG UNBOUND closed witness mapping only part of the tensors: G6 unknown (no absence certified)",
          need(mtp | {"mtp.extra.weight"}, unbound(good)), ([], "G6"))
    check("NEG UNBOUND record with a registry error: G6 unknown", need(mtp, unbound(dict(noarch, registry_error="OSError: x"))), ([], "G6"))
    check("VALID bound closed witness mapping only part: the unmapped ones need nothing (vLLM's loader shows it)",
          need(mtp | {"t5.rel"}, good), (sorted(mtp), None))
    ls = has("loader_statement")
    l = ls(good, 2) if ls else ""
    check("VALID the loader-branch wording names vLLM version, source id, revision and the count",
          all(x in l for x in ("0.30.0", "src1", REV, "loader maps none of these 2")), True)
    st = has("no_mtp_statement")
    s = st(noarch) if st else ""
    check("NEG the leftover wording names vLLM version, source id and revision, never 'no runtime'",
          all(x in s for x in ("0.30.0", "src1", REV, "no MTP module for this config")) and "no runtime" not in s, True)

    # ---- 4. the stamp: the recorder's own files only ----
    tmp = tempfile.mkdtemp(prefix="selftest_witness_")
    try:
        files = set(getattr(CM, "WITNESS_FILES", ())) | {"worker_v2.py", "common.py"}
        for f in files:
            if os.path.exists(os.path.join(HERE, f)):
                os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)     # (F10: a vocabulary subfolder)
                shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
        T = module("witness_io", legacy, tmp) if os.path.exists(os.path.join(tmp, "witness_io.py")) and not legacy else module("common", legacy, tmp)
        pid = getattr(T, "witness_producer_id", None)
        p0 = pid() if pid else None
        open(os.path.join(tmp, "worker_v2.py"), "a").write("\n# verdict-only change\n")
        p1 = pid() if pid else None
        open(os.path.join(tmp, "common.py"), "a").write("\n# grading-rule change (e.g. runtime_credit_needed wording)\n")
        p1b = pid() if pid else None
        open(os.path.join(tmp, "dag.py"), "a").write("\n# recorder change\n")
        p2 = pid() if pid else None
        open(os.path.join(tmp, "witness_io.py"), "a").write("\n# recorder helper change\n")
        p3 = pid() if pid else None
        check("VALID a verdict-only file change keeps the witness stamp", p0 is not None and p0 == p1, True)
        check("VALID a grading-rule change in common.py keeps the witness stamp", p0 is not None and p0 == p1b, True)
        check("NEG a recorder file change makes the stamp differ (records stale)", p0 is not None and p1b != p2, True)
        check("NEG a recorder helper (witness_io) change makes the stamp differ", p0 is not None and p2 != p3, True)
        check("VALID the recorder helpers are the very ones the grader uses (one ckpt_split)",
              getattr(CM, "ckpt_split", None) is not None and getattr(sys.modules.get("witness_io"), "ckpt_split", 0) is getattr(CM, "ckpt_split", None), True)
        check("VALID the stamp covers both witness scripts", all(f in getattr(CM, "WITNESS_FILES", ()) for f in ("mtp_witness_run.py", "vllm_witness.py")), True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 5. the producer's error path keeps identity + the registry result (stubbed vLLM / hub / transformers) ----
    out = tempfile.mkdtemp(prefix="selftest_witness_run_")
    try:
        fake_h = os.path.join(out, "harness_v2"); os.makedirs(fake_h)
        vdir = os.path.join(out, "vllm"); os.makedirs(vdir); open(os.path.join(vdir, "__init__.py"), "w").write("# stub\n")
        recs = {}
        _real_network_error = importlib.import_module("noweights").network_error     # the recorder's real classifier
        saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.config", "vllm.config.speculative", "vllm.transformers_utils",
                                                  "vllm.transformers_utils.config", "vllm_witness", "lowcost",
                                                  "noweights", "huggingface_hub", "transformers", "common")}
        try:
            for case in ("A", "B", "C", "D", "F", "V", "N", "W", "O", "R"):
                for k in saved: sys.modules.pop(k, None)
                sys.modules["common"] = CM
                m = lambda n: sys.modules.setdefault(n, types.ModuleType(n))
                v = m("vllm"); v.__version__ = "0.30.0"; v.__file__ = os.path.join(vdir, "__init__.py")
                m("vllm.config"); sp = m("vllm.config.speculative")
                class SpeculativeConfig:
                    @staticmethod
                    def hf_config_override(c, case=case):
                        if case in ("C", "N", "W", "O"): c.architectures = ["Qwen3_5MTP"]
                        if case == "D": c.architectures = ["EagleMistralLarge3ForCausalLM"]     # vLLM's draft, no "MTP" in the name
                        if case == "V": c.model_type, c.n_predict = "deepseek_mtp", 1          # config changed, no arch added
                        return c
                sp.SpeculativeConfig = SpeculativeConfig
                nw = m("noweights"); nw.PIN_LOG = {"pinned": 7, "other_revision": ["stale"]}   # a previous repo's log
                nw.pin = lambda repo, sha, nw=nw: nw.PIN_LOG.__setitem__("pinned", nw.PIN_LOG["pinned"] + 1)
                nw.network_error = _real_network_error
                class HfApi:
                    def model_info(self, repo, case=case):
                        if case == "F": raise ConnectionError("hub unreachable")
                        return types.SimpleNamespace(sha=REV)
                m("huggingface_hub").HfApi = HfApi
                class AutoConfig:
                    @staticmethod
                    def from_pretrained(repo, case=case, **k):
                        if case == "B": raise OSError("We couldn't connect to 'https://huggingface.co' to load the config")
                        if case == "R":
                            import httpx
                            raise httpx.ConnectTimeout("handshake timed out")             # the registry stage hits the network
                        return types.SimpleNamespace(architectures=["FooForCausalLM"], model_type="foo")
                m("transformers").AutoConfig = AutoConfig
                m("vllm.transformers_utils")
                m("vllm.transformers_utils.config").get_config = lambda repo, trust_remote_code, revision=None, **k: \
                    AutoConfig.from_pretrained(repo)          # (vLLM's own config parser; stubbed)
                m("lowcost").fetch_headers = lambda repo: ({"model.embed.weight": ((4, 4), "F32")}, None)
                W = m("vllm_witness"); W.SETTINGS = {"n_tokens": 8, "run_seeds": [0, 1]}
                def build(repo, case=case):
                    if case == "N":
                        import socket
                        raise socket.gaierror("nodename nor servname provided")      # a network failure (by type)
                    if case == "W":                                                    # vLLM wraps it: ValueError from ConnectTimeout
                        import httpx
                        try:
                            raise httpx.ConnectTimeout("handshake timed out")
                        except httpx.ConnectTimeout as ce:
                            raise ValueError("Error parsing config for x") from ce
                    if case == "O":                                                    # a repo fact, not the network
                        raise OSError("x does not appear to have a file named config.json")
                    raise NotImplementedError("Unsupported speculative method: 'mtp'")
                W.build = build
                sys.argv = ["mtp_witness_run.py", f"probe/case{case}"]
                g = {"__name__": "__main__", "__file__": os.path.join(fake_h, "mtp_witness_run.py")}
                exec(compile(source("mtp_witness_run.py", legacy), g["__file__"], "exec"), g)
                recs[case] = json.load(open(os.path.join(out, "results_v2", "_witness", "vllm", f"probe__case{case}.json")))
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        A, B, C, D, F, V, N, Wr, O, Rg = (recs[k] for k in "ABCDFVNWOR")
        idk = ("schema", "revision", "producer_id", "vllm_version", "vllm_source_id", "runtime")
        check("NEG a failed run keeps its identity (schema, revision, producer, vLLM version + source, runtime)",
              all(A.get(k) for k in idk), True)
        check("NEG registry answered 'no MTP' then the build failed: registry_checked kept", A.get("registry_checked"), True)
        check("NEG registry raised then the build failed: registry_error kept (not a 'no MTP' answer)", bool(B.get("registry_error")), True)
        check("NEG ... and the two records differ beyond the repo name",
              {k: v for k, v in A.items() if k not in ("repo", "secs", "peak_rss_gb", "runtime")} !=
              {k: v for k, v in B.items() if k not in ("repo", "secs", "peak_rss_gb", "runtime")}, True)
        pid_now = CM.witness_producer_id() if has("witness_producer_id") else None
        check("VALID the failed-run record carries the current producer id", A.get("producer_id") == pid_now and pid_now is not None, True)
        if rc:
            n, u = (rc(mtp, A, REV) if takes_binding else rc(mtp, A))
            check("VALID A (bound, registry completed, no MTP) gives no credit need and no unknown", (sorted(n), u), ([], None))
            n, u = (rc(mtp, B, REV) if takes_binding else rc(mtp, B))
            check("NEG B (registry raised) is an unknown, never 'no runtime'", bool(u), True)
            n, u = (rc(mtp, C, REV) if takes_binding else rc(mtp, C))
            check("VALID C (MTP arch, build failed) needs credit for every MTP tensor", sorted(n), sorted(mtp))
        check("VALID C keeps the registry's MTP arch", C.get("vllm_mtp_arch"), "Qwen3_5MTP")
        check("NEG a draft arch vLLM substitutes without 'MTP' in its name is still its MTP module (evidence, not a name)",
              D.get("vllm_mtp_arch"), "EagleMistralLarge3ForCausalLM")
        check("NEG a failure before the registry is asked never writes a 'no MTP' answer", "vllm_mtp_arch" in F, False)
        check("NEG (F14 e) the recorder marks a network failure by type on the failed record", bool(N.get("network_error")), True)
        check("VALID (F14 e) a model error is not marked as a network failure", "network_error" in C, False)
        check("NEG (F14 e) a network error vLLM wrapped (ValueError from ConnectTimeout) is still marked", bool(Wr.get("network_error")), True)
        check("VALID (F14 e) a repo-fact OSError (missing config.json) is not marked as network", "network_error" in O, False)
        check("NEG (F14 e) a network failure at the registry stage is marked for re-recording", bool(Rg.get("registry_network_error")), True)
        if rc and takes_binding:
            check("NEG (F14 e) that record is a G6 unknown for the grader", (rc(mtp, N, REV)[1] or ("",))[0], "G6")
            check("NEG (F14 e) the wrapped network failure is a G6 unknown too", (rc(mtp, Wr, REV)[1] or ("",))[0], "G6")
            check("VALID (F14 e) the repo-fact failure still needs credit for every MTP tensor (an observed failure)",
                  sorted(rc(mtp, O, REV)[0]), sorted(mtp))
        if rc and takes_binding:
            check("NEG override changed model_type / n_predict without an arch (V4.1-like): recorded evidence gives a G3 unknown",
                  (rc(mtp, V, REV)[1] or ("",))[0], "G3")
        check("NEG the failed-run record carries its own repo's pin log (not a previous repo's)", A.get("revision_pin"), {"pinned": 1, "other_revision": []})
    finally:
        shutil.rmtree(out, ignore_errors=True)

    # ---- 6. wiring: the call sites really use the rules (a reverted site fails here) ----
    W2, VW, MR = source("worker_v2.py", legacy), source("vllm_witness.py", legacy), source("mtp_witness_run.py", legacy)
    wiring = {
        "worker_v2: execution credit uses witness_execution_credit": "common.witness_execution_credit(WIT, R.get(\"revision\"))" in W2,
        "worker_v2: runtime credit is bound to the graded revision": "common.runtime_credit_needed(unbuilt, WIT, R.get(\"revision\"))" in W2,
        "worker_v2: the leftover certification carries the scoped no-MTP statement": "common.no_mtp_statement(WIT)" in W2,
        "vllm_witness: run_forward draws inputs per run index": "witness_io.witness_inputs(n_tokens, H, run_index)" in VW
                                                                and "manual_seed(0)" not in VW,
        "vllm_witness: the recorder carries the input fingerprints": "rec.input_fingerprints" in VW,
        "mtp_witness_run: the two runs use the two settings seeds": "run_forward(model, dcfg, n_tok, md, seed_a)" in MR
                                                                    and "run_forward(model, dcfg, n_tok, md, seed_b)" in MR,
        "mtp_witness_run: distinctness is recorded, not folded into closed (an undistinct pair is a G5a unknown)":
            'R["distinct_inputs"] = ' in MR and 'and R["distinct_inputs"]' not in MR,
        "mtp_witness_run: no name test on vLLM's arch": '"MTP" in a' not in MR,
        "mtp_witness_run: the registry is asked on vLLM's own parsed config (F14 follow-up a)":
            "get_config(repo, trust_remote_code=False, revision=R[\"revision\"])" in MR and "AutoConfig.from_pretrained" not in MR,
        "mtp_witness_run: the recorder imports its helpers from witness_io, not the grading rules": "common" not in MR,
        "vllm_witness: the recorder does not import the grading rules": "import common" not in VW,
        "worker_v2: the loader-branch statement is emitted": "common.loader_statement(WIT, len(unbuilt))" in W2,
        "mtp_witness_run: the stamp is hashed once at recorder import": "PRODUCER_ID = witness_io.witness_producer_id()" in MR
                                                                         and 'R["producer_id"] = PRODUCER_ID' in MR,
        "worker_v2: the certification states the witness credit status": "vLLM witness credit ({R['executed_by_witness'].get('credit')})" in W2,
    }
    for k, v in wiring.items():
        check(f"wiring: {k}", v, True)
    import ast
    VWm = next((ast.literal_eval(n.value) for n in ast.parse(VW).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "SETTINGS" for t in n.targets)), None)
    check("vllm_witness: two different run seeds in SETTINGS", bool(VWm) and len(set(VWm.get("run_seeds", []))) == 2, True)

    # ---- 7. the stamp covers the recorder's whole local import closure (a new local import cannot escape it) ----
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    seen, todo = set(), ["mtp_witness_run", "vllm_witness"]
    while todo:
        m = todo.pop()
        if m in seen: continue
        seen.add(m)
        for n in ast.walk(ast.parse(source(m + ".py", legacy))):
            names = ([a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else
                     [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module else
                     [n.args[0].value.split(".")[0]] if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "__import__"
                     and n.args and isinstance(n.args[0], ast.Constant) else [])
            todo += [x for x in names if x in local]
    check("NEG the recorder's local import closure equals WITNESS_FILES", sorted(seen),
          sorted(f[:-3] for f in getattr(CM, "WITNESS_FILES", ()) if f.endswith(".py")))
    # F10: the data file the classifier reads is stamped too (a reviewed-op change makes every record stale)
    _oc = source("opclass.py", legacy)
    # F10-24: the recorder's transparency is the ONE shared rule (witness_io.witness_transparent, also the grader's), and
    # that rule never reads the rejected-vocabulary note
    _mw, _wi = source("mtp_witness_run.py", legacy), source("witness_io.py", legacy)
    _wt = _wi[_wi.find("def witness_transparent"):_wi.find("def witness_producer_id")] if "def witness_transparent" in _wi else ""
    _body = _wt.split('"""')[-1] if _wt.count('"""') >= 2 else _wt
    check("NEG F10-24: the recorder's transparency is witness_transparent(R), whose rule ignores vocabulary_errors",
          'R["transparent"] = witness_transparent(R)' in _mw and bool(_wt) and "vocabulary" not in _body, True)
    check("NEG the classifier's vocabulary file is in WITNESS_FILES", ("opacity_reviewed.yaml" not in _oc) or
          "everchanging/opacity_reviewed.yaml" in getattr(CM, "WITNESS_FILES", ()), True)

    print(f"\n{bad} unexpected" + (" (legacy: expected to fail)" if legacy else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
