"""The library's own verdict on which shipped tensors it loads: every weight file of the repo is mirrored locally as a
SPARSE stub (real header bytes, zero-filled data, no disk used), then transformers' from_pretrained runs on the meta
device (no tensor data in RAM) with output_loading_info=True. File selection, renames, merges/splits and ignore lists
are all the library's own; nothing is re-implemented here."""
import os, json, struct, tempfile, shutil
from huggingface_hub import HfApi, HfFileSystem

WEIGHT_EXT = (".safetensors",)


def _read_header(fs, path):
    """The real header bytes of one safetensors file (first 8 bytes = header length). huggingface_hub's own http
    backoff retries transient network errors; anything that still fails propagates (the caller records it)."""
    with fs.open(path, "rb", block_size=1 << 16) as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, fh.read(n)


def mirror_sparse(repo, dest, headers=None, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    api, fs = HfApi(), HfFileSystem()
    info = api.model_info(repo, files_metadata=True)
    headers = {} if headers is None else headers        # file -> {tensor: (shape, dtype)} from the real header bytes
    weights = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(WEIGHT_EXT)]
    # every shard's header is fetched in parallel (sharded repos have up to hundreds of files)
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        import noweights
        got = list(ex.map(lambda fz: _read_header(fs, f"{noweights.ref(repo)}/{fz[0]}"), weights))   # pinned revision
    finally:                                            # on a timeout: do not wait for reads still in flight
        ex.shutdown(wait=False, cancel_futures=True)
    for (f, size), (n, hdr) in zip(weights, got):
        p = os.path.join(dest, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as out:
            out.write(struct.pack("<Q", n)); out.write(hdr); out.truncate(size)
        headers[f] = {k: (tuple(v["shape"]), v["dtype"]) for k, v in json.loads(hdr).items() if k != "__metadata__"}
    for s in info.siblings:
        f, size = s.rfilename, s.size or 0
        if f.endswith((".json", ".txt", ".model")) and size < 50_000_000:
            p = os.path.join(dest, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            local = api.hf_hub_download(repo, f, cache_dir=os.path.join(dest, ".dl"))
            shutil.copy(local, p)
    shutil.rmtree(os.path.join(dest, ".dl"), ignore_errors=True)
    return len(weights)


def headers_read_names(opened, headers):
    return [k for f in opened for k in headers.get(f, {})]


def hypothesis(repo, cls, aux_keys=(), config=None, load_kw=None):
    """The COMPLETE load hypothesis a library load report is evidence for (F04): the repo (+ its pinned revision), the
    class (module + qualified name, not its bare name), every from_pretrained keyword, the auxiliary-key classification
    and the config the loader is given (its full serialisation, incl. attention implementation and dtype), plus the
    installed library. Two requests share a report only if every field is equal."""
    import hashlib, transformers, torch
    rev = None
    try:
        import noweights
        rev = noweights.ref(repo) if not (os.path.isabs(str(repo)) and os.path.isdir(str(repo))) else None
    except Exception:
        rev = None
    cfg_txt = None
    if config is not None:
        try:
            cfg_txt = config.to_json_string(use_diff=False)
        except Exception:
            cfg_txt = repr(sorted(vars(config).items(), key=lambda x: x[0]))
        cfg_txt += f"|attn={getattr(config, '_attn_implementation_internal', None)}|dtype={getattr(config, 'dtype', None)}"
        cfg_txt += "|" + json.dumps({k: v for k, v in (config_view(config) or {}).items()
                                     if k.rsplit(".", 1)[-1].startswith(("_attn_implementation", "_experts_implementation"))},
                                    sort_keys=True)          # (every sub-config's private implementation choice)
    h = {"repo": str(repo), "revision_ref": rev, "class": f"{cls.__module__}.{cls.__qualname__}",
         "load_kw": {k: str(v) for k, v in sorted((load_kw or {}).items())}, "aux_keys": sorted(aux_keys),
         "config_sha": hashlib.sha256(cfg_txt.encode()).hexdigest()[:16] if cfg_txt is not None else None,
         "transformers": transformers.__version__, "torch": torch.__version__}
    h["key"] = hashlib.sha256(json.dumps(h, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return h


class Authority:
    """The repo mirrored ONCE as sparse stubs; report(cls, ...) runs the library's from_pretrained over the same stubs.
    Reports are cached by the COMPLETE load hypothesis (F04: class, config, keywords, auxiliary keys, revision, library),
    never by class name: a request with any different field gets its own load. close() deletes."""
    def __init__(self, repo):
        self.repo, self.d, self.headers, self.n, self.cache = repo, tempfile.mkdtemp(prefix="libload_"), {}, None, {}

    def mirror(self):
        if self.n is None:
            if os.path.isabs(self.repo) and os.path.isdir(self.repo):   # a local checkpoint (grading self-test fixtures)
                shutil.rmtree(self.d, ignore_errors=True); self.d, self.local = self.repo, True
                self.n = 0
                for f in sorted(os.path.relpath(os.path.join(r_, x), self.repo) for r_, _, xs in os.walk(self.repo) for x in xs):
                    if f.endswith(WEIGHT_EXT):
                        with open(os.path.join(self.repo, f), "rb") as fh:
                            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
                        self.headers[f] = {k: (tuple(v["shape"]), v["dtype"]) for k, v in hdr.items() if k != "__metadata__"}
                        self.n += 1
            else:
                self.n = mirror_sparse(self.repo, self.d, self.headers)
        return self.n

    def key(self, cls, aux_keys=(), config=None, **load_kw):
        return hypothesis(self.repo, cls, aux_keys, config, load_kw)["key"]

    def lookup(self, cls, aux_keys=(), config=None, **load_kw):
        """The cached report for exactly this hypothesis, or None (never a report of another hypothesis)."""
        return self.cache.get(self.key(cls, aux_keys, config, **load_kw))

    def forget(self, cls, aux_keys=(), config=None, **load_kw):
        self.cache.pop(self.key(cls, aux_keys, config, **load_kw), None)

    def report(self, cls, aux_keys=(), config=None, **load_kw):
        """One report per complete load hypothesis. `config` is copied (the loader must not change the caller's), and
        the hypothesis is computed from the copy BEFORE the load (a load that mutates its config cannot re-key it)."""
        import copy
        hyp = hypothesis(self.repo, cls, aux_keys, config, load_kw)
        if hyp["key"] not in self.cache:
            kw = dict(load_kw)
            if config is not None:
                kw["config"] = copy.deepcopy(config)
            r = _report(self, cls, aux_keys, **kw)
            if isinstance(r, dict):
                r["hypothesis"] = hyp
            self.cache[hyp["key"]] = r
        r = self.cache[hyp["key"]]
        if isinstance(r, dict) and r.get("hypothesis", hyp)["key"] != hyp["key"]:   # invariant: never cross-served
            raise RuntimeError("authority cache served a report of another load hypothesis")
        return r

    def close(self):
        if not getattr(self, "local", False):
            shutil.rmtree(self.d, ignore_errors=True)


def _report(A, cls, aux_keys=(), **kw):
    import torch
    import transformers.modeling_utils as MU
    import transformers.core_model_loading as CML
    d = A.d
    opened, orig = set(), MU.safe_open
    def recording_safe_open(path, *a, **k):                # observe which files the library reads; behaviour unchanged
        opened.add(os.path.relpath(str(path), d)); return orig(path, *a, **k)
    # F04: the library's own conversion record: which file keys each target is built from, and with which operations
    conversions, orig_add = {}, CML.WeightTransform.add_tensor
    def recording_add(self_, target_key, source_key, source_pattern, future):
        e = conversions.setdefault(target_key, {"sources": set(), "ops": [], "quant_op": None, "transform": type(self_).__name__})
        e["sources"].add(source_key)
        e["ops"] = [type(o).__name__ for o in (getattr(self_, "operations", None) or [])]
        q = getattr(self_, "quantization_operation", None)
        e["quant_op"] = type(q).__name__ if q is not None else None
        return orig_add(self_, target_key, source_key, source_pattern, future)
    renamed_to, orig_rename = {}, CML.rename_source_key
    def recording_rename(source_key, *a, **k):             # observe the library's own key renames; behaviour unchanged
        out = orig_rename(source_key, *a, **k)
        renamed_to.setdefault(source_key, set()).add(out[0])   # every name the library gave this key (incl. its fallback)
        return out
    headers = A.headers
    n = A.mirror()
    d = A.d                                                # a local checkpoint is read in place (mirror() sets it)
    if n == 0:
        return {"error": "no safetensors weight files"}
    MU.safe_open = recording_safe_open
    CML.rename_source_key = recording_rename
    CML.WeightTransform.add_tensor = recording_add
    # tied weights shipped twice: the library compares their VALUES (torch.equal) to decide whether to tie, which
    # meta tensors cannot do. Keep the tie the config builds and record that the check could not run.
    tie_unchecked, orig_equal = [], torch.equal
    def meta_equal(a, b, *args, **kwargs):
        if getattr(a, "is_meta", False) or getattr(b, "is_meta", False):
            tie_unchecked.append([list(a.shape), list(b.shape)])
            return tuple(a.shape) == tuple(b.shape)        # values unknowable on meta; different shapes are never equal
        return orig_equal(a, b, *args, **kwargs)
    torch.equal = meta_equal
    # load provenance (G1): which targets the library itself set, and how it filled every other tensor (loadtrace)
    import loadtrace
    set_ids, pre, tracer_box = {}, {}, {}
    orig_spf, orig_fin = CML.set_param_for_module, MU.PreTrainedModel._finalize_model_loading
    orig_fin_attr = MU.PreTrainedModel.__dict__["_finalize_model_loading"]      # the staticmethod object itself
    def recording_spf(model_, target_name, param_value, loading_info, *a, **k):
        mm = len(loading_info.mismatched_keys)
        r = orig_spf(model_, target_name, param_value, loading_info, *a, **k)
        if len(loading_info.mismatched_keys) == mm:
            mp, _, pn = target_name.rpartition(".")
            try:
                set_ids[target_name] = getattr(model_.get_submodule(mp) if mp else model_, pn)   # the object (an id can be recycled)
            except Exception:
                set_ids[target_name] = None
        return r
    def traced_finalize(model_, load_config, loading_info):
        pre.update(missing=set(loading_info.missing_keys), unexpected=set(loading_info.unexpected_keys),
                   mismatched={str(x[0]) if isinstance(x, (tuple, list)) else str(x) for x in loading_info.mismatched_keys},
                   conversion_errors={k: str(v)[-300:] for k, v in loading_info.conversion_errors.items()},
                   quantizer=type(load_config.hf_quantizer).__name__ if load_config.hf_quantizer else None)
        tr = loadtrace.InitTracer(loaded=set_ids); tracer_box["t"] = tr   # knows what the loader set (F03)
        with tr:
            return orig_fin(model_, load_config, loading_info)
    CML.set_param_for_module = recording_spf
    MU.PreTrainedModel._finalize_model_loading = staticmethod(traced_finalize)
    def _load(ignore):
        with torch.device("meta"):
            return cls.from_pretrained(d, output_loading_info=True, device_map="meta", ignore_mismatched_sizes=ignore, **kw)
    refusal, refused_keys = None, set()
    try:
        try:
            model, info = _load(False)                     # exactly as a user loads it (EXECUTION_PLAN P0.5)
        except RuntimeError as e:
            if not pre.get("mismatched"):
                raise
            # the library itself refuses this checkpoint for this class (mismatched shapes). A forced load follows only
            # as diagnostic evidence (which keys it discards, how it fills them): it never certifies anything.
            refused_keys = set(pre["mismatched"])
            refusal = {"error": f"{type(e).__name__}: {str(e)[:300]}", "mismatched": sorted(refused_keys)[:50],
                       "count": len(refused_keys)}
            for box in (opened, renamed_to, set_ids, pre, tracer_box, conversions): box.clear()
            tie_unchecked.clear()
            model, info = _load(True)
    finally:
        MU.safe_open = orig
        CML.rename_source_key = orig_rename
        CML.WeightTransform.add_tensor = orig_add
        torch.equal = orig_equal
        CML.set_param_for_module = orig_spf
        MU.PreTrainedModel._finalize_model_loading = orig_fin_attr                # restored as the staticmethod it was
    loaded = {k for k, _ in model.named_parameters()} - set(info.get("missing_keys", []))
    # size accounting: every tensor in the files the library read must land in the model's state (params or
    # persistent buffers, tied names counted per name) or be reported by the library as unexpected/mismatched.
    read = {k: v for f in opened for k, v in headers.get(f, {}).items()}
    def nel(s):
        x = 1
        for i in s: x *= int(i)
        return x
    missing = set(info.get("missing_keys", []))
    sd = model.state_dict(keep_vars=True)
    buffer_names = {n for n, _ in model.named_buffers(remove_duplicate=False)}
    for mn, mod in model.named_modules():               # non-persistent buffers too (e.g. stored position_ids)
        for bn in getattr(mod, "_non_persistent_buffers_set", set()):
            buffer_names.add(f"{mn}.{bn}" if mn else bn)
    # tied names are one tensor: an alias of a tensor that the file already fills is not extra state
    read_names = set(headers_read_names(opened, headers))
    read_ids = {id(sd[k]) for k in sd if k in read_names}
    state, seen_ids = {}, set()
    for k, t in sd.items():
        if k in missing or k in buffer_names:
            continue
        if id(t) in read_ids and k not in read_names:
            continue                                    # tied alias of a shipped tensor
        if id(t) in seen_ids and k not in read_names:
            continue
        seen_ids.add(id(t)); state[k] = t.numel()
    reported = set(info.get("unexpected_keys", [])) | {str(x[0]) for x in info.get("mismatched_keys", [])}
    # the library reports keys under ITS names (after renames): the checkpoint source(s) of each reported key
    src_of = {}
    for s_, names in renamed_to.items():
        for n_ in names:
            if n_ in reported: src_of.setdefault(n_, []).append(s_)
    reported = reported | {s_ for v in src_of.values() for s_ in v}      # file names too, for the size accounting
    read_numel = sum(nel(s) for k, (s, _) in read.items())
    reported_numel = sum(nel(read[k][0]) for k in reported if k in read)
    # same-name tensors are accounted directly; only the rest (renamed / merged / split / dropped) is compared by size
    # a buffer renamed on load (e.g. relative_position_index): same last name part AND same shape as a model buffer
    #   (an INTEGER/bool table in the file whose name part and element count match a model buffer, e.g. Swin's
    #    144x144 relative_position_index stored flat as 20,736 in the model; learned weights are never integer tables)
    buf_sig = set()
    for bn, b in model.named_buffers(remove_duplicate=False):
        buf_sig.add((bn.rsplit(".", 1)[-1], b.numel()))
    for mn, mod in model.named_modules():
        for bn in getattr(mod, "_non_persistent_buffers_set", set()):
            b = getattr(mod, bn, None)
            if b is not None: buf_sig.add((bn, b.numel()))
    INT = ("I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "BOOL")
    def stored_buffer(k, s, d):
        return d in INT and (k.rsplit(".", 1)[-1], nel(s)) in buf_sig
    # keys the library DECLARES it ignores (e.g. MTP) are counted by the declared-unbuilt check, not as silent drops
    aux = set(aux_keys)                        # auxiliary tensors (e.g. FP8 scales), as the harness classifies them
    import re as _re
    ign = [_re.compile(p) for p in (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])]
    declared = lambda k: any(p.search(k) for p in ign)
    rest_read = {k: nel(s) for k, (s, d) in read.items() if k not in state and k not in reported and k not in buffer_names
                 and k not in sd and not stored_buffer(k, s, d) and not declared(k) and k not in aux}
    # a persistent buffer the library filled under a NEW name (e.g. timm_wrapper's "timm_model." prefix, a
    # conversion-mapping rename like ".conv.batch_norm" -> ".conv.norm"): paired by the library's OWN rename record
    # (file key -> model key) and only if the library actually filled that buffer (it marks every tensor it sets
    # with _is_hf_initialized). Buffers are kept out of `state` above, so without this they look like drops.
    paired = set()
    for k in list(rest_read):
        for bn in sorted(renamed_to.get(k, ())):
            if bn != k and bn in buffer_names and bn in sd and bn not in read and bn not in paired \
                    and getattr(sd[bn], "_is_hf_initialized", False) and sd[bn].numel() == rest_read[k]:
                paired.add(bn); del rest_read[k]; break
    # a shipped tensor the library maps onto a TIED target while its tie value check could not run: it is either an
    # equal duplicate (discarded) or the real separate weight — named here, never counted as an unidentified drop
    tie_dups = []
    if tie_unchecked:
        sizes = {n_: t.numel() for n_, t in model.named_parameters(remove_duplicate=False)}
        free = {t_: sizes.get(t_) for t_ in (getattr(model, "all_tied_weights_keys", None) or {}).keys()}
        for k in sorted(rest_read):
            hit = next((t_ for t_ in ([k] + sorted(renamed_to.get(k, ()))) if t_ in free), None)
            if hit is not None and free[hit] == rest_read[k]:       # exactly the target's size, one key per target
                tie_dups.append(k); del rest_read[k]; free.pop(hit)
    rest_state = {k: n for k, n in state.items() if k not in read}
    # quantized codes the installed library does not interpret (no quantizer), from the checkpoint headers alone
    # (config.json may not say: metadata can live elsewhere, or be stale)
    float_params = {n_ for n_, p_ in model.named_parameters(remove_duplicate=False) if p_.is_floating_point()}
    qev = quantized_evidence(read, {k: renamed_to.get(k) or {k} for k in read}, float_params, set(sd),
                             {b.rsplit(".", 1)[-1] for b in buffer_names}, aux, declared)
    # refused keys the quantized codes do not explain (an unrelated float mismatch is still the library's refusal)
    ev_keys = {k for k, _ in qev}
    _srcs = {}
    for s_, names in renamed_to.items():
        for n_ in names: _srcs.setdefault(n_, set()).add(s_)
    refusal_unexplained = sorted(k for k in (refused_keys if refusal else ()) if not ((_srcs.get(k) or {k}) & ev_keys))
    quantizer = pre.get("quantizer")
    load_state = load_state_of(qev, quantizer, refusal)
    int_ev = int_table_evidence(model, read, renamed_to, set_ids, buffer_names, declared, aux)
    try:
        provenance = _provenance(model, info, set_ids, pre, tracer_box.get("t"), tie_unchecked, headers, opened, d)
    except Exception as e:                  # a bug here must not throw away the library's load report: say so instead
        import traceback
        provenance = {"error": f"{type(e).__name__}: {str(e)[:200]} at {traceback.extract_tb(e.__traceback__)[-1].lineno}"}
    state_numel = sum(state.values())
    try:                                    # F04: the loaded model's identity (G2 compares the executed model against it)
        identity = model_identity(model)
        identity["conversions"] = {t: {"sources": sorted(v["sources"]), "source_dtypes": sorted({read[x][1] for x in v["sources"] if x in read}),
                                       "aux_sources": sorted(x for x in v["sources"] if x in set(aux_keys)),
                                       "ops": v["ops"], "quant_op": v["quant_op"]} for t, v in conversions.items()}
        identity["quantizer"] = pre.get("quantizer")
        identity["quantized_codes"] = sorted({t_ for k_, _ in qev for t_ in (renamed_to.get(k_) or {k_})})
        identity["load_kw"] = {k: str(v) for k, v in kw.items() if k != "config"}
    except Exception as e:
        identity = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    du = int(os.popen(f"du -sk '{d}'").read().split()[0])
    pshape = {k: t.numel() for k, t in model.named_parameters()}
    return {"files_read": sorted(opened), "files_all": sorted(headers), "read_keys": sorted(read),
            "read_numel": read_numel, "reported_numel": reported_numel, "state_numel": state_numel,
            "unaccounted_numel": max(0, sum(rest_read.values()) - sum(rest_state.values())),
            "unmatched_read_keys": sorted(rest_read)[:200], "renamed_buffers_paired": len(paired),
            "unexpected": sorted(info.get("unexpected_keys", [])), "missing": sorted(info.get("missing_keys", [])),
            "mismatched": [list(map(str, x)) for x in info.get("mismatched_keys", [])],
            "loaded_params": len(loaded), "model_params": sum(1 for _ in model.named_parameters()),
            "missing_param_numel": sum(pshape.get(k, 0) for k in info.get("missing_keys", [])),
            "reported_sources": {k: sorted(v) for k, v in src_of.items()},
            "tie_value_check_unevaluable": tie_unchecked, "tie_unchecked_shipped": sorted(tie_dups),
            "stub_disk_kb": du, "provenance": provenance, "identity": identity,
            # load authority state (P0.5): loaded | refused | quantized_interpretation_unknown; a refused load's fields
            # come from the forced diagnostic recovery (evidence only)
            "load_state": load_state, "refusal": refusal, "forced_diagnostic_recovery": refusal is not None,
            "quantized_evidence": [list(x) for x in qev[:20]], "quantized_evidence_count": len(qev),
            "refusal_unexplained_by_quantization": refusal_unexplained[:50],
            # F05: what each shipped integer / bool tensor is, by the library's own placement (never by dtype alone)
            "int_table_evidence": int_ev}


def int_table_evidence(model, read, renamed_to, set_ids, buffer_names, declared, aux=()):
    """F05: for every integer / bool checkpoint tensor the library read (file key), the library's own evidence of what
    it is, or None (meaning not established):
      loaded_buffer      : the loader filled a persistent buffer with it (checkpoint state: a G3 obligation)
      loaded_param       : the loader filled a parameter with it
      nonpersistent_slot : it names a non-persistent buffer: the library declares that table derived (never loaded)
      declared_ignorable : the library declares the key ignorable (_keys_to_ignore_on_load_unexpected)
      recomputed         : HINT only: the model computes its own buffer of the same name part and size elsewhere
                           (the shipped values are never compared: not evidence, the key stays an unknown)
      mask_shape         : a 1x1xNxN mask whose N is a position length of the model
    Quantized codes are judged by quantized_evidence, not here (an integer that lands in a float parameter is codes)."""
    import common
    try:
        pos = common.pos_lengths(model.config)
    except Exception:
        pos = set()
    sd = model.state_dict(keep_vars=True)
    pnames = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    nonpers = set()
    for mn, mod in model.named_modules():
        for bn in getattr(mod, "_non_persistent_buffers_set", set()):
            nonpers.add(f"{mn}.{bn}" if mn else bn)
    sig = common.recomputed_signatures(model, loaded=set(set_ids))
    out = {}
    for k, (s, d_) in read.items():
        if not (d_ in common.INT_DTYPES or d_ == "BOOL"):
            continue                                # (names in the auxiliary regex get evidence too: a name is not evidence)
        tg = set(renamed_to.get(k) or ()) | {k}
        n = 1
        for i in s: n *= int(i)
        if any(t in set_ids and t in sd and t in pnames for t in tg):
            ev = "loaded_param"
        elif any(t in set_ids and t in sd and t not in pnames for t in tg):
            ev = "loaded_buffer"
        elif any(t in nonpers for t in tg):
            ev = "nonpersistent_slot"
        elif declared(k) or any(declared(t) for t in tg):
            ev = "declared_ignorable"
        elif len(s) == 4 and s[0] == s[1] == 1 and s[2] == s[3] and s[2] in pos:
            ev = "mask_shape"
        elif (k.rsplit(".", 1)[-1], n) in sig:
            ev = "recomputed"
        else:
            ev = None
        out[k] = ev
    return out


def buffer_groups(model):
    """{state-dict name: (module class, buffer name)} for EVERY persistent buffer of the model, whatever its dtype
    (F05: an integer table is checkpoint state as much as a floating one; dtype is not meaning)."""
    sd = model.state_dict(keep_vars=True)
    pn = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    return {k: (type(model.get_submodule(k.rpartition(".")[0]) if "." in k else model).__name__, k.rpartition(".")[2])
            for k in sd if k not in pn}


def matcher_buffer_split(unused, matched, buffer_group, used, passes=None):
    """The name-matcher path (no library load): `unused` [(checkpoint key, numel)] of matched-but-unread tensors.
    Checkpoint keys matched onto a persistent buffer are judged by buffer_gaps, never as unread learned weights (F05):
    returns (unused without the buffers buffer_gaps does not fail, buffer names that are an R-BUF unknown)."""
    loaded = {h for h in matched.values() if h in buffer_group}
    bfail, bunk = buffer_gaps(buffer_group, loaded, used, passes)
    keep = [(c, n) for c, n in unused if matched.get(c) not in buffer_group or matched.get(c) in set(bfail)]
    return keep, bunk


def buffer_gaps(buffer_group, loaded, used, passes=None):
    """THE G3 rule for checkpoint-loaded persistent buffers (any dtype), on every path (library authority and name
    matcher): (fail, unknown) name lists.
      fail    : unread while a buffer of the same kind (module class, buffer name) WAS read, in a pass where this
                buffer's module also ran, or this module never ran in any pass (a module-level gap): a real coverage gap.
                (Per pass, F05: a counter that only the training pass touches, in a module the training pass skips but
                inference runs, is not shown skipped where its siblings were read: an unknown, not a failure.)
                passes = [(names read, modules run)] per credited pass; None = all reads as one pass;
      unknown : unread and no buffer of its kind was read in the modes tried: not proven dead (R-BUF 2026-10-13: trying
                modes never proves that no supported input reads it). Dead state needs evidence covering the approved
                scope (the library's own declaration); transformers 5.12.1 has no such declaration for buffers."""
    shipped = [k for k in buffer_group if k in loaded]
    unread = [k for k in shipped if k not in used]
    if passes is None:
        groups_read = {buffer_group[k] for k in shipped if k in used}
        fail = [k for k in unread if buffer_group[k] in groups_read]
    else:
        def ran(mod, rd, run):                      # the buffer's module executed in this pass
            return mod == "" or mod in run or any(n.startswith(mod + ".") for n in rd)
        per = [({buffer_group[k] for k in shipped if k in rd}, rd, run) for rd, run in passes]
        fail = []
        for k in unread:
            mod, g = k.rpartition(".")[0], buffer_group[k]
            if not any(g in gr for gr, _, _ in per):
                continue
            ever = any(ran(mod, rd, run) for _, rd, run in per)
            if not ever or any(g in gr and ran(mod, rd, run) for gr, rd, run in per):
                fail.append(k)
    return sorted(fail), sorted(k for k in unread if k not in set(fail))


def _final(tracer, t):
    """(class, write log) of a tensor the loader did not set, from the ordered provenance tracker (F03)."""
    import loadtrace
    if tracer is None:
        return "UNINITIALIZED", []
    return tracer.final_class(t), [f for f, _, _ in tracer.writes.get(id(loadtrace._root(t)), [])]


def _provenance(model, info, set_ids, pre, tracer, tie_unchecked, headers, opened, d):
    """G1/G2 evidence from the library's own load: every state-dict tensor the library did not fill from the checkpoint,
    with how it was filled (loadtrace classes), what the library silenced, conversion errors, weight files it never
    opened (and whether they only duplicate tensors it read), the raw vs library tie flag, and the loaded shapes."""
    import collections, math, loadtrace
    sd = model.state_dict(keep_vars=True)
    pnames = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    kept = {k for k in set_ids if k in sd and set_ids[k] is sd[k]}           # still the tensor the loader set
    kept_ids = {id(sd[k]) for k in kept}
    tied = dict(getattr(model, "all_tied_weights_keys", None) or {})
    nonloaded = {}
    mism = set(pre.get("mismatched") or ())      # filled only by a forced diagnostic load (the library refused these)
    for k, t in sd.items():
        if k in mism:
            nonloaded[k] = {"kind": "param" if k in pnames else "buffer", "numel": t.numel(), "class": "MISMATCHED_DISCARDED"}
            continue
        if k in kept:
            # F03: any change of a loaded tensor after loading is evidence (re-randomised, partly zeroed, rescaled, ...)
            w = tracer.writes.get(id(loadtrace._root(t)), []) if tracer else []
            lc = tracer.final_class(t, loaded=True) if tracer else "LOADED"
            if lc != "LOADED":
                nonloaded[k] = {"kind": "param" if k in pnames else "buffer", "numel": t.numel(),
                                "init_fns": ["(changed after load)"] + [f for f, _, _ in w][:5],
                                "class": {"RANDOM": "INIT_RANDOM"}.get(lc, lc)}
            continue
        kind = "param" if k in pnames else "buffer"
        if k in set_ids and (k in tied and tied[k] in sd and sd[tied[k]] is t):   # replaced by its tie source
            c = "TIED_BOTH_SHIPPED_VALUE_UNVERIFIED" if tie_unchecked else "TIED_REPLACED"
            nonloaded[k] = {"kind": kind, "class": c, "numel": t.numel(), "tie_source": tied.get(k)}
        elif k in set_ids and id(t) not in kept_ids:   # loaded, then replaced by something that is not its tie
            st, fns = _final(tracer, t)
            nonloaded[k] = {"kind": kind, "class": "INIT_" + st if st not in ("UNINITIALIZED", "UNPROVEN") else st,
                            "numel": t.numel(), "init_fns": ["(replaced after load)"] + fns[:5]}
        elif id(t) in kept_ids:
            continue                                            # tied alias of a loaded tensor: the same values
        else:
            st, fns = _final(tracer, t)
            c = "INIT_" + st if st not in ("UNINITIALIZED", "UNPROVEN") else st
            nonloaded[k] = {"kind": kind, "class": c, "numel": t.numel(), "init_fns": fns[:6]}
    # non-persistent buffers (not in the state dict) are assessed too (F03-5): certified only if computed / constant
    for n_, b_ in model.named_buffers(remove_duplicate=False):
        if n_ in sd or n_ in nonloaded:
            continue
        st, fns = _final(tracer, b_)
        if st not in ("DETERMINISTIC", "COMPUTED", "CKPT_DERIVED"):
            nonloaded[n_] = {"kind": "buffer_nonpersistent", "class": "INIT_" + st if st not in ("UNINITIALIZED", "UNPROVEN") else st,
                             "numel": b_.numel(), "init_fns": fns[:6]}
    for k, v in nonloaded.items():               # did the loader ever write this name from the checkpoint (P0.6 D2-C14)
        v["set_by_loader"] = k in set_ids
    # a tied alias the library replaced by its source carries the SOURCE's provenance when the source itself was not
    # loaded (e.g. a shipped alias head replaced by an unshipped, re-initialised source: random, not "shipped twice")
    # (random / unproven / discarded sources pass their class on; a source filled by deterministic or computed init means
    #  the shipped alias values were discarded for non-checkpoint values: never certified, an unknown; classes are read
    #  from a snapshot, so the result does not depend on iteration order)
    orig = {k: v["class"] for k, v in nonloaded.items()}
    for k, v in nonloaded.items():
        src = v.get("tie_source")
        if orig[k] in ("TIED_REPLACED", "TIED_BOTH_SHIPPED_VALUE_UNVERIFIED") and src in orig \
                and orig[src] not in ("TIED_REPLACED", "TIED_BOTH_SHIPPED_VALUE_UNVERIFIED"):
            inherit = orig[src] in RANDOM_CLASSES + ("UNPROVEN", "MISMATCHED_DISCARDED", "LOADED_MODIFIED", "LOADED_OVERWRITTEN")
            v["class"], v["inherited_from"] = (orig[src] if inherit else "TIED_ALIAS_DISCARDED"), src
    # weight files the library never opened: a file whose every tensor shape is already among the read tensors'
    # shapes (multiset inclusion) only duplicates them (e.g. a second export); anything else is unexamined weight
    read_named = {(k, s) for f in opened for k, (s, _) in headers.get(f, {}).items()}
    not_opened = []
    for f in sorted(set(headers) - set(opened)):
        sh = collections.Counter(s for s, _ in headers[f].values())
        not_opened.append({"file": f, "tensors": len(headers[f]), "numel": sum(math.prod(s) for s in sh.elements()),
                           "duplicate": bool(sh) and all((k, s) in read_named for k, (s, _) in headers[f].items())})
    raw = {}
    try:
        raw = json.load(open(os.path.join(d, "config.json")))
    except Exception:
        pass
    cfg = getattr(model, "config", None)
    try:
        lib_tie = getattr(cfg.get_text_config(), "tie_word_embeddings", None)
    except Exception:
        lib_tie = getattr(cfg, "tie_word_embeddings", None)
    raw_text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    return {"nonloaded": nonloaded,
            "conversion_errors": pre.get("conversion_errors") or {},
            "missing_silenced_by_ignore": sorted(pre.get("missing", set()) - set(info.get("missing_keys", [])) - set(tied))[:50],
            "unexpected_silenced_by_ignore": sorted(pre.get("unexpected", set()) - set(info.get("unexpected_keys", [])))[:50],
            "files_not_opened": not_opened, "quantizer": pre.get("quantizer"),
            "raw_tie_word_embeddings": raw.get("tie_word_embeddings", raw_text.get("tie_word_embeddings", "<absent>")),
            "library_tie_word_embeddings": lib_tie,
            "state_shapes": {k: list(t.shape) for k, t in sd.items()}}


def model_identity(model):
    """What G2 promises is the same in the loaded and the executed model (INTENT G2, F04): every module (name -> class),
    module objects registered under several names, every parameter (shape, dtype) and buffer (shape, dtype, persistent),
    and the tie partition (names that are ONE tensor object). Works on meta and real models alike."""
    sd = model.state_dict(keep_vars=True)
    def cname(x):
        t = type(x)
        return f"{t.__module__}.{t.__qualname__}"
    mods, mgroups, attrs = {}, {}, {}
    for n, x in model.named_modules(remove_duplicate=False):
        mods[n or "(root)"] = cname(x)
        mgroups.setdefault(id(x), []).append(n or "(root)")
        for k, v in vars(x).items():                # plain settings a forward reads (scaling, is_causal, eps, stride ...)
            # public settings only (private ones are loader bookkeeping, e.g. _use_kernels); MODULE_ATTRS_NOT_COMPARED:
            # where the model was loaded from (the library's load writes its own mirror path there)
            if k.startswith("_") or k == "training" or k in MODULE_ATTRS_NOT_COMPARED:
                continue
            if v is None or isinstance(v, (bool, int, float, str)) or (
                    isinstance(v, (tuple, list)) and len(v) <= 16 and all(x_ is None or isinstance(x_, (bool, int, float, str)) for x_ in v)):
                attrs[f"{n or '(root)'}.{k}"] = repr(v)
    groups = {}
    params = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        params[n] = [list(p.shape), str(p.dtype).replace("torch.", "")]
        groups.setdefault(id(p), []).append(n)
    buffers = {}
    for n, b in model.named_buffers(remove_duplicate=False):
        buffers[n] = [list(b.shape), str(b.dtype).replace("torch.", ""), n in sd]
        groups.setdefault(id(b), []).append(n)
    return {"modules": mods, "params": params, "buffers": buffers,
            "ties": sorted(sorted(g) for g in groups.values() if len(g) > 1),
            "module_ties": sorted(sorted(g) for g in mgroups.values() if len(g) > 1),
            "config": config_view(getattr(model, "config", None)), "module_attrs": attrs}


# config fields that are not the model's computation: precision (the declared float32 build, F04-2) and bookkeeping
# (quantization_config: how the checkpoint is STORED; judged by the quantized-layout / dequantization rules, F04-3)
CONFIG_NOT_COMPARED = ("dtype", "torch_dtype", "transformers_version", "_name_or_path", "_commit_hash", "quantization_config")
MODULE_ATTRS_NOT_COMPARED = ("name_or_path",)            # (mirrors the config's _name_or_path: a load location)


def _is_cfg(v):
    return any(c.__name__ in ("PreTrainedConfig", "PretrainedConfig") for c in type(v).__mro__)


def config_view(cfg, path=""):
    """The config the model was built from, flattened {path: repr(value)}, incl. every sub-config's private attention /
    experts implementation (to_dict omits them), minus CONFIG_NOT_COMPARED. None when there is no config."""
    if cfg is None or not _is_cfg(cfg):
        return None
    out = {}
    def walk(c, pre, seen):
        if id(c) in seen:
            return
        seen.add(id(c))
        try:
            d = c.to_dict()
        except Exception:
            d = {}
        def flat(prefix, val, depth=0):
            if isinstance(val, dict) and depth < 4:
                for k2, v2 in val.items():
                    if k2 not in CONFIG_NOT_COMPARED:
                        flat(f"{prefix}.{k2}", v2, depth + 1)
            else:
                out[prefix] = repr(val)
        for k, v in d.items():
            if k in CONFIG_NOT_COMPARED:
                continue
            flat(f"{pre}{k}", v)
        for k, v in vars(c).items():
            if k.startswith(("_attn_implementation", "_experts_implementation")):
                out[f"{pre}{k}"] = repr(v)
            elif _is_cfg(v):
                walk(v, f"{pre}{k}.", seen)
    walk(cfg, path, set())
    return out


FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
QUANT_STORAGE = ("F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "I8", "U8", "I16", "U16", "I32", "U32")   # header formats of codes


def _partition(groups, names):
    """A tie partition restricted to names present in both models, as a set of frozensets (size > 1 only)."""
    out = set()
    for g in groups:
        g2 = frozenset(n for n in g if n in names)
        if len(g2) > 1:
            out.add(g2)
    return out


def g2_compare(L, E, executed_build=None):
    """G2 executed = loaded (INTENT; F04), from the two identities (model_identity) plus the library's conversion record.
      L: the loaded identity (from the library's own from_pretrained, `identity` of a load report);
      E: the executed identity (model_identity of the model the recorder ran);
      executed_build: what the harness declares about its build ({"attn_implementation", "dtype", "zero_storage"}).
    Allowed and NAMED (evidence, not names of models or methods):
      - float16 / bfloat16 / float32 -> float32: the harness executes one float32 build; learned values are zeros by design
        (the declared abstraction); float64 -> float32 rounds values: a G2 unknown (F04-2);
      - a target the library itself built from quantized storage together with an auxiliary tensor (its own conversion
        record: >= 2 sources, >= 1 operation): dequantized at load by the library, named with its operations.
    Unknown (never FULL): quantized codes the library cast to floats without such a conversion; a loaded layout that keeps
      quantized (non-floating) parameters the executed model runs as floats (the harness, not the library, dequantized).
    Fail (observed difference -> PARTIAL): different module names or classes, module aliasing, parameter / buffer names,
      shapes, persistence, non-floating dtypes, or tie groups.
    Returns {"fail": {...}, "unknowns": [...], "named": {...}}."""
    import collections
    if not L or L.get("error") or not E or E.get("error"):
        why = (L or {}).get("error") or (E or {}).get("error") or "no identity"
        return {"fail": {}, "named": {}, "unknowns": [{"predicate": "G2", "what": f"the executed model could not be "
                f"compared with the library-loaded model ({why})", "items": [], "count": 0}]}
    fail, unknowns, named = collections.defaultdict(list), [], {}
    def unk(what, items):
        if items: unknowns.append({"predicate": "G2", "what": what, "items": sorted(items)[:20], "count": len(items)})
    # modules
    lm, em = L["modules"], E["modules"]
    fail["modules_differ"] = [f"{n}: {'loaded only' if n in lm else 'executed only'}" for n in sorted(set(lm) ^ set(em))]
    fail["module_classes_differ"] = [f"{n}: {lm[n].rsplit('.', 1)[-1]} loaded, {em[n].rsplit('.', 1)[-1]} executed"
                                     for n in sorted(set(lm) & set(em)) if lm[n] != em[n]]
    both_m = set(lm) & set(em)
    if _partition(L.get("module_ties", []), both_m) != _partition(E.get("module_ties", []), both_m):
        fail["module_aliasing_differs"] = [" = ".join(sorted(g)) for g in
                                           _partition(L.get("module_ties", []), both_m) ^ _partition(E.get("module_ties", []), both_m)][:20]
    # a quantized layout the loaded model keeps: a module whose loaded state holds non-floating tensors (codes, as
    # parameters or persistent buffers) that the executed module does not hold, where the executed module runs floating
    # parameters instead. The harness (not the library) interpreted it: the module (class, parameters, buffers) is
    # judged as ONE unknown, never as a structural failure (F04-3)
    lp, ep = L["params"], E["params"]
    lb, eb = L["buffers"], E["buffers"]
    isf = lambda dt: dt in FLOAT_DTYPES or dt.startswith("complex")
    par = lambda n: n.rpartition(".")[0] or "(root)"
    lstate = collections.defaultdict(dict); estate = collections.defaultdict(dict)
    for n, (_, dt) in lp.items(): lstate[par(n)][n] = dt
    for n, (_, dt, per) in lb.items():
        if per: lstate[par(n)][n] = dt
    for n, (_, dt) in ep.items(): estate[par(n)][n] = dt
    for n, (_, dt, per) in eb.items():
        if per: estate[par(n)][n] = dt
    # evidence PER MODULE: one of its own non-floating loaded tensors is a quantized code the library identified, or a
    # conversion target built from quantized storage (never "the model has a quantizer somewhere")
    codes = set(L.get("quantized_codes") or ())
    qtargets = {t for t, c in (L.get("conversions") or {}).items() if any(d in QUANT_STORAGE for d in c.get("source_dtypes", []))}
    is_code = lambda n: n in codes or n in qtargets or any(n.startswith(t.replace(".*.", ".") + ".") or n == t.replace(".*.", ".") for t in qtargets if "*" in t)
    qmods = {mo for mo, st in lstate.items()
             if any(not isf(dt) and estate[mo].get(n) != dt and is_code(n) for n, dt in st.items())
             and any(isf(dt) for n, dt in estate[mo].items() if n in ep)}
    # inside such a module only the code layout is the harness's interpretation: names present on ONE side, or
    # non-floating on the loaded side, are part of it; floating state present on both sides is still compared
    inq = lambda n: par(n) in qmods and (n not in (set(lp) | set(lb)) or n not in (set(ep) | set(eb))
                                         or not isf((lp.get(n) or lb.get(n))[1]))
    harness_deq, f64 = sorted(qmods), []
    fail["module_classes_differ"] = [x for x in fail["module_classes_differ"] if x.split(":", 1)[0] not in qmods]
    for n in sorted(set(lp) | set(ep)):
        if inq(n):
            continue
        if n not in ep:
            fail["parameters_differ"].append(f"{n}: loaded only")
        elif n not in lp:
            fail["parameters_differ"].append(f"{n}: executed only")
        elif lp[n][0] != ep[n][0]:
            fail["parameter_shapes_differ"].append(f"{n}: {lp[n][0]} loaded, {ep[n][0]} executed")
        elif lp[n][1] != ep[n][1] and not (lp[n][1] in FLOAT_DTYPES and ep[n][1] == "float32"):
            fail["parameter_dtypes_differ"].append(f"{n}: {lp[n][1]} loaded, {ep[n][1]} executed")
        elif lp[n][1] == "float64" and ep[n][1] == "float32":
            f64.append(n)
    unk("modules the library loads in a quantized layout (codes as non-floating parameters / buffers) that the executed "
        "model runs as floating modules: the harness, not the library, chose that interpretation", harness_deq)
    # buffers
    for n in sorted(set(lb) | set(eb)):
        if inq(n):
            continue
        if n not in eb:
            fail["buffers_differ"].append(f"{n}: loaded only")
        elif n not in lb:
            fail["buffers_differ"].append(f"{n}: executed only")
        else:
            (ls, ld, lper), (es, ed, eper) = lb[n], eb[n]
            if lper != eper:
                fail["buffer_persistence_differs"].append(f"{n}: {'persistent' if lper else 'non-persistent'} loaded")
            if ls != es:
                fail["buffer_shapes_differ"].append(f"{n}: {ls} loaded, {es} executed")
            elif ld != ed and not (ld in FLOAT_DTYPES and ed == "float32"):
                fail["buffer_dtypes_differ"].append(f"{n}: {ld} loaded, {ed} executed")
            elif ld == "float64" and ed == "float32":
                f64.append(n)
    # F04-2 (Soumil 2026-10-14): float16 / bfloat16 / float32 -> float32 loses nothing; float64 -> float32 rounds values
    unk("float64 tensors the executed float32 build rounds (the values run are not the loaded values)", f64)
    # ties (one tensor object under several names): the partitions must be equal
    names = (set(lp) | set(lb)) & (set(ep) | set(eb))           # (ties are compared across every name both carry)
    lt, et = _partition(L.get("ties", []), names), _partition(E.get("ties", []), names)
    if lt != et:
        fail["tie_groups_differ"] = ([f"loaded only: {' = '.join(sorted(g))}" for g in lt - et]
                                     + [f"executed only: {' = '.join(sorted(g))}" for g in et - lt])[:20]
    # named conversions: float32 build, and the library's own dequantization at load
    conv = collections.Counter(f"{lp[n][1]}->float32" for n in set(lp) & set(ep)
                               if lp[n][1] != ep[n][1] and lp[n][1] in FLOAT_DTYPES and ep[n][1] == "float32")
    conv.update(f"{lb[n][1]}->float32 (buffer)" for n in set(lb) & set(eb)
                if lb[n][1] != eb[n][1] and lb[n][1] in FLOAT_DTYPES and eb[n][1] == "float32")
    named["declared_abstraction"] = ("learned values executed as zeros (zero-storage build); one float32 build"
                                     + (f"; attention implementation {executed_build.get('attn_implementation')}" if executed_build else ""))
    if conv:
        named["float32_conversion"] = dict(conv)
    deq, cast = [], []
    for t, c in sorted((L.get("conversions") or {}).items()):
        qsrc = [d for d in c.get("source_dtypes", []) if d in QUANT_STORAGE]
        if not qsrc or not c.get("ops"):
            continue
        tgt_float = any(isf(lp[x][1]) for x in lp if x == t or x.startswith(t.replace(".*.", ".")))
        if not tgt_float:
            continue                            # the target stays quantized (not dequantized: judged above as a layout)
        if c.get("aux_sources"):                # the library's own op consumed the codes WITH an auxiliary (scale) tensor
            deq.append((t, tuple(c["ops"]), tuple(c["source_dtypes"])))
        else:                                   # codes turned into floats with no auxiliary tensor: not shown dequantized
            cast.append(t)
    covered = {t for t, _, _ in deq}
    cast = sorted(set(cast) | {t for t in codes if t in lp and lp[t][1] in FLOAT_DTYPES
                               and not any(t == x or t.startswith(x.replace(".*.", ".")) for x in covered)})
    if deq:
        named["dequantized_at_load_by_library"] = {
            "count": len(deq), "quantizer": L.get("quantizer"),
            "operations": sorted({" + ".join(o) for _, o, _ in deq}), "source_dtypes": sorted({d for _, _, ds in deq for d in ds}),
            "examples": [t for t, _, _ in deq][:5]}
    unk("quantized codes the library cast to floats without dequantizing them with their auxiliary tensor (the "
        "executed values are not the checkpoint's)", cast)
    # the config the executed model was built from vs the one the library loaded with (attention implementation, eps,
    # scaling ...; precision excluded: F04-2): a difference is not shown to change the computation -> unknown
    lc, ec = L.get("config"), E.get("config")
    if lc is not None and ec is not None:
        unk("config fields differ between the library-loaded and the executed model (whether that changes the "
            "computation is not shown)", [f"{k}: {lc.get(k)} loaded, {ec.get(k)} executed" for k in sorted(set(lc) | set(ec)) if lc.get(k) != ec.get(k)])
    la, ea = L.get("module_attrs"), E.get("module_attrs")
    if la is not None and ea is not None:
        # settings BOTH models carry (one a loader adds to its own model only, e.g. a quantizer's is_quantized flag, is
        # bookkeeping, not a setting the two builds share)
        both = {k for k in set(la) & set(ea) if k.rsplit(".", 1)[0] in set(lm) & set(em) and k.rsplit(".", 1)[0] not in qmods}
        # F04-2: a float setting the library computed with tensors in the checkpoint's half precision (e.g. drop-path
        # rates from torch.linspace) differs from the float32 build only in the last digits: precision, named
        half = sorted({dt for _, dt in lp.values() if dt in ("float16", "bfloat16")})
        rtol = {"float16": 4 * 2 ** -10, "bfloat16": 4 * 2 ** -7}.get(half[0]) if len(half) == 1 else None
        def _num(v):
            import ast as _a
            try:
                x = _a.literal_eval(v)
            except Exception:
                return None
            if isinstance(x, float):
                return (x,)
            if isinstance(x, (tuple, list)) and x and all(isinstance(y, float) for y in x):
                return tuple(x)
            return None
        def _close(a, b):
            xa, xb = _num(a), _num(b)
            return (rtol is not None and xa is not None and xb is not None and len(xa) == len(xb)
                    and all(abs(p_ - q_) <= rtol * max(abs(p_), abs(q_)) for p_, q_ in zip(xa, xb)))
        diff = [k for k in sorted(both) if la.get(k) != ea.get(k)]
        prec = [k for k in diff if _close(la[k], ea[k])]
        if prec:
            named["settings_computed_in_loaded_precision"] = {"dtype": half[0], "count": len(prec), "examples": prec[:5]}
        unk("module settings differ between the library-loaded and the executed model (whether that changes the "
            "computation is not shown)", [f"{k}: {la.get(k)} loaded, {ea.get(k)} executed" for k in diff if k not in prec])
    return {"fail": {k: v[:20] for k, v in fail.items() if v}, "unknowns": unknowns, "named": named,
            "counts": {k: len(v) for k, v in fail.items() if v}}


def g2_decide(loaded_identity, build_ids, final_identity, executed_build):
    """The worker's whole G2 decision (F04), one function so it is tested as it runs: compare the library-loaded identity
    with the FIRST executed build; every later build and the model at grading time must equal it (else an unknown).
    Returns (unknowns, failure items for T1 or [], the record for R["g2"])."""
    try:
        G = g2_compare(loaded_identity, build_ids[0] if build_ids else final_identity, executed_build)
        if build_ids and any(b != build_ids[0] for b in list(build_ids[1:]) + [final_identity]):
            G["unknowns"].append({"predicate": "G2", "what": "the executed model differs between its builds or changed "
                                  "during the run (modules / state / ties)", "items": [], "count": 0})
    except Exception as e:                       # a comparator error is an unknown, never a crashed row
        G = {"fail": {}, "named": {}, "counts": {}, "unknowns": [{"predicate": "G2", "what": "the executed model could not "
             f"be compared with the library-loaded model (comparator error: {type(e).__name__}: {str(e)[:120]})", "items": [], "count": 0}]}
    fail_items = [f"{k} ({(G.get('counts') or {}).get(k, len(v))}): " + "; ".join(v[:5]) for k, v in G["fail"].items()]
    return G["unknowns"], fail_items, {"named": G["named"], "fail": G["fail"], "counts": G.get("counts")}


RANDOM_CLASSES = ("INIT_RANDOM", "UNINITIALIZED")


def initialised_not_shipped(nonloaded, cap=40):
    """F05-4 (R-NORM kept): the persistent state the library fills itself (a deterministic / constant init) because the
    checkpoint does not ship it, named on the row so nobody reads it as learned. Never random state (G1 judges that),
    never derived non-persistent buffers, never a key the loader itself set (a shipped value the library replaced)."""
    return sorted(k for k, v in (nonloaded or {}).items()
                  if str(v.get("class", "")).startswith("INIT_") and v["class"] not in RANDOM_CLASSES
                  and v.get("kind") != "buffer_nonpersistent" and not v.get("set_by_loader"))[:cap]
QUANT_INT = ("I8", "U8", "I16", "U16", "I32", "U32")      # integer formats packed weights use (I64 / BOOL are tables)


def quantized_evidence(read, targets, float_params, state_names, buffer_parts, aux, ignorable=lambda k: False):
    """Checkpoint tensors that are quantized codes, from headers and the library's own placement, never from names of
    models or quantization methods:
      - an integer tensor (I8..U32) that lands in a floating parameter, or lands nowhere and is not a stored table
        of a buffer slot the model has;
      - an FP8 tensor that lands in a floating parameter or nowhere while its module ships an auxiliary (scale) tensor
        (unscaled FP8 converts exactly; scaled FP8 read as plain floats is the codes, not the values).
    read: {file key: (shape, dtype)}; targets: {file key: model names the library mapped it to}; float_params: floating
    parameter names; state_names: every state-dict name; buffer_parts: last name parts of the model's buffers;
    aux: the harness's auxiliary keys; ignorable(k): the library DECLARES it ignores k (_keys_to_ignore_on_load_*), so it
    is not the model's weight. An integer/bool 1x1xNxN table is a stored mask, not codes. Returns [(file key, why)]."""
    aux_mods = {a.rpartition(".")[0] for a in aux}
    out = []
    for k in sorted(read):
        s_, d_ = read[k]
        if k in aux or ignorable(k):
            continue
        if len(s_) == 4 and s_[0] == s_[1] == 1 and s_[2] == s_[3] and (d_ in QUANT_INT or d_ in ("I64", "BOOL")):
            continue                                       # a stored square mask (e.g. a causal attention.bias)
        lands = targets.get(k) or {k}
        into_float = any(t in float_params for t in lands)
        nowhere = not any(t in state_names for t in lands)
        if not (into_float or nowhere):
            continue                                       # lands in a buffer / integer slot: a stored table
        where = "into a floating parameter" if into_float else "placed nowhere"
        if d_ in QUANT_INT and not (nowhere and k.rsplit(".", 1)[-1] in buffer_parts):
            out.append((k, f"{d_} codes {where}"))
        elif str(d_).startswith("F8") and k.rpartition(".")[0] in aux_mods:
            out.append((k, f"{d_} {where}, with a scale tensor in the same module"))
    return out


def loaded_params(param_names, rep):
    """Parameters the library filled from the checkpoint: every parameter minus the reported-missing keys and minus
    the ones the loader never wrote (a missing key the library silences with _keys_to_ignore_on_load_missing is not
    in the reported list). Shipped tensors the library overwrote, replaced or discarded after loading stay: they were
    shipped (P0.6 D2-C14)."""
    nl = (rep.get("provenance") or {}).get("nonloaded") or {}
    never = {k for k, v in nl.items() if v.get("kind") == "param" and v.get("set_by_loader") is False
             and v.get("class") != "MISMATCHED_DISCARDED"}
    return set(param_names) - set(rep.get("missing") or []) - never


def load_state_of(quantized_ev, quantizer, refusal):
    """quantized codes with no quantizer -> quantized_interpretation_unknown (whatever else happened: a refusal of
    packed codes is a consequence); else the library's own refusal -> refused; else loaded."""
    if quantized_ev and not quantizer:
        return "quantized_interpretation_unknown"
    return "refused" if refusal else "loaded"


def refuses(rep):
    """R-REFUSE (Soumil 2026-10-13): the installed library refuses its own default load of this checkpoint (reproduced
    on the real header layout): a refused load, or quantized codes plus a refusal the codes do not explain."""
    if not rep or "error" in rep:
        return False
    st = rep.get("load_state", "loaded")
    return st == "refused" or (st == "quantized_interpretation_unknown" and bool(rep.get("refusal_unexplained_by_quantization")))


def authority(rep):
    """What a library load report may certify (P0.5). Returns {"certifies": bool, "state", "unknown": text or None}.
    loaded: certifies. refused: the report is the forced diagnostic recovery; it still names the discarded keys (the
    refusal fails T1 through g1_assess's library_refuses_load), it certifies nothing else on its own.
    quantized_interpretation_unknown: no quantizer for quantized codes: the load is not the checkpoint's meaning."""
    if not rep or "error" in rep:
        return {"certifies": False, "state": None, "unknown": None, "fail": {}}
    st = rep.get("load_state", "loaded")
    if st == "quantized_interpretation_unknown":
        ev = rep.get("quantized_evidence") or []
        un = rep.get("refusal_unexplained_by_quantization") or []
        return {"certifies": False, "state": st, "fail": ({"library_refuses_load": list(un)} if un else {}),
                "unknown": f"the installed library has no quantizer for this checkpoint's quantized tensors "
                           f"({rep.get('quantized_evidence_count', len(ev))}: e.g. {', '.join(e[0] + ' ' + e[1] for e in ev[:2])})"}
    return {"certifies": True, "state": st, "unknown": None, "fail": {}}
TIE_UNVERIFIED = ("TIED_BOTH_SHIPPED_VALUE_UNVERIFIED", "TIED_REPLACED")


def g1_assess(prov, used, exec_shapes, buffer_group, passes=None):
    """The G1/G2 verdict inputs (INTENT.md; decisions in G1_LOAD_PROVENANCE_2026-10-09.md), from the library's load
    provenance and the recorded execution.
      prov         : _provenance(...) of the library load of the executed class
      used         : names of every parameter / buffer some recorded op read (tied aliases included)
      exec_shapes  : {state-dict name: shape} of the executed model
      buffer_group : {name: (module class, buffer name)} for the executed model's persistent buffers of ANY dtype
                     (buffer_groups; F05)
      passes       : [(names read, modules run)] per credited pass (buffer_gaps' per-pass sibling evidence; None = one pass)
    Returns {"fail": {...}, "unknowns": [...], "leftovers": {...}}: a non-empty fail fails T1 (PARTIAL); unknowns make an
    otherwise FULL row FULL_UNVERIFIED (incl. buffers unread in the modes tried, R-BUF); leftovers is empty since R-BUF."""
    if prov.get("error"):
        return {"fail": {}, "leftovers": {},
                "unknowns": [{"predicate": "G1", "what": f"load provenance could not be computed ({prov['error']})",
                              "items": [], "count": 0}]}
    nl = prov.get("nonloaded") or {}
    rnd = sorted(k for k, v in nl.items() if v["class"] in RANDOM_CLASSES)
    fail = {"random_init_executed": [k for k in rnd if k in used and nl[k]["class"] == "INIT_RANDOM"],   # random values
            "uninitialized_executed": [k for k in rnd if k in used and nl[k]["class"] == "UNINITIALIZED"],  # allocation garbage
            "conversion_errors": sorted(prov.get("conversion_errors") or {}),
            # the library refuses the default load (mismatched shapes): its own refusal, not a random fill (P0.5)
            "library_refuses_load": sorted(k for k, v in nl.items() if v["class"] == "MISMATCHED_DISCARDED")}
    unknowns = []
    def unk(pred, what, items):
        if items: unknowns.append({"predicate": pred, "what": what, "items": list(items)[:20], "count": len(items)})
    unk("G1", "learned tensors the checkpoint does not ship, randomly initialised or left uninitialised by the library, read "
        "by no recorded op",
        [k for k in rnd if k not in used])
    # F03-1 (Soumil 2026-10-13): shipped values the library changed while loading (padding rows zeroed, rescaled, ...)
    unk("G1", "shipped tensors the library changed after loading them (the executed values are not the shipped ones)",
        sorted(k for k, v in nl.items() if v["class"] in ("LOADED_MODIFIED", "LOADED_OVERWRITTEN")))
    unk("G1", "tensors filled by a copy of unknown provenance during the library's load", sorted(k for k, v in nl.items() if v["class"] == "UNPROVEN"))
    unk("G1", "tied tensors shipped twice: the library ties them only if their values are equal (not checkable without "
        f"the weights; config.json tie_word_embeddings={prov.get('raw_tie_word_embeddings')}, library "
        f"{prov.get('library_tie_word_embeddings')})", sorted(k for k, v in nl.items() if v["class"] in TIE_UNVERIFIED))
    unk("G1", "shipped tensors discarded for their tie source's non-checkpoint initialisation (the tie source is not "
        "shipped)", sorted(k for k, v in nl.items() if v["class"] == "TIED_ALIAS_DISCARDED"))
    unk("G1", "weight files the library never opens that are not duplicates of what it reads",
        [f"{x['file']} ({x['tensors']} tensors)" for x in prov.get("files_not_opened") or [] if not x["duplicate"]])
    ls = prov.get("state_shapes") or {}
    diff = sorted(k for k in set(ls) | set(exec_shapes) if list(ls.get(k, [])) != list(exec_shapes.get(k, [])) or (k in ls) != (k in exec_shapes))
    unk("G2", "the executed model's state differs from the library-loaded model's (names / shapes)", diff)
    # checkpoint-loaded persistent buffers (ANY dtype, F05) no op read: a sibling of the same kind WAS read -> a real
    # gap (fail); none of that kind was read in the modes tried -> not proven dead (R-BUF: an unknown, below)
    gap_fail, gap_unknown = buffer_gaps(buffer_group, {k for k in buffer_group if k not in nl}, used, passes)
    fail["buffers_unread_while_siblings_read"] = gap_fail
    # R-BUF (Soumil 2026-10-13): no recorded run read them in the modes tried, but trying modes never proves that no
    # supported input reads them: not proven dead -> an unknown (FULL_UNVERIFIED), never FULL_LEFTOVERS
    unk("G3", "checkpoint-loaded buffers no recorded run read in the modes tried (not proven dead: no supported input is "
        "shown to leave them unread)", gap_unknown)
    leftovers = {}
    return {"fail": {k: v for k, v in fail.items() if v}, "unknowns": unknowns,
            "leftovers": {k: v for k, v in leftovers.items() if v}}
