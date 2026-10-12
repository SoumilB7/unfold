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


class Authority:
    """The repo mirrored ONCE as sparse stubs; report(cls) runs the library's from_pretrained for any class over the
    same stubs (class selection compares candidates; the chosen class's report is reused for T1). close() deletes."""
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

    def report(self, cls, aux_keys=(), config=None):
        """One report per class. `config` is the config the worker executes (copied: the loader must not change it)."""
        import copy
        if cls.__name__ not in self.cache:
            kw = {"config": copy.deepcopy(config)} if config is not None else {}
            self.cache[cls.__name__] = _report(self, cls, aux_keys, **kw)
        return self.cache[cls.__name__]

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
        tr = loadtrace.InitTracer(); tracer_box["t"] = tr
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
            for box in (opened, renamed_to, set_ids, pre, tracer_box): box.clear()
            tie_unchecked.clear()
            model, info = _load(True)
    finally:
        MU.safe_open = orig
        CML.rename_source_key = orig_rename
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
    try:
        provenance = _provenance(model, info, set_ids, pre, tracer_box.get("t"), tie_unchecked, headers, opened, d)
    except Exception as e:                  # a bug here must not throw away the library's load report: say so instead
        import traceback
        provenance = {"error": f"{type(e).__name__}: {str(e)[:200]} at {traceback.extract_tb(e.__traceback__)[-1].lineno}"}
    state_numel = sum(state.values())
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
            "stub_disk_kb": du, "provenance": provenance,
            # load authority state (P0.5): loaded | refused | quantized_interpretation_unknown; a refused load's fields
            # come from the forced diagnostic recovery (evidence only)
            "load_state": load_state, "refusal": refusal, "forced_diagnostic_recovery": refusal is not None,
            "quantized_evidence": [list(x) for x in qev[:20]], "quantized_evidence_count": len(qev),
            "refusal_unexplained_by_quantization": refusal_unexplained[:50]}


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
            w = tracer.writes.get(id(t), []) if tracer else []
            if any(kd == "random" for _, _, kd in w):   # loaded, then re-randomised in place by the library's init code
                nonloaded[k] = {"kind": "param" if k in pnames else "buffer", "numel": t.numel(),
                                "init_fns": ["(overwritten after load)"] + [f for f, _, _ in w][:5], "class": "INIT_RANDOM"}
            continue
        kind = "param" if k in pnames else "buffer"
        if k in set_ids and (k in tied and tied[k] in sd and sd[tied[k]] is t):   # replaced by its tie source
            c = "TIED_BOTH_SHIPPED_VALUE_UNVERIFIED" if tie_unchecked else "TIED_REPLACED"
            nonloaded[k] = {"kind": kind, "class": c, "numel": t.numel(), "tie_source": tied.get(k)}
        elif k in set_ids and id(t) not in kept_ids:   # loaded, then replaced by something that is not its tie
            st, fns = loadtrace.classify(tracer.writes.get(id(t), []) if tracer else [])
            st = st or "UNINITIALIZED"
            nonloaded[k] = {"kind": kind, "class": "INIT_" + st if st not in ("UNINITIALIZED", "UNPROVEN") else st,
                            "numel": t.numel(), "init_fns": ["(replaced after load)"] + fns[:5]}
        elif id(t) in kept_ids:
            continue                                            # tied alias of a loaded tensor: the same values
        else:
            st, fns = loadtrace.classify(tracer.writes.get(id(t), []) if tracer else [])
            c = "INIT_" + st if st not in ("UNINITIALIZED", "UNPROVEN") else st
            nonloaded[k] = {"kind": kind, "class": c, "numel": t.numel(), "init_fns": fns[:6]}
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
            inherit = orig[src] in RANDOM_CLASSES + ("UNPROVEN", "MISMATCHED_DISCARDED")
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


RANDOM_CLASSES = ("INIT_RANDOM", "UNINITIALIZED")
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


def load_state_of(quantized_ev, quantizer, refusal):
    """quantized codes with no quantizer -> quantized_interpretation_unknown (whatever else happened: a refusal of
    packed codes is a consequence); else the library's own refusal -> refused; else loaded."""
    if quantized_ev and not quantizer:
        return "quantized_interpretation_unknown"
    return "refused" if refusal else "loaded"


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


def g1_assess(prov, used, exec_shapes, buffer_group):
    """The G1/G2 verdict inputs (INTENT.md; decisions in G1_LOAD_PROVENANCE_2026-10-09.md), from the library's load
    provenance and the recorded execution.
      prov         : _provenance(...) of the library load of the executed class
      used         : names of every parameter / buffer some recorded op read (tied aliases included)
      exec_shapes  : {state-dict name: shape} of the executed model
      buffer_group : {name: (module class, buffer name)} for the executed model's FLOATING persistent buffers
    Returns {"fail": {...}, "unknowns": [...], "leftovers": {...}}: a non-empty fail fails T1 (PARTIAL); unknowns make an
    otherwise FULL row FULL_UNVERIFIED; leftovers make it FULL_LEFTOVERS (measured, listed)."""
    if prov.get("error"):
        return {"fail": {}, "leftovers": {},
                "unknowns": [{"predicate": "G1", "what": f"load provenance could not be computed ({prov['error']})",
                              "items": [], "count": 0}]}
    nl = prov.get("nonloaded") or {}
    rnd = sorted(k for k, v in nl.items() if v["class"] in RANDOM_CLASSES)
    fail = {"random_init_executed": [k for k in rnd if k in used],                 # random values flow into the run
            "conversion_errors": sorted(prov.get("conversion_errors") or {}),
            # the library refuses the default load (mismatched shapes): its own refusal, not a random fill (P0.5)
            "library_refuses_load": sorted(k for k, v in nl.items() if v["class"] == "MISMATCHED_DISCARDED")}
    unknowns = []
    def unk(pred, what, items):
        if items: unknowns.append({"predicate": pred, "what": what, "items": list(items)[:20], "count": len(items)})
    unk("G1", "learned tensors the checkpoint does not ship, randomly initialised by the library, read by no recorded op",
        [k for k in rnd if k not in used])
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
    # shipped floating buffers no op read: a sibling of the same kind WAS read -> a real gap (fail); none of that kind
    # was read in any mode -> dead state (measured, a leftover)
    shipped_bufs = [k for k in buffer_group if k not in nl]
    groups_read = {buffer_group[k] for k in shipped_bufs if k in used}
    unread = [k for k in shipped_bufs if k not in used]
    fail["buffers_unread_while_siblings_read"] = sorted(k for k in unread if buffer_group[k] in groups_read)
    leftovers = {"buffers_never_read_in_any_mode": sorted(k for k in unread if buffer_group[k] not in groups_read)}
    return {"fail": {k: v for k, v in fail.items() if v}, "unknowns": unknowns,
            "leftovers": {k: v for k, v in leftovers.items() if v}}
