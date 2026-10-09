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
        got = list(ex.map(lambda fz: _read_header(fs, f"{repo}/{fz[0]}"), weights))
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
                   conversion_errors={k: str(v)[-300:] for k, v in loading_info.conversion_errors.items()},
                   quantizer=type(load_config.hf_quantizer).__name__ if load_config.hf_quantizer else None)
        tr = loadtrace.InitTracer(); tracer_box["t"] = tr
        with tr:
            return orig_fin(model_, load_config, loading_info)
    CML.set_param_for_module = recording_spf
    MU.PreTrainedModel._finalize_model_loading = staticmethod(traced_finalize)
    try:
        with torch.device("meta"):
            model, info = cls.from_pretrained(d, output_loading_info=True, device_map="meta", ignore_mismatched_sizes=True, **kw)
    finally:
        MU.safe_open = orig
        CML.rename_source_key = orig_rename
        torch.equal = orig_equal
        CML.set_param_for_module = orig_spf
        MU.PreTrainedModel._finalize_model_loading = orig_fin
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
            "stub_disk_kb": du, "provenance": provenance}


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
    for k, t in sd.items():
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
            "conversion_errors": sorted(prov.get("conversion_errors") or {})}
    unknowns = []
    def unk(pred, what, items):
        if items: unknowns.append({"predicate": pred, "what": what, "items": list(items)[:20], "count": len(items)})
    unk("G1", "learned tensors the checkpoint does not ship, randomly initialised by the library, read by no recorded op",
        [k for k in rnd if k not in used])
    unk("G1", "tensors filled by a copy of unknown provenance during the library's load", sorted(k for k, v in nl.items() if v["class"] == "UNPROVEN"))
    unk("G1", "tied tensors shipped twice: the library ties them only if their values are equal (not checkable without "
        f"the weights; config.json tie_word_embeddings={prov.get('raw_tie_word_embeddings')}, library "
        f"{prov.get('library_tie_word_embeddings')})", sorted(k for k, v in nl.items() if v["class"] in TIE_UNVERIFIED))
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
