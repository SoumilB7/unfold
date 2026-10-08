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
            return True
        return orig_equal(a, b, *args, **kwargs)
    torch.equal = meta_equal
    try:
        with torch.device("meta"):
            model, info = cls.from_pretrained(d, output_loading_info=True, device_map="meta", ignore_mismatched_sizes=True, **kw)
    finally:
        MU.safe_open = orig
        CML.rename_source_key = orig_rename
        torch.equal = orig_equal
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
        targets = set((getattr(model, "all_tied_weights_keys", None) or {}).keys())
        for k in list(rest_read):
            if k in targets or any(n_ in targets for n_ in renamed_to.get(k, ())):
                tie_dups.append(k); del rest_read[k]
    rest_state = {k: n for k, n in state.items() if k not in read}
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
            "stub_disk_kb": du}
