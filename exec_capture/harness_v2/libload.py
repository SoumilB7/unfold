"""The library's own verdict on which shipped tensors it loads: every weight file of the repo is mirrored locally as a
SPARSE stub (real header bytes, zero-filled data, no disk used), then transformers' from_pretrained runs on the meta
device (no tensor data in RAM) with output_loading_info=True. File selection, renames, merges/splits and ignore lists
are all the library's own; nothing is re-implemented here."""
import os, json, struct, tempfile, shutil
from huggingface_hub import HfApi, HfFileSystem

WEIGHT_EXT = (".safetensors",)


def mirror_sparse(repo, dest, headers=None):
    api, fs = HfApi(), HfFileSystem()
    info = api.model_info(repo, files_metadata=True)
    n_stub, disk_bytes = 0, 0
    headers = {} if headers is None else headers        # file -> {tensor: (shape, dtype)} from the real header bytes
    for s in info.siblings:
        f, size = s.rfilename, s.size or 0
        p = os.path.join(dest, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if f.endswith(WEIGHT_EXT):
            with fs.open(f"{repo}/{f}", "rb", block_size=1 << 16) as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = fh.read(n)
            with open(p, "wb") as out:
                out.write(struct.pack("<Q", n)); out.write(hdr); out.truncate(size)
            headers[f] = {k: (tuple(v["shape"]), v["dtype"]) for k, v in json.loads(hdr).items() if k != "__metadata__"}
            n_stub += 1
        elif f.endswith((".json", ".txt", ".model", ".py")) and size < 50_000_000 and not f.endswith(".py"):
            local = api.hf_hub_download(repo, f, cache_dir=os.path.join(dest, ".dl"))
            shutil.copy(local, p)
    shutil.rmtree(os.path.join(dest, ".dl"), ignore_errors=True)
    return n_stub


def headers_read_names(opened, headers):
    return [k for f in opened for k in headers.get(f, {})]


def library_load_report(repo, cls, aux_keys=(), **kw):
    """-> dict(unexpected, missing, mismatched, loaded_param_names) exactly as the library reports them."""
    import torch
    import transformers.modeling_utils as MU
    d = tempfile.mkdtemp(prefix="libload_")
    opened, orig = set(), MU.safe_open
    def recording_safe_open(path, *a, **k):                # observe which files the library reads; behaviour unchanged
        opened.add(os.path.relpath(str(path), d)); return orig(path, *a, **k)
    try:
        headers = {}
        n = mirror_sparse(repo, d, headers)
        if n == 0:
            return {"error": "no safetensors weight files"}
        MU.safe_open = recording_safe_open
        try:
            with torch.device("meta"):
                model, info = cls.from_pretrained(d, output_loading_info=True, device_map="meta", ignore_mismatched_sizes=True, **kw)
        finally:
            MU.safe_open = orig
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
        rest_state = {k: n for k, n in state.items() if k not in read}
        state_numel = sum(state.values())
        du = int(os.popen(f"du -sk '{d}'").read().split()[0])
        return {"files_read": sorted(opened), "files_all": sorted(headers), "read_keys": sorted(read),
                "read_numel": read_numel, "reported_numel": reported_numel, "state_numel": state_numel,
                "unaccounted_numel": max(0, sum(rest_read.values()) - sum(rest_state.values())),
                "unmatched_read_keys": sorted(rest_read)[:200],
                "unexpected": sorted(info.get("unexpected_keys", [])), "missing": sorted(info.get("missing_keys", [])),
                "mismatched": [list(map(str, x)) for x in info.get("mismatched_keys", [])],
                "loaded_params": len(loaded), "model_params": sum(1 for _ in model.named_parameters()),
                "stub_disk_kb": du}
    finally:
        shutil.rmtree(d, ignore_errors=True)
