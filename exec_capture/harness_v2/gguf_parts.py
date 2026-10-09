"""GGUF parts list: tensor names + shapes from the GGUF header (range reads, no weight download), renamed to the
library's parameter names by transformers' own GGUF loader map (get_gguf_hf_weights_map). Unmapped tensors keep
their GGUF names, so they surface as "not in the library class" instead of disappearing."""
import re, struct, collections
from huggingface_hub import HfFileSystem

GGUF_SCALAR = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
    6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
T_STRING, T_ARRAY = 8, 9


class NeedMore(Exception):
    def __init__(self, want):
        self.want = want


class Cur:
    """Cursor over a growable byte buffer; raises NeedMore(min_len) instead of IndexError."""
    __slots__ = ("buf", "pos")

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def need(self, n):
        if self.pos + n > len(self.buf):
            raise NeedMore(self.pos + n)

    def read(self, n):
        self.need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u8(self):
        return self.read(1)[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def gstr(self):
        n = self.u64()
        self.need(n)
        return self.read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype == T_STRING:
            return self.gstr()
        if vtype == T_ARRAY:
            atype = self.u32()
            n = self.u64()
            out = []
            # cap materialised array preview to keep this fast/light; still consumes all bytes
            for i in range(n):
                out.append(self.value(atype) if len(out) < 8 else self._skip_value(atype))
            return {"_array_type": atype, "_len": n, "sample": out[:8]}
        if vtype in GGUF_SCALAR:
            fmt, size = GGUF_SCALAR[vtype]
            return struct.unpack("<" + fmt, self.read(size))[0]
        raise ValueError(f"unknown gguf value type {vtype}")

    def _skip_value(self, vtype):
        if vtype == T_STRING:
            self.gstr()
            return None
        if vtype == T_ARRAY:
            atype = self.u32()
            n = self.u64()
            for _ in range(n):
                self._skip_value(atype)
            return None
        if vtype in GGUF_SCALAR:
            _, size = GGUF_SCALAR[vtype]
            self.read(size)
            return None
        raise ValueError(f"unknown gguf value type {vtype}")


def parse_gguf_header(buf):
    """Parse as much of the header as `buf` allows. Raises NeedMore(n) if buf is too short."""
    c = Cur(buf)
    magic = c.read(4)
    if magic != b"GGUF":
        raise ValueError(f"bad magic {magic!r}")
    version = c.u32()
    tensor_count = c.u64()
    kv_count = c.u64()
    kv = {}
    for _ in range(kv_count):
        key = c.gstr()
        vtype = c.u32()
        val = c.value(vtype)
        kv[key] = val
    tensors = []
    for _ in range(tensor_count):
        name = c.gstr()
        n_dims = c.u32()
        dims = [c.u64() for _ in range(n_dims)]
        ggml_type = c.u32()
        offset = c.u64()
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
    return {"version": version, "tensor_count": tensor_count, "kv_count": kv_count,
            "kv": kv, "tensors": tensors, "header_bytes": c.pos}




def fetch_and_parse(fs, path, cap=(1 << 27)):
    size = 1 << 20
    with fs.open(path, "rb", block_size=(1 << 22)) as f:
        while size <= cap:
            f.seek(0)
            buf = f.read(size)
            try:
                h = parse_gguf_header(buf)
                h["_bytes"] = buf[:h["header_bytes"]]
                return h
            except NeedMore as nm:
                size = max(size * 2, nm.want + (1 << 20))
    raise RuntimeError("gguf header exceeds cap")


COMPANION_KEYWORDS = ("mmproj", "projector", "draft", "dflash", "speculative",
                      "medusa", "eagle-", "vision-encoder", "vision_encoder")


def pick_gguf_files(files):
    """files: list of {'name':..., 'size':...} under the repo (already flattened one
    level into quant-named subfolders, e.g. unsloth's BF16/, UD-Q4_K_XL/, ...).
    Returns (main_group, mmproj_files) where main_group is a list of file infos
    belonging to the smallest legitimate non-companion gguf (a single file, or all
    shards of a split -00001-of-000NN- group)."""
    ggufs = [f for f in files if f["name"].lower().endswith(".gguf")]
    mmproj = [f for f in ggufs if "mmproj" in f["name"].lower() or "projector" in f["name"].lower()]
    companions = [f for f in ggufs if any(k in f["name"].lower() for k in COMPANION_KEYWORDS)]
    mains = [f for f in ggufs if f not in companions]

    def split_base(name):
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name, re.I)
        if m:
            return name[:m.start()], int(m.group(2))
        return name[:-5], 1

    groups = collections.defaultdict(list)
    for f in mains:
        base, _ = split_base(f["name"])
        groups[base].append(f)
    if not groups:
        return [], mmproj
    group_sizes = {b: sum(f["size"] for f in fl) for b, fl in groups.items()}
    max_size = max(group_sizes.values())
    # Real quantization variants of the SAME model cluster within roughly a 3-4x size
    # band (Q2 .. BF16); anything under 1/8th of the largest surviving candidate is
    # almost certainly yet another un-named companion artifact rather than a smaller
    # real quantization, so drop it before picking "smallest".
    survivors = {b: s for b, s in group_sizes.items() if s >= max_size / 8}
    best_base = min(survivors, key=survivors.get)
    chosen = sorted(groups[best_base], key=lambda f: f["name"])
    return chosen, mmproj



GGML_DTYPE = {0: "F32", 1: "F16", 30: "BF16", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64"}


def fetch_gguf_parts(repo):
    """-> (raw {gguf_name: (torch_shape, dtype)}, kv, file names). Shapes are GGUF dims reversed (ggml ne order)."""
    fs = HfFileSystem()
    files = [{"name": f["name"].split(repo + "/", 1)[-1], "size": f.get("size") or 0}
             for f in fs.find(repo, detail=True).values() if f["name"].lower().endswith(".gguf")]
    main, _ = pick_gguf_files(files)
    raw, kv = {}, {}
    first = None
    for f in main:                                   # every shard of the picked quantization
        h = fetch_and_parse(fs, f"{repo}/{f['name']}")
        kv = kv or h["kv"]
        if first is None: first = (f["name"], f["size"], h["_bytes"])
        for t in h["tensors"]:
            raw[t["name"]] = (tuple(int(d) for d in reversed(t["dims"])), GGML_DTYPE.get(t["type"], f"Q{t['type']}"))
    return raw, kv, [f["name"] for f in main], first


def library_config(first, stub_dir):
    """The library's own GGUF rule: config + architecture come from the GGUF header (from_pretrained(gguf_file=...)).
    The header bytes are written into a SPARSE file of the real size (no disk used, tensor data never read)."""
    import os, transformers
    name, size, header = first
    fn = os.path.basename(name)
    path = os.path.join(stub_dir, fn)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.truncate(max(size, len(header)))
    return transformers.AutoConfig.from_pretrained(stub_dir, gguf_file=fn)


def rename_to_library(raw, model):
    """Rename with the library's own GGUF loader map; returns ({name: (shape, dtype)}, unmapped gguf names)."""
    import transformers.modeling_gguf_pytorch_utils as G
    arch = model.config.model_type
    proc_cls = G.TENSOR_PROCESSORS.get(arch, G.TensorProcessor)
    try:
        proc = proc_cls(config=model.config.to_dict())
    except TypeError:
        proc = proc_cls()
    name_map = G.get_gguf_hf_weights_map(model, proc)
    out, unmapped = {}, []
    for gname, (shape, dt) in raw.items():
        hf = name_map.get(gname) or name_map.get(gname.rsplit(".", 1)[0])
        if hf and not hf.endswith((".weight", ".bias")) and gname.endswith((".weight", ".bias")):
            hf = hf + "." + gname.rsplit(".", 1)[1]
        if hf:
            out[hf] = (shape, dt)
        else:
            out[gname] = (shape, dt); unmapped.append(gname)
    return out, unmapped
