"""Shared scoring for every mechanism.

Ground truth = the tensors the checkpoint actually ships (safetensors headers).
A mechanism "loses architecture" for every checkpoint weight it never accounts for.
Matching is generic: exact name, else unique same-shape suffix match, else
per-expert tensors summed into a fused tensor with the same prefix. No model names.
"""
import re, math, collections

AUX = re.compile(r"(\.biases|inv_freq|position_ids|num_batches_tracked|_scales|weight_global_scale|input_global_scale|_scale_inv|weight_scale(_2)?|input_scale|output_scale|k_scale|v_scale|zero_point|\.g_idx|\.qzeros|\.scales|\.SCB|absmax|quant_map|quant_state|nested_|_blocks_scale)$")
INT_DTYPES = {"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"}


def module_pattern(name):
    parts = name.split(".")[:-1]
    return ".".join("*" if p.isdigit() else p for p in parts) or "(root)"


def top_groups(named_numel, n=6):
    agg = collections.Counter()
    for name, k in named_numel:
        agg[module_pattern(name)] += k
    return [(g, v) for g, v in agg.most_common(n)]


def ckpt_split(T):
    """T: {name: (shape, dtype)} -> (weights, aux, packed_flag). Packed-quantized names are
    normalized to the plain weight name (qweight/weight_packed/_blocks -> weight)."""
    weights, aux = {}, {}
    packed = any(d in INT_DTYPES for k, (s, d) in T.items()
                 if not AUX.search(k) and re.search(r"(weight|qweight|weight_packed|_blocks)$", k))
    def quant_scale(k):
        if not k.endswith(".scale"):
            return False
        sib = T.get(k[: -len(".scale")] + ".weight")
        return sib is not None and (sib[1] in INT_DTYPES or str(sib[1]).startswith("F8"))
    for k, (s, d) in T.items():
        if AUX.search(k) or quant_scale(k):
            aux[k] = (tuple(s), d); continue
        if packed:
            k = re.sub(r"\.(qweight|weight_packed)$", ".weight", k)
            k = re.sub(r"_blocks$", "", k)
        weights[k] = (tuple(s), d)
    return weights, aux, packed


def library_rename(model):
    """The installed library's own checkpoint->model key mapping (the rules it applies when loading weights)."""
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import rename_source_key, WeightRenaming, WeightConverter
    except Exception:
        return None
    try:
        tr = get_model_conversion_mapping(model)
    except Exception:
        return None
    ren = [t for t in tr if isinstance(t, WeightRenaming)]
    conv = [t for t in tr if isinstance(t, WeightConverter)]
    msd = model.state_dict()
    prefix = getattr(model, "base_model_prefix", None)
    def f(key):
        try:
            return rename_source_key(key, ren, conv, prefix, msd)[0]
        except Exception:
            return key
    return f


def library_ignored(model):
    """The library's declared-ignore patterns: class attributes AND the instance's own (set at build time, e.g.
    Gemma-3n/4 list their KV-shared k/v projections per built model)."""
    pats = list(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
    for cls in type(model).__mro__:
        pats += list(getattr(cls, "_keys_to_ignore_on_load_unexpected", None) or [])
    return [re.compile(p) for p in dict.fromkeys(pats)]


def match(weights, H, alias, packed, rename=None):
    """Returns (matched: ckpt_name -> husk_name or group tag, unmatched_ckpt, unmatched_husk)."""
    matched, used_h = {}, set()
    # 0) the library's own loading rules: rename, and many-to-one converters (fused experts, qkv, ...)
    if rename is not None:
        tgt = collections.defaultdict(list)
        for ck in weights:
            t = rename(ck)
            if t in H:
                tgt[t].append(ck)
        for t, cks in tgt.items():
            if packed or sum(math.prod(weights[c][0]) for c in cks) == math.prod(H[t]):
                for c in cks: matched[c] = t if len(cks) == 1 else "fused:" + t.rsplit(".", 1)[0]
                used_h.add(t)
    idx = collections.defaultdict(list)
    for hk in H:
        seg = hk.split(".")
        for k in range(2, len(seg) + 1):
            idx[".".join(seg[-k:])].append(hk)
    # 1) exact
    for ck in weights:
        if ck in H and (packed or weights[ck][0] == H[ck]):
            matched[ck] = ck; used_h.add(ck)
    # 2) unique suffix with same shape (renamed prefixes, e.g. model.language_model vs language_model.model)
    for ck in weights:
        if ck in matched:
            continue
        seg = ck.split(".")
        for k in range(len(seg), 1, -1):
            cands = [h for h in idx.get(".".join(seg[-k:]), []) if h not in used_h and (packed or H[h] == weights[ck][0])]
            if len(cands) == 1:
                matched[ck] = cands[0]; used_h.add(cands[0]); break
            if len(cands) > 1:
                break
    # 3) per-expert tensors summed into fused husk tensors under the same prefix
    groups = collections.defaultdict(list)
    for ck in weights:
        if ck not in matched:
            m = re.match(r"(.*\.experts)\.\d+\.", ck)
            if m:
                groups[m.group(1)].append(ck)
    for gp, cks in groups.items():
        seg = gp.split(".")
        hk_prefix = None
        for k in range(len(seg), 1, -1):
            suf = ".".join(seg[-k:])
            prefs = {h[: h.index(suf) + len(suf)] for h in H if (suf + ".") in h}
            if len(prefs) == 1:
                hk_prefix = prefs.pop(); break
            if len(prefs) > 1:
                exact = [p for p in prefs if p == gp]
                if exact:
                    hk_prefix = exact[0]; break
        if not hk_prefix:
            continue
        hks = [h for h in H if h.startswith(hk_prefix + ".") and h not in used_h]
        if hks and (packed or sum(math.prod(weights[c][0]) for c in cks) == sum(math.prod(H[h]) for h in hks)):
            for c in cks:
                matched[c] = "fused:" + hk_prefix
            used_h.update(hks)
    # 4) renamed leaves: deepest common ancestor whose unmatched ckpt and husk tensors have identical shape multisets
    un_c0 = [c for c in weights if c not in matched]
    prefixes = sorted({".".join(c.split(".")[:i]) for c in un_c0 for i in range(1, len(c.split(".")) - 1)},
                      key=lambda p: -p.count("."))
    for p in prefixes:
        cks = [c for c in weights if c not in matched and c.startswith(p + ".")]
        hks = [h for h in H if h not in used_h and h.startswith(p + ".")]
        if cks and hks and sorted(weights[c][0] for c in cks) == sorted(H[h] for h in hks):
            for c in cks: matched[c] = "renamed:" + p
            used_h.update(hks)
    # tied aliases: if any alias of a husk tensor was used, all aliases are used
    for names in alias.values():
        if any(n in used_h for n in names):
            used_h.update(names)
    un_c = [c for c in weights if c not in matched]
    un_h = [h for h in H if h not in used_h]
    return matched, un_c, un_h, used_h


def numel(shape):
    return math.prod(shape) if shape else 1


def looks_like_stored_buffer(shape, dtype):
    """Masks and counters shipped in old checkpoints: integer/bool, scalars, or 1x1xNxN matrices."""
    return (dtype in INT_DTYPES or dtype == "BOOL" or len(shape) == 0
            or (len(shape) == 4 and shape[0] == shape[1] == 1 and shape[2] == shape[3]))


def split_library_ignored(model, W, rename=None):
    """Checkpoint keys the library declares it will not load, split into
    (a) stored buffers (the model has a buffer at that path, e.g. causal masks) and
    (b) learned weights the library does not build (e.g. multi-token-prediction layers)."""
    pats = library_ignored(model)
    buf_names = {n for n, _ in model.named_buffers()}
    for mod_name, mod in model.named_modules():
        for bname in getattr(mod, "_non_persistent_buffers_set", set()):
            buf_names.add(f"{mod_name}.{bname}" if mod_name else bname)
    # the library applies these patterns only to keys it could NOT place (it silences their "unexpected" report);
    # a key the built model has a slot for is loaded whatever the pattern says (patterns are merged from child
    # models unprefixed, e.g. T5's "decoder" or GPT-2's "attn.bias", so they can match loaded keys).
    slots = set(model.state_dict().keys())
    ignored = [k for k in W if any(p.search(k) for p in pats)
               and k not in slots and not (rename and rename(k) in slots)]
    buffers, unbuilt = [], []
    for k in ignored:
        t = rename(k) if rename else k
        shape, dtype = W[k]
        stored_buffer = (k in buf_names or t in buf_names or dtype in INT_DTYPES or dtype == "BOOL"
                         or len(shape) == 0 or (len(shape) == 4 and shape[0] == shape[1] == 1 and shape[2] == shape[3]))
        (buffers if stored_buffer else unbuilt).append(k)
    return buffers, unbuilt
