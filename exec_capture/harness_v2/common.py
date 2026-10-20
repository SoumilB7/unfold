"""Shared scoring for every mechanism.

Ground truth = the tensors the checkpoint actually ships (safetensors headers).
A mechanism "loses architecture" for every checkpoint weight it never accounts for.
Matching is generic: exact name, else unique same-shape suffix match, else
per-expert tensors summed into a fused tensor with the same prefix. No model names.
"""
import re, math, collections
from witness_io import (AUX, INT_DTYPES, ckpt_split, runtime, WITNESS_SCHEMA, WITNESS_FILES,  # noqa: F401
                        WITNESS_STRUCTURAL_INPUTS, witness_producer_id, vllm_source_id, witness_inputs,
                        input_fingerprints, witness_transparent)                     # (re-exported: the recorder's code, F14)



# the code that decides a verdict (INTENT.md G6). Drawing code (exec_to_ir), reports and self-tests are not in it, so
# a drawing-only change never changes the grading identity.
GRADING_FILES = ("worker_v2.py", "worker_v2_diffusion.py", "run_batch.py", "dag.py", "common.py", "libload.py",
                 "loadtrace.py", "lowcost.py", "inputs.py", "gguf_parts.py", "noweights.py", "synth.py", "opclass.py",
                 "mtp_witness_run.py", "vllm_witness.py", "capture.py", "class_rules.py",
                 "diffusion_rules.py", "witness_io.py",
                 "everchanging/opacity_reviewed.yaml",      # capture.py records graded passes; F10: the reviewed-op vocabulary
                 "attempts.py", "everchanging/f09_value_shape_ops.yaml")   # F09: taint carry + py_inputs; F09-7 vocabulary


def grading_id():
    """Content hash of the grading code: two results with the same id were graded by byte-identical code, whichever
    copy (worktree or run mirror) ran them."""
    import hashlib, os
    here = os.path.dirname(os.path.abspath(__file__)); h = hashlib.sha256()
    for f in GRADING_FILES:
        p = os.path.join(here, f)
        h.update(f.encode()); h.update(open(p, "rb").read() if os.path.exists(p) else b"<missing>")
    return h.hexdigest()[:12]





def module_pattern(name):
    parts = name.split(".")[:-1]
    return ".".join("*" if p.isdigit() else p for p in parts) or "(root)"


def top_groups(named_numel, n=6):
    agg = collections.Counter()
    for name, k in named_numel:
        agg[module_pattern(name)] += k
    return [(g, v) for g, v in agg.most_common(n)]





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


QUANT_INT = ("I8", "U8", "I16", "U16", "I32", "U32")      # integer formats quantized weights use (I64 / BOOL are tables)
_NON_QUANT_AUX = re.compile(r"(inv_freq|position_ids|num_batches_tracked)$")
_PACKED_NAME = re.compile(r"(qweight|weight_packed|_blocks)$")


def quantized_weight_keys(T):
    """Checkpoint keys (raw, and as ckpt_split names them) that are quantized LEARNED weights, from the checkpoint alone:
    a raw packed name (qweight / weight_packed / _blocks), or I8..U32 codes whose module ships a quantization auxiliary
    tensor (a scale / zero point). Such keys are never stored tables (P0.6 B4)."""
    aux = ckpt_split(T)[1]
    qmods = {k.rpartition(".")[0] for k in aux if not _NON_QUANT_AUX.search(k)}
    out = set()
    for k, (s, d) in T.items():
        if k in aux:
            continue
        if _PACKED_NAME.search(k) or (d in QUANT_INT and k.rpartition(".")[0] in qmods):
            out.add(k); out.add(re.sub(r"_blocks$", "", re.sub(r"\.(qweight|weight_packed)$", ".weight", k)))
    return out


def pos_lengths(c):
    """The position lengths a config declares (its own and its sub-configs')."""
    out = set()
    subs = [v for v in vars(c).values() if hasattr(v, "to_dict") and hasattr(v, "model_type")] if hasattr(c, "__dict__") else []
    for sc in [c] + [getattr(c, a) for a in dir(c) if a.endswith("_config") and hasattr(getattr(c, a, None), "to_dict")] + subs:
        for f in ("n_positions", "max_position_embeddings", "n_ctx", "max_seq_len", "block_size"):
            v = getattr(sc, f, None)
            if isinstance(v, int) and v > 0: out.add(v)
    return out


def stored_constant(key, shape, dtype, *, quantized=frozenset(), pos=frozenset(), declared=False, buffer_slot=False):
    """THE rule for "a shipped tensor is stored state (a table / mask / counter), not a learned weight", on every path
    (P0.6 B4 + C7; D-C7 ruling 2026-10-12 "strict unless declared"; F05 "by meaning, not dtype"):
      - it lands in a buffer slot of the built model;
      - never a quantized learned weight (quantized_weight_keys);
      - an integer / bool tensor ONLY with evidence about THIS key (F05): it lands in a buffer slot (above), the
        library declares the key ignorable, or it is a 1x1xNxN mask whose N is a position length (the C7 shape rule).
        dtype alone is never evidence: an integer table can be routing / control state (int_table_unknown names the
        rest, never hidden). A same-named table the model computes ELSEWHERE is a hint, not evidence: the shipped
        values are not compared (zero storage);
      - a float 1x1xNxN matrix whose N is a position length of the model (a stored mask), or any N if the library
        declares the key ignorable;
      - a 0-d float ONLY when the library declares the key ignorable (a learned scalar such as CLIP's logit_scale is
        never hidden)."""
    if buffer_slot:
        return True
    if key in quantized:
        return False
    if dtype in INT_DTYPES or dtype == "BOOL":
        return bool(declared or (len(shape) == 4 and shape[0] == shape[1] == 1 and shape[2] == shape[3] and shape[2] in pos))
    if len(shape) == 4 and shape[0] == shape[1] == 1 and shape[2] == shape[3]:
        return declared or shape[2] in pos
    if len(shape) == 0:
        return declared
    return False


def is_int_table(dtype, key=None, quantized=frozenset()):
    """An integer / bool checkpoint tensor that is not a quantized learned weight (a table, index, counter, mask or
    control state: its meaning needs evidence, F05)."""
    return (dtype in INT_DTYPES or dtype == "BOOL") and key not in quantized


def int_table_unknown(keys, hints=None):
    """F05 class (d): shipped integer / bool tensors the model does not load, with no evidence of what they are (not
    declared ignorable, no buffer slot, not a table the model recomputes, not a position-length mask). Not a learned
    weight (no failure) and never hidden: their relevance is unknown (G1)."""
    keys, hints = sorted(keys), hints or {}
    return {"predicate": "G1", "hints": {k: hints[k] for k in keys[:20] if hints.get(k)}, "what": "shipped integer / bool tables the model does not load, with no evidence of what "
            "they are (not declared ignorable, no buffer slot, not recomputed by the model): relevance unknown",
            "items": keys[:20], "count": len(keys)}


def recomputed_signatures(model, loaded=()):
    """(last name part, element count) of every buffer the built model computes itself instead of loading: every
    non-persistent buffer (the library declares it derived, never saved), and persistent buffers not in `loaded` (the
    names the loader filled from the checkpoint). A shipped integer table with the same name part and size is the
    HINT that it is the library's recomputed table; it is not evidence (the shipped values are never compared)."""
    out, loaded = set(), set(loaded)
    for mn, mod in model.named_modules():
        npb = getattr(mod, "_non_persistent_buffers_set", set())
        for bn, b in mod._buffers.items():
            if b is None:
                continue
            full = f"{mn}.{bn}" if mn else bn
            if bn in npb or full not in loaded:
                out.add((bn, b.numel()))
    return out


def looks_like_stored_buffer(shape, dtype):
    """Masks and counters shipped in old checkpoints: integer/bool, scalars, or 1x1xNxN matrices."""
    return (dtype in INT_DTYPES or dtype == "BOOL" or len(shape) == 0
            or (len(shape) == 4 and shape[0] == shape[1] == 1 and shape[2] == shape[3]))


def buffer_names(model):
    """Every buffer path of the built model: registered buffers plus non-persistent ones."""
    names = {n for n, _ in model.named_buffers()}
    for mod_name, mod in model.named_modules():
        for bname in getattr(mod, "_non_persistent_buffers_set", set()):
            names.add(f"{mod_name}.{bname}" if mod_name else bname)
    return names


def split_library_ignored(model, W, rename=None, quantized=frozenset()):
    """Checkpoint keys the library declares it will not load, split into
    (a) stored buffers (the model has a buffer at that path, e.g. causal masks) and
    (b) learned weights the library does not build (e.g. multi-token-prediction layers)."""
    pats = library_ignored(model)
    buf_names = buffer_names(model)
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
        stored_buffer = stored_constant(k, shape, dtype, quantized=quantized, declared=True,     # the library declares it
                                        buffer_slot=(k in buf_names or t in buf_names))
        (buffers if stored_buffer else unbuilt).append(k)
    return buffers, unbuilt


def after_grading_exception(R, unknowns, err):
    """A worker's last resort (sys.excepthook). Before grading the row is FAIL. After a verdict was set, the steps that
    follow grading (revision pin, unknown promotion) did not run, so a FULL* verdict cannot stand certified: it becomes
    FULL_UNVERIFIED with a G6 unknown naming the exception. An observed failure (PARTIAL / FAIL / OUT) stays."""
    v = R.get("verdict")
    if R.get("library_refuses_checkpoint") and v != "OUT":          # R-REFUSE holds whatever crashed after it
        R["verdict"] = "FAIL"
        R["reason"] = R.get("reason") or "library refuses this checkpoint"
        return R
    if not v:
        R["verdict"] = "FAIL"
    elif v in ("FULL", "FULL_LEFTOVERS", "FULL_UNVERIFIED"):
        unknowns.append({"predicate": "G6", "what": f"the run raised after grading ({err[:200]}): the revision pin and "
                         "the unknown promotion were not evaluated", "items": [], "count": 0})
        R["verdict"] = "FULL_UNVERIFIED"
        R["unknowns"] = unknowns
    return R


# ---- F14 (Soumil 2026-10-13): a vLLM witness record counts only when it names what made it (schema, checkpoint
#      revision, the recorder's own code, vLLM version + source, library versions) and its two stability runs are shown
#      to have received different inputs. Records without that identity give an unknown, never credit or absence. ----
WITNESS_IDENTITY = ("schema", "revision", "producer_id", "vllm_version", "vllm_source_id", "runtime", "revision_pin")


def distinct_inputs(fa, fb, structural=WITNESS_STRUCTURAL_INPUTS):
    """Both runs fingerprinted the same input names and EVERY value input (not a function of the token count) differs.
    Missing or malformed fingerprints show nothing: not distinct."""
    if not isinstance(fa, dict) or not isinstance(fb, dict) or not fa or set(fa) != set(fb):
        return False
    value = set(fa) - set(structural)
    return bool(value) and all(isinstance(fa[k], str) and fa[k] and fa[k] != fb[k] for k in value)


def witness_identity_gaps(wit, revision, producer=None):
    """What a witness record lacks to be bound to the row graded at `revision` and to the current recorder code
    (empty = bound). The revision must also be SHOWN: the record's pin log names no read of another revision."""
    if not isinstance(wit, dict):
        return ["not a witness record"]
    producer = witness_producer_id() if producer is None else producer
    gaps = [f"no {k}" for k in WITNESS_IDENTITY if not wit.get(k)]
    if wit.get("schema") and wit["schema"] != WITNESS_SCHEMA:
        gaps.append(f"schema {wit['schema']} (current {WITNESS_SCHEMA})")
    if revision is None:
        gaps.append("graded revision unknown")
    elif wit.get("revision") and wit["revision"] != revision:
        gaps.append(f"recorded for revision {wit['revision']}, graded {revision}")
    if wit.get("producer_id") and wit["producer_id"] != producer:
        gaps.append("made by another version of the witness code")
    if wit.get("runtime") and not (isinstance(wit["runtime"], dict) and wit["runtime"].get("torch")):
        gaps.append("library versions malformed")
    pin = wit.get("revision_pin")
    if pin and not (isinstance(pin, dict) and isinstance(pin.get("other_revision"), list)):
        gaps.append("revision pin log malformed")
    elif pin and pin["other_revision"]:
        gaps.append("hub reads named another revision (" + ", ".join(map(str, pin["other_revision"][:3])) + ")")
    elif pin and not (isinstance(pin.get("pinned"), int) and pin["pinned"] > 0):
        gaps.append("no hub read was pinned to the revision")
    return gaps


def witness_execution_credit(wit, revision, producer=None):
    """Why a closed witness's execution credit is not proven: [(predicate, why)], empty = proven. The producer's own
    `stable_structure` is not trusted alone: input distinctness is re-derived from the fingerprints."""
    out = []
    gaps = witness_identity_gaps(wit, revision, producer)
    if gaps:
        out.append(("G6", "the record is not bound to this checkpoint and the current witness code (" + ", ".join(gaps) + ")"))
    ins = wit.get("inputs") if isinstance(wit.get("inputs"), dict) else {}
    fp = lambda r: ins.get(r).get("fingerprints") if isinstance(ins.get(r), dict) else None
    if wit.get("stable_structure") is not True or not distinct_inputs(fp("a"), fp("b")):
        out.append(("G5a", "its two runs are not shown to have the same structure on different inputs"))
    if not witness_transparent(wit):                # (F10-24: recomputed from the record's evidence, the producer's rule)
        why = []
        if "op_full" not in wit:
            why.append("old record without full op names")
        else:
            if wit.get("python_built_unexplained") != 0:    # (round 4, N4/N5: name F09 when it is the cause)
                why.append("tensors built from Python values the recording cannot explain (F09)" if
                           wit.get("python_built_unexplained") else "no F09 classification of Python-built tensors (older record)")
            if (wit.get("opaque_ops") or wit.get("opacity_unknown_ops") or wit.get("decomposition_disagrees")
                    or wit.get("external_nonscalar") != 0 or not why):
                why.append("opaque ops or tensors from outside the recording")
        out.append(("G4", "; ".join(why)))
    return out


def no_mtp_statement(wit):
    """The scoped wording of a bound "no MTP module" record (never "no runtime executes it")."""
    return (f"vLLM {wit.get('vllm_version')} (source {wit.get('vllm_source_id')}) has no MTP module for this config "
            f"at revision {wit.get('revision')}")


def loader_statement(wit, n):
    """The scoped wording when vLLM has an MTP module but its own loader maps none of the n remaining tensors."""
    return (f"vLLM {wit.get('vllm_version')} (source {wit.get('vllm_source_id')})'s MTP loader maps none of these {n} "
            f"tensors at revision {wit.get('revision')}")


def runtime_credit_needed(unbuilt, wit, revision=None, producer=None):
    """DB-3, reading B (Soumil 2026-10-12): which declared-unbuilt tensors need runtime (vLLM witness) credit before the
    library's own "unused" declaration can certify them as leftovers. Only tensors a runtime is known to run do: the
    checkpoint keys vLLM's own loader maps (executed or loaded) when vLLM has an MTP module for this config; all of
    them when it has one but the witness never mapped (a failed run); none when vLLM has no MTP module (nothing runs
    them). A missing or errored witness record is an unknown, not a failure; a failed witness run (not closed / a mapping
    error) needs credit for every one. Witness keys are compared under ckpt_split names.
    F14: no statement of a record counts unless it is bound (witness_identity_gaps); an unbound record says nothing
    (like no record) and is a G6 unknown. A "no MTP module" answer also needs the registry stage shown completed.
    Returns (need, (predicate, what) or None)."""
    unbuilt = set(unbuilt)
    if not unbuilt:
        return set(), None
    if not wit or "vllm_mtp_arch" not in wit:
        return set(), ("G3", "no runtime witness record: whether a runtime executes the declared-unbuilt tensors is not established")
    gaps = witness_identity_gaps(wit, revision, producer)
    if gaps:
        return set(), ("G6", "the runtime witness record is not bound (" + ", ".join(gaps) + ")")
    if wit.get("registry_error"):
        return set(), ("G3", f"the runtime registry could not be read ({str(wit['registry_error'])[:120]})")
    if not wit.get("vllm_mtp_arch"):
        if wit.get("registry_checked") is not True:
            return set(), ("G6", "the runtime's \"no MTP module\" answer is not shown (registry stage not completed)")
        ov = wit.get("registry_override")
        if not (isinstance(ov, dict) and isinstance(ov.get("before"), dict) and isinstance(ov.get("after"), dict)):
            return set(), ("G6", "the runtime's \"no MTP module\" answer is not shown (its registry evidence is not recorded)")
        if ov["before"] != ov["after"]:
            return set(), ("G3", "vLLM's MTP override changed this config (" + ", ".join(
                k for k in sorted(set(ov["before"]) | set(ov["after"])) if ov["before"].get(k) != ov["after"].get(k))
                + ") without naming an MTP module: whether vLLM runs the declared-unbuilt tensors is not established")
        return set(), None
    if not wit.get("closed") or wit.get("mapping_error"):
        if wit.get("network_error"):              # (F14 e) cut short by the network: not evidence about vLLM (re-record)
            return set(), ("G6", f"the runtime witness was cut short by a network failure ({str(wit['network_error'])[:100]}): re-record it")
        return unbuilt, None                      # a failed witness run proves nothing about what vLLM runs: all of them
    norm = lambda k: re.sub(r"_blocks$", "", re.sub(r"\.(qweight|weight_packed)$", ".weight", k))   # as ckpt_split names
    mapped = {norm(k) for k in (wit.get("executed_checkpoint_keys") or []) + (wit.get("loaded_not_used_keys") or [])}
    return (unbuilt & mapped if mapped else unbuilt), None
