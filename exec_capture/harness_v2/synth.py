"""Input synthesizer for a module whose forward never ran (a pipeline stopped early, or no processor builds its inputs).

Every input is chosen by its ARGUMENT NAME, with sizes taken from the module's OWN config. Ambiguous cases produce a
short ladder of candidates (image latent / video latent / packed token sequence; each text width the config offers).
Each candidate is a real forward call: a wrong guess fails loudly and the next one is tried. Nothing about the
structure is assumed; only input values and shapes are chosen.
"""
import inspect
import torch

S = 16            # spatial size of a latent (small: architecture does not depend on it)
L = 8             # text tokens

LATENT = ("sample", "hidden_states", "x", "latents", "latent_model_input", "noisy_latents")
TIME = ("timestep", "timesteps", "t", "sigma", "decoder_noise_time", "noise_level")


def _cfg(m):
    c = getattr(m, "config", None)
    return c if c is not None else {}


def _subconfigs(c):
    yield c
    for a in (list(c.keys()) if isinstance(c, dict) else dir(c)):
        if a.endswith("_config") and not a.startswith("_"):
            s = c.get(a) if isinstance(c, dict) else getattr(c, a, None)
            if s is not None and (isinstance(s, dict) or hasattr(s, "to_dict")):
                yield s


def _get(c, *names):
    for n in names:
        for sc in _subconfigs(c):                      # the module's own config first, then its sub-configs
            v = sc.get(n) if isinstance(sc, dict) else getattr(sc, n, None)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], int):
                v = v[-1]
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                yield v


def _first(c, *names, default=None):
    return next(_get(c, *names), default)


def _text_dims(c):
    seen = []
    for v in _get(c, "cross_attention_dim", "joint_attention_dim", "caption_channels", "text_dim", "text_embed_dim",
                  "encoder_hid_dim", "context_dim", "clip_text_in_channels", "c_clip_text", "text_hidden_size",
                  "encoder_hidden_size", "condition_dim", "hidden_size"):
        if v not in seen: seen.append(v)
    return seen or [768]


def _pooled_dims(c):
    seen = []
    for v in _get(c, "pooled_projection_dim", "projection_dim", "clip_text_pooled_in_channels", "c_clip_text_pooled",
                  "projection_class_embeddings_input_dim", "pooled_dim"):
        if v not in seen: seen.append(v)
    return seen or [768]


def _latent_shapes(c):
    # parts that do not declare in_channels (e.g. some VAEs) take a picture: 3 channels
    C = _first(c, "in_channels", "in_chans", "input_channels", "num_channels", "c_in", default=3)
    out = [("image", (1, C, S, S)), ("video", (1, C, 3, S, S)), ("tokens", (1, (S // 2) * (S // 2), C)), ("audio", (1, C, 64))]
    # video parts patch/compress time: other frame counts, so the one matching the config's temporal factor runs
    out += [(f"video{t}", (1, C, t, S, S)) for t in (1, 2, 4, 5, 8, 9, 13, 17)]
    return out


def _image_hw(c):
    for sc in _subconfigs(c):
        v = sc.get("image_size") if isinstance(sc, dict) else getattr(sc, "image_size", None)
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
            return tuple(v)
        if isinstance(v, int) and v > 0:
            return (v, v)
    return (224, 224)


def required_params(fn):
    sig = inspect.signature(fn).parameters
    return [p for p, prm in sig.items() if p != "self" and prm.default is prm.empty
            and prm.kind not in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD)], sig


def candidates(module, fn=None, limit=16, seed=0):
    """Yield (label, kwargs) candidates for module.forward (or fn)."""
    fn = fn or module.forward
    req, sig = required_params(fn)
    c = _cfg(module)
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g) * 0.02
    names = list(sig)
    wants_text = any("encoder_hidden_states" in p or p in ("context", "clip_text") for p in names)
    tdims = _text_dims(c) if wants_text else [None]
    n = 0
    # every latent shape with the primary text width first, then the other widths (breadth before depth)
    order = [(ls, tdims[0]) for ls in _latent_shapes(c)] + [(ls, td) for td in tdims[1:] for ls in _latent_shapes(c)]
    for (lk, lshape), td in order:
        kw = {}
        seq = lshape[1] if lk == "tokens" else (lshape[-1] * lshape[-2] // 4 if lk.startswith(("image", "video")) else 8)
        for p in names:
            if p in ("self", "kwargs", "return_dict"): continue
            q = p.lower(); required = p in req
            if q.endswith("input_ids") and (required or not any(x in sig for x in LATENT)):
                kw[p] = torch.randint(0, max(2, min(1000, _first(c, "vocab_size", default=1000))), (1, L), generator=g)
            elif q == "clip_input":
                hw = _image_hw(c)
                kw[p] = rnd(1, 3, *hw)
            elif q in ("pixel_values", "image_input") and (required or not any(x in sig for x in LATENT) and not any(x.endswith("input_ids") for x in sig)):
                hw = _image_hw(c)
                kw[p] = rnd(1, _first(c, "num_channels", default=3), *hw)
            elif q == "input_features" and (required or not any(x in sig for x in LATENT)):
                mel = _first(c, "num_mel_bins", "feature_size", default=80)
                src = _first(c, "max_source_positions", default=None)
                kw[p] = rnd(1, mel, 2 * src if src else 3000)
            elif q in LATENT or (q.endswith("_latents") and required):
                kw[p] = rnd(*lshape)
            elif q in ("controlnet_cond", "control_cond", "cond_image", "pixels", "image"):
                if required or q == "controlnet_cond":
                    kw[p] = rnd(1, _first(c, "conditioning_channels", default=3), S * 8, S * 8)
            elif q in TIME:
                kw[p] = torch.tensor([500.0])
            elif q in ("timestep_ratio",):
                kw[p] = torch.tensor([0.5])
            elif "encoder_hidden_states" in q and "mask" not in q or q in ("context", "clip_text"):
                if required or q in ("encoder_hidden_states",):
                    kw[p] = rnd(1, L, td or 768)
            elif "pooled" in q:
                kw[p] = rnd(1, _pooled_dims(c)[0]) if "clip_text_pooled" not in q else rnd(1, 1, _pooled_dims(c)[0])
            elif q in ("img_ids",):
                kw[p] = torch.zeros(seq, 3)
            elif q in ("txt_ids",):
                kw[p] = torch.zeros(L, 3)
            elif q == "guidance":
                kw[p] = torch.tensor([1.0])
            elif q in ("class_labels",) and (required or _first(c, "num_class_embeds") or getattr(c, "class_embed_type", None)):
                cet = getattr(c, "class_embed_type", None)
                if cet in ("projection", "simple_projection"):
                    kw[p] = rnd(1, _first(c, "projection_class_embeddings_input_dim", default=1024))
                else:
                    kw[p] = torch.tensor([0])
            elif q == "proj_embedding":
                kw[p] = rnd(1, _first(c, "embedding_dim", "embedding_proj_dim", default=768))
            elif q == "added_cond_kwargs" and _first(c, "projection_class_embeddings_input_dim") and getattr(c, "addition_embed_type", None) == "text_time":
                tid = _first(c, "addition_time_embed_dim", default=256)
                kw[p] = {"text_embeds": rnd(1, _first(c, "projection_class_embeddings_input_dim") - 6 * tid), "time_ids": torch.zeros(1, 6)}
            elif q in ("img_shapes",):
                kw[p] = [[(1, S // 2, S // 2)]]
            elif q in ("txt_seq_lens",):
                kw[p] = [L]
            elif "mask" in q and required:
                kw[p] = torch.ones(1, L)
            elif q in ("fps",) and required:
                kw[p] = 8
            elif required and (q.endswith("_ids") or q.endswith("_id") or "index" in q or "token" in q):
                kw[p] = torch.zeros(1, L, dtype=torch.long)
            elif required and "embed" in q and not q.endswith(("_ids", "_id")) and "index" not in q:
                kw[p] = rnd(1, _first(c, "embedding_dim", "projection_dim", "embed_dim", "hidden_size", default=768))
            elif required:
                # an input this synthesizer has no name rule for: the shape varies across candidates (each one is a
                # real forward call that fails loudly when wrong): sequence of values / sequence of vectors / picture
                H = td or _first(c, "hidden_size", "d_model", default=64)
                ctx = _first(c, "context_length", "context_len", "max_position_embeddings", default=32)
                ctx = min(ctx, 512)
                kw[p] = {"image": rnd(1, 3, S * 8, S * 8), "tokens": rnd(1, ctx),
                         "audio": torch.zeros(1, ctx, dtype=torch.long)}.get(lk, rnd(1, L, H))
        yield f"latent={lk},text={td}", kw
        n += 1
        if n >= limit: return


def flat_tensors(kw, prefix):
    """{name: tensor} for every tensor in kwargs (nested dicts/lists included) so the recorder knows them as inputs."""
    out = {}
    def walk(v, name):
        if isinstance(v, torch.Tensor): out[name] = v
        elif isinstance(v, dict):
            for k, x in v.items(): walk(x, f"{name}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v): walk(x, f"{name}.{i}")
    for k, v in kw.items(): walk(v, f"{prefix}.{k}")
    return out
