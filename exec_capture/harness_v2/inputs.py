"""Build real inputs for a model from (a) its forward() signature, (b) the repo's own processor components and
(c) config fields. Nothing is chosen by model name. Returns ([(label, kwargs), ...], notes, processor_errors).

Order of evidence for every modality:
  the repo's processor (with its chat template)  ->  the processor's own placeholder tokens  ->  the base model's
  processor (fine-tunes often ship none)  ->  sub-processors (image processor / feature extractor) alone.
`variant=True` keeps every SHAPE identical and changes only CONTENT (used by the stability check)."""
import inspect, os
TOKEN_BUDGET = int(os.environ.get('BENCH_TOKEN_BUDGET', '1024'))
import numpy as np
import torch

TEXT = "A photo of a small red cat sitting on a wooden table next to a window."


def _load_components(repo):
    import transformers as T
    comps = {}
    for key, loader in (("processor", T.AutoProcessor), ("tokenizer", T.AutoTokenizer),
                        ("image_processor", T.AutoImageProcessor), ("feature_extractor", T.AutoFeatureExtractor),
                        ("video_processor", getattr(T, "AutoVideoProcessor", None))):
        if loader is None:
            continue
        try:
            comps[key] = loader.from_pretrained(repo, trust_remote_code=False)
        except Exception as e:
            comps[key + "_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    p = comps.get("processor")
    for attr in ("tokenizer", "image_processor", "feature_extractor", "video_processor"):
        if attr not in comps and p is not None and getattr(p, attr, None) is not None:
            comps[attr] = getattr(p, attr)
    return comps


def _proc(repo):
    comps = _load_components(repo)
    if not any(k in comps for k in ("processor", "tokenizer", "image_processor", "feature_extractor")):
        # fine-tunes often ship weights only: use the declared base model's processor
        try:
            from huggingface_hub import HfApi
            cd = HfApi().model_info(repo).card_data
            base = getattr(cd, "base_model", None) if cd is not None else None
            base = base[0] if isinstance(base, list) and base else base
            if isinstance(base, str) and base != repo:
                bc = _load_components(base)
                bc = {k: v for k, v in bc.items() if not k.endswith("_error")}
                if bc:
                    comps.update(bc); comps["processor_from_base_model"] = base
        except Exception:
            pass
    return comps


def _tensorize(d, dtype):
    out = {}
    for k, v in dict(d).items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dtype) if v.is_floating_point() else v
        elif isinstance(v, np.ndarray) and v.dtype != object:
            t = torch.from_numpy(v)
            out[k] = t.to(dtype) if t.is_floating_point() else t
    return out


def _chat(proc, part_type):
    msgs = [{"role": "user", "content": [{"type": part_type}, {"type": "text", "text": "Describe this."}]}]
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        return None


def _prompts(proc, part_type):
    """Candidate prompts: chat template, then the processor's own placeholder tokens, then plain text."""
    cands = []
    t = _chat(proc, part_type) if proc is not None else None
    if t: cands.append(t)
    cands.append("Describe this.")            # processors that insert their own placeholders need plain text
    for attr in (f"{part_type}_token", "boi_token" if part_type == "image" else None, f"start_{part_type}_token"):
        tokv = getattr(proc, attr, None) if (proc is not None and attr) else None
        if isinstance(tokv, str) and tokv:
            cands.append(f"{tokv} Describe this.")
    cands += [f"<{part_type}> Describe this.", "Describe this."]
    seen, out = set(), []
    for c in cands:
        if c not in seen: seen.add(c); out.append(c)
    return out


def _cfg_get(cfg, *names, default=None):
    for c in (cfg, getattr(cfg, "vision_config", None), getattr(cfg, "text_config", None), getattr(cfg, "decoder", None)):
        if c is None: continue
        for n in names:
            v = getattr(c, n, None)
            if v is not None:
                return v
    return default


def mask_positions(kw, cfg):
    """bool_masked_pos for a pass: one flag per patch (patch/tubelet sizes from the config; a wrong size fails loudly)."""
    pv = kw.get("pixel_values")
    ps = getattr(getattr(cfg, "vision_config", cfg), "patch_size", None) or getattr(cfg, "patch_size", 16)
    ps = ps[0] if isinstance(ps, (list, tuple)) else ps
    if pv is not None and pv.dim() >= 4:
        n = (pv.shape[-1] // ps) * (pv.shape[-2] // ps) * (pv.shape[1] // getattr(cfg, "tubelet_size", 1) if pv.dim() == 5 else 1)
    else:
        n = next((v.shape[-1] for v in kw.values() if isinstance(v, torch.Tensor) and v.dim() == 2), 16)
    return (torch.arange(n) % 2 == 0).unsqueeze(0)


def build_passes(repo, model, cfg, dtype=torch.float32, text_len_variant=False):
    variant = text_len_variant
    sig = inspect.signature(model.forward).parameters
    comps = _proc(repo)
    passes, notes = [], []
    extra_passes = []                  # run only if weights remain unused after the main passes
    if comps.get("processor_from_base_model"):
        notes.append(f"processor taken from base model {comps['processor_from_base_model']}")
    tok, img, fe, vid, proc = (comps.get(k) for k in ("tokenizer", "image_processor", "feature_extractor", "video_processor", "processor"))
    if proc is not None:                  # processor fields left empty by the repo: take the same-named value from the config
        for attr in ("num_query_tokens", "image_seq_length", "num_image_tokens"):
            if getattr(proc, attr, "absent") is None:
                val = _cfg_get(cfg, attr)
                if val is not None:
                    try: setattr(proc, attr, val); notes.append(f"processor.{attr} taken from config ({val})")
                    except Exception: pass
    dec = None
    if "decoder_input_ids" in sig:                 # the forward takes a decoder: give it a start token
        start = getattr(cfg, "decoder_start_token_id", None)
        if start is None:
            start = _cfg_get(cfg, "decoder_start_token_id", "bos_token_id", default=0) or 0
        dec = torch.tensor([[int(start)] * 4])
        ncb = _cfg_get(cfg, "num_codebooks")
        if ncb:
            dec = dec.repeat(int(ncb), 1)                      # codec decoders take one row per codebook

    required_all = {p for p, prm in sig.items() if prm.default is prm.empty and p != "self"
                    and prm.kind not in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD)}

    def finish(kw, label):
        kw = {k: v for k, v in kw.items() if k in sig}
        if "bool_masked_pos" in required_all and "bool_masked_pos" not in kw and kw:
            try:
                kw["bool_masked_pos"] = mask_positions(kw, cfg); notes.append(f"{label}: required bool_masked_pos built per patch")
            except Exception:
                pass
        if dec is not None and "decoder_input_ids" in sig:
            kw.setdefault("decoder_input_ids", dec)
        if kw:
            passes.append((label, kw))

    # --- text -----------------------------------------------------------------------------------------------------
    if "input_ids" in sig:
        if tok is not None:
            kw = _tensorize(tok(TEXT, return_tensors="pt"), dtype)
            special = set(getattr(tok, "all_special_ids", []) or [])
            if "input_ids" in kw and sum(int(i) not in special for i in kw["input_ids"].flatten()) < 4:
                # the test text is outside this tokenizer's alphabet: use ids from its own vocabulary
                vocab = [i for i in range(len(tok)) if i not in special][:1000]
                if vocab:
                    g = torch.Generator().manual_seed(1 if variant else 0)
                    kw = {"input_ids": torch.tensor([vocab])[:, torch.randint(0, len(vocab), (16,), generator=g)]}
                    if "attention_mask" in sig: kw["attention_mask"] = torch.ones_like(kw["input_ids"])
                    notes.append("test text tokenized to <4 tokens: ids drawn from the tokenizer's own vocabulary")
            if variant and "input_ids" in kw:
                kw["input_ids"] = kw["input_ids"].roll(1, dims=-1)
        else:
            g = torch.Generator().manual_seed(1 if variant else 0)
            kw = {"input_ids": torch.randint(0, max(2, min(1000, getattr(cfg, "vocab_size", 1000) or 1000)), (1, 16), generator=g)}
            notes.append("no tokenizer: random token ids")
        finish(kw, "text")

    # --- image / video-as-pixel_values -----------------------------------------------------------------------------
    IMG_SLOTS = ("pixel_values", "pixel_values_images", "image_patches", "images")
    img_slots = [a for a in IMG_SLOTS if a in sig]
    placeholder_ids = [int(v) for v in (_cfg_get(cfg, "image_token_id"), _cfg_get(cfg, "image_token_index"),
                                        _cfg_get(cfg, "image_token_id_") ) if isinstance(v, int)]
    def _has_placeholder(cand):
        """An image prompt is valid only if the processor actually inserted the image placeholder (when one exists)."""
        ids = cand.get("input_ids")
        if ids is None or not placeholder_ids or "input_ids" not in sig:
            return True
        return any(bool((ids == p).any()) for p in placeholder_ids)
    if img_slots:
        from PIL import Image
        color = (120, 60, 30) if not variant else (30, 90, 160)
        image = Image.new("RGB", (336, 336), color)
        image2 = Image.new("RGB", (336, 336), tuple(255 - c for c in color))
        nframes = _cfg_get(cfg, "num_frames")
        kw, errs = None, []
        attempts = []
        if proc is not None and "input_ids" in sig and tok is not None:
            attempts += [("processor+text", lambda p=p: proc(images=image, text=p, return_tensors="pt")) for p in _prompts(proc, "image")]
            box_args = [a for a in ("input_boxes", "input_points") if a in sig]
            attempts += [("processor+text+boxes", lambda: proc(images=image, text="a cat", input_boxes=[[[20, 20, 200, 200]]], return_tensors="pt"))] if box_args else []
        if proc is not None:
            if "input_boxes" in sig:
                attempts.append(("processor+boxes", lambda: proc(images=image, input_boxes=[[[20, 20, 200, 200]]], return_tensors="pt")))
            if "input_points" in sig:
                attempts.append(("processor+points", lambda: proc(images=image, input_points=[[[[100, 100]]]], return_tensors="pt")))
            attempts.append(("processor image-only", lambda: proc(images=image, return_tensors="pt")))
        if nframes:                                   # the config declares a clip length: frames come first
            frame_attempts = []
            for sub in (vid, img, proc):
                if sub is None: continue
                frame_attempts.append(("frames", lambda s=sub: s([[image] * int(nframes)], return_tensors="pt")))
                frame_attempts.append(("frames-flat", lambda s=sub: s([image] * int(nframes), return_tensors="pt")))
            attempts = frame_attempts + attempts
        for sub in (img, vid):
            if sub is None: continue
            attempts.append(("image processor", lambda s=sub: s(images=image, return_tensors="pt")))
            attempts.append(("image pair", lambda s=sub: s(images=[[image, image2]], return_tensors="pt")))
        # fill by name: every argument the processor takes (prompt images, masks, class lists, task text)
        mask_img = Image.new("L", (336, 336), 255)
        for sub in (proc, img):
            if sub is None: continue
            try:
                psig = inspect.signature(sub.__call__ if sub is proc else getattr(sub, "preprocess", sub.__call__)).parameters
            except (TypeError, ValueError):
                continue
            fill = {}
            for pn in psig:
                if pn == "images": fill[pn] = image
                elif "mask" in pn and "image" not in pn.replace("mask", ""): fill[pn] = mask_img
                elif "image" in pn: fill[pn] = image2
            if not fill: continue
            texts = [None] + (["a cat.", ["cat", "dog"]] if "text" in psig else [])
            for tx in texts:
                attempts.append((f"processor fill-by-name{'' if tx is None else '+text'}",
                                 lambda s=sub, f=fill, tx=tx: s(**f, **({"text": tx} if tx is not None else {}), return_tensors="pt")))
        required = [p for p, prm in sig.items() if prm.default is prm.empty and p != "self"
                    and prm.kind not in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD)]
        first = None
        for name, fn in attempts:
            try:
                cand = _tensorize(fn(), dtype)
                if "pixel_values" in cand and "pixel_values" not in sig and "pixel_values_images" in sig:
                    cand["pixel_values_images"] = cand.pop("pixel_values")        # same tensor, the forward's own name
                if any(k in cand for k in IMG_SLOTS + ("pixel_values_videos",)):
                    if not _has_placeholder(cand):
                        errs.append(f"{name}: processor inserted no image placeholder"); continue
                    if first is None: first = (cand, name, fn)
                    if all(r in cand for r in required):   # covers every input the forward requires
                        kw = cand; win_fn = fn; notes.append(f"image input via {name}"); break
                    errs.append(f"{name}: lacks required {[r for r in required if r not in cand][:4]}")
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {str(e)[:70]}")
        if kw is None and first is not None:
            kw, win_fn = first[0], first[2]; notes.append(f"image input via {first[1]} (no candidate covered every required input)")
        # token budget: attention memory grows with the square of the sequence. When the processor's own size limits
        # make the image sequence longer than the budget, lower those limits (same code path, smaller picture).
        size_objs = [o for o in dict.fromkeys([proc, getattr(proc, "image_processor", None), img]) if o is not None]
        saved = {}
        def _scale_limits(scale):
            for o in size_objs:
                for a_ in ("min_pixels", "max_pixels"):
                    v = saved.setdefault((id(o), a_), getattr(o, a_, None))
                    if isinstance(v, int) and v > 0:
                        try: setattr(o, a_, max(28 * 28 * 4, int(v * scale)))
                        except Exception: pass
                sz = saved.setdefault((id(o), "size"), getattr(o, "size", None))
                if isinstance(sz, dict):
                    new_sz = dict(sz)
                    for k_, v_ in sz.items():
                        if isinstance(v_, int) and "pixels" in k_: new_sz[k_] = max(28 * 28 * 4, int(v_ * scale))
                        elif isinstance(v_, int) and k_ == "longest_edge": new_sz[k_] = max(56, int(v_ * scale ** 0.5))
                    try: o.size = new_sz
                    except Exception: pass
        def _restore_limits():
            for o in size_objs:
                for a_ in ("min_pixels", "max_pixels", "size"):
                    if (id(o), a_) in saved and saved[(id(o), a_)] is not None:
                        try: setattr(o, a_, saved[(id(o), a_)])
                        except Exception: pass
        def _seq(c):
            ids = c.get("input_ids")
            return int(ids.shape[-1]) if ids is not None else 0
        main_scale = 1.0
        if kw and _seq(kw) > TOKEN_BUDGET:
            for k_ in range(1, 6):
                _scale_limits(0.25 ** k_)
                try:
                    c_ = _tensorize(win_fn(), dtype)
                    if "pixel_values" in c_ and "pixel_values" not in sig and "pixel_values_images" in sig:
                        c_["pixel_values_images"] = c_.pop("pixel_values")
                except Exception:
                    break
                if _has_placeholder(c_) and _seq(c_) < _seq(kw):
                    notes.append(f"image sequence {_seq(kw)} > budget {TOKEN_BUDGET}: processor size limits x{0.25 ** k_:g} -> {_seq(c_)}")
                    kw, main_scale = c_, 0.25 ** k_
                    if _seq(c_) <= TOKEN_BUDGET: break
            _scale_limits(main_scale) if main_scale < 1 else _restore_limits()
        if kw:
            finish(kw, "image")
            if "input_ids" in sig and "input_ids" not in kw and tok is not None and not placeholder_ids:
                # the processor gave pixels only, the forward also reads text: run both together
                try:
                    t = _tensorize(tok(TEXT, return_tensors="pt"), dtype)
                    finish({**kw, **{k: v for k, v in t.items() if k in ("input_ids", "attention_mask")}}, "image+text")
                    notes.append("image+text pass: tokenizer ids merged into image-only processor output")
                except Exception as e:
                    notes.append(f"image+text merge failed: {type(e).__name__}: {str(e)[:60]}")
            try:                                       # a second resolution: modules chosen by image size
                big = Image.new("RGB", (1024, 768), color)
                if main_scale < 1:
                    _scale_limits(main_scale * 2)         # twice the main pass's pixels: a different, bounded size
                for name2, fn2 in attempts:
                    if name2 != next(n for n in notes[::-1] if n.startswith("image input via")).split("via ", 1)[1].split(" (")[0]:
                        continue
                    cand2 = None
                    if name2.startswith("processor+text"):
                        for p in _prompts(proc, "image"):
                            try:
                                c2 = _tensorize(proc(images=big, text=p, return_tensors="pt"), dtype)
                                if _has_placeholder(c2): cand2 = c2; break
                            except Exception:
                                continue
                    elif name2 == "processor image-only":
                        cand2 = _tensorize(proc(images=big, return_tensors="pt"), dtype)
                    elif name2 == "image processor" and img is not None:
                        cand2 = _tensorize(img(images=big, return_tensors="pt"), dtype)
                    if cand2 is not None:
                        if "pixel_values" in cand2 and "pixel_values" not in sig and "pixel_values_images" in sig:
                            cand2["pixel_values_images"] = cand2.pop("pixel_values")
                        if _seq(cand2) > 3 * TOKEN_BUDGET:
                            notes.append(f"large-image variant skipped: {_seq(cand2)} tokens > {3 * TOKEN_BUDGET} (memory)")
                        else:
                            extra_passes.append(("image:large", {k: v for k, v in cand2.items() if k in sig}))
                    break
            except Exception as e:
                notes.append(f"large-image variant failed: {type(e).__name__}: {str(e)[:60]}")
            finally:
                _restore_limits()
            # every prompt type the forward accepts gets its own pass (each exercises a different path)
            for arg, prompt in (("input_boxes", [[[20, 20, 200, 200]]]), ("input_points", [[[[100, 100]]]])):
                if arg in sig and proc is not None:
                    try:
                        extra = _tensorize(proc(images=image, **{arg: prompt}, return_tensors="pt"), dtype)
                        if arg in extra:
                            finish(extra, f"image+{arg}")
                    except Exception as e:
                        notes.append(f"{arg} pass failed: {type(e).__name__}: {str(e)[:60]}")
        elif errs:
            notes += errs[:4]

    # --- audio -----------------------------------------------------------------------------------------------------
    audio_keys = [k for k in ("input_features", "input_values", "audio_values", "input_audio_embeds") if k in sig]
    if audio_keys or ("input_ids" in sig and fe is not None and hasattr(fe, "sampling_rate")):
        sr = int(getattr(fe, "sampling_rate", 16000) or 16000) if fe is not None else 16000
        freq = 440 if not variant else 660
        wav = (0.1 * np.sin(2 * np.pi * freq * np.arange(int(sr * 1.0)) / sr)).astype(np.float32)
        kw = None
        if proc is not None and "input_ids" in sig and tok is not None:
            for p in _prompts(proc, "audio"):
                for key in ("audio", "audios"):
                    try:
                        kw = _tensorize(proc(**{key: wav}, text=p, sampling_rate=sr, return_tensors="pt"), dtype); break
                    except Exception as e:
                        notes.append(f"audio processor ({key}) failed: {type(e).__name__}: {str(e)[:70]}")
                if kw: break
        if kw is None and fe is not None:
            try:
                kw = _tensorize(fe(wav, sampling_rate=sr, return_tensors="pt"), dtype)
            except Exception as e:
                notes.append(f"feature extractor failed: {type(e).__name__}: {str(e)[:70]}")
        if kw:
            finish(kw, "audio")

    # --- video (separate argument) --------------------------------------------------------------------------------
    if any(k in sig for k in ("pixel_values_videos", "video_values")) and (vid is not None or proc is not None):
        frames = [np.full((224, 224, 3), (60 if not variant else 140) + 10 * i, dtype=np.uint8) for i in range(8)]
        kw = None
        vid_ids = [int(v) for v in (_cfg_get(cfg, "video_token_id"), _cfg_get(cfg, "video_token_index")) if isinstance(v, int)]
        for p in (_prompts(proc, "video") if proc is not None else [None]):
            try:
                cand = _tensorize(proc(videos=[frames], text=p or "Describe.", return_tensors="pt"), dtype)
                ids = cand.get("input_ids")
                if vid_ids and ids is not None and not any(bool((ids == t).any()) for t in vid_ids):
                    notes.append("video prompt without video placeholder: skipped"); continue
                kw = cand; break
            except Exception as e:
                notes.append(f"video processor failed: {type(e).__name__}: {str(e)[:70]}")
        if kw is None and vid is not None:
            try:
                kw = _tensorize(vid(videos=[frames], return_tensors="pt"), dtype)
            except Exception as e:
                notes.append(f"video sub-processor failed: {type(e).__name__}: {str(e)[:70]}")
        if kw:
            finish(kw, "video")

    # --- time series (windows sized from config) --------------------------------------------------------------------
    if "past_values" in sig:
        g = torch.Generator().manual_seed(1 if variant else 0)
        ctx = int(_cfg_get(cfg, "context_length", "context_len", default=32) or 32)
        lags = _cfg_get(cfg, "lags_sequence", default=None)
        past = ctx + (max(lags) if lags else 0)
        pred = int(_cfg_get(cfg, "prediction_length", "horizon_length", default=8) or 8)
        chans = int(_cfg_get(cfg, "num_input_channels", "input_size", "n_input_channels", default=1) or 1)
        ntf = int(_cfg_get(cfg, "num_time_features", default=0) or 0)
        shape = (1, past, chans) if (chans > 1 or "num_input_channels" in cfg.to_dict()) else (1, past)
        kw = {"past_values": torch.randn(*shape, generator=g)}
        if "past_observed_mask" in sig: kw["past_observed_mask"] = torch.ones(*shape)
        if "past_time_features" in sig and ntf: kw["past_time_features"] = torch.randn(1, past, ntf, generator=g)
        if "future_time_features" in sig and ntf: kw["future_time_features"] = torch.randn(1, pred, ntf, generator=g)
        nsc = int(_cfg_get(cfg, "num_static_categorical_features", default=0) or 0)
        nsr = int(_cfg_get(cfg, "num_static_real_features", default=0) or 0)
        if "static_categorical_features" in sig and nsc: kw["static_categorical_features"] = torch.zeros(1, nsc, dtype=torch.long)
        if "static_real_features" in sig and nsr: kw["static_real_features"] = torch.randn(1, nsr, generator=g)
        finish(kw, "time_series")

    # --- last resort: the library's own dummy inputs -----------------------------------------------------------------
    if not passes:
        try:
            d = model.dummy_inputs
            if d:
                passes.append(("dummy_inputs", _tensorize(d, dtype)))
                notes.append("used model.dummy_inputs")
        except Exception:
            notes.append(f"no input path: forward needs {[k for k, p in sig.items() if p.default is inspect.Parameter.empty][:6]}")
    errs = {k: v for k, v in comps.items() if k.endswith("_error")}
    if hasattr(model, "__dict__"):
        model.__dict__["_bench_extra_passes"] = extra_passes
    return passes, notes, errs
