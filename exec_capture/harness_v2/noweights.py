"""Hard guard: this process may NEVER download model weights.
Any download of a weight file through huggingface_hub raises. Header reads (HTTP range requests) are unaffected."""
import re
WEIGHT = re.compile(r"\.(safetensors|bin|pt|pth|ckpt|gguf|onnx|onnx_data|msgpack|h5|tflite|mlmodel|npz|pkl)$", re.I)


class WeightDownloadBlocked(RuntimeError):
    pass


def install():
    import huggingface_hub, huggingface_hub.file_download as fd
    orig = fd.hf_hub_download
    def guarded(repo_id, filename, *a, **k):
        name = filename if isinstance(filename, str) else str(filename)
        sub = k.get("subfolder") or ""
        if WEIGHT.search(name):
            raise WeightDownloadBlocked(f"blocked weight download: {repo_id}/{sub + '/' if sub else ''}{name}")
        return orig(repo_id, filename, *a, **k)
    fd.hf_hub_download = guarded
    huggingface_hub.hf_hub_download = guarded
    try:
        import transformers.utils.hub as th
        if hasattr(th, "hf_hub_download"): th.hf_hub_download = guarded
    except Exception:
        pass
    try:
        import diffusers.utils.hub_utils as dh
        if hasattr(dh, "hf_hub_download"): dh.hf_hub_download = guarded
    except Exception:
        pass
    # snapshot_download is also how transformers fetches TOKENIZER files: keep it, but never let a weight file through
    BLOCK = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "*.onnx_data", "*.msgpack",
             "*.h5", "*.tflite", "*.mlmodel", "*.npz", "*.pkl"]
    import huggingface_hub._snapshot_download as sd
    orig_snap = sd.snapshot_download
    def filtered_snapshot(*a, **k):
        ig = k.get("ignore_patterns") or []
        ig = [ig] if isinstance(ig, str) else list(ig)
        k["ignore_patterns"] = ig + BLOCK
        return orig_snap(*a, **k)
    sd.snapshot_download = filtered_snapshot
    huggingface_hub.snapshot_download = filtered_snapshot
    try:
        import huggingface_hub.hf_api as ha
        if hasattr(ha, "snapshot_download"): ha.snapshot_download = filtered_snapshot
    except Exception:
        pass


# ---- one checkpoint revision per run (INTENT.md G6) ------------------------------------------------------------------
# Every hub read of the pinned repo (config, tokenizer, processor, model card, headers, the library-load mirror) uses the
# commit resolved once at the start, so a repo updated mid-run cannot mix revisions. Calls that name no revision (or
# "main") are pointed at the pinned commit; a call naming another revision is counted (PIN_LOG) and left alone.
PIN, PIN_LOG = {}, {"pinned": 0, "other_revision": []}
_API_METHODS = ("model_info", "list_repo_files", "list_repo_tree", "parse_safetensors_file_metadata", "file_exists",
                "get_paths_info", "hf_hub_download", "snapshot_download")


def _pinned(repo_id, kwargs):
    sha = PIN.get(repo_id)
    if sha is None:
        return kwargs
    rev = kwargs.get("revision")
    if rev in (None, "main"):
        kwargs["revision"] = sha; PIN_LOG["pinned"] += 1
    elif rev != sha and len(PIN_LOG["other_revision"]) < 20:
        PIN_LOG["other_revision"].append(str(rev))
    return kwargs


def ref(repo):
    """A HfFileSystem path prefix for the pinned revision of repo ("repo@sha"), or repo itself when not pinned."""
    return f"{repo}@{PIN[repo]}" if repo in PIN else repo


def pin(repo, sha):
    import huggingface_hub, huggingface_hub.hf_api as ha, huggingface_hub.file_download as fd
    import huggingface_hub._snapshot_download as sd
    PIN[repo] = sha
    if getattr(pin, "installed", False):
        _rebind_all()                        # modules imported since the last pin() may hold unpinned functions
        return
    pin.installed = True
    for name in _API_METHODS:
        if not hasattr(ha.HfApi, name):
            continue
        orig = getattr(ha.HfApi, name)
        def wrapped(self, repo_id, *a, __orig=orig, **k):
            return __orig(self, repo_id, *a, **_pinned(repo_id, k))
        setattr(ha.HfApi, name, wrapped)
        if hasattr(huggingface_hub, name) and name not in ("hf_hub_download", "snapshot_download"):
            setattr(huggingface_hub, name, getattr(ha.api, name))          # the module-level bound shortcuts
    # the cache lookup: transformers resolves a downloaded file with try_to_load_from_cache(revision=None -> "main"),
    # which a commit-pinned download never writes; the lookup must use the pinned commit too
    for mod in [m for m in (_mod("huggingface_hub"), _mod("huggingface_hub.file_download"), _mod("transformers.utils.hub"),
                            _mod("diffusers.utils.hub_utils")) if m]:
        cur = getattr(mod, "try_to_load_from_cache", None)
        if cur is None or getattr(cur, "_pinned", False):
            continue
        def lookup(repo_id, filename, *a, __cur=cur, **k):
            if a:                                       # cache_dir positional
                k.setdefault("cache_dir", a[0]); a = a[1:]
            if repo_id in PIN and k.get("revision") in (None, "main"):
                k["revision"] = PIN[repo_id]
            return __cur(repo_id, filename, *a, **k)
        lookup._pinned = True
        setattr(mod, "try_to_load_from_cache", lookup)
    # the module-level download functions (already wrapped by install(): wrap the current ones again)
    for mod in [fd, huggingface_hub, sd] + [m for m in (_mod("transformers.utils.hub"), _mod("diffusers.utils.hub_utils"),
                                                         _mod("huggingface_hub.hf_api")) if m]:
        for name in ("hf_hub_download", "snapshot_download"):
            cur = getattr(mod, name, None)
            if cur is None or getattr(cur, "_pinned", False):
                continue
            def wrapped_fn(repo_id, *a, __cur=cur, **k):
                return __cur(repo_id, *a, **_pinned(repo_id, k))
            wrapped_fn._pinned = True
            setattr(mod, name, wrapped_fn)
    # modules that bound a hub function at import (e.g. `from huggingface_hub import hf_hub_download` in diffusers or in
    # a worker) still hold the unpinned object: rebind every loaded module's reference to the pinned one (LAST, once the
    # pinned functions exist)
    _rebind_all()


def _mod(name):
    import importlib
    try:
        return importlib.import_module(name)
    except Exception:
        return None



_HUB_NAMES = ("hf_hub_download", "snapshot_download", "try_to_load_from_cache", "model_info", "list_repo_files",
              "list_repo_tree", "get_paths_info", "file_exists", "parse_safetensors_file_metadata")


def _rebind_all():
    """After pin(): any module attribute named like a hub function that is not the pinned one is replaced by the pinned
    one (taken from the module that defines it), so nothing reads main through an early import."""
    import sys, huggingface_hub, huggingface_hub.file_download as fd, huggingface_hub._snapshot_download as sd
    pinned = {}
    for name in _HUB_NAMES:
        for src in (huggingface_hub, fd, sd):
            f = getattr(src, name, None)
            if f is not None and (getattr(f, "_pinned", False) or name not in ("hf_hub_download", "snapshot_download",
                                                                               "try_to_load_from_cache")):
                pinned[name] = f; break
    for mod in list(sys.modules.values()):
        if mod is None or mod in (huggingface_hub,):
            continue
        for name, f in pinned.items():
            try:
                cur = mod.__dict__.get(name)
            except Exception:
                break
            if cur is not None and cur is not f and callable(cur) and getattr(cur, "__module__", "").startswith(("huggingface_hub", "noweights")):
                try:
                    setattr(mod, name, f)
                except Exception:
                    pass


def revision_outcome(repos=None):
    """What the run's fresh HF cache actually holds: for each repo, the snapshot commits and refs written. A pinned run
    must hold exactly the pinned commit and no refs/main (anything else = some read went to another revision)."""
    import os
    from huggingface_hub.constants import HF_HUB_CACHE
    out = {}
    for repo, sha in PIN.items():
        if repos and repo not in repos: continue
        d = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
        snaps = sorted(os.listdir(os.path.join(d, "snapshots"))) if os.path.isdir(os.path.join(d, "snapshots")) else []
        refs = sorted(os.listdir(os.path.join(d, "refs"))) if os.path.isdir(os.path.join(d, "refs")) else []
        out[repo] = {"pinned": sha, "snapshots": snaps, "refs": refs, "ok": set(snaps) <= {sha} and not refs}
    return out


def network_error(e):
    """The exception chain is a network / hub-availability failure (never evidence about a model): returns a short
    description, else None. By type, not by name."""
    import socket, ssl
    kinds = [ConnectionError, socket.gaierror, ssl.SSLError]       # not socket.timeout: an alias of builtin TimeoutError
    http_answer, http_misuse = (), ()
    try:
        import httpx; kinds.append(httpx.TransportError); http_answer = (httpx.HTTPError,)
        http_misuse = (httpx.StreamError,)
    except Exception: pass
    def raised_in_http_client(x):
        """Raised by the HTTP client's own code (its innermost frame is in httpx / httpcore), e.g. a closed client.
        Typed httpx errors are handled by type above (a TransportError is network; an HTTPStatusError is the server's
        answer, e.g. a 404, never network)."""
        tb, last = x.__traceback__, None
        while tb is not None:
            last, tb = tb, tb.tb_next
        mod = str(last.tb_frame.f_globals.get("__name__", "")) if last is not None else ""
        # (an argument-type error or a consumed stream raised inside httpx is a caller bug, not the network)
        return (mod.split(".")[0] in ("httpx", "httpcore") and not isinstance(x, http_answer)
                and not isinstance(x, (TypeError,) + http_misuse))
    try:
        import requests; kinds.append(requests.exceptions.ConnectionError); kinds.append(requests.exceptions.Timeout)
    except Exception: pass
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled, HfHubHTTPError
        kinds += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:
        HfHubHTTPError = None
    try:                                                  # pydantic (vLLM's config dataclasses) wraps an exception
        from pydantic_core import ValidationError as _PydVE    # raised in validation without __cause__ / __context__:
    except Exception:                                     # its own record of it is ctx["error"] (F14 e)
        _PydVE = None
    def _wrapped(x):
        if _PydVE is not None and isinstance(x, _PydVE):
            try:
                return [d["ctx"]["error"] for d in x.errors() if isinstance((d.get("ctx") or {}).get("error"), BaseException)]
            except Exception:
                return []
        return []
    chain, todo, seen, hit = [], [e], set(), False
    while todo and len(chain) < 10:
        x = todo.pop(0)
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        chain.append(type(x).__name__)
        if isinstance(x, tuple(kinds)) or raised_in_http_client(x):
            hit = True
        if HfHubHTTPError is not None and isinstance(x, HfHubHTTPError):
            code = getattr(getattr(x, "response", None), "status_code", None)
            if code is not None and (code >= 500 or code == 429): hit = True
        if "couldn't connect to" in str(x) or "We couldn't connect" in str(x):
            hit = True                                    # transformers' own wording for an unreachable hub
        todo += [x.__cause__ or x.__context__] + _wrapped(x)
    return " <- ".join(chain) if hit else None
