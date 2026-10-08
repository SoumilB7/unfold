"""Run the strict capture check over a slice of the catalog.
Usage: run_batch.py --shard K/N [--categories a,b] [--workers 2] [--limit 0]
- one subprocess per model, hard timeout, resumable (existing verdicts are skipped)
- every model downloads its small files (config/tokenizer/processor) into a throwaway cache that is deleted
  immediately after; only the header table (tiny JSON) and the result JSON are kept
- GGUF / adapter repos are recorded as OUT with a pointer, and their base model is added as its own job
- aborts cleanly if free disk drops below --min-free-gb"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="1/1"); ap.add_argument("--categories", default="")
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--min-free-gb", type=float, default=3.0)
ap.add_argument("--min-free-mem-pct", type=int, default=25, help="start a job only when the system reports this much memory free")
ap.add_argument("--max-worker-gb", type=float, default=4.0, help="footprint cap per worker while running in parallel (above it: deferred to solo)")
ap.add_argument("--solo-worker-gb", type=float, default=8.0, help="footprint cap when a deferred job runs alone")
ap.add_argument("--critical-mem-pct", type=int, default=15, help="below this system free %%, the biggest running job is deferred")
ap.add_argument("--out", default=os.path.join(ROOT, "results_v2"))
ap.add_argument("--catalog", default=os.path.join(ROOT, "catalog", "index.json"))
ap.add_argument("--python", default=sys.executable, help="interpreter for workers (e.g. an isolated newer-library venv)")
ap.add_argument("--repos-file", default="", help="only run repos listed in this file (one per line)")
a = ap.parse_args()
k, n = map(int, a.shard.split("/"))
rows = json.load(open(a.catalog))
cats = {c for c in a.categories.split(",") if c}
rows = [r for r in rows if not cats or r.get("category") in cats]
wanted = {l.strip() for l in open(a.repos_file) if l.strip()} if a.repos_file else None
if wanted is not None:   # keep catalog rows that are wanted OR whose base model is wanted (base models run as their own job)
    def _base(r):
        b = r.get("base_model"); return (b[0] if b else None) if isinstance(b, list) else b
    rows = [r for r in rows if r["repo"] in wanted or _base(r) in wanted]
rows = sorted(rows, key=lambda r: r["repo"])
rows = [r for i, r in enumerate(rows) if i % n == k - 1]
known = {r["repo"] for r in json.load(open(a.catalog))}
jobs = []
for r in rows:
    lib = (r.get("library_name") or "").lower()
    is_gguf = lib in ("gguf", "llama.cpp") or r.get("quant_format") == "gguf"
    diffusion_row = "diffusion" in (r.get("category") or "") or bool(r.get("diffusers_pipeline_class"))
    if is_gguf and not diffusion_row:
        # measured directly: the library's gguf_file rule (config + parts list from the GGUF header, no weights)
        jobs.append(("run", r))
        b = r.get("base_model")
        if isinstance(b, list): b = b[0] if b else None
        if b and b not in known:
            jobs.append(("run", {**r, "repo": b, "sources": [f"base_of:{r['repo']}"], "library_name": None, "quant_format": None}))
            known.add(b)
    elif lib in ("gguf", "ggml", "llama.cpp", "mlx", "transformers.js") or r.get("quant_format") == "gguf" or r.get("negative_control") == "adapter_lora":
        jobs.append(("pointer", r))
        b = r.get("base_model")
        if isinstance(b, list): b = b[0] if b else None
        if b and b not in known:
            jobs.append(("run", {**r, "repo": b, "sources": [f"base_of:{r['repo']}"], "library_name": None}))
            known.add(b)
    else:
        jobs.append(("run", r))
if wanted is not None:
    jobs = [j for j in jobs if j[1]["repo"] in wanted]
if a.limit:
    jobs = jobs[: a.limit]


def out_path(r):
    return os.path.join(a.out, r.get("category") or "uncategorized", r.get("subcategory") or "other", r["repo"].replace("/", "__") + ".json")


def _free_mem_pct():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        return int(out.rsplit("free percentage:", 1)[1].strip().rstrip("%"))
    except Exception:
        return 100


def _footprint_gb(pid):
    """phys_footprint from proc_pid_rusage (what Activity Monitor shows; includes compressed/swapped pages)."""
    import ctypes
    try:
        lib = _footprint_gb.lib = getattr(_footprint_gb, "lib", None) or ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * 32)()
        if lib.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:      # RUSAGE_INFO_V2
            return _rss_gb(pid)
        return buf[9] / 1e9                                                  # uuid(2 x u64) + 7 fields -> phys_footprint
    except Exception:
        return _rss_gb(pid)


RUNNING = {}                    # pid -> repo, for the system-level guard
DEFERRED = []                   # jobs stopped by the guard, run alone at the end


def _rss_gb(pid):
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() or 0) / 1048576
    except Exception:
        return 0.0


def free_gb():
    return shutil.disk_usage("/").free / 1e9


def run(job, solo=False):
    kind, r = job
    p = out_path(r)
    if os.path.exists(p):
        try:
            if json.load(open(p)).get("verdict"):
                return r["repo"], "cached", 0
        except Exception:
            pass
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if kind == "pointer":
        json.dump({"repo": r["repo"], "verdict": "OUT", "reason": f"{r.get('library_name') or r.get('quant_format') or 'adapter'} repo: "
                   f"architecture is measured on its base model {r.get('base_model')}", "catalog": r}, open(p, "w"), indent=1)
        return r["repo"], "OUT(pointer)", 0
    if free_gb() < a.min_free_gb:
        return r["repo"], "SKIPPED(low disk)", 0
    is_diff = bool(r.get("diffusers_pipeline_class")) or (r.get("library_name") or "") == "diffusers"
    worker = os.path.join(HERE, "worker_v2_diffusion.py" if is_diff else "worker_v2.py")
    tmp = tempfile.mkdtemp(prefix="hfc_", dir=os.path.join(ROOT, ".tmp_hf"))
    env = dict(os.environ, HF_HUB_CACHE=tmp, BENCH_THREADS="2", TOKENIZERS_PARALLELISM="false",
               BENCH_HEADER_CACHE=os.path.join(ROOT, "cache_v2", "headers"))
    t = time.time(); status = "done"
    try:
        job_timeout = 2400 if is_diff else 1800          # optional stages stop early enough to save the result
        env["BENCH_OPTIONAL_BUDGET_S"] = str(job_timeout - 400)
        while _free_mem_pct() < a.min_free_mem_pct:          # start gate: do not start a job into memory pressure
            time.sleep(5)
        # the runner's kill budget: measured by both sides on the monotonic clock (it pauses while the machine sleeps,
        # like the kill timer below), so a sleep never makes the worker think its time is used up
        env["BENCH_JOB_TIMEOUT_S"] = str(job_timeout)
        proc = subprocess.Popen([a.python, worker, r["repo"], p], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        RUNNING[proc.pid] = r["repo"]
        t_start, last_sys = time.monotonic(), 0.0
        cap = a.solo_worker_gb if solo else a.max_worker_gb
        try:
            while proc.poll() is None:
                if time.monotonic() - t_start > job_timeout:
                    proc.kill(); proc.wait(); raise subprocess.TimeoutExpired(worker, job_timeout)
                fp = _footprint_gb(proc.pid)
                over = fp > cap
                if not over and time.monotonic() - last_sys > 5:        # system guard: stop the biggest running job
                    last_sys = time.monotonic()
                    if _free_mem_pct() < a.critical_mem_pct:
                        sizes = {pid: _footprint_gb(pid) for pid in list(RUNNING)}
                        over = max(sizes, key=sizes.get) == proc.pid
                if over:
                    proc.kill(); proc.wait()
                    status = "memcap" if solo else "deferred"
                    break
                time.sleep(2)
        finally:
            RUNNING.pop(proc.pid, None)
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        tmp_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(tmp) for f in fs if not os.path.islink(os.path.join(dp, f))) / 1e6
        heavy = [f for dp, _, fs in os.walk(tmp) for f in fs if f.rsplit(".", 1)[-1].lower() in ("safetensors", "bin", "pt", "pth", "gguf", "onnx", "ckpt", "h5", "msgpack")]
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        R = json.load(open(p))
    except Exception:
        R = {"repo": r["repo"], "verdict": "FAIL", "reason": "worker crashed without writing a result (likely out of memory or segfault)"}
    if status == "deferred":                                   # not a verdict: it runs alone at the end
        DEFERRED.append(job)
        try: os.remove(p)
        except OSError: pass
        return r["repo"], "DEFERRED(memory)", round(time.time() - t, 1)
    if status == "memcap":
        R = {"repo": r["repo"], "verdict": "FAIL", "reason": f"memory: footprint exceeded {a.solo_worker_gb} GB even running alone"}
    if status == "timeout" and not R.get("verdict"):
        R["verdict"], R["reason"] = "FAIL", "timeout"
    if not R.get("verdict"):
        R["verdict"], R["reason"] = "FAIL", R.get("reason") or "worker ended without a verdict"
    R["catalog"] = {kk: r.get(kk) for kk in ("category", "subcategory", "sources", "pipeline_tag", "library_name", "mechanism_tags", "total_params", "quant_format", "gated", "remote_code")}
    R["job_secs"] = round(time.time() - t, 1)
    R["tmp_cache_mb"] = round(tmp_mb, 2); R["weight_files_downloaded"] = heavy
    json.dump(R, open(p, "w"), indent=1, default=str)
    return r["repo"], R["verdict"], R["job_secs"]


os.makedirs(os.path.join(ROOT, ".tmp_hf"), exist_ok=True)
print(f"shard {a.shard}: {len(jobs)} jobs, {a.workers} workers, free disk {free_gb():.1f} GB", flush=True)
done = 0
with cf.ThreadPoolExecutor(a.workers) as ex:
    for repo, v, s in ex.map(run, jobs):
        done += 1
        print(f"[{done}/{len(jobs)}] {v:18s} {s:7.1f}s {repo}", flush=True)
if DEFERRED:                                   # heavy jobs, one at a time, with more headroom
    print(f"SOLO PHASE: {len(DEFERRED)} jobs deferred for memory", flush=True)
    for i, job in enumerate(list(DEFERRED)):
        DEFERRED.remove(job)
        repo, v, s = run(job, solo=True)
        print(f"[solo {i + 1}] {v:18s} {s:7.1f}s {repo}", flush=True)
print(f"SHARD DONE {a.shard} | free disk {free_gb():.1f} GB", flush=True)
