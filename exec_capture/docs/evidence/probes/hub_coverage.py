import collections, warnings, json; warnings.filterwarnings("ignore")
import transformers, diffusers
from huggingface_hub import HfApi
api = HfApi()
def scan(tag, limit):
    rows = []
    for m in api.list_models(pipeline_tag=tag, sort="downloads", limit=limit, expand=["config", "safetensors", "library_name", "gated"]):
        cfg = m.config or {}
        archs = cfg.get("architectures") or []
        remote = bool(cfg.get("auto_map"))
        in_lib = bool(archs) and all(hasattr(transformers, a) for a in archs)
        if tag == "text-to-image":
            cls = (cfg.get("diffusers") or {}).get("_class_name") if isinstance(cfg.get("diffusers"), dict) else None
            in_lib = bool(cls) and hasattr(diffusers, cls); archs = [cls] if cls else archs
        rows.append(dict(id=m.id, lib=m.library_name, archs=archs, remote=remote, in_lib=in_lib, st=m.safetensors is not None, gated=bool(m.gated)))
    return rows
for tag, n in [("text-generation", 300), ("image-text-to-text", 100), ("text-to-image", 100)]:
    R = scan(tag, n)
    buildable = [r for r in R if r["in_lib"] and not r["remote"]]
    remote = [r for r in R if r["remote"]]
    unknown = [r for r in R if not r["in_lib"] and not r["remote"]]
    both = [r for r in buildable if r["st"]]
    print(f"\n== top {len(R)} '{tag}' models by downloads")
    print(f"  code in installed library, no remote code : {len(buildable):3d} ({100*len(buildable)/len(R):.0f}%)")
    print(f"     ...and safetensors headers available  : {len(both):3d} ({100*len(both)/len(R):.0f}%)")
    print(f"  needs repository remote code             : {len(remote):3d} ({100*len(remote)/len(R):.0f}%)   e.g. {[r['id'] for r in remote[:4]]}")
    print(f"  no recognizable class / other library    : {len(unknown):3d} ({100*len(unknown)/len(R):.0f}%)   e.g. {[(r['id'], r['lib']) for r in unknown[:4]]}")
    print(f"  gated (needs HF token)                   : {sum(r['gated'] for r in R)}")
    print(f"  distinct architecture classes            : {len({tuple(r['archs']) for r in R})}")
