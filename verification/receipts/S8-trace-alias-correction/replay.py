from pathlib import Path
import importlib.util, json, tempfile
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("reporter",root/"scripts/report_s8_demonstration.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
actual=Path("/private/tmp/unfold-s8-final-4560449/sdxl/ordinary")
case=m.read_case(actual)
with tempfile.TemporaryDirectory(prefix="s8-trace-alias-") as raw:
    out=Path(raw)
    m.read_case=lambda path:case
    positive=m.claim_traces(out)
    assert len(positive)==3 and all(not row["chain_gaps"] for row in positive)
    denoiser=case["observation"]["page"]["cards"]["denoiser"]
    previous=denoiser["node_ids"]
    denoiser["node_ids"]=[]
    missing=m.claim_traces(out)
    assert all("stage overview" in row["chain_gaps"] for row in missing)
    denoiser["node_ids"]=previous
    result={"positive_canonical_stages":[row["overview_stage"]["id"]for row in positive],"positive_chain_gaps":[row["chain_gaps"]for row in positive],"missing_visible_stage_rejected":True}
    print(json.dumps(result,indent=2))
