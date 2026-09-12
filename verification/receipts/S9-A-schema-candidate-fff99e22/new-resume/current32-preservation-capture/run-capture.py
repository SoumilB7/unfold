"""Root-only serial preservation observation; never calls a baseline writer."""
from pathlib import Path
import argparse,dataclasses,importlib.util,json,os,sys,traceback
from capture_contract import validate,HERE
p=argparse.ArgumentParser(); p.add_argument('--lane',choices=['current32','baseline'],required=True)
p.add_argument('--source-manifest-sha256',required=True); a=p.parse_args()
before=validate(a.lane,a.source_manifest_sha256)
plan=json.loads((HERE/'plan.json').read_bytes()); lane=plan['lanes'][a.lane]
tree=Path(lane['tree']); out=Path(lane['output']); out.mkdir(parents=True,exist_ok=False)
(out/'pins-before.json').write_text(json.dumps(before,indent=2)+'\n')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'),PYTEST_ADDOPTS='')
spec=importlib.util.spec_from_file_location('preservation32_verify',tree/'scripts/verify_commit.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
result=None
try:
    result=m._run_lane(m.Lane('preservation-'+a.lane,(sys.executable,str(HERE/'run-tests.py'),
         '--lane',a.lane,'--source-manifest-sha256',a.source_manifest_sha256)),tree,out)
finally:
    after=validate(a.lane,a.source_manifest_sha256)
    (out/'pins-after.json').write_text(json.dumps(after,indent=2)+'\n')
report_path=out/'capture/report.json'
report=json.loads(report_path.read_bytes()) if report_path.is_file() else {}
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'pins_equal':before==after,
    'packet_complete':report.get('packet_complete',False),
    'preservation_passed':result.passed and before==after and report.get('raw_preservation_passed',False),
    'authority':'Original test failures preserved. A complete packet does not mean preservation passed or approval.'}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
raise SystemExit(0 if record['preservation_passed'] else 1)
