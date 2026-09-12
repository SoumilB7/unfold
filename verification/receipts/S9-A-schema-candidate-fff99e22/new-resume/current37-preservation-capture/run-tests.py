"""Child entry: validate before importing pytest or any production module."""
from pathlib import Path
import argparse,json,os,sys,traceback
from capture_contract import validate,HERE
p=argparse.ArgumentParser(); p.add_argument('--lane',choices=['current37','baseline'],required=True)
p.add_argument('--source-manifest-sha256',required=True); a=p.parse_args()
before=validate(a.lane,a.source_manifest_sha256)
plan=json.loads((HERE/'plan.json').read_bytes()); lane=plan['lanes'][a.lane]
tree=Path(lane['tree']); out=Path(lane['output'])
if Path.cwd().resolve()!=tree: raise ValueError('Child cwd differs from exact tree')
sys.path.insert(0,str(tree))
os.environ['UNFOLD_S9A32_OUTPUT_CAPTURE']=str(out/'capture')
os.environ['PYTEST_ADDOPTS']=''
(out/'child-pins-before.json').write_text(json.dumps(before,indent=2)+'\n')
code=2; report=None
try:
    import pytest
    import capture_preservation as observer
    code=int(pytest.main(['-q','-o','addopts=','tests/test_preservation.py'],plugins=[observer]))
    report=observer.report_capture(out/'capture')
except BaseException:
    (out/'child-failure.txt').write_text(traceback.format_exc())
    raise
finally:
    after=validate(a.lane,a.source_manifest_sha256)
    (out/'child-pins-after.json').write_text(json.dumps(after,indent=2)+'\n')
    if before!=after: raise ValueError('Child source/input/artifact/script bracket changed')
if not report or not report['packet_complete']: raise SystemExit(2)
raise SystemExit(code)
