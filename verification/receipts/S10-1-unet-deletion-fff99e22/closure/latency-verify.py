"""Serial deletion-only qualification; no baseline writers or model-source changes."""
import dataclasses,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
assert not sys.flags.optimize
ROOT=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
TREE=ROOT/'.claude/worktrees/verify-s10-1-deletion'
OUT=Path('/private/tmp/unfold-s10-1/qualification-latency-ratchet')
OUT.mkdir(exist_ok=False)
(OUT/'verify.py').write_bytes(Path(__file__).read_bytes())
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(OUT/'cache'))
spec=importlib.util.spec_from_file_location('verify_s10_isolated',TREE/'scripts/verify_commit.py');v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
v._stage_external_artifacts(ROOT,TREE)
files=json.loads(Path('/private/tmp/unfold-s10-1/owned-files.json').read_text())
changed=[p for p in files if p.endswith('.py') and (TREE/p).is_file()]
lanes=[v.Lane('latency-contract',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_s81_latency.py'))]
for lane in lanes:
 for arg in lane.command:
  if arg.startswith('tests/'):
   assert (TREE/arg).is_file(),arg
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),TREE/'scripts/verify_commit.py',TREE/'test_support/tree_state.py')}
rows=[];error=None
try:
 for lane in lanes:
  r=v._run_lane(lane,TREE,OUT);rows.append({**dataclasses.asdict(r),'log_path':str(r.log_path),'passed':r.passed})
  (OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(lane.name,r.returncode,r.passed,flush=True)
  if not r.passed:break
except BaseException as exc:
 error={'type':type(exc).__name__,'message':str(exc)}
finally:
 end={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in pins}
 result={'status':'PASS' if error is None and pins==end and all(x['passed'] for x in rows) and len(rows)==len(lanes) else 'FAIL','tree':str(TREE),'lanes':rows,'tools_before':pins,'tools_after':end,'error':error}
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['status']=='PASS'
