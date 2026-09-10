"""Check the regenerated census at a committed tree with unchanged gate helpers."""
from pathlib import Path
import dataclasses,hashlib,importlib.util,json,os,subprocess,sys
if not __debug__:raise RuntimeError('Assertions required')
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');out=Path('/private/tmp/unfold-s81-linux-census-check');out.mkdir(exist_ok=False)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();spec=importlib.util.spec_from_file_location('census_check_verifier',root/'scripts/verify_commit.py');v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
tree=v._add_worktree(commit,'census-check','s81-linux-'+commit[:8]);v._stage_external_artifacts(root,tree);os.environ['UNFOLD_EVIDENCE_CACHE_DIR']='/private/tmp/unfold-s81-linux-census-diagnostic/cache'
initial=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();lane=v.Lane('census-check',(sys.executable,'scripts/census.py','--check'));r=v._run_lane(lane,tree,out)
record={'status':'PASS_COMMITTED_CENSUS_CHECK' if r.passed else 'FAIL','commit':commit,'lane':{**dataclasses.asdict(r),'log_path':str(r.log_path),'passed':r.passed},'worktree':str(tree),'tool_sha256':initial};assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==initial
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');assert r.passed;print(json.dumps(record))
