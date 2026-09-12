"""Run the unchanged census producer once into scratch; committed tree remains unchanged."""
from pathlib import Path
import dataclasses,hashlib,importlib.util,json,os,subprocess,sys
if not __debug__:raise RuntimeError('Assertions required')
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');out=Path('/private/tmp/unfold-s81-linux-census-diagnostic');out.mkdir(exist_ok=False)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert commit=='7f5e560e8ad29d7e51b7033393e35d1eeaa5a060'
spec=importlib.util.spec_from_file_location('census_verifier',root/'scripts/verify_commit.py');v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
tree=v._add_worktree(commit,'census','s81-linux-'+commit[:8]);v._stage_external_artifacts(root,tree);os.environ['UNFOLD_EVIDENCE_CACHE_DIR']=str(out/'cache')
code="import sys; from pathlib import Path; from scripts import census; census.DOC=Path(sys.argv[1]); sys.argv=[sys.argv[0]]; raise SystemExit(census.main())"
lane=v.Lane('census-producer',(sys.executable,'-c',code,str(out/'COR5_NET1_MIGRATION_DEBT.md')))
tool=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();r=v._run_lane(lane,tree,out)
record={'status':'PASS_SCRATCH_PRODUCER' if r.passed else 'FAIL','commit':commit,'worktree':str(tree),'lane':{**dataclasses.asdict(r),'log_path':str(r.log_path),'passed':r.passed},'producer_sha256':hashlib.sha256((tree/'scripts/census.py').read_bytes()).hexdigest(),'tool_sha256':tool,'original_sha256':hashlib.sha256((tree/'docs/COR5_NET1_MIGRATION_DEBT.md').read_bytes()).hexdigest()}
if (out/'COR5_NET1_MIGRATION_DEBT.md').exists():record['candidate_sha256']=hashlib.sha256((out/'COR5_NET1_MIGRATION_DEBT.md').read_bytes()).hexdigest()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==tool
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');assert r.passed;print(json.dumps(record))
