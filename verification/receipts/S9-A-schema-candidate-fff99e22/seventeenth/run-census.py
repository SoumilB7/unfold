from pathlib import Path
import dataclasses,importlib.util,json,os,sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');tree=root/'.claude/worktrees/verify-s9-a-seventeenth';out=Path('/private/tmp/unfold-s9a-seventeenth/census-lane');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_census_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
result=module._run_lane(module.Lane('fact-census',(sys.executable,'/private/tmp/unfold-s9a-seventeenth/census.py')),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed};(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
