from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentyeighth'
out=Path('/private/tmp/unfold-s9a-twentyeighth/full-bracket');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
command=(sys.executable,str(tree/'scripts/pytest_file_bracket.py'),'--workers','4','--log-dir',str(out/'batches'),'--ignore','tests/test_preservation.py','--ignore','tests/test_identity_guard.py','--ignore','tests/test_u2_r4_structural_multiset.py','--ignore','tests/test_u2_r8_blocking_nets.py','tests')
result=module._run_lane(module.Lane('full-remainder',command),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
