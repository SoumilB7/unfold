from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentieth'
out=Path('/private/tmp/unfold-s9a-sparse-twentieth-llama-lane');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_reader_claims.py','tests/test_expert_storage.py','tests/test_attention_mechanism.py','tests/test_s9_input_position_claims.py','tests/test_position_fixed.py','tests/test_consumer_firewall.py','tests/test_structural_writes.py')
result=module._run_lane(module.Lane('actual-sparse-config-html',(sys.executable,'/private/tmp/unfold-s9a-sparse-witness-preparation/run_sparse.py','--checkout',str(tree),'--plan','/private/tmp/unfold-s9a-twentieth/sparse-plan.json','--witness','llama-7b','--output','/private/tmp/unfold-s9a-sparse-twentieth-llama','--publish-dir','/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S9-schema/sparse-twentieth-llama')),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
