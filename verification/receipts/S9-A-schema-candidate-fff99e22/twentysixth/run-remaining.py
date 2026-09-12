from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentysixth'
out=Path('/private/tmp/unfold-s9a-twentysixth/remaining-qualification');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_code_evidence.py','tests/test_coverage.py','tests/test_diffusion.py','tests/test_everchanging.py','tests/test_legacy_reader_quarantine.py','tests/test_reader_exceptions.py','tests/test_s81_latency.py','tests/test_s8_unet_routing.py','tests/test_u4_f_cross_surface_closure.py','tests/test_identity_guard.py','tests/test_u2_r4_structural_multiset.py','tests/test_u2_r8_blocking_nets.py')
result=module._run_lane(module.Lane('remaining-changed-and-authority',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests)),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
