from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentyseventh'
out=Path('/private/tmp/unfold-s9a-twentyseventh/qualification');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_s9_limited_document.py','tests/test_reader_claims.py','tests/test_s9_reader_placement.py','tests/test_s9_projection_claim_requirements.py','tests/test_structural_writes.py','tests/test_u2_r4_structural_multiset.py','tests/test_u2_r8_blocking_nets.py','tests/test_code_evidence.py::test_norm_placement_defaults_two_unknown_tiers','tests/test_diffusion.py::test_config_tokens_do_not_change_typed_grouping_or_period_detection','tests/test_diffusion.py::test_moe_dit_counts_do_not_manufacture_a_router','tests/test_identity_guard.py','tests/test_s81_latency.py::test_measured_baseline_rows_match_the_committed_sample_bytes')
result=module._run_lane(module.Lane('limited-document-writer-authority',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests)),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
