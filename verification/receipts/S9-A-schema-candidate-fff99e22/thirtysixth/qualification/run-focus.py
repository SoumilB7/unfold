from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-thirtysixth'
out=Path('/private/tmp/unfold-s9a-thirtysixth/qualification');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
result=module._run_lane(module.Lane('qualification',(sys.executable,'-m','pytest','-q','tests/test_s9_shared_alias_defaults.py','tests/test_s9_layer_count_alias_defaults.py','tests/test_s9_repeated_schedule_visibility.py','tests/test_s9_layer_count_defaults.py','tests/test_attention_softcap.py','tests/test_submodel_parity.py','tests/test_reader_claims.py','tests/test_s9_presentation_chips.py','tests/test_projection_obligations.py','tests/test_s9_reader_placement.py','tests/test_s7_reconciliation.py','tests/test_structural_writes.py','tests/test_fact_registry.py','tests/test_position_linear_bias.py','tests/test_cross_attention_schedule.py','tests/test_s9_repeated_schedule_qualification.py','tests/test_s9_default_consumption.py','tests/test_s9_projection_claim_requirements.py')),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
