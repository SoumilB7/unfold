from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentyninth'
out=Path('/private/tmp/unfold-s9a-twentyninth/qualification-r2');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_s9_child_class_default.py','tests/test_s9_slot_config_class_address.py','tests/test_s9_scalar_decisions.py','tests/test_s9_finite_unknown_coverage.py','tests/test_s9_class_default_value.py','tests/test_s9_limited_document.py','tests/test_config_access.py','tests/test_u2_r2_raw_consume_debt.py','tests/test_isolation.py','tests/test_attention_storage.py','tests/test_fact_ledger.py','tests/test_s6_physics.py::test_production_can_request_isolated_inventory_but_cannot_construct_in_parent','tests/test_position_initialization.py','tests/test_expanded_json.py::test_expanded_json_completes_qwen2_audio_sparse_text_config','tests/test_smoke.py::test_qwen2_audio_sparse_text_config_keeps_only_source_bound_structure','tests/test_smoke.py::test_qwen2_audio_code_evidence_does_not_mark_config_partial','tests/test_smoke.py::test_modality_host_looks_through_declared_wrappers','tests/test_h7_diffusion.py::test_metamorphic_harness_holds_on_a_diffusion_reference')
# Pin exact test selection before execution; the banner RETURN remains open.
(out/'scope.json').write_text(json.dumps({'tests':tests,'known_source_return':'stats_banner event must not place occurrences; excluded from acceptance, corrected snapshot required'},indent=2)+'\n')
result=module._run_lane(module.Lane('correction-focus',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests)),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
