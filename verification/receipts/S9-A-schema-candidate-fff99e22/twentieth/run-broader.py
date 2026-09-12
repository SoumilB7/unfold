from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-twentieth'
out=Path('/private/tmp/unfold-s9a-twentieth/broader-qualification');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_fact_registry.py','tests/test_s9_input_position_claims.py', 'tests/test_position_fixed.py', 'tests/test_consumer_firewall.py', 'tests/test_structural_writes.py', 'tests/test_s9_norm_claims.py', 'tests/test_weight_tying.py', 'tests/test_decoder_norm.py', 'tests/test_embedding_bookend.py', 'tests/test_program_index.py', 'tests/test_ffn_width.py', 'tests/test_expert_storage.py', 'tests/test_expert_width.py', 'tests/test_diffusion_reader_claims.py', 'tests/test_diffusion_root.py', 'tests/test_diffusion_stack.py', 'tests/test_diffusion_block.py', 'tests/test_diffusion_stream.py', 'tests/test_diffusion_conditioning.py', 'tests/test_diffusion_bookends.py', 'tests/test_diffusion_schema.py', 'tests/test_diffusion_config_binding.py', 'tests/test_reader_claims.py', 'tests/test_s9_presentation_chips.py', 'tests/test_s9_fact_unknowns.py', 'tests/test_s9_checkpoint_metadata.py', 'tests/test_evidence_facts.py', 'tests/test_s8_card_projection_receipts.py', 'tests/test_attention_softcap.py', 'tests/test_attention_mechanism.py', 'tests/test_config_access.py', 'tests/test_s9_class_default_value.py','tests/test_s9_operand_channels.py','tests/test_s9_config_annotation_mro.py', 'tests/test_s9_reader_declarations.py', 'tests/test_framework_config.py', 'tests/test_position_absolute.py', 'tests/test_position_relative_bias.py', 'tests/test_attention_geometry.py', 'tests/test_cross_attention_replacement.py', 'tests/test_mtp.py', 'tests/test_projector_evidence.py', 'tests/test_projector_width.py', 'tests/test_projector_lineage.py', 'tests/test_projector_chain.py', 'tests/test_ffn_mechanism.py', 'tests/test_router.py')
result=module._run_lane(module.Lane('schema-proof-broader',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests)),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
