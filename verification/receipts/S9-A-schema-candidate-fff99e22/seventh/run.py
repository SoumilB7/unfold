from pathlib import Path
import dataclasses, importlib.util, json, os, sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
tree=root/'.claude/worktrees/verify-s9-a-seventh'
out=Path('/private/tmp/unfold-s9a-seventh/qualification');out.mkdir(exist_ok=False)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',UNFOLD_EVIDENCE_CACHE_DIR=str(out/'cache'))
spec=importlib.util.spec_from_file_location('s9_first_verify',tree/'scripts/verify_commit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module._stage_external_artifacts(root,tree)
tests=('tests/test_s9_norm_claims.py','tests/test_weight_tying.py','tests/test_decoder_norm.py','tests/test_embedding_bookend.py','tests/test_program_index.py','tests/test_ffn_width.py','tests/test_expert_storage.py','tests/test_expert_width.py','tests/test_diffusion_reader_claims.py','tests/test_diffusion_root.py','tests/test_diffusion_stack.py','tests/test_diffusion_block.py','tests/test_diffusion_stream.py','tests/test_diffusion_conditioning.py','tests/test_diffusion_bookends.py','tests/test_diffusion_schema.py','tests/test_diffusion_config_binding.py','tests/test_reader_claims.py','tests/test_s9_presentation_chips.py','tests/test_s9_fact_unknowns.py','tests/test_s9_checkpoint_metadata.py','tests/test_evidence_facts.py','tests/test_s8_card_projection_receipts.py','tests/test_attention_softcap.py','tests/test_attention_mechanism.py','tests/test_config_access.py')
result=module._run_lane(module.Lane('schema-proof-metadata',(sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests)),tree,out)
record={**dataclasses.asdict(result),'log_path':str(result.log_path),'passed':result.passed}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if result.passed else 1)
