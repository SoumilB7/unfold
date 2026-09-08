"""Materialize exact reviewed fixture literals only; no model imports or evaluation."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.checkout);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
base='a83d704906850257fadd3904904635d218af7365'
old_support=subprocess.check_output(['git','show',base+':test_support/__init__.py'],cwd=root,text=True);old_tests=subprocess.check_output(['git','show',base+':tests/test_diffusion.py'],cwd=root,text=True);current_support=(root/'test_support/__init__.py').read_text();corpus_path=root/'tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json';corpus=json.loads(corpus_path.read_text())['config']
def literal(source,name,function=None):
    tree=ast.parse(source);body=tree.body if function is None else next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    node=next(n for n in body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return ast.literal_eval(node.value),{'line':node.lineno,'source':ast.get_source_segment(source,node)}
invalid,invalid_site=literal(old_support,'SDXL_UNET');corrected,corrected_site=literal(current_support,'SDXL_UNET');fields=('addition_time_embed_dim','projection_class_embeddings_input_dim','attention_head_dim');assert {k:v for k,v in corrected.items() if k not in fields}==invalid
assert {k:corrected[k] for k in fields}=={k:corpus[k] for k in fields}=={'addition_time_embed_dim':256,'projection_class_embeddings_input_dim':2816,'attention_head_dim':[5,10,20]}
svd,svd_site=literal(old_tests,'SVD_UNET');deepfloyd,deep_site=literal(old_tests,'DEEPFLOYD','test_unet_stage_part_kinds_resolve_for_all_block_variants');assert 'num_attention_heads' not in deepfloyd
bridge={k:v for k,v in invalid.items() if k not in ('_repo_id','text_encoder_2','addition_embed_type')};bridge['encoder_hid_dim']=4096;assert all(k not in bridge for k in fields)
function=next(n for n in ast.parse(old_tests).body if isinstance(n,ast.FunctionDef) and n.name=='test_encoder_hid_dim_draws_the_projection_in_the_text_cond_drill');node=next(n for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].slice,ast.Constant) and n.targets[0].slice.value=='_text_encoder_configs');bridge['_text_encoder_configs']=ast.literal_eval(node.value)
cases={'sdxl-invalid-original':invalid,'sdxl-corrected-shared':corrected,'svd-exact':svd,'encoder-bridge':bridge,'deepfloyd-second':deepfloyd}
for name,config in cases.items():
    destination=out/(name+'.json');assert not destination.exists();destination.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest={'original_commit':base,'original_source_hashes':{'test_support/__init__.py':sha(old_support.encode()),'tests/test_diffusion.py':sha(old_tests.encode())},'current_support_sha256':sha(current_support.encode()),'exact_corpus_config_file_sha256':sha(corpus_path.read_bytes()),'corrected_shared_added_values':{k:corrected[k] for k in fields},'original_shared_site':invalid_site,'corrected_shared_site':corrected_site,'svd_site':svd_site,'deepfloyd_second_site':deep_site,'bridge_encoder_configs_site':{'line':node.lineno,'source':ast.get_source_segment(old_tests,node)},'bridge_derivation':'Exact original test transform applied to original a83 shared fixture: omit _repo_id,text_encoder_2,addition_embed_type; encoder_hid_dim4096 and literal _text_encoder_configs. No corrected constructor fields added; omitted head dimension remains the original class-default input.','inputs':{name:{'file':name+'.json','sha256':sha((out/(name+'.json')).read_bytes())} for name in cases},'limits':'SVD num_frames25 is merely declared input here, not an executed frame-count or temporal-formula assertion. No model/fixture imports executed.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'inputs':manifest['inputs'],'corrected_shared_added_values':manifest['corrected_shared_added_values']}))
