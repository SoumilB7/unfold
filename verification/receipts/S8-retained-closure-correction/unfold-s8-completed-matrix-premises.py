import dataclasses,gzip,hashlib,importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('matrix_premises',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
original=Path('/private/tmp/unfold-s8-final-verification-96d0c1e/verification/s7')
rows=[]
for target in g._targets()[:26]:
 print(target['slug'],flush=True)
 p=original/'models'/f"{target['slug']}.json.gz";artifact=json.loads(gzip.decompress(p.read_bytes()))
 label,config=g._read_payload(root/target['input']);context=ParseContext.build(config)
 runtime=artifact['inventory_provenance']['resolved_class']['qualname']
 before=dict(context.source_bundle.component_architectures);after={**before,'root':runtime}
 same_bundle=before==after
 index=build_program_index(context.source_bundle);address=resolve_component_root(index,context.source_bundle,'root')
 result=read_diffusion_root_topology(index,address)
 assert not result.has_value or result.value.kind!='u_shaped',target['slug']
 rows.append({'target':target,'model_artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_context_architectures':before,'runtime_addressed_architectures':after,'same_context_bundle':same_bundle,'topology_kind':result.value.kind if result.has_value else None,'unet_cutover_selected':False,'initial_index_fingerprint':g._portable_source_index_fingerprint(index)})
(out/'completed26-premises.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
print('26 original bundle identities and non-UNet selection premises assessed',flush=True)
