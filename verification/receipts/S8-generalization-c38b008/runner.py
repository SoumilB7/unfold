"""Finite serial actual component regressions; no blessing or source edits."""
from pathlib import Path
import sys,json,subprocess,time,hashlib,shutil,re,xml.etree.ElementTree as ET,gzip
checkout=Path('/private/tmp/unfold-s8-compiler-final');out=Path('/private/tmp/unfold-s8-render-phase-c38-v2');old=Path('/private/tmp/unfold-s8-generalization-293c2d9');sys.path[:0]=[str(out/'scripts'),str(checkout)]
import demonstrate_s8_unet as demo
demo.ROOT=checkout
_tree_sources=demo._tree_sources
from report_s8_demonstration import report,blocks
from model_unfolder.preview import svg_views
from model_unfolder.ir import ModelIR
from model_unfolder.expanded import build_expanded
pin=json.loads((out/'phase-pin.json').read_text());assert _tree_sources()==pin['production']
review=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/final-c38b008');root=out/'generalization';root.mkdir(exist_ok=False)
expected={'sdxl-refiner':([4,1,4],[384,4,3,3],2259526660),'sdxl-inpainting':([3,1,3],[320,9,3,3],2567478084),'ddpm-cifar10':([4,1,4],[128,3,3,3],35746307)}
provenance=json.loads((old/'config-downloads.json').read_text());(root/'config-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');(root/'runner.py').write_bytes(Path(__file__).read_bytes());results=[]
for name,(counts,shape,total) in expected.items():
 target=root/name;target.mkdir();raw=(old/name/'input.json').read_bytes();pinned=next(x for x in provenance if x['witness']==name);assert hashlib.sha256(raw).hexdigest()==pinned['input_sha256'];(target/'input.json').write_bytes(raw)
 cmd=['python3',str(out/'phase-entry.py'),'--checkout',str(checkout),'--scripts',str(out/'scripts'),'--input',str(target/'input.json'),'--output',str(target),'--condition','ordinary','--expect-source-sha256',pin['production']['sha256']]
 print(json.dumps({'event':'started','witness':name,'condition':'ordinary'}),flush=True);start=time.monotonic()
 with (target/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=checkout,stdout=log,stderr=subprocess.STDOUT)
 row={'witness':name,'command':cmd,'returncode':r.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'commit':pin['production_commit'],'scope':'Published denoiser-only config; installed-version reconstruction, no weights or original historical library claim.'};case=target/'ordinary';page=case/'page.html'
 if page.exists():
  html=page.read_bytes();row.update(html_bytes=len(html),html_sha256=hashlib.sha256(html).hexdigest());shutil.copyfile(page,review/(name+'-ordinary.html'))
 (target/'run.json').write_text(json.dumps(row,indent=2)+'\n');assert _tree_sources()==pin['production']
 if r.returncode:print((target/'run.log').read_text()[-5000:],flush=True);raise SystemExit(r.returncode)
 checks=report(target)['conditions']['ordinary']['checks'];failures=[x for x in checks if x['status']=='FAIL'];ir=json.loads((case/'ir.json').read_text());inv=json.loads((case/'inventory.json').read_text());u=ir['extras']['unet'];render=ir['extras']['render'];cards=list(blocks(ir));modules={x['path']:x for x in inv['modules']};all_ids=set();architecture_ids=set()
 for label,svg in svg_views(page.read_text()):
  ids={node.get('data-id') for node in ET.fromstring(svg).iter() if 'uf-node' in node.get('class','').split() and node.get('data-id')};all_ids.update(ids)
  if label=='architecture':architecture_ids.update(ids)
 citations={x['source_instance_path'] for x in cards if 'source_instance_path' in x};placed={x['source_instance_path'] for x in cards if 'source_instance_path' in x and x.get('id') in all_ids};paths=set(modules)
 stages=[sum(bool(re.fullmatch(r'down_blocks\.\d+',x)) for x in paths),int('mid_block'in paths),sum(bool(re.fullmatch(r'up_blocks\.\d+',x)) for x in paths)];conv=next(x['shape'] for x in modules['conv_in']['parameters'] if x['name']=='weight');params=u['parameter_shapes']['total'];forbidden=[x for x in all_ids if re.fullmatch(r'encoder_\d+',x) or x in {'scheduler','vae_decode','prompt','noise','latent'}];inputs=render.get('component_input_ids',());sample_ids={x['id'] for x in render['loop_blocks'] if x['id'] in inputs and x.get('label')=='sample'}
 expanded=build_expanded(ModelIR(**ir));(case/'expanded.json.gz').write_bytes(gzip.compress((json.dumps(expanded,indent=2)+'\n').encode(),mtime=0))
 assertions={'stage_counts':stages==counts,'input_conv_shape':conv==shape,'parameter_total':params==total,'no_orphan_occurrence':not(citations-paths),'component_scope':render.get('component_scope')=='denoiser','no_invented_components_in_any_svg':not forbidden,'root_sample_input_in_actual_overview':bool(sample_ids & architecture_ids),'expanded_has_no_sampling_loop':'sampling_loop'not in expanded,'unconditional_has_no_context':name!='ddpm-cifar10' or not u.get('context_routes')}
 audit={'assertions':assertions,'stage_counts':stages,'input_conv':conv,'parameter_total':params,'constructed_paths':sorted(paths),'canonical_paths':sorted(citations),'actual_svg_paths':sorted(placed),'unplaced':sorted(paths-placed),'unplaced_classes':{p:modules[p]['class_ref'] for p in sorted(paths-placed)},'orphan_citations':sorted(citations-paths),'forbidden_svg_ids':forbidden,'architecture_svg_ids':sorted(architecture_ids),'sample_input_ids':sorted(sample_ids),'context_routes':u.get('context_routes'),'unbound_bookend_paths':u.get('unbound_bookend_paths'),'scope':'Placement is not execution closure; no grouped occurrence inferred from ancestor placement.'};(target/'scope-placement-shape-audit.json').write_text(json.dumps(audit,indent=2)+'\n');row.update(report_failures=failures,regression_failures=[key for key,value in assertions.items() if not value]);results.append(row);(root/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'event':'completed',**row}),flush=True)
 if failures or row['regression_failures']:raise SystemExit(2)
