"""Read-only component projection review from saved actual model facts.

No model construction, source-reader campaign, pytest or production edits.
"""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
expected={
 'model_unfolder/adapters/diffusor/parser.py':'2da1f1dc41eaec94cf2defa7fc6a35df8dd7a645eaa77ea08bfdacd2dfcb1758',
 'model_unfolder/adapters/diffusor/unet_projection.py':'39027ed5586949ad407a5f57b645d694d5d64c3f9a4c670ed02b97f23516478d',
 'model_unfolder/renderers/html/views_diffusion.py':'5be3e118f2a1b0d058299de9c4ecd910d895406c3720363b0bdfe6a2ee8042b1',
 'model_unfolder/expanded/loop.py':'accf5e4d2756c70f35ec156c6bdbf749335c007a5bdcd3f298f82c0baa5c8075'}
contents={name:(ROOT/name).read_bytes() for name in expected}
assert {name:hashlib.sha256(raw).hexdigest() for name,raw in contents.items()}==expected
from model_unfolder.adapters.diffusor.unet_projection import project_unet
from model_unfolder.expanded.loop import build_sampling_loop
from model_unfolder.renderers.html.views_diffusion import _build_loop_view,_stub_info
source=ROOT/'verification/receipts/S8-generalization-293c2d9/sdxl-refiner/ordinary'
raw=json.loads(gzip.decompress((source/'facts.json.gz').read_bytes()))
facts={key:SimpleNamespace(value=row['value']) for key,row in raw.items()}
out=ROOT/'verification/receipts/S8-component-scope-independent'
rows={}
for label,handoffs in {
 'denoiser_only':{},
 'scheduler_only':{'component_presence':{'scheduler':True,'vae':False,'text_encoders':False}},
 'vae_only':{'component_presence':{'scheduler':False,'vae':True,'text_encoders':False},'vae':{'latent_channels':4}},
 'encoder_only':{'component_presence':{'scheduler':False,'vae':False,'text_encoders':True},'text_encoders':['Supplied encoder'],'text_encoder_specs':[{'name':'Supplied encoder'}]},
 'full_pipeline':{'component_presence':{'scheduler':True,'vae':True,'text_encoders':True},'text_encoders':['Supplied encoder'],'text_encoder_specs':[{'name':'Supplied encoder'}],'scheduler':'Supplied scheduler','vae':{'latent_channels':4}},
}.items():
 ir=project_unet(facts=facts,handoffs=handoffs,name='Saved refiner projection',architecture='UNet2DConditionModel').to_dict()
 render=ir['extras']['render'];svg=_build_loop_view(ir,_stub_info(),label)
 (out/(label+'.svg')).write_text(svg)
 rows[label]={'scope':render.get('component_scope'),'block_ids':[block['id'] for block in render['loop_blocks']],
              'declared_input_labels':[block['label'] for block in render['loop_blocks'] if block['id'] in render.get('component_input_ids',[])],
              'sampling_loop_present':build_sampling_loop(ir['extras']) is not None}
 if label=='denoiser_only':
  assert not any(key in rows[label]['block_ids'] for key in ('scheduler','vae_decode','encoder_0','text_encoder','noise','image'))
  assert not rows[label]['sampling_loop_present']
 if label=='scheduler_only':assert 'scheduler' in rows[label]['block_ids']
 if label=='vae_only':assert 'vae_decode' in rows[label]['block_ids']
 if label=='encoder_only':assert 'encoder_0' in rows[label]['block_ids']
 if label=='full_pipeline':assert rows[label]['sampling_loop_present']
# Exact source declaration supplies the comparison; names are not inferred.
installed=Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/diffusers/models/unets/unet_2d_condition.py')
cls=next(node for node in ast.parse(installed.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='UNet2DConditionModel')
forward=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
params=[arg.arg for arg in (*forward.args.posonlyargs,*forward.args.args,*forward.args.kwonlyargs) if arg.arg!='self']
assert set(rows['denoiser_only']['declared_input_labels'])<=set(params)
for name in expected:assert (ROOT/name).read_bytes()==contents[name]
for name,raw in contents.items():(out/(name.replace('/','__')+'.txt')).write_bytes(raw)
print(json.dumps({'reviewed_sha256':expected,'unchanged_during_probe':True,'actual_saved_fact_source':str(source),'root_declared_formals':params,'sample_input_card_present':'sample' in rows['denoiser_only']['declared_input_labels'],'results':rows},indent=2,sort_keys=True))
