from pathlib import Path
import json,re,hashlib,collections
R=Path('/private/tmp/unfold-s8-compiler-final');P=Path('/private/tmp/unfold-s8-render-phase-c38-v2/sdxl');O=Path(__file__).resolve().parent
old=json.loads((P/'legacy/ir.json').read_text());new=json.loads((P/'ordinary/ir.json').read_text());f=json.loads((P/'ordinary/facts.json').read_text());q=json.loads((P/'ordinary/qualified-facts.json').read_text());cards=json.loads((P/'ordinary/observation.json').read_text())['page']['cards'];page=(P/'ordinary/page.html').read_text()
def walk(x):
 if isinstance(x,dict):
  if 'id'in x and any(k in x for k in ('kind','title','children','view')):yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
oldroot=next(b for b in old['extras']['render']['loop_blocks']if b['id']=='denoiser');obs={b['id']:b for b in walk(oldroot)};nbs={b['id']:b for b in walk(new['extras']['render'])};assert len(obs)==53
population=f['root.denoiser.constructed_modules']['value'];shapes=f['root.denoiser.constructed_parameter_shapes']['value'];ffns=f['root.denoiser.ffn_mechanisms']['value'];cells=f['root.denoiser.cell_connections']['value'];spatial=f['root.denoiser.spatial_mechanisms']['value'];primitives=f['root.denoiser.runtime_primitives']['value'];context=f['root.denoiser.context_connections']['value']
assert len(ffns)==70 and all(v['activation']=='gelu'and v['gated']and v['projection_mode']=='fused_gate_up'for v in ffns.values())
resnets=sorted(cells);assert len(resnets)==17
stages={**{f'unet_down_{i}':f'down_blocks.{i}'for i in range(3)},'unet_mid':'mid_block',**{f'unet_up_{i}':f'up_blocks.{i}'for i in range(3)}}
def cid(path):return 'instance_'+path.replace('.','__')
def targets(paths,keys):
 out=[]
 for path in paths:
  c=cid(path);assert path in population and c in cards and c in nbs
  relevant=[k for k in keys if k in nbs[c].get('source_fact_keys',[])];assert relevant,(c,keys)
  out.append({'occurrence':path,'card_id':c,'fact_keys':relevant,'actual_svg_count':cards[c]['svg_count'],'actual_facts':cards[c]['facts']})
 return out
def claim(text,status,author,paths=(),keys=(),limit=None):
 return {'assertion':text,'disposition':status,'old_author':author,'new_targets':targets(paths,keys)if paths else [],'support_limit':limit}
K=lambda s:'root.denoiser.'+s
rows=[]
for oid,b in obs.items():
 cs=[];stage=next((v for k,v in stages.items()if oid==k or oid.startswith(k+'__')),None)
 sr=[p for p in resnets if p.startswith(stage+'.resnets.')]if stage else []
 tf=sorted(p for p,v in population.items()if stage and p.startswith(stage+'.attentions.')and v['class_name']=='Transformer2DModel')
 sf=[p for p in ffns if stage and p.startswith(stage+'.')]
 if oid in stages:
  oldst=next(s for s in old['extras']['unet'].get('down',[])+[old['extras']['unet'].get('mid',{})]+old['extras']['unet'].get('up',[])if s.get('id')==oid)
  assert len(sr)==oldst['resnets'];assert shapes['parameters'][sr[0]+'.conv2.weight']['shape'][0]==oldst['channels']
  cs.append(claim(f"Stage existence; {oldst['resnets']} ResNet occurrences; old width {oldst['channels']} now explicit conv2 output-weight extent",'carried_proven','unet.py:55–113(config arithmetic); parser.py:72–94(mid source presence)',[stage]+sr,[K('constructed_modules'),K('constructed_parameter_shapes')], 'Containment/count and raw shape dimensions retained; no count-derived execution claim.'))
  if oldst['attn']:
   assert tf
   depths={p:len(population[p+'.transformer_blocks']['children'])for p in tf};assert all(d==oldst['transformers']for d in depths.values())
   cs.append(claim(f"Transformer2D construction and {oldst['transformers']} nested blocks per attention wrapper",'carried_proven','patterns.py:532–582 construction traversal; unet.py:55–113 depth from config',tf+[p+'.transformer_blocks'for p in tf],[K('constructed_modules')],str(depths)))
  cs.append(claim('Stage-internal paired ResNet→attention repetition or mid sandwich execution; direction-implied unconditional order','demoted_unsupported_old_author','unet.py:669–734; block_views/unet.py:588–660 installs stage flow from direction/attn/count',sr,[K('cell_connections')],'Construction reader never proved this stage-wide tensor-flow sequence; new per-cell routes and conditional root invocation only.'))
  if oid.startswith('unet_up_'):
   cs.append(claim('Matching down-stage skip directly enters each up stage','carried_bounded_with_named_limitation','unet.py:830–846 stage description; old U renderer assumes matching stage rail',[stage],[K('stage_join_connections'),K('constructed_stage_relations')],'Actual concat→child connection retained; exact bank lineage/optional FreeU and unconditional stage pairing remain explicit unknown.'))
  if oldst.get('sample'):
   path=stage+('.downsamplers.0'if oid.startswith('unet_down')else'.upsamplers.0');v=spatial[path]
   cs.append(claim('Spatial 2× resampling','carried_proven'if v['effect']=='reduce'else'demoted_unsupported_old_author','unet.py:80–113 sample=position; :735–743 hardcodes stride2 or nearest-neighbor2×',[path],[K('spatial_mechanisms')], 'Stride2 proved.'if v['effect']=='reduce'else'Interpolate applied-function retained; operand=None, resize direction is investigation_missing. Old 2×/nearest claim had no source reader premise.'))
 elif '__resnet' in oid:
  selected=sr[:1]if oid.endswith('_pre')else sr[1:2]if oid.endswith('_post')else sr
  cs.append(claim('Constructed ResNet cells and component operations','carried_proven','unet.py:669–734 aggregate constructor',[p for p in selected],[K('constructed_modules'),K('cell_connections')]))
  cs.append(claim('Complete norm→activation→conv→time-add→norm→activation→conv→residual bypass and divide','demoted_unsupported_old_author','unet.py:290–337 generic ResnetBlock2D prose; block_views/unet.py:669–721 literal residual skeleton',selected,[K('cell_connections'),K('cell_arithmetic')],'Five positive local edges/two fragments remain per cell; conditional time-add and return arithmetic have unresolved lineage/operand/scale boundaries. Old renderer had no selected forward proof for the full route.'))
 elif oid.endswith('__transformer'):
  assert tf;cs.append(claim('Constructed Transformer2D and actual nested transformer depth','carried_proven','patterns.py:532–582 proves reachable attention+FFN construction; unet.py:636–660 adds aggregate card',tf+[p+'.transformer_blocks'for p in tf],[K('constructed_modules')]))
  cs.append(claim('Self-attention→cross-attention→FFN execution order and head-count/head-dimension split','demoted_unsupported_old_author','unet.py:115–146 heads_hd convention; :636–660 sequence prose; block_views/unet.py:728 onward fixed sequence',tf,[K('constructed_modules')],'Construction traversal proves presence, not connections, mask, query role, head split, or execution order. FFN computation itself is retained separately.'))
 elif oid.endswith('__ff')or oid=='block':
  ps=sf if stage else sorted(ffns)
  cs.append(claim('Anchored GEGLU activation/gating; old projection storage unresolved','carried_proven','parser.py:55–71; patterns.py:425–492 anchored activation; unet.py:382–393/449–460 FFN card',ps,[K('ffn_mechanisms'),K('constructed_parameter_shapes')],'All70 exact FFNs have fused gate/up projection, gelu and multiplication plus output projection and shapes. Old after-attention order text was template-only and is not imported.'))
 elif oid.endswith('__selfattn')or oid.endswith('__crossattn')or oid=='opaque_mixer':
  member='attn1'if oid.endswith('__selfattn')else'attn2'if oid.endswith('__crossattn')else None
  ps=sorted(p for p,v in population.items()if (not stage or p.startswith(stage+'.'))and (p.endswith('.'+member)if member else p.endswith(('.attn1','.attn2'))))
  cs.append(claim('Attention occurrence/projection storage and unresolved attention internals','carried_proven','patterns.py:532–582 construction; unet.py:374–380 kind=None deliberately limits mechanism',ps,[K('constructed_modules'),K('constructed_parameter_shapes')]))
  if member:
   cs.append(claim('Full mask, spatial latent query role, self/cross distinction, conventional text K/V and legacy heads','demoted_unsupported_old_author','unet.py:115–146 head convention; :374–380 mask/full+cross flag; :403–450 fixed query/text prose',ps,[K('context_connections'),K('constructed_modules')],'External-context proof exists only at '+str(sum(p in context for p in ps))+' selected occurrences; distinct query lineage and source-to-K/V semantics remain unresolved. No old forward/processor reader established these stronger assertions.'))
 elif oid.startswith('unet_op_'):
  suffix=oid.removeprefix('unet_op_');mapping={'norm1':['norm1','nonlinearity'],'norm2':['norm2','nonlinearity'],'conv1':['conv1'],'conv2':['conv2'],'temb':['time_emb_proj'],'residual':['conv_shortcut']};ps=[p+'.'+m for p in resnets for m in mapping[suffix]if p+'.'+m in population]
  if ps:cs.append(claim('Constructed '+suffix+' primitives and parameter extents','carried_proven','unet.py:290–337 literal shared operation card',ps,[K('runtime_primitives'),K('constructed_parameter_shapes'),K('constructed_modules')]))
  cs.append(claim('Operation application/connection within complete residual path','carried_bounded_with_named_limitation','unet.py:290–337 and block_views/unet.py:669–721 literal template',resnets,[K('cell_connections'),K('cell_arithmetic')],'GroupNorm/SiLU/conv primitives and five source edges are proved; time-add and residual-return operands retain visible partial lineage. Padding/stride1 generic prose not separately promoted from kernel shape.'))
 elif oid.endswith('__downsample')or oid.endswith('__upsample'):
  down=oid.endswith('__downsample');path=stage+('.downsamplers.0'if down else'.upsamplers.0');cs.append(claim('Stride2 convolution'if down else'Nearest-neighbor2× upsample then convolution','carried_proven'if down else'demoted_unsupported_old_author','unet.py:735–743 hardcoded sample description',[path,path+'.conv'],[K('spatial_mechanisms'),K('runtime_primitives'),K('constructed_parameter_shapes')],'Downsample stride2 applied-function retained.'if down else'Interpolate operation and conv existence/shape retained; 2×/nearest/order was template prose and remains limited.'))
 elif oid in ('unet_conv_in','unet_conv_out'):
  ps=['conv_in']if oid.endswith('_in')else['conv_norm_out','conv_act','conv_out'];cs.append(claim('Input4→320 or output320→4 convolution; output norm/SiLU components','carried_proven','unet.py:774–783/942–952 uses config channels and fixed bookend prose',ps,[K('constructed_modules'),K('runtime_primitives'),K('constructed_parameter_shapes'),K('primary_state_ports')],'Exact 3×3 parameter shapes retained; root call ports bind constructed targets under explicit conditions. Predicted-noise semantic output is not proved by shape.'))
 elif oid in ('unet_text_cond','text_concat_op','cross_attention_states'):
  ps=sorted(context);cs.append(claim('768+1280=2048 encoder concatenation, same text K/V for every cross stage','demoted_unsupported_old_author','unet.py:849–940 component-width arithmetic and fixed concat/KV prose; :403–438 cross-state prose',ps,[K('context_connections')],'Encoder existence/width retained outside UNet; no pipeline concat/producer→K/V proof supplied old claim. New60 formal→context-port claims are narrower; query/meaning remains visible unknown.'))
 elif oid=='denoiser':
  cs.append(claim('Down/mid/up constructed architecture; all-stage fixed2× rules, global4× downscale and noise prediction','carried_bounded_with_named_limitation','unet.py:186–242 repeats config-derived stage/downscale and fixed prediction prose',list(stages.values()),[K('constructed_modules'),K('constructed_stage_relations'),K('primary_state_ports')],'Exact3/1/3stage occurrences and source-port boundaries retained. Global unconditional composition, noise semantics and every-stage halving were not source-proven by old geometry.'))
 else:raise AssertionError(oid)
 assert cs;rows.append({'old_card_id':oid,'old_assertion_payload':{k:v for k,v in b.items()if k!='children'},'claims':cs,'verdict':'no_lost_proved_drawing_identified'})
assert {r['old_card_id']for r in rows}==set(obs)
# Every generated target is an actual card; all cited fact kinds must be qualified.
for row in rows:
 for c in row['claims']:
  for t in c['new_targets']:
   for k in t['fact_keys']:assert q[k]['claim_kind']==q[k]['proof']['claim_kind']
# Actual drawings, beyond card existence/citation.
starts=list(re.finditer(r'data-card-id="([^"]+)"',page));chunks={m.group(1):page[m.start():(starts[i+1].start()if i+1<len(starts)else len(page))]for i,m in enumerate(starts)}
overview=re.findall(r'<svg\b.*?</svg>',chunks['denoiser'],re.S)[0]
assert all('data-id="'+cid(path)+'"'in overview for path in stages.values())
for path in ffns:
 svg=re.findall(r'<svg\b.*?</svg>',chunks[cid(path)],re.S)[0]
 assert len(re.findall('marker-end=',svg))==7
for path in resnets:
 assert primitives[path+'.nonlinearity']['function']=='silu'
 assert all(primitives[path+'.'+norm]['function']=='groupnorm'for norm in ('norm1','norm2'))
 assert all(primitives[path+'.'+conv]['kind']=='conv2d'and shapes['parameters'][path+'.'+conv+'.weight']['shape'][-2:]==[3,3]for conv in ('conv1','conv2'))
assert primitives['conv_act']['function']=='silu'
files=['model_unfolder/adapters/diffusor/unet.py','model_unfolder/adapters/diffusor/parser.py','model_unfolder/evidence/patterns.py','model_unfolder/renderers/html/block_views/unet.py'];pins={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in files}
(O/'ledger.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');summary={'checkpoint':'c38b008af8153302d85bbaf2a8e3593841d87e1a','legacy_unique_cards':len(rows),'semantic_assertions':sum(len(r['claims'])for r in rows),'dispositions':dict(collections.Counter(c['disposition']for r in rows for c in r['claims'])),'qualified_ffn_count':len(ffns),'new_resnet_count':len(resnets),'source_pins':pins,'scope_verdict':'ACCEPT named preservation/limitation mapping; no actual lost old proved drawing found in the53-card legacy UNet subtree. Not approval to bless output.'};(O/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
