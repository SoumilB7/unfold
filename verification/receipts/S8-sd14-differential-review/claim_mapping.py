"""Read-only SD14-specific preservation/demotion review; no output blessing."""
from pathlib import Path
import json,gzip,hashlib,re,collections,math
O=Path(__file__).resolve().parent;R=Path('/private/tmp/unfold-s8-render-phase-c38-v2');P=R/'sd-v1-4';S=Path('/private/tmp/unfold-s8-compiler-final')
load=lambda p:json.loads(p.read_text());cases={n:{k:load(P/n/(k+'.json')) for k in ('ir','facts','qualified-facts','observation')} for n in ('legacy','ordinary')};old,new=cases['legacy'],cases['ordinary'];facts=new['facts'];proofs=new['qualified-facts'];inv=load(P/'ordinary/inventory.json');modules={m['path']:m for m in inv['modules']}
def walk(v):
 if isinstance(v,dict):
  if 'id' in v and any(k in v for k in ('kind','view','children')):yield v
  for x in v.values():yield from walk(x)
 elif isinstance(v,list):
  for x in v:yield from walk(x)
blocks={n:{b['id']:b for b in walk(c['ir']['extras']['render'])} for n,c in cases.items()};canonical={b['source_instance_path']:b for b in blocks['ordinary'].values() if 'source_instance_path' in b and not b.get('target')};cards=new['observation']['page']['cards'];shapes=facts['root.denoiser.constructed_parameter_shapes']['value']
assert set(facts['root.denoiser.constructed_modules']['value'])==set(modules)
chips=[]
for b in blocks['ordinary'].values():
 if 'source_instance_path' not in b or b.get('target'):continue
 path=b['source_instance_path']
 for chip in b.get('facts',[]):
  if chip.endswith(' parameters in subtree'):
   assert chip in cards[b['id']]['facts'];assert chip==f"{shapes['by_module'][path]:,} parameters in subtree";chips.append({'occurrence':path,'card_id':b['id'],'actual_chip':chip})
archive_checks={}
for key,q in proofs.items():
 hashes=set(re.findall(r'sha256:([0-9a-f]{64}):','\n'.join((q.get('proof') or {}).get('evidence_refs',[]))));mapping=q.get('source_archive',{}).get('artifacts',{})
 for h in hashes:assert hashlib.sha256(gzip.decompress((P/'ordinary'/mapping[h]).read_bytes())).hexdigest()==h
 archive_checks[key]={'claim_kind':q['claim_kind'],'archived_span_sources':sorted(hashes),'proof_summary_present':bool(q.get('proof'))}
old_u=old['ir']['extras']['unet'];ffn=facts['root.denoiser.ffn_mechanisms']['value'];ffn_rows=[]
for path,value in ffn.items():
 b=canonical[path];card=cards[b['id']];assert value['activation']=='gelu' and value['gated'] is True and value['projection_mode']=='fused_gate_up';assert 'Activation: gelu' in card['facts'] and 'Projection storage: fused_gate_up' in card['facts'];assert b['view']=='runtime_ffn';assert 'root.denoiser.ffn_mechanisms' in b['source_fact_keys']
 required={b['id']+'__op_'+suffix for suffix in ('gate_up_proj','gate_up_split','activation','multiply','down_proj')};assert required<=set(card['node_ids']);ffn_rows.append({'occurrence':path,'actual_card':b['id'],'canonical_fact':'root.denoiser.ffn_mechanisms','fact_value':value,'actual_quantity_and_mechanism_chips':card['facts'],'actual_operation_nodes':sorted(required),'proof_kind':'applied_function'})
assert len(ffn_rows)==16 and old_u['transformer_ffn_act']=='geglu'
stages=[]
for old_st in old_u['down']+[old_u['mid']]+old_u['up']:
 sid=old_st['id'];path='mid_block' if sid=='unet_mid' else sid.replace('unet_down_','down_blocks.').replace('unet_up_','up_blocks.');b=canonical[path];assert b['id'] in cards;resnets=sorted(p for p in modules if re.fullmatch(re.escape(path)+r'\.resnets\.\d+',p));attention=sorted(p for p in modules if re.fullmatch(re.escape(path)+r'\.attentions\.\d+',p));actual_ffn=[r for r in ffn_rows if r['occurrence'].startswith(path+'.')]
 assert len(resnets)==old_st['resnets'];assert bool(attention) or not old_st.get('attn');assert all(canonical[p]['id'] in cards for p in resnets+attention)
 stages.append({'legacy_stage':sid,'actual_occurrence':path,'class_ref':modules[path]['class_ref'],'legacy_attention_present':old_st.get('attn'),'actual_attention_occurrences':attention,'actual_resnet_occurrences':resnets,'retained_ffn_cards':[x['actual_card'] for x in actual_ffn],'actual_stage_card':b['id'],'actual_stage_facts':cards[b['id']]['facts'],'bounded_result':'Construction and actual module cards retained; aggregate execution order not inferred.'})
assert len(stages)==9;mid=next(x for x in stages if x['actual_occurrence']=='mid_block');assert old_u['mid_present'] is True and old_u['mid']['attn'] is False;assert len(mid['actual_attention_occurrences'])==1;mid_default=facts['root.denoiser.declared_constructor_defaults']['value']['mid_block_type'];assert mid_default=={'checkpoint':'omitted','provenance':'class_default','value':'UNetMidBlock2DCrossAttn'}
contexts=[]
for path,value in facts['root.denoiser.context_connections']['value'].items():
 b=canonical[path];card=cards[b['id']];assert 'External context input proven' in card['facts'];assert value['cross_attention']=='investigation_missing';assert any('distinct query lineage remains open' in x for x in card['facts']);contexts.append({'occurrence':path,'actual_card':b['id'],'fact':value,'actual_facts':card['facts']})
assert len(contexts)==15
cells=[]
for path,value in facts['root.denoiser.cell_connections']['value'].items():
 b=canonical[path];card=cards[b['id']];assert value['coverage']=='positive_only';assert len(value['connections'])==5;assert value['unresolved'] in card['facts'];cells.append({'occurrence':path,'actual_card':b['id'],'proven_connections':value['connections'],'unresolved':value['unresolved'],'arithmetic_fact_present':path in facts['root.denoiser.cell_arithmetic']['value']})
assert len(cells)==22
source_ranges={'model_unfolder/adapters/diffusor/parser.py':[(52,132),(151,169),(883,936)],'model_unfolder/adapters/diffusor/unet.py':[(59,147),(292,331),(350,450),(605,631)],'model_unfolder/evidence/patterns.py':[(425,493),(532,582),(624,660)],'model_unfolder/renderers/html/block_views/unet.py':[(667,724),(727,755)]};source={}
for name,ranges in source_ranges.items():
 data=(S/name).read_bytes();lines=data.decode().splitlines();source[name]={'sha256':hashlib.sha256(data).hexdigest(),'snippets':['\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,min(hi,len(lines))+1)) for lo,hi in ranges]}
demotions={
 'resnet_template':{'old_author':'unet.py:_unet_resnet_ops/_resnet_card and renderer build_unet_resnet_view compose a fixed norm/activation/conv/timestep/residual program from display fields, without a per-cell connection proof.','new_result':'22 actual cell cards retain five source-established edges each plus conditional arithmetic. Optional/helper connections remain explicit limitations. Full old residual program is not re-proved.','evidence':'cell_connections/cell_arithmetic; all22actual cards checked'},
 'attention_template':{'old_author':'patterns.unet_stage_attn_cell_from_files establishes reachable attention+FFN construction roles. unet.py:_unet_transformer_subblocks and build_unet_transformer_view then emit fixed self→cross→FFN order, full mask and text/KV role labels. The construction predicate alone does not prove these connections.','new_result':'All actual attention/FFN modules retained;16 FFNs have applied-function proofs.15 downstream attention cards prove external context input while retaining unknown distinct-query role. Mid attention is newly constructed/visible but role remains unknown. No complete attention algorithm or global order is asserted.'},
 'head_geometry':{'old_author':'parse_unet.heads_hd interprets config attention_head_dim as legacy head count when num_attention_heads absent, then divides channel width.','new_result':'Old aggregate head8/head_dim40/80/160 labels are not retained as qualified mechanism claims. Exact projection parameter shapes remain; global head-role decomposition still needs an appropriate value/mechanism proof.'},
 'spatial_template':{'old_author':'parse_unet assumes sample at every stage except last and downscale=2**(n-1).','new_result':'Six exact spatial operation facts replace the aggregate recipe: three reduce Conv2d operations with operand2 and three interpolate resize operations with operandNone (resize factor unresolved), with actual occurrence cards. Full spatial execution composition is not inferred from stage count.'},
 'mid_default':{'old_author':'Legacy source mid presence was true; missing config mid_block_type left attn=False.','new_result':'Resolved constructor default identifies actual mid block and its one attention container plus FFN. This adds previously omitted structure and preserves known mid existence.','resolved_default':mid_default}}
for path,value in facts['root.denoiser.spatial_mechanisms']['value'].items():assert canonical[path]['id'] in cards
result={'status':'bounded preservation and named conservative demotions reviewed; remaining authority/config findings retained; no blessing','actual_subtree_quantity_chips':chips,'archived_proofs':archive_checks,'legacy_stage_to_actual_occurrence':stages,'ffn_preservation':ffn_rows,'context_limits':contexts,'cell_connection_fragments':cells,'named_template_demotions':demotions,'source_provenance':source}
(O/'claim-preservation-and-demotions.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
# Keep all exact raw rows; attach specific old/new block locations and bounded causes.
rows=json.loads(gzip.decompress((O/'per-output-review.json.gz').read_bytes()))
def ancestor(ir,pointer):
 parts=pointer.strip('/').split('/');last=None;value=ir
 for part in parts:
  if isinstance(value,dict) and 'id' in value:last=value
  try:value=value[int(part)] if isinstance(value,list) else value[part.replace('~1','/').replace('~0','~')]
  except (KeyError,ValueError,IndexError,TypeError):break
 if isinstance(value,dict) and 'id' in value:last=value
 return {k:last[k] for k in ('id','view','kind','source_instance_path','source_fact_keys') if k in last} if last else None
for row in rows:
 pointer=row['original_delta']['path'];row['legacy_block_anchor']=ancestor(old['ir'],pointer);row['new_block_anchor']=ancestor(new['ir'],pointer)
 keys=(row['new_block_anchor'] or {}).get('source_fact_keys',[]);row['cited_fact_claim_kinds']={k:proofs[k]['claim_kind'] for k in keys if k in proofs}
 row['specific_review_links']=[]
 sid=(row['legacy_block_anchor'] or {}).get('id','')
 if sid.startswith('unet_'):
  if 'resnet' in sid or sid.startswith('unet_op_'):row['specific_review_links'].append('claim-preservation-and-demotions.json#/named_template_demotions/resnet_template')
  if any(t in sid for t in ('attn','transformer','__ff')):row['specific_review_links'].append('claim-preservation-and-demotions.json#/named_template_demotions/attention_template')
  row['specific_review_links'].append('claim-preservation-and-demotions.json#/legacy_stage_to_actual_occurrence')
 if pointer.startswith('/extras/unet/'):
  field=pointer.split('/')[3];key='spatial_template' if field in ('downscale','down','up') else 'attention_template' if field in ('cross_attention_dim','kv_label','kv_modality','transformer_ffn_act') else 'mid_default' if field in ('mid','mid_present','declares_mid_block_type') else None
  if key:row['specific_review_links'].append('claim-preservation-and-demotions.json#/named_template_demotions/'+key)
 if pointer.startswith('/extras/render/loop_edges') or sid in ('encoder_0','image'):row['specific_review_links'].append('named-removals.json')
 if row['review_group']=='audit':row['bounded_disposition']='Retained config accounting finding; root disposition required, no automatic waiver.'
 elif row['review_group']=='shapes':row['bounded_disposition']='Exact inventory sum and all706actual subtree chips independently matched.'
 else:row['bounded_disposition']='Named source/display cause and preservation mappings attached where available; original blocking/candidate status retained, no automatic approval.'
(O/'per-output-review.json.gz').write_bytes(gzip.compress((json.dumps(rows,indent=2,sort_keys=True)+'\n').encode(),mtime=0))
print(json.dumps({'rows':len(rows),'actual_quantity_chips':len(chips),'stages':len(stages),'ffns':len(ffn_rows),'context_limits':len(contexts),'cell_fragments':len(cells)}))
