from pathlib import Path
import json,gzip,hashlib,math,re,xml.etree.ElementTree as ET
P=Path('/private/tmp/unfold-s8-final-f664467/sdxl/ordinary');O=Path(__file__).resolve().parent
pins={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('facts.json','qualified-facts.json','inventory.json','ir.json','page.html')}
f=json.loads((P/'facts.json').read_text());q=json.loads((P/'qualified-facts.json').read_text());ir=json.loads((P/'ir.json').read_text());inv=json.loads((P/'inventory.json').read_text());page=(P/'page.html').read_text()
def walk(x,path=''):
 if isinstance(x,dict):
  yield path,x
  for k,v in x.items():yield from walk(v,path+'.'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from walk(v,path+f'[{i}]')
key='root.denoiser.primary_state_ports';valuepath='.regions[3].route.iteration_result.when_true.arguments[1].route.when_false.when_false.when_false'
routes=dict(walk(f[key]['value']));r=routes[valuepath]
assert r['target_binding']['targets']==['time_embedding'] and r['kind']=='call_result'
sha='052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26';call=f'sha256:{sha}:1084:14:1084:55';previous=f'sha256:{sha}:1083:16:1083:69'
assert r['call_source']==call and r['arguments'][0]['port']=='0' and r['arguments'][0]['route']['call_source']==previous
assert r['arguments'][1]=={'port':'1','route':{'formal':'timestep_cond','kind':'formal'}}
assert q[key]['claim_kind']==q[key]['proof']['claim_kind']=='connection' and call in str(q[key]['proof']) and previous in str(q[key]['proof'])
raw=gzip.decompress((P/'cited-source'/f'{sha}.py.gz').read_bytes());assert hashlib.sha256(raw).hexdigest()==sha
source=raw.decode().splitlines();assert 'emb = self.time_embedding(t_emb, timestep_cond)' in source[1083]
bs={b['id']:b for _,b in walk(ir['extras']['render'])if 'id'in b}
cid='unet_primary_region_3__iteration__when_true__arg_1__when_false__when_false__when_false';b=bs[cid];boundary=bs[cid+'__call']
assert b['view']=='runtime_port_route' and key in b['source_fact_keys'] and boundary['target']=='instance_time_embedding'
assert boundary['detail']['invocation_conditions']==r['target_binding']['conditions']
def card(cid):
 start=page.index('data-card-id="'+cid+'"');end=page.find('data-card-id="',start+15);return page[start:end] if end>=0 else page[start:]
def svg_nodes(svg):
 return [e.attrib['data-id'] for e in ET.fromstring(svg).iter() if 'uf-node' in e.attrib.get('class','').split()]
svg=re.findall(r'<svg\b.*?</svg>',card(cid),re.S)[0];assert len(re.findall('marker-end=',svg))==3
assert set(svg_nodes(svg))=={cid+'__arg_0',cid+'__arg_1',cid+'__call'}
(O/'time-embedding-call.svg').write_text(svg)
overview=re.findall(r'<svg\b.*?</svg>',card('denoiser'),re.S)[0];assert 'unet_primary_region_3' in svg_nodes(overview) and 'instance_time_embedding' in svg_nodes(overview)
canonical=bs['instance_time_embedding'];assert key in canonical['source_fact_keys'] and 'root.denoiser.constructed_parameter_shapes' in canonical['source_fact_keys']
params={m['path']+'.'+p['name']:p['shape'] for m in inv['modules'] if m['path']=='time_embedding' or m['path'].startswith('time_embedding.') for p in m['parameters']};total=sum(math.prod(s) for s in params.values());assert total==2050560==f['root.denoiser.constructed_parameter_shapes']['value']['by_module']['time_embedding'];assert '2,050,560 parameters in subtree' in card('instance_time_embedding')
for child,shape in [('linear_1','1280'),('linear_2','1280')]:assert shape in card('instance_time_embedding__'+child)
result={'claim':'The result of get_time_embed at line 1083 feeds argument 0 of the exact constructed time_embedding invocation at line 1084; root formal timestep_cond feeds argument 1. Call result boundary is visible; internal computation and argument-to-result semantic dependence remain unresolved.','fact_key':key,'value_path':valuepath,'claim_kind':q[key]['claim_kind'],'proof_kind':q[key]['proof']['proof_kind'],'source_sha256':sha,'source_lines':{'1083':source[1082],'1084':source[1083]},'source_refs':[previous,call],'overview_ids':['unet_primary_region_3','instance_time_embedding'],'routed_drill_card_id':cid,'boundary_id':cid+'__call','canonical_quantity_card_id':'instance_time_embedding','svg_nodes':svg_nodes(svg),'svg_arrows':3,'conditions':r['target_binding']['conditions'],'parameter_shapes':params,'parameter_total':total,'quantity_fact':'root.denoiser.constructed_parameter_shapes.value.by_module.time_embedding','artifact_pins':pins,'limitation':'Overview places the repeat region and time_embedding bookend. Actual input/call/result arrows are in the source-port drill, not on the time_embedding containment bookend or inside its linear/activation child drill. No complete conditioning function or all-path execution asserted.'}
assert pins=={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in pins}
(O/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'qualified_connection':True,'routed_drill_arrows':3,'params':total,'artifact_pins_unchanged':True}))
