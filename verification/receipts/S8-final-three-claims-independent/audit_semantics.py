from pathlib import Path
import json,gzip,hashlib,math,re,xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent;CASE=Path('/private/tmp/unfold-s8-final-f664467/sdxl/ordinary')
traces=json.loads((OUT/'claim-traces.json').read_text());facts=json.loads((CASE/'facts.json').read_text());qualified=json.loads((CASE/'qualified-facts.json').read_text());inventory=json.loads((CASE/'inventory.json').read_text());ir=json.loads((CASE/'ir.json').read_text());page=(CASE/'page.html').read_text();modules={x['path']:x for x in inventory['modules']}
def walk(x):
 if isinstance(x,dict):
  if 'id' in x:yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
blocks={x['id']:x for x in walk(ir['extras']['render']) if not x.get('target')}
source_specs={'attention':('0192998d3979533cdbc8eef9c8f783d0b4a4a547dec9e4388a17c9ae3b1c8ea8',1709,1742),'activation':('ab1767e8e44e7d4bf1cb18299ba33654329e2711fa983b1423fc4fe12de2c3ab',103,123),'resnet':('b601e45d78a640e66b05822d3d008e40a039b92bb126abd1a9bdef6124666d84',319,365),'downsample':('2852f787a39f6f679578b9992c37b114c4b76cb9449b473267490429c22c396c',97,147)}
sources={}
for name,(fingerprint,lo,hi) in source_specs.items():
 raw=gzip.decompress((CASE/'cited-source'/f'{fingerprint}.py.gz').read_bytes());assert hashlib.sha256(raw).hexdigest()==fingerprint
 lines=raw.decode().splitlines();sources[name]={'sha256':fingerprint,'lines':[lo,hi],'text':'\n'.join(f'{i+1}: {lines[i]}' for i in range(lo-1,hi))}
(OUT/'source-excerpts.json').write_text(json.dumps(sources,indent=2,sort_keys=True)+'\n')
rows=[]
for trace in traces:
 occurrence=trace['occurrence'];key=trace['canonical_fact']['key'];block=blocks[trace['block']['id']]
 assert not trace['chain_gaps'] and block['source_instance_path']==occurrence and key in block['source_fact_keys']
 assert qualified[key]['claim_kind']==trace['canonical_fact']['claim_kind']==qualified[key]['proof']['claim_kind']
 assert trace['overview_stage']['drawn_in_actual_denoiser_svg']
 parameters={path+'.'+p['name']:p for path,m in modules.items() if path==occurrence or path.startswith(occurrence+'.') for p in m['parameters']}
 total=sum(math.prod(p['shape']) for p in parameters.values())
 shapes=facts['root.denoiser.constructed_parameter_shapes']['value'];assert total==shapes['by_module'][occurrence]
 for name,p in parameters.items():assert p['shape']==shapes['parameters'][name]['shape']
 assert f'{total:,} parameters in subtree' in trace['actual_card']['facts']
 cid=block['id'];start=page.index('data-card-id="'+cid+'"');end=page.find('data-card-id="',start+15);chunk=page[start:end] if end>=0 else page[start:]
 svgs=re.findall(r'<svg\b.*?</svg>',chunk,re.S);assert len(svgs)==trace['actual_card']['svg_count']
 svg_rows=[]
 for n,svg in enumerate(svgs):
  tree=ET.fromstring(svg);nodeids=[x.attrib['data-id'] for x in tree.iter() if 'uf-node' in x.attrib.get('class','').split()]
  svg_rows.append({'nodes':nodeids,'arrow_count':len(re.findall('marker-end=',svg))})
 rows.append({'occurrence':occurrence,'claim_kind':qualified[key]['claim_kind'],'proof_kind':qualified[key]['proof']['proof_kind'],'fact_status':facts[key]['status'],'fact_completeness':facts[key].get('completeness'),'stage':trace['overview_stage'],'block_id':cid,'parameter_sum':total,'direct_parameter_shapes':{n:p['shape'] for n,p in parameters.items()},'actual_drills':svg_rows})
ffn=traces[0]['canonical_fact']['value'];prefix=traces[0]['occurrence']
assert ffn=={'activation':'gelu','gated':True,'input_projection':prefix+'.net.0.proj','output_projection':prefix+'.net.2','projection_mode':'fused_gate_up'}
assert modules[ffn['input_projection']]['framework_primitive']['key']=='linear'
assert modules[ffn['output_projection']]['framework_primitive']['key']=='linear'
assert modules[prefix+'.net.1']['framework_primitive']['key']=='dropout' and modules[prefix+'.net.1']['init_attributes']['p']==0
assert rows[0]['parameter_sum']==5120*640+5120+640*2560+640==4920960
cell=traces[1]['canonical_fact']['value'];prefix=traces[1]['occurrence'];attrs=modules[prefix]['init_attributes'];assert attrs['upsample'] is None and attrs['downsample'] is None and attrs['time_embedding_norm']=='default'
assert len(cell['connections'])==5 and cell['coverage']=='positive_only'
assert [s['arrow_count'] for s in rows[1]['actual_drills'][:2]]==[2,3]
for edge in cell['connections']:
 for endpoint in ('source','target'):
  member=cell['calls'][edge[endpoint]]['member'];assert member in modules
assert rows[1]['parameter_sum']==2255040
spatial=traces[2]['canonical_fact']['value'];prefix=traces[2]['occurrence'];conv=modules[prefix+'.conv'];assert spatial=={'effect':'reduce','operand':2,'primitive':'torch.nn.Conv2d'}
assert modules[prefix]['init_attributes']['norm'] is None and modules[prefix]['init_attributes']['use_conv'] is True and modules[prefix]['init_attributes']['padding']==1
assert conv['framework_primitive']['key']=='conv2d' and conv['init_attributes']['stride']==[2,2]
assert rows[2]['parameter_sum']==320*320*3*3+320==921920
pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CASE.iterdir() if p.is_file()}
out={'three_claims':rows,'runtime_primitive_claim_kind':qualified['root.denoiser.runtime_primitives']['claim_kind'],'artifact_pins':pins,'semantic_scope':'Selected GEGLU computation, five positive cell edges, and selected stride-2 Conv2d are independently source/instance/shape matched; no universal execution claim. Spatial drill shows containment and prose, not an explicit flow route.'}
(OUT/'semantic-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'selected_parameter_sums':[r['parameter_sum'] for r in rows],'svg_arrow_counts':[[s['arrow_count'] for s in r['actual_drills']] for r in rows],'runtime_primitive_claim_kind':out['runtime_primitive_claim_kind']},indent=2))
