from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,re
HERE=Path(__file__).resolve().parent;old=HERE.parent/'browser36/results-r2';r=HERE/'results';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Scripts(HTMLParser):
 def __init__(self):super().__init__();self.current=None;self.rows=[]
 def handle_starttag(self,tag,attrs):
  if tag=='script':self.current={'attributes':dict(attrs),'text':''}
 def handle_data(self,data):
  if self.current is not None:self.current['text']+=data
 def handle_endtag(self,tag):
  if tag=='script' and self.current is not None:self.rows.append(self.current);self.current=None
result=load(r/'result.json');assert result['status']=='PASS_BOUNDED_INTERACTIONS' and not result['findings'];cases={}
for case in result['cases']:
 a=load(old/case/'result.json');b=load(r/case/'result.json');assert len(a['interactions'])==len(b['interactions'])
 for x,y in zip(a['interactions'],b['interactions']):
  assert x['clicked_node']==y['clicked_node']
  if 'card' in x:
   assert x['card']['text']==y['card']['text'];assert [z['text'] for z in x['card']['chip_nodes']]==[z['text'] for z in y['card']['chip_nodes']];assert y['card']['scrollWidth']<=y['card']['clientWidth']
   for oc,nc in zip(x['card']['chip_nodes'],y['card']['chip_nodes']):
    for k in ['background','color','border']:assert oc['style'][k]==nc['style'][k]
  else:assert x['text']==y['text'] and x['card_id']==y['card_id']
 for phase in ['initial_dom','final_dom']:
  for field in ['elements','cards','op_nodes','chip_kinds','visible_cards']:assert a[phase][field]==b[phase][field]
 paths=[Path(k) for folder in (old,r) for k in load(folder/'input-pins.json') if k.endswith('/'+Path((a if folder==old else b)['input']).name) and str(k).endswith((a if folder==old else b)['input'])]
 assert len(paths)==2;pages=[p.read_text() for p in paths];mounts=[re.search(r'#(uf-[a-f0-9]+) \.uf-fact \{',s).group(1) for s in pages]
 parsed=[]
 for page in pages:s=Scripts();s.feed(page);parsed.append(s.rows)
 assert len(parsed[0])==len(parsed[1]);js=[0,0];data=[0,0];payload_pairs=[]
 for x,y in zip(*parsed):
  if 'data-uf-card-payload' in x['attributes']:
   assert case=='sdxl' and re.fullmatch('[0-9a-f]{64}',x['attributes']['data-uf-card-payload']) and re.fullmatch('[0-9a-f]{64}',y['attributes']['data-uf-card-payload']);payload_pairs.append([x['attributes']['data-uf-card-payload'],y['attributes']['data-uf-card-payload']])
 for x,y in zip(*parsed):
  typ=x['attributes'].get('type','').lower();expected_attrs={k:(v.replace(mounts[0],mounts[1]) if isinstance(v,str) else v) for k,v in x['attributes'].items()}
  if 'data-uf-card-payload' in expected_attrs:expected_attrs['data-uf-card-payload']=payload_pairs[0][1]
  assert expected_attrs==y['attributes']
  if typ in ('','text/javascript','application/javascript','module'):
   expected_js=x['text'].replace(mounts[0],mounts[1])
   for before_digest,after_digest in payload_pairs:expected_js=expected_js.replace(before_digest,after_digest)
   assert expected_js==y['text'];js[0]+=len(x['text'].encode());js[1]+=len(y['text'].encode())
  else:data[0]+=len(x['text'].encode());data[1]+=len(y['text'].encode())
 ac=a['interactions'][0]['card'];bc=b['interactions'][0]['card'];cases[case]={'before_scroll_width':ac['scrollWidth'],'after_scroll_width':bc['scrollWidth'],'card_client_width':bc['clientWidth'],'full_card_and_chip_text_exact':True,'DOM_counts_card_ids_and_four_border_colors_unchanged':True,'actual_visible_default_chip_count':len([x for x in bc['chip_nodes'] if x['kind']=='class_default']),'ready_at_load_event_ms':b['load_ms'],'click_to_card_ms':b['interactions'][0]['click_to_card_ms'],'HTML_bytes_before_after':[len(s.encode()) for s in pages],'executable_JS_bytes_before_after':js,'executable_JS_size_delta':js[1]-js[0],'exact_payload_selector_pairs':payload_pairs,'JS_comparison_scope':'Raw JS differs only exact mount and SDXL content-addressed payload selector values. Size delta measured without normalization; no output approval normalization.','nonexecutable_script_data_bytes_before_after':data,'page_errors':b['page_errors']}
styles=load(r/'four-style-probe.json')['styles'];assert len(styles)==4 and all(x['borderRadius']=='4px' and x['rect']['width']<=750 for x in styles)
verdict={'verdict':'PASS_BOUNDED_ACTUAL37_BROWSER_QA','actual_runtime_manifest_sha256':sha(r/'manifest.json'),'cases':cases,'four_style_scope':'Actual model cards emit class_default in these three checked examples; the other kinds are evaluated with separately labelled synthetic long-token nodes under the actual37 stylesheet. This does not claim four actual fact emitters.','new_toggles':0,'new_colors':0,'full_text_preserved':True,'screenshots_visually_inspected':['musicgen-sparse/card-viewport.png','bloom-sparse/card-viewport.png','sdxl/card-viewport.png','four-style-probe.png'],'residuals':[{'id':'W1','owner':'S9-D','source':'model_unfolder/labels.py:382','finding':'Existing MusicGen card prose says projected image states; actual drawing labels Encoded text. Recorded without broadening A or inferring source mechanism from browser.'},{'id':'W2','owner':'S9-C/L5','finding':'Existing Bloom prose says every head attends over the sequence while mask is unresolved; wording should be proof-sourced. No further A change.'},'Full JSON default labels remain verbose; no information-design redesign in this bounded correction.'],'limits':'Six actual37 captured pages; published bytes matched root saved outputs before browser; headless Chrome152 at1440x1000, fresh profile, fonts blocked. Incidental load/click times under concurrent matrix are not latency baselines. No exhaustive drill/mobile/accessibility check or output blessing.'}
(HERE/'visual-verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');print('verdict',sha(HERE/'visual-verdict.json'));print(json.dumps({k:{'html':v['HTML_bytes_before_after'],'JS':v['executable_JS_bytes_before_after'],'data':v['nonexecutable_script_data_bytes_before_after']} for k,v in cases.items()}))
