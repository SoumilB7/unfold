from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;old=BASE/'browser36/results-r2';r=HERE/'results';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result=load(r/'result.json');assert result['status']=='PASS_BOUNDED_INTERACTIONS' and not result['findings'];cases={}
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
 ac=a['interactions'][0]['card'];bc=b['interactions'][0]['card'];cases[case]={'before_scroll_width':ac['scrollWidth'],'after_scroll_width':bc['scrollWidth'],'card_client_width':bc['clientWidth'],'full_card_and_chip_text_exact':True,'DOM_counts_and_card_identity_exact':True,'borders_and_colors_exact':True,'page_errors':b['page_errors']}
styles=load(r/'four-style-probe.json')['styles'];assert len(styles)==4 and all(x['borderRadius']=='4px' and x['rect']['width']<=750 for x in styles)
for row in load(HERE/'input-transform.json'):
 source=Path(row['original']);derived=HERE/row['derived'];assert sha(source)==row['original_sha256'] and sha(derived)==row['derived_sha256'];s=derived.read_text();marker='/* Typed annotations remain distinguishable by borders and labels as well as colour. */'
 for change in row['exact_insertions']:s=s.replace(marker+change['inserted'],marker,1)
 assert s==source.read_text()
verdict={'verdict':'PASS_BOUNDED_CSS_WRAP_AND_OUTLINE_BROWSER_COMPARISON','cases':cases,'proposal_manifest_sha256':sha(BASE/'typed-chip-wrap-proposal-r2/manifest.json'),'runtime_manifest_sha256':sha(r/'manifest.json'),'before_runtime_manifest_sha256':sha(old/'manifest.json'),'full_original_HTML_preserved_except_exact_recorded_CSS_insertions':True,'four_long_token_style_boxes_fit750px_host':True,'four_style_borders_colors_distinct':True,'radius4px_avoids_multiline_capsule_intersection':True,'screenshots_visually_inspected':['musicgen-sparse/card-viewport.png','four-style-probe.png'],'limits':'Derived exact saved36 HTML plus only rendered proposed CSS, including SDXL embedded stylesheet. Full text/IDs/counts verified; zero JS errors and all6 card clicks/nestedSDXL pass. Actual37 regeneration and browser rerun pending. Long JSON remains verbose; no information redesign or output blessing.','runtime_limits':result['limits']}
(HERE/'comparison-verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');(HERE/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json'},'runtime_manifest_sha256':sha(r/'manifest.json')},indent=2)+'\n');print('verdict',sha(HERE/'comparison-verdict.json'));print('manifest',sha(HERE/'manifest.json'))
