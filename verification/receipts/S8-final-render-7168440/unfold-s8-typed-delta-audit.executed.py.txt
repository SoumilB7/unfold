from pathlib import Path
import json,re,html,hashlib,gzip
root=Path('/private/tmp/unfold-s8-typed-phase-7168440')
rows=[]
for result in sorted(root.glob('*/result.json')):
 case=result.parent;record=json.loads(result.read_text());assert record['status']=='PASS'
 original=(Path(record['base_case'])/'page.html').read_text();actual=(case/'page.html').read_text();delta=json.loads((case/'ir-deltas.json').read_text());changed=[]
 expected=original
 for row in delta['block_fields']:
  if row['field']!='title':continue
  old='data-card-id="'+row['id']+'" data-card-size="compact"><div class="uf-card-title">'+html.escape(row['before'])+'</div>'
  new='data-card-id="'+row['id']+'" data-card-size="compact"><div class="uf-card-title">'+html.escape(row['after'])+'</div>'
  count=expected.count(old);assert count>0,(case.name,row['id']);expected=expected.replace(old,new);changed.append({'id':row['id'],'count':count,'old_title':row['before'],'new_title':row['after']})
 entry=delta['component_entry']
 if entry:
  for cls,old,new in (('uf-section-label','SAMPLING LOOP',entry['title']),('uf-section-sub','Denoiser applied iteratively · click it to open its architecture',entry['subtitle'])):
   oldhtml='<span class="'+cls+'">'+old+'</span>';newhtml='<span class="'+cls+'">'+html.escape(new)+'</span>';assert expected.count(oldhtml)==1;expected=expected.replace(oldhtml,newhtml)
 assert expected==actual,(case.name,'HTML changed beyond exact card titles and explicit component header')
 events=json.loads((case/'event-deltas.json').read_text());assert events['removed']==[]
 for event in events['added']:
  assert event['view'] in ('card_fact_lines','runtime_ffn_fact'),event
  assert set(event['facts_projected'])<={'root.denoiser.ffn_mechanisms','root.denoiser.declared_constructor_defaults','root.denoiser.spatial_mechanisms'}
  assert event['facts_projected'] and event['node_ids'] and event['block_path']
 assert record['svg_deltas']==[]
 row={'case':case.name,'actual_html_only_declared_card_titles_and_component_header':True,'activation_card_changes':changed,'component_header_change':entry,'svg_bytes_unchanged':True,'prior_events_retained':True,'added_exact_receipt_events':len(events['added']),'before_event_count':events['before_count'],'after_event_count':events['after_count'],'new_html_sha256':record['html_sha256'],'old_html_sha256':record['prior_html_sha256']}
 (case/'actual-delta-audit.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');rows.append(row)
(root/'actual-delta-audit.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print(json.dumps({'cases':len(rows),'all_exact':True,'event_counts':{r['case']:r['added_exact_receipt_events'] for r in rows}}))
