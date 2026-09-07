from pathlib import Path
import json,gzip,hashlib,re,sys,xml.etree.ElementTree as E
root=Path(sys.argv[1]);before=(root/'before/page.html').read_text();after=(root/'after/page.html').read_text();a=json.loads((root/'before/result.json').read_text());b=json.loads((root/'after/result.json').read_text());assert not b['dotted_arrows'] and not b['dotted_boundaries'];assert not a['dotted_arrows'];assert a['render_event_sha256']==b['render_event_sha256'];assert gzip.decompress((root/'before/render-events.json.gz').read_bytes())==gzip.decompress((root/'after/render-events.json.gz').read_bytes());assert a['source_artifact_pins']==b['source_artifact_pins'];sys.path.insert(0,'/private/tmp/unfold-s8-solid-review-f7baef5')
from model_unfolder.preview import svg_views
old=list(svg_views(before));new=list(svg_views(after));assert [n for n,s in old]==[n for n,s in new];changes=[];removed_dash=0;identity_routes=[]
for (name,x),(_,y) in zip(old,new):
 if x==y:continue
 assert name=='denoiser',name
 expected=re.sub(r' stroke-dasharray="(?:5 4|5 3)"','',x);removed_dash+=len(re.findall(r' stroke-dasharray="(?:5 4|5 3)"',x));expected=expected.replace('Conditional slot targets · dashed links','Conditional slot identity links').replace('Conditional call targets · dashed links','Conditional call identity links');assert y==expected,'SVG delta exceeds exact stroke/caption edit'
 tx,ty=E.fromstring(x),E.fromstring(y)
 def ids(t):return sorted(n.get('data-id') for n in t.iter() if n.get('data-id'))
 assert ids(tx)==ids(ty)
 def routes(t):return [dict(n.attrib) for n in t.iter() if n.get('data-route-kind')]
 rx,ry=routes(tx),routes(ty)
 for r in rx:r.pop('stroke-dasharray',None)
 assert rx==ry
 for n in ty.iter():
  if n.get('data-route-kind')=='conditional_call_target':
   assert not any('marker' in k for k in n.attrib);identity_routes.append(dict(n.attrib))
 changes.append(name);(root/'denoiser-before.svg').write_text(x);(root/'denoiser-after.svg').write_text(y)
 # The same exact SVG replacement must account for the full HTML difference.
 before=before.replace(x,y)
assert before==after,'non-SVG or additional HTML delta';result={'changed_svg_labels':changes,'removed_dash_attributes':removed_dash,'identity_routes':identity_routes,'route_and_node_metadata_identical':True,'identity_links_have_no_arrowheads':True,'only_exact_stroke_caption_delta':True,'all_render_events_identical':True,'render_event_count':b['render_event_count'],'before_dotted_boundary_count':len(a['dotted_boundaries']),'after_dotted_boundary_count':0,'after_dotted_arrow_count':0,'html_sha256':b['actual_html_sha256'],'blessed':False};(root/'comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ('changed_svg_labels','removed_dash_attributes','render_event_count','html_sha256')}))
