from pathlib import Path
import json,hashlib,re,shutil,xml.etree.ElementTree as ET
from model_unfolder.preview import _visual_hash,svg_to_png
from test_support.preservation import html_meta,_canon_bytes
src=Path('/private/tmp/unfold-s8-current-captures-fb1f871/17-view-cause-ledger.json')
out=Path('verification/receipts/S8-nonunet-view-geometry-independent-fb1f871')
rows=json.loads(src.read_text());expected=json.loads(Path('tests/preservation_expected_manifest.json').read_text())['witnesses'];ns='{http://www.w3.org/2000/svg}'
def digest(b):return hashlib.sha256(b).hexdigest()
def read(svg):
 root=ET.fromstring(svg);boxes={};nodes={}
 for g in root.iter(ns+'g'):
  key=g.get('data-id')
  if not key:continue
  nodes[key]=tuple(''.join(t.itertext())for t in g.iter(ns+'text'))
  r=g.find(ns+'rect');c=g.find(ns+'circle')
  if r is not None:
   x,y,w,h=[float(r.get(k))for k in ('x','y','width','height')];boxes[key]={'x':x,'y':y,'w':w,'h':h,'cx':x+w/2,'cy':y+h/2,'shape':'rect'}
  elif c is not None:
   x,y,r=[float(c.get(k))for k in ('cx','cy','r')];boxes[key]={'x':x-r,'y':y-r,'w':2*r,'h':2*r,'cx':x,'cy':y,'shape':'circle'}
 paths=[tuple(map(float,re.findall(r'-?\d+(?:\.\d+)?',p.get('d'))))for p in root.iter(ns+'path')if p.get('marker-end')]
 return root,boxes,nodes,paths
results=[]
for i,row in enumerate(rows):
 slug=row['slug'];dest=out/f'{i:02d}-{slug}';dest.mkdir(exist_ok=True)
 old=Path(row['old_svg']).read_text();new=Path(row['current_svg']).read_text();result=json.loads(Path(row['result_path']).read_text());oldpage=Path(row['result_path']).with_name('page.html').read_text();assert result['old_views']==expected[slug]['views'];assert digest(_canon_bytes(html_meta(oldpage)))==expected[slug]['surfaces']['html_meta']
 assert _visual_hash(new)==row['change']['current'][1] and _visual_hash(old)==row['change']['old'][1]
 graph=row['actual_records'][0]['graph'];region=row['actual_records'][0]['region'];assert all(r['graph']==graph and r['region']==region for r in row['actual_records'])
 records=json.loads(Path(row['graph_record_path']).read_text());olds=[r for r in records if r['phase']=='old_parallel:'+slug and r['visual_hash']==row['change']['old'][1]];assert olds and all(r['graph']==graph and r['region']==region for r in olds)
 nr,boxes,nodes,paths=read(new);orr,ob,on,op=read(old);assert nodes==on
 proofs=[];stems={}
 for edge in row['actual_records'][0]['empty_lane_edges']:
  s,t=edge['source'],edge['target'];assert any(e['src']==s and e['dst']==t for e in region['edges']);sg,tg=boxes[s],boxes[t];tap=sg['y']-16
  targety=tg['cy'] if tg['shape']=='circle' else tg['y']+tg['h']+6
  candidates=[p for p in paths if abs(p[1]-(tap-12))<.01 and abs(p[-1]-targety)<.01];assert candidates,(slug,s,t)
  for p in candidates:
   obstacles=[k for k,g in boxes.items()if k not in(s,t)and g['shape']=='rect'and min(p[1],p[-1])<g['y']+g['h']and max(p[1],p[-1])>g['y']and g['x']<p[0]<g['x']+g['w']];assert not obstacles,(slug,p,obstacles)
  stems[s]=sum(1 for l in nr.iter(ns+'line')if abs(float(l.get('x1'))-sg['cx'])<.01 and abs(float(l.get('x2'))-sg['cx'])<.01 and abs(float(l.get('y1'))-sg['y'])<.01 and abs(float(l.get('y2'))-tap)<.01)
  assert stems[s]==1,(slug,s,stems[s])
  proofs.append({'source':s,'target':t,'canonical_edge':True,'matching_actual_arrow_rails':[p[0]for p in candidates],'intersected_rectangles':[]})
 for phase,svg in [('old',old),('current',new)]:
  (dest/f'{phase}.svg').write_text(svg);svg_to_png(svg,str(dest/f'{phase}.png'),scale=1)
 (dest/'graph-and-region.json').write_text(json.dumps({'graph':graph,'region':region},indent=2)+'\n')
 results.append({'index':i,'slug':slug,'view':row['change']['current'][0],'old_locked_views_recovered':True,'old_html_meta_recovered':True,'canonical_graph_region_identical':True,'node_ids_and_labels_unchanged':True,'source_stems':stems,'restored_edges':proofs,'old_svg_sha256':digest(old.encode()),'current_svg_sha256':digest(new.encode()),'native_paths':[str(dest/'old.png'),str(dest/'current.png')]})
(out/'geometry-checks.json').write_text(json.dumps(results,indent=2)+'\n');(out/'input-ledger.json').write_bytes(src.read_bytes());shutil.copyfile(__file__,out/'review.py')
print('PASS',len(results),'views',len(set(r['slug']for r in results)),'models',sum(len(r['restored_edges'])for r in results),'listed empty edges')
