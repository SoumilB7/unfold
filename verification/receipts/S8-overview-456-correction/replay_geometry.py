from pathlib import Path
import sys,json,re,subprocess,xml.etree.ElementTree as ET,hashlib,shutil
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');sys.path.insert(0,str(root))
from model_unfolder.renderers.html.block_views.unet import build_unet_constructed_view
p=Path('/private/tmp/unfold-s8-final-4560449/sdxl/ordinary/ir.json');ir=json.loads(p.read_text())
def walk(v):
 if isinstance(v,dict):
  yield v
  for child in v.values():yield from walk(child)
 elif isinstance(v,list):
  for child in v:yield from walk(child)
block=next(v for v in walk(ir['extras']['render']) if v.get('id')=='denoiser')
out=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/final-4560449/overview-correction');out.mkdir(exist_ok=True)
before=out/'before.svg';shutil.copyfile(out.parent/'native/03__denoiser.svg',before)
after=out/'after.svg';after.write_text(build_unet_constructed_view(ir,{},'overview_correction',block));subprocess.run(['rsvg-convert',str(after),'-o',str(out/'after.png')],check=True)
NS={'s':'http://www.w3.org/2000/svg'}
def audit(path):
 tree=ET.fromstring(path.read_text());boxes={}
 for group in tree.findall('.//s:g',NS):
  if 'uf-node' not in group.get('class','').split():continue
  rect=group.find('s:rect',NS)
  if rect is not None:
   x,y,w,h=[float(rect.get(k)) for k in ('x','y','width','height')];boxes[group.get('data-id')]=(x,y,x+w,y+h)
 findings=[];routes=[]
 for group in tree.findall('.//s:g',NS):
  if group.get('data-route-kind')!='primary_state_port':continue
  route=(group.get('data-source'),group.get('data-target'));routes.append(route)
  for path in group.findall('s:path',NS):
   numbers=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',path.get('d'))];points=list(zip(numbers[::2],numbers[1::2]))
   for a,b in zip(points,points[1:]):
    for name,(l,t,r,d) in boxes.items():
     if name in route:continue
     if ((a[0]==b[0] and l<a[0]<r and max(min(a[1],b[1]),t)<min(max(a[1],b[1]),d)) or
         (a[1]==b[1] and t<a[1]<d and max(min(a[0],b[0]),l)<min(max(a[0],b[0]),r))):
      findings.append({'source':route[0],'target':route[1],'crossed_card':name,'segment':[a,b]})
 return {'nodes':sorted(boxes),'routes':sorted(routes),'crossings':findings}
a,b=audit(before),audit(after)
result={'scope':'Renderer-only correction on saved actual4560449 IR; not a fresh candidate page or blessing','ir_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256((root/'model_unfolder/renderers/html/block_views/unet.py').read_bytes()).hexdigest(),'before':a,'after':b,'nodes_preserved':a['nodes']==b['nodes'],'primary_route_pairs_preserved':a['routes']==b['routes']}
(out/'geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('before','after')}));print('before crossings',a['crossings']);print('after crossings',b['crossings']);assert result['nodes_preserved'] and result['primary_route_pairs_preserved'] and not b['crossings']
