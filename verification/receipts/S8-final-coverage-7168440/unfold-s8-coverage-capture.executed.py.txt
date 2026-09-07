"""Run the unchanged coverage entry point, retaining actual UNet render outputs."""
import gzip,hashlib,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv.pop(1));out=Path(sys.argv.pop(1));out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('actual_coverage',root/'scripts/coverage.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
from model_unfolder.diagram import Diagram
state={};real_inputs=g._inputs;real_html=Diagram.to_html

def save(name,value):
 p=out/state['slug']/name;p.parent.mkdir(parents=True,exist_ok=True)
 data=value.encode() if isinstance(value,str) else (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 p.write_bytes(gzip.compress(data,mtime=0))
def inputs():
 for item in real_inputs():
  state.clear();state['slug']=item[1].stem;state['capture']=item[1].stem in ('stable-diffusion-xl-base-1-0','sd-v1-4')
  yield item

def html(self,*args,**kwargs):
 rendered=real_html(self,*args,**kwargs)
 if state.get('capture') and state.setdefault('diagram_id',id(self))==id(self):
  save('page.html.gz',rendered);save('ir.json.gz',self.ir.to_dict());save('display-ir.json.gz',self.to_ir())
 return rendered
with patch.object(g,'_inputs',inputs),patch.object(Diagram,'to_html',html):
 raise SystemExit(g.main())
