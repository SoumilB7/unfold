from pathlib import Path
from html.parser import HTMLParser
import json,hashlib
class Trace(HTMLParser):
 def __init__(self):super().__init__();self.stack=[];self.nodes=[];self.cards=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs);parent=self.stack[-1][1].copy() if self.stack else {};parent.update({k:a[k] for k in ('data-depth','data-card-id') if k in a});self.stack.append((tag,parent))
  if a.get('data-id')=='unet_context_0':self.nodes.append(parent)
  if a.get('data-card-id')=='unet_context_0':self.cards.append(parent)
  if tag in ('meta','br','hr','input','img','link'):self.stack.pop()
 def handle_startendtag(self,t,a):self.handle_starttag(t,a);self.handle_endtag(t)
 def handle_endtag(self,t):
  while self.stack:
   tag,_=self.stack.pop()
   if tag==t:break
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/final-f7baef5');out={}
for case in ('sdxl-ordinary','sd14-ordinary'):
 p=root/(case+'.html');t=Trace();t.feed(p.read_text());out[case]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'context_nodes':t.nodes,'context_cards':t.cards}
print(json.dumps(out,indent=2));Path('/private/tmp/unfold-s8-context-click-audit/results.json').write_text(json.dumps(out,indent=2)+'\n')
