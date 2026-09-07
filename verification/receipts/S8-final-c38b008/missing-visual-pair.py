from pathlib import Path
import sys,json,hashlib
from html.parser import HTMLParser
sys.path.insert(0,'/private/tmp/unfold-s8-compiler-final')
from model_unfolder.preview import svg_views,svg_to_png
root=Path(__file__).parent;out=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/final-c38b008/render-phase/missing-pair');out.mkdir(exist_ok=True)
class Done(Exception):pass
class Card(HTMLParser):
 def __init__(self,text):super().__init__(convert_charrefs=False);self.text=text;self.depth=0;self.end=None;self.lines=[0]+[i+1 for i,c in enumerate(text) if c=='\n']
 def handle_starttag(self,tag,attrs):
  if tag=='div':self.depth+=1
 def handle_endtag(self,tag):
  if tag=='div':
   self.depth-=1
   if self.depth==0:
    line,col=self.getpos();self.end=self.text.index('>',self.lines[line-1]+col)+1;raise Done
manifest=[]
container='<none>';ffn='instance_down_blocks__1__attentions__0__transformer_blocks__0__ff'
for condition,ids in [('ordinary',[ffn]),('missing',[ffn])]:
 case=root/'sdxl'/condition;page=(case/'page.html').read_text();views=list(svg_views(page));obs=json.loads((case/'observation.json').read_text());records={}
 for cid in ids:
  svg=next(svg for label,svg in views if label==cid);stem=condition+('-container' if cid==container else '-new-ffn');(out/(stem+'.svg')).write_text(svg);svg_to_png(svg,str(out/(stem+'.png')),scale=1.0)
  start=page.rfind('<div',0,page.index('data-card-id="'+cid+'"'));parser=Card(page[start:])
  try:parser.feed(page[start:])
  except Done:pass
  assert parser.end;card=page[start:start+parser.end];(out/(stem+'-actual-card.html')).write_text(card)
  records[cid]=obs['page']['cards'][cid]
  manifest.append({'condition':condition,'card_id':cid,'svg':stem+'.svg','png':stem+'.png','actual_card_html':stem+'-actual-card.html','html_sha256':hashlib.sha256(page.encode()).hexdigest(),'card_facts':records[cid].get('facts'), 'card_text_scope':'Exact actual card HTML/text; numeric prose is not included by native SVG pipeline'})
 # Keep actual visible page lines surrounding banner numbers without inventing HTML.
 text=obs['page']['text'];lines=text.splitlines();banner=[line for line in lines if 'LAYERS' in line or 'PARAMS' in line or '2.57' in line or '2.66' in line];(out/(condition+'-banner-text.json')).write_text(json.dumps(banner,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Actual missing-evidence SVG pair</h1><p>Native SVG pixels; exact card HTML links retain quantity prose.</p>'+''.join('<h2>'+row['condition']+' '+row['card_id']+'</h2><img style="max-width:95%" src="'+row['png']+'"><p><a href="'+row['actual_card_html']+'">Exact actual card HTML</a></p>' for row in manifest))
print(json.dumps(manifest,indent=2))
