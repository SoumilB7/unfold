"""Actual saved diagnostic comparison; standard library only."""
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,gzip,difflib,re,collections
HERE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def delta(a,b,path=()):
 if type(a)is not type(b):return [{'path':list(path),'before':a,'after':b}]
 if isinstance(a,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   if k not in a:out.append({'path':list(path+(k,)),'added':b[k]})
   elif k not in b:out.append({'path':list(path+(k,)),'removed':a[k]})
   else:out.extend(delta(a[k],b[k],path+(k,)))
  return out
 if isinstance(a,list):
  out=[]
  for i in range(max(len(a),len(b))):
   if i>=len(a):out.append({'path':list(path+(i,)),'added':b[i]})
   elif i>=len(b):out.append({'path':list(path+(i,)),'removed':a[i]})
   else:out.extend(delta(a[i],b[i],path+(i,)))
  return out
 return [] if a==b else [{'path':list(path),'before':a,'after':b}]
class Nodes(HTMLParser):
 def __init__(self):super().__init__();self.stack=[];self.chips=[];self.card_ids=[];self.payloads=0;self.chip=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if a.get('data-card-id'):self.card_ids.append(a['data-card-id'])
  if tag=='script' and 'data-uf-card-payload' in a:self.payloads+=1
  if 'data-chip-kind' in a:
   assert self.chip is None
   self.chip={'kind':a['data-chip-kind'],'owner':a.get('data-chip-owner'),'card':next((x[1].get('data-card-id') for x in reversed(self.stack) if x[1].get('data-card-id')),None),'tag':tag,'classes':a.get('class'),'text':'','depth':len(self.stack)}
  if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append((tag,a))
 def handle_endtag(self,tag):
  if self.chip is not None and tag==self.chip['tag'] and len(self.stack)==self.chip['depth']+1:self.chips.append(self.chip);self.chip=None
  for i in range(len(self.stack)-1,-1,-1):
   if self.stack[i][0]==tag:self.stack=self.stack[:i];break
 def handle_data(self,data):
  if self.chip is not None:self.chip['text']+=data
BASE=HERE.parents[1];RULE=(BASE/'typed-chip-wrap-proposal-r2/inserted-rule.txt').read_text();marker='/* Typed annotations remain distinguishable by borders and labels as well as colour. */'
roots=[Path('/private/tmp/unfold-s9a-'+g)/(k+'-default-diagnostic') for g in ('thirtysixth','thirtyseventh') for k in ('bloom','layer')];pins={str(p):sha(p.read_bytes()) for root in roots for p in sorted(root.rglob('*')) if p.is_file()}
for root in roots:
 r=load(root/'result.json');assert r['passed'] and r['before']==r['after'] and r['artifacts_before']==r['artifacts_after']
summary={}
for kind,cases in [('bloom',('ordinary','missing_root_count')),('layer',('ordinary','missing_decoder_count','explicit_null','explicit_zero'))]:
 for case in cases:
  ds=[Path('/private/tmp/unfold-s9a-'+g)/(kind+'-default-diagnostic/capture')/case for g in ('thirtysixth','thirtyseventh')];out=HERE/(kind+'-'+case);out.mkdir();counts={}
  for name in sorted({p.name for d in ds for p in d.iterdir() if p.suffix=='.json'}):
   values=[load(d/name) for d in ds];changes=delta(*values);save(out/(name+'.deltas.json'),changes);counts[name]=len(changes)
   if name in ['input.json','result.json','native-facts.json','ir.json']:assert not changes,(kind,case,name)
  result=load(ds[1]/'result.json');row={'result':result,'json_delta_counts':counts,'native_facts_IR_inputs_results_exact':True}
  if result['error'] is not None:
   assert case in ['explicit_null','explicit_zero'] and all(not(d/'page.html').exists() for d in ds);row['limited_unchanged']=True
  else:
   pages=[(d/'page.html').read_text() for d in ds];mounts=[re.search(r'#(uf-[a-f0-9]+) \.uf-fact \{',s).group(1) for s in pages];assert pages[0].count(marker)==pages[1].count(marker)==1
   insertion=RULE.replace('{mount_id}',mounts[1]).replace('{{','{').replace('}}','}').rstrip();paired=pages[0].replace(mounts[0],mounts[1]);expected=paired.replace(marker,marker+'\n'+insertion);assert expected==pages[1]
   nodes=[]
   for s in pages:p=Nodes();p.feed(s);nodes.append(p)
   assert nodes[0].chips==nodes[1].chips and nodes[0].card_ids==nodes[1].card_ids and nodes[0].payloads==nodes[1].payloads==0
   (out/'html.raw.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='actual36',tofile='actual37')).encode(),mtime=0));save(out/'exact-css-and-mount-delta.json',{'before_mount':mounts[0],'after_mount':mounts[1],'inserted_css':insertion,'scope':'Finite raw cause accounting; not an output acceptance normalization.'})
   row.update(actual_typed_chips=nodes[1].chips,card_ids=nodes[1].card_ids,html_bytes=[len(p.encode()) for p in pages],css_only_after_exact_recorded_mount_pair=True)
  summary[kind+'-'+case]=row
assert pins=={str(p):sha(p.read_bytes()) for root in roots for p in sorted(root.rglob('*')) if p.is_file()};save(HERE/'input-pins.json',pins);save(HERE/'verdict.json',{'verdict':'PASS_ACTUAL37_FINITE_DEFAULT_DIAGNOSTIC_SCOPE','cases':summary,'scope':'Actual root-produced37 HTML, not CSS-derived preview. All six inputs/results/native facts/IR exact36; four successful HTML pages differ only source-reviewed9-line typed CSS insertion plus exact random mount identity. Full chip text and card IDs unchanged. Null/zero remain limited. Browser visibility and actual37 fleet reviewed separately; no output blessing.'});save(HERE/'manifest.json',{'files':{str(p.relative_to(HERE)):sha(p.read_bytes()) for p in sorted(HERE.rglob('*')) if p.is_file()}});print('verdict',sha((HERE/'verdict.json').read_bytes()));print('manifest',sha((HERE/'manifest.json').read_bytes()))
