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
ROOT=Path('/private/tmp')
def root(g,k):return ROOT/('unfold-s9a-'+g)/(k+'-default-diagnostic')
roots=[root(g,k) for g,k in [('thirtyfourth','bloom'),('thirtyfourth','layer'),('thirtyfifth','layer'),('thirtysixth','bloom'),('thirtysixth','layer')]]
pins={str(p):sha(p.read_bytes()) for r in roots for p in sorted(r.rglob('*')) if p.is_file()}
for r in roots:
 packet=load(r/'result.json');assert packet['before']==packet['after'] and packet['artifacts_before']==packet['artifacts_after']
 if 'thirtysixth' in str(r):assert packet['passed'] and packet['returncode']==0
summary={}
for kind,prior,cases,key,card in [('layer','thirtyfifth',('ordinary','missing_decoder_count','explicit_null','explicit_zero'),'decoder.attention.cross_attention_schedule','cross_attn'),('bloom','thirtyfourth',('ordinary','missing_root_count'),'decoder.attention.position_schedule','attn')]:
 for case in cases:
  name=kind+'-'+case;out=HERE/name;out.mkdir();a,b=[root(g,kind)/'capture'/case for g in (prior,'thirtysixth')];results=[load(p/'result.json') for p in (a,b)];native=[load(p/'native-facts.json') for p in (a,b)]
  assert load(a/'input.json')==load(b/'input.json');ds={}
  for file in sorted({x.name for p in (a,b) for x in p.iterdir() if x.suffix=='.json'}):
   vals=[load(p/file) if (p/file).exists() else None for p in (a,b)];changes=delta(*vals);save(out/(file+'.deltas.json'),changes);ds[file]=len(changes)
  row={'before_generation':prior,'results':results,'full_json_delta_counts':ds,'native_equal':native[0]==native[1]}
  if case in ('explicit_null','explicit_zero'):
   assert native[0]==native[1] and results[0]==results[1] and results[1]['error']['type']=='ConfigParseError'
   assert all(not(p/'ir.json').exists() and not(p/'page.html').exists() for p in(a,b));row['limited_without_default_rescue']=True;summary[name]=row;continue
  ir=load(b/'ir.json');fact=native[1][key];default=case!='ordinary';expected=2 if kind=='bloom' and default else 70 if kind=='bloom' else 24
  assert len(ir['layers'])==len(fact['value'])==expected and fact['status']==('class_default' if default else 'code_and_config')
  ref=fact['presentation_reference'];assert ref['claim_proof']['claim_kind']=='relation'
  for layer in ir['layers']:
   block=next(v for v in layer['blocks'] if v['id']==card);assert block['source_fact_keys'].count(key)==1 and 'source_instance_path' not in block
   chips=block.get('presentation_chips',[]);assert len(chips)==int(default)
   for chip in chips:assert chip['chip_kind']=='class_default' and chip['references']==[ref]
  page=(b/'page.html').read_text();n=Nodes();n.feed(page);assert n.chip is None and n.payloads==0
  assert len(n.chips)==int(default)
  for chip in n.chips:assert chip['tag']=='span' and chip['kind']=='class_default' and chip['owner']=='decoder.attention' and chip['card']==card and chip['text']=='class_default · '+key.rsplit('.',1)[1]+': '+json.dumps(fact['value'],ensure_ascii=False,sort_keys=True)
  row.update(layers=expected,actual_html_chip_nodes=n.chips,inline_card_ids=n.card_ids,deferred_payloads=0,page_sha256=sha(page.encode()),page_bytes=len(page.encode()))
  if kind=='bloom' and default:
   assert results[0]['error']['type']=='ConfigParseError' and results[1]['error'] is None and key not in native[0]
   assert set(native[1])-set(native[0])=={key} and all(native[1][k]==v for k,v in native[0].items())
   prepared=load(b/'prepared-documents.json')['root'];assert prepared['failure'] is None and 'n_layer' not in prepared['checkpoint'] and prepared['overlay']['n_layer']==2
   row['cause']='Exact current root prepared class default n_layer=2, joined through supported alias in shared operand resolver; former34 root returned ConfigParseError. Two source-backed ALiBi schedule rows and existing-card class_default span now present.'
  else:
   assert native[0]==native[1] and results[0]==results[1]
   oldir=load(a/'ir.json');ird=delta(oldir,ir)
   if kind=='bloom':
    assert len(ird)==70 and all(x['path'][-1]=='source_fact_keys' and x.get('added')==[key] for x in ird);row['cause']='70 exact qualified position schedule references on existing attention cards; native proof/value unchanged.'
   elif default:
    oldabs=oldir['extras']['config_access']['absent_default'];newabs=ir['extras']['config_access']['absent_default'];removed={'root:encoder_layers','root:n_blocks','root:n_layer','root:n_layers','root:num_blocks','root:num_layers'}
    assert set(oldabs)-set(newabs)==removed and not(set(newabs)-set(oldabs));assert all(x['path'][:3]==['extras','config_access','absent_default'] for x in ird)
    reference34=load(root('thirtyfourth','layer')/'capture'/case/'ir.json')['extras']['config_access']['absent_default'];assert newabs==reference34
    row['alias_probe_removal']={'removed':sorted(removed),'added':[],'raw_position_deltas':len(ird),'inventory_exactly_equals34':True};row['cause']='36 shared resolver removes35 parser duplicate alias probes; native class-default proof/card annotations unchanged.'
   else:assert not ird;row['cause']='No native or IR delta.'
   oldpage=(a/'page.html').read_text();mounts=[re.search(r'uf-[0-9a-f]{6,32}\b',p).group() for p in (oldpage,page)];paired=oldpage.replace(mounts[0],mounts[1]);assert paired==page
   (out/'raw-html.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(oldpage.splitlines(True),page.splitlines(True),fromfile=prior,tofile='thirtysixth')).encode(),mtime=0));row['html_exact_recorded_mount_pair_only']={'before':mounts[0],'after':mounts[1],'scope':'Raw cause accounting only; not accepted output normalization.'}
  summary[name]=row
assert pins=={str(p):sha(p.read_bytes()) for r in roots for p in sorted(r.rglob('*')) if p.is_file()}
save(HERE/'input-pins.json',pins)
save(HERE/'verdict.json',{'verdict':'PASS_FINITE_SAVED_SPARSE_DEFAULT_SCOPE','cases':summary,'limitations':'Exact saved native proof summaries, IR, HTML nodes and producer records inspected; no browser layout/interaction measurement, new proof qualification, occurrence-placement inference, runtime or output approval. Original34 Bloom failed packet is preserved as the comparison limitation, not relabelled green. All raw JSON deltas retained including process-local document tokens.','input_pins_sha256':sha((HERE/'input-pins.json').read_bytes())})
save(HERE/'manifest.json',{'files':[{'path':str(p.relative_to(HERE)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file()]})
print(json.dumps({'verdict_sha256':sha((HERE/'verdict.json').read_bytes()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes()),'cases':{k:{'native_equal':v['native_equal'],'layers':v.get('layers'),'html_chips':len(v.get('actual_html_chip_nodes',[])),'limited':v.get('limited_without_default_rescue',False)} for k,v in summary.items()}},indent=2))
