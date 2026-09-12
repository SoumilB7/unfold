"""Actual saved diagnostic comparison; standard library only."""
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,gzip,difflib,re,collections
HERE=Path(__file__).resolve().parent
ROOTS=[Path('/private/tmp/unfold-s9a-'+g+'/layer-default-diagnostic') for g in ('thirtyfourth','thirtyfifth')]
KEY='decoder.attention.cross_attention_schedule'
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
pins={str(p):sha(p.read_bytes()) for root in ROOTS for p in sorted(root.rglob('*')) if p.is_file()}
for root in ROOTS:
 r=load(root/'result.json');assert r['passed'] and r['returncode']==0 and r['before']==r['after'] and r['artifacts_before']==r['artifacts_after']
summary={}
for case in ('ordinary','missing_decoder_count','explicit_null','explicit_zero'):
 out=HERE/case;out.mkdir(exist_ok=False);dirs=[r/'capture'/case for r in ROOTS];documents={};counts={}
 for name in sorted(set(p.name for d in dirs for p in d.iterdir())):
  paths=[d/name for d in dirs];assert all(p.is_file() for p in paths)
  if name.endswith('.json'):
   values=[load(p) for p in paths];documents[name]=values;changes=delta(*values);save(out/(name+'.deltas.json'),changes);counts[name]=len(changes)
 assert documents['input.json'][0]==documents['input.json'][1]
 assert documents['result.json'][0]==documents['result.json'][1]
 native=documents['native-facts.json'];assert native[0]==native[1]
 row={'json_delta_counts':counts,'result':documents['result.json'][1],'native_records_exactly_equal':True}
 if case in ('explicit_null','explicit_zero'):
  assert row['result']['error']['type']=='ConfigParseError' and row['result']['layers'] is None
  assert all(not (d/'page.html').exists() and not (d/'ir.json').exists() for d in dirs)
  row['limited_result_unchanged']=True
 else:
  irs=documents['ir.json'];all_changes=delta(*irs)
  assert len(irs[0]['layers'])==len(irs[1]['layers'])==24
  access_changes=[c for c in all_changes if c['path'][:3]==['extras','config_access','absent_default']]
  changes=[c for c in all_changes if c not in access_changes]
  aliases_before=irs[0]['extras']['config_access']['absent_default'];aliases_after=irs[1]['extras']['config_access']['absent_default']
  expected_aliases={'root:encoder_layers','root:n_blocks','root:n_layer','root:n_layers','root:num_blocks','root:num_layers'} if case=='missing_decoder_count' else set()
  assert set(aliases_after)-set(aliases_before)==expected_aliases and not set(aliases_before)-set(aliases_after)
  assert aliases_after==sorted(set(aliases_before)|expected_aliases)
  row['alias_probe_inventory_delta']={'added':sorted(expected_aliases),'removed':[], 'raw_position_deltas':access_changes, 'cause':'35 parser absent-count supported-alias probes record these missing reads; no supplied value/tier/proof changes.'}
  assert all(c['path'][0]=='layers' and c['path'][2]=='blocks' and c['path'][-1] in {'source_fact_keys','presentation_chips','presentation_path','presentation_aliases'} and 'added' in c for c in changes)
  assert all(irs[1]['layers'][c['path'][1]]['blocks'][c['path'][3]]['id']=='cross_attn' for c in changes)
  fact=native[1][KEY];assert fact['status']==('class_default' if case=='missing_decoder_count' else 'code_and_config')
  reference=fact['presentation_reference'];assert reference['claim_proof']['claim_kind']=='relation'
  for layer in irs[1]['layers']:
   blocks=[b for b in layer['blocks'] if b['id']=='cross_attn'];assert len(blocks)==1;block=blocks[0]
   assert block['source_fact_keys'].count(KEY)==1 and 'source_instance_path' not in block
   chips=block.get('presentation_chips',[])
   assert len(chips)==int(case=='missing_decoder_count')
   for chip in chips:assert chip['chip_kind']=='class_default' and chip['references']==[reference]
  pages=[(d/'page.html').read_text() for d in dirs];parsers=[]
  for page in pages:n=Nodes();n.feed(page);assert n.chip is None;parsers.append(n)
  assert all(n.payloads==0 for n in parsers),'MusicGen actual HTML is inline; a new packed location requires explicit audit'
  assert parsers[0].card_ids==parsers[1].card_ids
  chips=parsers[1].chips;assert not parsers[0].chips
  assert len(chips)==int(case=='missing_decoder_count')
  for chip in chips:
   assert chip['tag']=='span' and chip['kind']=='class_default' and chip['owner']=='decoder.attention' and chip['card']=='cross_attn'
   assert chip['text']=='class_default · cross_attention_schedule: '+json.dumps(fact['value'],ensure_ascii=False,sort_keys=True)
  mounts=[re.search(r'uf-[0-9a-f]{6,32}\b',page).group() for page in pages]
  paired=pages[0].replace(mounts[0],mounts[1]);a,b=paired.splitlines(True),pages[1].splitlines(True)
  hunks=[{'operation':tag,'before_lines':[i,j],'after_lines':[k,l],'before':''.join(a[i:j]),'after':''.join(b[k:l])} for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if tag!='equal']
  svgs=[re.findall(r'<svg\b.*?</svg>',p,re.S) for p in (paired,pages[1])];assert svgs[0]==svgs[1]
  if case=='ordinary':assert not hunks
  else:
   assert len(hunks)==1
   exact='<div class="uf-card-facts"><span class="uf-fact uf-chip-class_default" data-chip-kind="class_default" data-chip-owner="decoder.attention">'
   assert exact in hunks[0]['after'] and 'data-chip-kind=' not in hunks[0]['before']
  (out/'html.raw.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='actual34',tofile='actual35')).encode(),mtime=0));save(out/'html.exact-paired-mount-hunks.json',hunks)
  row.update(ir_metadata_addition_count=len(changes),metadata_field_counts=dict(collections.Counter(c['path'][-1] for c in changes)),actual_html_chips=chips,inline_card_ids=parsers[1].card_ids,deferred_payloads=0,svg_count=len(svgs[1]),all_svg_bytes_equal_after_exact_recorded_mount_pairing=True,html_bytes=[len(p.encode()) for p in pages],html_sha256=[sha(p.encode()) for p in pages],paired_mounts=mounts,paired_mount_comparison_authority='Finite raw cause accounting only; not an output acceptance normalization.',html_hunk_count=len(hunks))
 summary[case]=row
assert pins=={str(p):sha(p.read_bytes()) for root in ROOTS for p in sorted(root.rglob('*')) if p.is_file()}
save(HERE/'input-pins.json',pins);save(HERE/'verdict.json',{'verdict':'PASS_SAVED_MUSICGEN_DIAGNOSTIC_SCOPE','cases':summary,'scope':'Native values/status/proof snapshots and layers unchanged. Ordinary card citations add metadata only; sparse default adds one actual span in inline cross_attn card, with exact native reference and text, no new SVG/card. Null/zero remain limited. No CSS-string shortcut, browser visibility claim, new proof/occurrence inference, model/test runtime or blessing.','input_pins_sha256':sha((HERE/'input-pins.json').read_bytes())});save(HERE/'manifest.json',{'files':[{'path':p.relative_to(HERE).as_posix(),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file()]});print(json.dumps({'verdict_sha256':sha((HERE/'verdict.json').read_bytes()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes()),'cases':{k:{'native_equal':v['native_records_exactly_equal'],'IR_metadata_additions':v.get('ir_metadata_addition_count'),'HTML_chip_nodes':len(v.get('actual_html_chips',[])),'limited':v.get('limited_result_unchanged',False)} for k,v in summary.items()}},indent=2))
