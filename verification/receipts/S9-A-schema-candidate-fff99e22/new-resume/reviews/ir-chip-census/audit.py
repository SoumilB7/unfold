"""Saved JSON/HTML audit only. No project modules, model runtime or browser."""
from pathlib import Path
from html.parser import HTMLParser
import collections,gzip,hashlib,json,re

OUT=Path(__file__).resolve().parent
CAPTURE=Path('/private/tmp/unfold-s9a-twentysixth/current-pages/capture')
REASONS={'investigation_missing','structure_unaccounted','mechanism_unresolved'}
KINDS={'unresolved','pending_design','class_default','symbolic'}
META={'fact_provenance','config_access','config_audit','source_provenance','ship_findings','presentation_chips','presentation_unresolved_values','presentation_reference','unknown_reason','presentation_path','presentation_aliases'}

def sha(b):return hashlib.sha256(b).hexdigest()
def read(path):
 b=path.read_bytes();return b,json.loads(gzip.decompress(b) if path.suffix=='.gz' else b)
def stable(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def at(d,p):
 for x in p:d=d[x]
 return d

def walk(d,p=(),exclude=frozenset()):
 yield p,d
 if isinstance(d,dict):
  for k,v in d.items():
   if k not in exclude:yield from walk(v,(*p,k),exclude)
 elif isinstance(d,list):
  for i,v in enumerate(d):yield from walk(v,(*p,i),exclude)

def envelopes(d):
 def blocks(bs,p):
  if not isinstance(bs,list):return
  for i,b in enumerate(bs):
   if isinstance(b,dict):
    yield (*p,i),b
    yield from blocks(b.get('children'),(*p,i,'children'))
 for i,l in enumerate(d.get('layers') or []):
  yield from blocks(l.get('blocks'),('layers',i,'blocks'))
 render=(d.get('extras') or {}).get('render') or {}
 for k in ('model_blocks','loop_blocks'):yield from blocks(render.get(k),('extras','render',k))
 if isinstance(render.get('loop_region'),dict):yield ('extras','render','loop_region'),render['loop_region']

def spec_unknowns(d):
 result={}
 def literal(m,p,keys):
  if isinstance(m,dict):
   for k in keys:
    if m.get(k)=='unknown':result[(*p,k)]='literal_unknown'
 def null(m,p,keys):
  if isinstance(m,dict):
   for k in keys:
    if k in m and m[k] is None:result[(*p,k)]='null_unknown'
 literal(d,(),('embedding_norm_kind','final_norm_kind'));null(d,(),('hidden_size','tie_word_embeddings'))
 for i,l in enumerate(d.get('layers') or []):
  p=('layers',i);literal(l,p,('norm_kind','norm_placement','residual_topology'))
  for name in ('attention','cross_attention'):
   a=l.get(name);q=(*p,name)
   literal(a,q,('kind','mask','position_kind','position_application'))
   null(a,q,('kind','mask','qk_norm','cached','output_projection','bias','scores_scaled'))
   if isinstance(a,dict):
    if a.get('kind') in (None,'mha','gqa','mqa'):null(a,q,('projection_mode',))
    if a.get('cross_attention') is True:null(a,q,('cross_kv_source_kind',))
  f=l.get('ffn');q=(*p,'ffn');literal(f,q,('kind','activation'));null(f,q,('kind','activation','gated'))
  if isinstance(f,dict):
   if f.get('kind') in (None,'dense','moe'):null(f,q,('projection_mode',))
   if f.get('kind')=='moe':null(f,q,('expert_projection_mode',))
  if l.get('residual_topology') in ('parallel','fused_parallel'):null(l,p,('parallel_norm_count',))
 return result

def reason_issue(reason):
 if not isinstance(reason,dict) or set(reason)!={'reason_class','concrete_reason','investigation'}:return 'missing_or_nonclosed_reason'
 if reason['reason_class'] not in REASONS:return 'invalid_reason_class'
 if not isinstance(reason['concrete_reason'],str) or not reason['concrete_reason']:return 'missing_concrete_reason'
 if reason['reason_class']=='mechanism_unresolved':
  if not isinstance(reason['investigation'],dict):return 'class3_missing_investigation'
  return 'class3_live_reader_authority_not_replayable_from_json'
 if reason['investigation'] is not None:return 'class1_or_2_has_investigation'
 return None

def textpath(path):
 s=''
 for i,p in enumerate(path):
  if type(p) is int:s+='['+str(p)+']'
  elif p.isidentifier():s+=('.' if i else '')+p
  else:s+='['+json.dumps(p,ensure_ascii=True)+']'
 return s

class Page(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True);self.chips=collections.Counter();self.bad=[];self.groups=[];self.group=None;self.depth=0;self.read=None;self.text=[];self.styles=[];self.instyle=False
 def handle_starttag(self,tag,attrs):
  a=dict(attrs);classes=a.get('class','').split()
  if tag=='style':self.instyle=True
  if 'data-chip-kind' in a:
   k=a['data-chip-kind'];self.chips[k]+=1
   if k not in KINDS or 'uf-chip-'+k not in classes:self.bad.append('HTML chip kind/style mismatch')
   if k=='unresolved' and a.get('data-reason-class') not in REASONS:self.bad.append('HTML unresolved chip missing class')
   if k=='pending_design' and not a.get('data-decision-ref'):self.bad.append('HTML pending chip missing decision')
  if 'uf-unknown-value-group' in classes:
   self.group={'scope':'','suffixes':[]};self.depth=1
  elif self.group is not None and tag=='section':self.depth+=1
  if self.group is not None:
   if a.get('data-annotation-kind')=='unresolved-value-group':self.group.update(state=a.get('data-value-state'),reason_class=a.get('data-reason-class'));self.read='label';self.text=[]
   if 'uf-value-scope' in classes:self.read='scope';self.text=[]
   if 'uf-value-paths' in classes:self.read='suffixes';self.text=[]
 def handle_data(self,s):
  if self.instyle:self.styles.append(s)
  if self.read:self.text.append(s)
 def handle_endtag(self,tag):
  if tag=='style':self.instyle=False
  if self.group is not None:
   if self.read and ((self.read=='label' and tag=='span') or (self.read=='scope' and tag=='code') or (self.read=='suffixes' and tag=='pre')):
    value=''.join(self.text);self.group[self.read]=value.splitlines() if self.read=='suffixes' else value;self.read=None
   if tag=='section':
    self.depth-=1
    if self.depth==0:self.groups.append(self.group);self.group=None

def phase(d):
 findings=[];rows=[];reason_counts=collections.Counter();specs=spec_unknowns(d);annotations={};seen=set();envs=dict(envelopes(d));chip_occ=collections.Counter();unique={};outside=[]
 for row in (d.get('extras') or {}).get('presentation_unresolved_values',[]):
  path=tuple(row.get('path',()));issue=reason_issue(row.get('unknown_reason'))
  if path in seen:findings.append({'path':list(path),'issue':'duplicate_spec_annotation'})
  seen.add(path)
  if specs.get(path)!=row.get('value_state'):findings.append({'path':list(path),'issue':'annotation_not_an_expected_schema_unknown'})
  if issue:findings.append({'path':list(path),'issue':issue})
  else:reason_counts[row['unknown_reason']['reason_class']]+=1
  annotations[path]=(row.get('value_state'),row.get('unknown_reason'))
 for p in specs.keys()-seen:findings.append({'path':list(p),'issue':'schema_unknown_missing_reason_annotation'})
 unknown_envs=[]
 for p,v in envs.items():
  unknown=(v.get('resolved') is False or v.get('kind')=='unknown' or (v.get('status') in ('unknown','ambiguous','oracle_missing') and 'value' in v) or v.get('unknown_reason') is not None)
  if unknown:
   reason=v.get('unknown_reason');issue=reason_issue(reason)
   if issue:findings.append({'path':list(p),'issue':issue})
   else:reason_counts[reason['reason_class']]+=1
   unknown_envs.append(list(p));annotations[p]=('unresolved_envelope',reason)
 for p,v in walk(d):
  if not isinstance(v,dict):continue
  for i,c in enumerate(v.get('presentation_chips',[]) if isinstance(v.get('presentation_chips'),list) else []):
   k=c.get('chip_kind');chip_occ[k]+=1
   canonical=v.get('presentation_path',list(p));sig=stable([canonical,i,v.get('id'),c]);unique[sig]=c
   issue=None;refs=c.get('references',[])
   if k not in KINDS:issue='invalid_chip_kind'
   elif not refs:issue='chip_missing_references'
   elif k=='unresolved':issue=reason_issue(c.get('unknown_reason'))
   elif k=='class_default' and any(r.get('status')!='class_default' or r.get('claim_proof') is None for r in refs):issue='class_default_missing_qualified_default'
   elif k in ('pending_design','symbolic') and any(r.get('completeness')!='complete' or r.get('claim_proof') is None for r in refs):issue='pending_or_symbolic_missing_complete_proof'
   if k=='pending_design' and not c.get('decision_ref'):issue='pending_missing_decision'
   if k=='symbolic' and not c.get('parameters'):issue='symbolic_missing_parameters'
   if issue:findings.append({'path':list(p),'issue':issue})
   for r in refs:
    ledger=(d.get('extras') or {}).get('fact_provenance',{}).get(r.get('fact_key'),{})
    if ledger.get('presentation_reference')!=r:findings.append({'path':list(p),'issue':'chip_reference_differs_from_ledger'})
 for p,v in walk(d,exclude=META):
  if isinstance(v,dict) and p not in envs:
   markers=[]
   if v.get('resolved') is False:markers.append('resolved_false')
   if v.get('kind') in ('unknown','unresolved'):markers.append('kind_'+v['kind'])
   if v.get('status') in ('unknown','ambiguous','oracle_missing') and ('value' in v or 'id' in v):markers.append('status_'+v['status'])
   if markers:outside.append({'path':list(p),'markers':markers,'reason_issue':reason_issue(v.get('unknown_reason'))})
  elif v=='unknown' and p not in specs and (not p or p[-1] not in ('kind','status')):
   outside.append({'path':list(p),'markers':['literal_unknown_outside_declared_slots'],'reason_issue':'outside_finite_schema_annotation'})
 facts=(d.get('extras') or {}).get('fact_provenance',{});unknown_facts=[]
 for k,v in facts.items():
  if v.get('status') in ('unknown','ambiguous','oracle_missing'):
   issue=reason_issue(v.get('unknown_reason'));unknown_facts.append({'key':k,'reason_class':(v.get('unknown_reason') or {}).get('reason_class'),'issue':issue})
   if issue:findings.append({'fact_key':k,'issue':issue})
 class3=[]
 for p,v in walk(d):
  if isinstance(v,dict) and v.get('reason_class')=='mechanism_unresolved':class3.append({'path':list(p),'issue':reason_issue(v)})
 result={'schema_unknown_count':len(specs),'architectural_unknown_envelope_count':len(unknown_envs),'serialized_annotation_count':len(annotations),'annotation_reason_counts':dict(reason_counts),'unknown_fact_count':len(unknown_facts),'unknown_fact_reason_counts':dict(collections.Counter(r['reason_class'] for r in unknown_facts)),'chip_serialized_occurrences':dict(chip_occ),'chip_distinct_canonical_declarations':dict(collections.Counter(c.get('chip_kind') for c in unique.values())),'class3_records':class3,'findings':findings,'outside_finite_coverage_markers':outside}
 return result,annotations

summary={'scope':'saved source/JSON/HTML only; no live presentation_census event replay','models':[]}
for folder in sorted(CAPTURE.iterdir()):
 if not folder.is_dir():continue
 pins=[];reports={};annotations=None
 for name in ('ir-before-render.json.gz','ir-after-render.json.gz'):
  raw,d=read(folder/name);pins.append({'path':'capture/'+folder.name+'/'+name,'sha256':sha(raw)})
  report,ann=phase(d);reports[name]=report
  if name=='ir-after-render.json.gz':annotations=ann
 raw=(folder/'page.html.gz').read_bytes();pins.append({'path':'capture/'+folder.name+'/page.html.gz','sha256':sha(raw)})
 page=Page();html=gzip.decompress(raw).decode();page.feed(html)
 packed_chips=collections.Counter();payloads=[]
 for match in re.finditer(r'<script\b[^>]*data-uf-card-payload[^>]*>(.*?)</script>',html,re.S):
  payload=json.loads(match.group(1));assert payload['version']==1
  templates=[]
  for parts in payload['templates']:
   fragment=''.join(part if isinstance(part,str) else payload['svgs'][part] for part in parts)
   parser=Page();parser.feed(fragment);templates.append(parser.chips)
  ordered=[piece for piece in payload['canonical'] if type(piece) is int]
  assert ordered==list(range(len(payload['cards'])))
  placements=[i for depth,indices in payload['containers'] for i in indices]
  assert sorted(placements)==list(range(len(payload['cards'])))
  for template,binding,own in payload['cards']:packed_chips.update(templates[template])
  payloads.append({'card_count':len(payload['cards']),'template_count':len(templates),'canonical_and_container_card_indices_complete':True})
 total_chips=page.chips+packed_chips
 expected=set()
 for p,(state,reason) in annotations.items():
  if isinstance(reason,dict):expected.add((textpath(p),state,reason.get('reason_class')))
 actual=set()
 for g in page.groups:
  for suffix in g['suffixes']:
   p=g['scope'] if suffix=='(this value)' else g['scope']+suffix
   actual.add((p,g.get('state'),g.get('reason_class')))
 css=''.join(page.styles);styles={}
 for k in KINDS:
  m=re.search(r'\.uf-chip-'+k+r'\s*\{([^}]+)\}',css);styles[k]=m.group(1).strip() if m else None
 row={'slug':folder.name,'inputs':pins,'phases':reports,'html':{'chip_emissions_by_kind':dict(total_chips),'inline_chip_emissions':dict(page.chips),'packed_card_chip_emissions':dict(packed_chips),'packed_payloads':payloads,'unknown_group_count':len(page.groups),'unique_rendered_unknown_paths':len(actual),'missing_annotation_paths':sorted(expected-actual,key=repr),'unexpected_annotation_paths':sorted(actual-expected,key=repr),'findings':page.bad,'chip_style_rules':styles,'four_styles_present_and_distinct':all(styles.values()) and len(set(styles.values()))==4},'before_after_metadata_counts_equal':{k:reports['ir-before-render.json.gz'][k]==reports['ir-after-render.json.gz'][k] for k in ('schema_unknown_count','architectural_unknown_envelope_count','serialized_annotation_count','chip_distinct_canonical_declarations','unknown_fact_count')}}
 (OUT/(folder.name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 summary['models'].append({'slug':folder.name,'schema_unknowns':reports['ir-after-render.json.gz']['schema_unknown_count'],'envelope_unknowns':reports['ir-after-render.json.gz']['architectural_unknown_envelope_count'],'unknown_facts':reports['ir-after-render.json.gz']['unknown_fact_count'],'metadata_findings':len(reports['ir-after-render.json.gz']['findings']),'outside_markers':len(reports['ir-after-render.json.gz']['outside_finite_coverage_markers']),'missing_rendered_unknown_paths':len(expected-actual),'unexpected_rendered_unknown_paths':len(actual-expected),'chip_declarations':reports['ir-after-render.json.gz']['chip_distinct_canonical_declarations'],'html_chip_emissions':dict(total_chips),'class3_records':len(reports['ir-after-render.json.gz']['class3_records']),'four_styles_present_and_distinct':row['html']['four_styles_present_and_distinct']})
summary['totals']={k:sum(r[k] for r in summary['models']) for k in ('schema_unknowns','envelope_unknowns','unknown_facts','metadata_findings','outside_markers','missing_rendered_unknown_paths','unexpected_rendered_unknown_paths','class3_records')}
for k in ('chip_declarations','html_chip_emissions'):
 c=collections.Counter()
 for r in summary['models']:c.update(r[k])
 summary['totals'][k]=dict(c)
summary['totals']['models']=len(summary['models'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['totals'],indent=2))
