"""Saved JSON/HTML audit only. No project modules, model runtime or browser."""
from pathlib import Path
from html.parser import HTMLParser
import collections,gzip,hashlib,json,re

OUT=Path(__file__).resolve().parent

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

def base_envelopes(d):
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

def base_spec_unknowns(d):
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


def envelopes(d):
 yield from base_envelopes(d)
 opaque=((d.get('extras') or {}).get('render') or {}).get('opaque_layer_block')
 def blocks(b,p):
  if isinstance(b,dict):
   yield p,b
   for i,c in enumerate(b.get('children') or []):yield from blocks(c,(*p,'children',i))
 yield from blocks(opaque,('extras','render','opaque_layer_block'))

def spec_unknowns(d):
 result=base_spec_unknowns(d)
 def layer(value,p):
  if isinstance(value,dict):
   for q,state in base_spec_unknowns({'layers':[value]}).items():result[(*p,*q[2:])]=state
 def submodel(value,p):
  if not isinstance(value,dict):return
  for i,g in enumerate(value.get('groups') or []):layer(g,(*p,'groups',i))
  for i,g in enumerate(value.get('sub_models') or []):submodel(g,(*p,'sub_models',i))
 for p,b in envelopes(d):
  detail=b.get('detail')
  if not isinstance(detail,dict):continue
  layer({k:detail[k] for k in ('attention','cross_attention','ffn') if k in detail},(*p,'detail'))
  submodel(detail.get('sub_model'),(*p,'detail','sub_model'))
 modalities=(d.get('extras') or {}).get('modalities')
 if isinstance(modalities,dict):
  inputs=modalities.get('inputs') or {}
  for name in ('vision','video','audio','conditioning'):
   value=inputs.get(name)
   if not isinstance(value,dict):continue
   for i,stage in enumerate(value.get('pipeline') or []):
    if isinstance(stage,dict) and stage.get('operation')=='unknown':result[('extras','modalities','inputs',name,'pipeline',i,'operation')]='literal_unknown'
  fusion=modalities.get('fusion') or {}
  for key in ('operation','target'):
   if fusion.get(key)=='unknown':result[('extras','modalities','fusion',key)]='literal_unknown'
 return result

class CountPage(Page):
 def __init__(self):
  super().__init__();self.nodes=0;self.cards=0;self.banner_markers=0;self.receipt_nodes=collections.Counter()
 def handle_starttag(self,tag,attrs):
  super().handle_starttag(tag,attrs);a=dict(attrs);self.nodes+=1
  self.cards+=('uf-card-detail' in a.get('class','').split())
  if a.get('data-uf-receipt-node'):
   self.receipt_nodes[a['data-uf-receipt-node']]+=1
   self.banner_markers+=(a['data-uf-receipt-node']=='stats_hidden_size')
 def handle_startendtag(self,tag,attrs):self.handle_starttag(tag,attrs)

def html_audit(raw,annotations):
 page=CountPage();html=gzip.decompress(raw).decode();page.feed(html)
 packed=collections.Counter();payloads=[];deferred_nodes=0;deferred_cards=0
 for match in re.finditer(r'<script\b[^>]*data-uf-card-payload[^>]*>(.*?)</script>',html,re.S):
  payload=json.loads(match.group(1));assert payload['version']==1
  templates=[]
  for parts in payload['templates']:
   fragment=''.join(part if isinstance(part,str) else payload['svgs'][part] for part in parts)
   parser=CountPage();parser.feed(fragment);templates.append(parser)
  ordered=[piece for piece in payload['canonical'] if type(piece) is int]
  assert ordered==list(range(len(payload['cards'])))
  placements=[i for depth,indices in payload['containers'] for i in indices]
  assert sorted(placements)==list(range(len(payload['cards'])))
  for template,binding,own in payload['cards']:
   packed.update(templates[template].chips);deferred_nodes+=templates[template].nodes;deferred_cards+=templates[template].cards
  payloads.append({'card_count':len(payload['cards']),'template_count':len(templates),'json_bytes':len(match.group(1).encode()),'canonical_and_container_card_indices_complete':True})
 expected={(textpath(p),state,reason.get('reason_class')) for p,(state,reason) in annotations.items() if isinstance(reason,dict)}
 actual={(g['scope'] if suffix=='(this value)' else g['scope']+suffix,g.get('state'),g.get('reason_class')) for g in page.groups for suffix in g['suffixes']}
 css=''.join(page.styles);styles={}
 for k in KINDS:
  match=re.search(r'\.uf-chip-'+k+r'\s*\{([^}]+)\}',css);styles[k]=match.group(1).strip() if match else None
 return {'html_bytes':len(html.encode()),'gzip_bytes':len(raw),'serialized_element_count':page.nodes,'inline_detail_cards':page.cards,'deferred_template_expanded_element_count':deferred_nodes,'deferred_template_expanded_detail_cards':deferred_cards,'packed_payloads':payloads,'inline_chips':dict(page.chips),'packed_chips':dict(packed),'all_chips':dict(page.chips+packed),'banner_scalar_markers':page.banner_markers,'receipt_node_counts':dict(page.receipt_nodes),'unknown_groups':len(page.groups),'rendered_unique_unknown_paths':len(actual),'missing_unknown_paths':sorted(expected-actual,key=repr),'unexpected_unknown_paths':sorted(actual-expected,key=repr),'four_styles_present_and_distinct':all(styles.values()) and len(set(styles.values()))==4,'styles':styles,'findings':page.bad,'accounting_limit':'HTMLParser serialized elements and template multiplicity; no JS execution, DOM measurement, browser visibility or click proof.'}

def native_counts(rows):
 return {'facts':len(rows),'status':dict(collections.Counter(v.get('status') for v in rows.values())),'claim_kind':dict(collections.Counter(v.get('claim_kind') for v in rows.values())),'qualified':sum(v.get('claim_proof') is not None for v in rows.values()),'qualified_kinds':dict(collections.Counter(v.get('claim_kind') for v in rows.values() if v.get('claim_proof') is not None)),'without_proof':sum(v.get('claim_proof') is None for v in rows.values())}

def native_delta(old,new):
 rows=[]
 for key in sorted(old.keys() | new.keys()):
  if key not in old or key not in new:rows.append({'key':key,'membership':'added' if key in new else 'removed','before':old.get(key),'after':new.get(key)});continue
  fields={field:{'before':old[key].get(field),'after':new[key].get(field)} for field in ('value','status','completeness','claim_kind','claim_proof','claim_readers','unknown_reason') if stable(old[key].get(field))!=stable(new[key].get(field))}
  if fields:rows.append({'key':key,'fields':fields})
 return rows

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--source-manifest',type=Path,required=True);ap.add_argument('--source-manifest-sha256',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 assert sha(args.source_manifest.read_bytes())==args.source_manifest_sha256
 assert not args.output.exists(),'new output directory required; previous evidence never overwritten'
 plan=json.loads(args.plan.read_text());slugs=[t['slug'] for t in plan['targets']]
 actual=sorted(p.name for p in args.capture.iterdir() if p.is_dir())
 assert len(slugs)==39 and sorted(slugs)==actual,'all39 exact capture membership required'
 result=json.loads((args.capture/'result.json').read_text())
 args.output.mkdir(parents=True)
 summary={'source_manifest_sha256':args.source_manifest_sha256,'plan_sha256':sha(args.plan.read_bytes()),'capture_result_sha256':sha((args.capture/'result.json').read_bytes()),'models':[],'scope':'Saved JSON/HTML only; portable proofs are display summaries, not reconstructed live authority. No normalization or blessing. Scalar banner receipts never prove occurrence placement; actual30 matrix join remains required.'}
 for target in plan['targets']:
  slug=target['slug'];folder=args.capture/slug;reference=args.reference/slug
  pins=[{'path':p.name,'sha256':sha(p.read_bytes())} for p in sorted(folder.iterdir()) if p.is_file()]
  assert not (folder/'failure.txt').exists(),f'{slug}: capture failure is blocking'
  address=json.loads((folder/'address-evidence.json').read_text())
  assert address['input_sha256']==target['input_sha256'] and address['resolved_class']==target['resolved_class']
  native=json.loads((folder/'declarations.json').read_text());old=json.loads((reference/'declarations.json').read_text())
  phases={};all_annotations={}
  for name in ('ir-before-render.json.gz','ir-after-render.json.gz','diagram-ir-before-render.json.gz','diagram-ir-after-render.json.gz'):
   raw,document=read(folder/name);phases[name],all_annotations[name]=phase(document)
  html=html_audit((folder/'page.html.gz').read_bytes(),all_annotations['diagram-ir-after-render.json.gz'])
  _,old_ir=read(reference/'ir-after-render.json.gz');old_phase,old_annotations=phase(old_ir)
  old_html=html_audit((reference/'page.html.gz').read_bytes(),old_annotations)
  census=json.loads((folder/'presentation-census.json').read_text());events=json.loads((folder/'actual-emission-counts.json').read_text())
  row={'slug':slug,'inputs':pins,'native_before':native_counts(old),'native_after':native_counts(native),'native_deltas':native_delta(old,native),'phases':phases,'html':html,'html_reference26':old_html,'live_capture_census':census,'live_capture_event_counts':events,'reference26_schema_counts':{k:old_phase[k] for k in ('schema_unknown_count','architectural_unknown_envelope_count','serialized_annotation_count')},'raw_before_after_render_equal':{prefix:(folder/(prefix+'before-render.json.gz')).read_bytes()==(folder/(prefix+'after-render.json.gz')).read_bytes() for prefix in ('ir-','diagram-ir-')}}
  (args.output/(slug+'.json')).write_text(json.dumps(row,indent=2)+'\n')
  summary['models'].append({'slug':slug,'native':row['native_after'],'native_delta_keys':len(row['native_deltas']),'unknown_scalar_addresses':phases['diagram-ir-after-render.json.gz']['schema_unknown_count'],'unknown_envelopes':phases['diagram-ir-after-render.json.gz']['architectural_unknown_envelope_count'],'metadata_findings':sum(len(v['findings']) for v in phases.values()),'html_bytes':html['html_bytes'],'element_count':html['serialized_element_count'],'missing_rendered_paths':len(html['missing_unknown_paths']),'unexpected_rendered_paths':len(html['unexpected_unknown_paths']),'four_styles_distinct':html['four_styles_present_and_distinct'],'class3_records':len(phases['diagram-ir-after-render.json.gz']['class3_records']),'capture_census_findings':len(census.get('findings',[]))})
 assert sha(args.source_manifest.read_bytes())==args.source_manifest_sha256
 summary['totals']={k:sum(row[k] for row in summary['models']) for k in ('native_delta_keys','unknown_scalar_addresses','unknown_envelopes','metadata_findings','html_bytes','element_count','missing_rendered_paths','unexpected_rendered_paths','class3_records','capture_census_findings')}
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['totals'],indent=2))

if __name__=='__main__':main()
