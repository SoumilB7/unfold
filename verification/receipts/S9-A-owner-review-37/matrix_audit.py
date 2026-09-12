"""Owner's independent matrix audit: committed verification/s7 (HEAD) vs candidate37."""
import json,gzip,glob,os,sys,collections
C='/private/tmp/unfold-s9a-thirtyseventh/matrix/'
H='verification/s7/'
out={'models':{}, 'totals':collections.Counter()}
def occ_id(o):
    pv=o['provenance']; 
    return json.dumps(pv.get('instance_path') or pv, sort_keys=True)[:400]
tot=out['totals']
for f in sorted(glob.glob(H+'models/*.json.gz')):
    name=os.path.basename(f)
    cf=C+'models/'+name
    if not os.path.exists(cf): out['models'][name]={'error':'missing in candidate'}; continue
    a=json.load(gzip.open(f)); b=json.load(gzip.open(cf))
    A=a['table']['occurrences']; B=b['table']['occurrences']
    r={'occ_head':len(A),'occ_cand':len(B)}
    tot['occ_head']+=len(A); tot['occ_cand']+=len(B)
    r['identity_order_equal']= [occ_id(o) for o in A]==[occ_id(o) for o in B]
    r['construction_equal']=sum(1 for x,y in zip(A,B) if x['construction']==y['construction'])==len(A)
    r['execution_equal']=sum(1 for x,y in zip(A,B) if x['execution']==y['execution'])==len(A)
    r['provenance_equal']=sum(1 for x,y in zip(A,B) if x['provenance']==y['provenance'])==len(A)
    pk=('kind','parent','rule','block_ids','reason','reason_class')
    diffs=collections.Counter()
    for x,y in zip(A,B):
        for k in pk:
            if x['projection'].get(k)!=y['projection'].get(k): diffs[k]+=1
    r['projection_field_diffs']=dict(diffs)
    # citations
    removed=0; head_unq=0; still_unq=0; new_unq=0; newly_cited=0; kind_changed=0; head_cited=0
    for x,y in zip(A,B):
        px,py=x['projection'],y['projection']
        fx=set(px.get('fact_keys') or []); fy=set(py.get('fact_keys') or [])
        head_cited+=len(fx)
        removed+=len(fx-fy); newly_cited+=len(fy-fx)
        ux=set(px.get('unqualified_fact_keys') or []); uy=set(py.get('unqualified_fact_keys') or [])
        head_unq+=len(ux); still_unq+=len(ux&uy); new_unq+=len(uy-ux)
        kx=px.get('fact_claim_kinds') or {}; ky=py.get('fact_claim_kinds') or {}
        if isinstance(kx,dict) and isinstance(ky,dict):
            for k in kx:
                if k in ky and kx[k]!=ky[k]: kind_changed+=1
    r.update(dict(head_cited=head_cited,removed_citations=removed,newly_cited=newly_cited,head_unqualified=head_unq,head_unqualified_still=still_unq,new_unqualified=new_unq,claim_kind_changed=kind_changed))
    for k in ('head_cited','removed_citations','newly_cited','head_unqualified','head_unqualified_still','new_unqualified','claim_kind_changed'): tot[k]+=r[k]
    r['relations_head']=len(a['table'].get('relations') or []); r['relations_cand']=len(b['table'].get('relations') or [])
    r['relations_equal']=a['table'].get('relations')==b['table'].get('relations')
    r['blocking_head']=a.get('blocking_findings'); r['blocking_cand']=b.get('blocking_findings')
    r['schedules_equal']=a.get('construction_schedules')==b.get('construction_schedules')
    out['models'][name]=r
out['totals']=dict(tot)
json.dump(out,open(sys.argv[1],'w'),indent=1)
print(json.dumps(out['totals'],indent=1))
bad=[n for n,r in out['models'].items() if r.get('error') or not r.get('identity_order_equal') or not r.get('construction_equal') or not r.get('execution_equal') or r.get('projection_field_diffs') or r.get('removed_citations') or not r.get('relations_equal') or not r.get('schedules_equal')]
print('models needing a look:',bad)
for n in bad[:12]:
    print(n, {k:v for k,v in out['models'][n].items() if k in ('identity_order_equal','construction_equal','execution_equal','provenance_equal','projection_field_diffs','removed_citations','relations_head','relations_cand','relations_equal','schedules_equal','blocking_head','blocking_cand')})
