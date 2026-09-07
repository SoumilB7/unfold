"""Exact saved-case renderer migration; no model, reader execution, or blessing."""
from pathlib import Path
import argparse, copy, dataclasses, gzip, hashlib, json, subprocess, sys, time, traceback
from types import SimpleNamespace


def read(p):
    b=Path(p).read_bytes();return json.loads(gzip.decompress(b) if str(p).endswith('.gz') else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):
    if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
    if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [canonical(x) for x in v]
    if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
    return v
def walk(v):
    if isinstance(v,dict):
        if 'id' in v:yield v
        for x in v.values():yield from walk(x)
    elif isinstance(v,list):
        for x in v:yield from walk(x)
def manifest(root):
    return {str(p.relative_to(root)):sha(p) for folder in ('model_unfolder','physics') for p in sorted((root/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
def dump(p,v):Path(p).write_text(json.dumps(canonical(v),indent=2,sort_keys=True)+'\n')

def project(checkout,factcase,base):
    sys.path.insert(0,str(checkout))
    from model_unfolder.adapters.diffusor.unet_projection import project_unet
    facts={k:SimpleNamespace(value=v['value']) for k,v in read(factcase/'facts.json').items()}
    ir=read(base/'ir.json')
    return project_unet(facts=facts,handoffs={},name=ir['name'],architecture=ir['architecture']).to_dict()


def normalized_projection(v):
    v=copy.deepcopy(v);v.pop('construction_summary',None);v.pop('component_entry',None)
    render=v['extras']['render'];render.pop('component_scope',None);render.pop('component_input_ids',None)
    for b in walk(v):
        if b.get('kind')=='activation' and b.get('role')=='operation':
            b.pop('label',None);b.pop('title',None)
        detail=b.get('detail',{})
        detail.pop('fact_display_lines',None)
        if not detail:b.pop('detail',None)
    return v


def main(a):
    root=Path(a.checkout);base=Path(a.base_case);factcase=Path(a.fact_case);proofdir=Path(a.summary_proof_dir);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    before=manifest(root);dump(out/'source-before.json',before)
    pins={'script':sha(__file__),'base':{n:sha(base/n) for n in ('ir.json','render-input.json','page.html','render-events.json.gz')},'facts':{n:sha(factcase/n) for n in ('facts.json','input.json','inventory.json','qualified-facts.json')},'proofs':{p.name:sha(p) for p in sorted(proofdir.glob('*.json'))}}
    dump(out/'input-pins.json',pins)
    try:
        pr=read(proofdir/'result.json');proofs=read(proofdir/'proofs.json');summary=read(proofdir/'summary.json');savedfacts=read(factcase/'facts.json')
        assert pr['no_model_or_execution'] and pr['three_fact_values_exact']
        for name in ('facts','input','inventory'):assert pr[name+'_sha256']==pins['facts'][name+'.json'],name
        assert read(proofdir/'source-before.json')==read(proofdir/'source-after.json')
        assert all(before[k]==v for k,v in read(proofdir/'source-before.json').items())
        expected={'root.denoiser.constructed_modules':'existence','root.denoiser.constructed_parameter_shapes':'value','root.denoiser.constructed_stage_relations':'relation'}
        assert set(proofs)==set(expected)
        for key,kind in expected.items():
            assert proofs[key]['value']==savedfacts[key]['value'],key
            assert proofs[key]['summary']['claim_kind']==kind and proofs[key]['summary']['fact_id']==key
        assert summary==pr['summary']
        shapes=proofs['root.denoiser.constructed_parameter_shapes']['value'];stages=proofs['root.denoiser.constructed_stage_relations']['value'];population=proofs['root.denoiser.constructed_modules']['value']
        assert summary==dict(scope=shapes['scope'],parameter_count=shapes['total'],parameterized_module_count=shapes['parameterized_modules'],stage_count=sum(len(stages[k]) for k in ('producer_stages','intermediate_stages','consumer_stages')),shape_fact_key='root.denoiser.constructed_parameter_shapes',stage_relation_fact_key='root.denoiser.constructed_stage_relations',population_fact_key='root.denoiser.constructed_modules')
        assert set(shapes['by_module'])==set(population)
        pure={}
        for tag,checkout in (('old',a.old_checkout),('new',a.checkout)):
            p=out/(tag+'-pure-projection.json')
            with p.open('w') as stream:subprocess.run([sys.executable,__file__,'--project-only','--checkout',checkout,'--base-case',str(base),'--fact-case',str(factcase)],stdout=stream,check=True)
            pure[tag]=read(p)
        assert normalized_projection(pure['old'])==normalized_projection(pure['new']),'unexpected pure-projector change'
        oldblocks={b['id']:b for b in walk(pure['old'])};newblocks={b['id']:b for b in walk(pure['new'])};assert oldblocks.keys()==newblocks.keys()
        updates={};deltas=[]
        for bid,nb in newblocks.items():
            ob=oldblocks[bid];u={}
            for field in ('label','title'):
                if ob.get(field)!=nb.get(field):
                    assert nb.get('kind')=='activation' and nb.get('role')=='operation' and 'root.denoiser.ffn_mechanisms' in nb['source_fact_keys']
                    u[field]=nb[field];deltas.append({'id':bid,'field':field,'before':ob.get(field),'after':nb[field],'cause':'existing activation label vocabulary'})
            lines=nb.get('detail',{}).get('fact_display_lines')
            if lines!=ob.get('detail',{}).get('fact_display_lines'):
                assert lines and all(k in nb['source_fact_keys'] and values and all(line in nb.get('facts',[]) for line in values) for k,values in lines.items())
                u['fact_display_lines']=lines;deltas.append({'id':bid,'field':'detail.fact_display_lines','before':ob.get('detail',{}).get('fact_display_lines'),'after':lines,'cause':'receipt metadata for existing exact displayed fact lines'})
            if u:updates[bid]=u
        raw=read(base/'ir.json');capture=read(base/'render-input.json');changed=[]
        for b in walk(raw):
            if b['id'] not in updates:continue
            for field,value in updates[b['id']].items():
                if field=='fact_display_lines':
                    assert b.get('detail',{}).get(field)==oldblocks[b['id']].get('detail',{}).get(field)
                    assert all(k in b['source_fact_keys'] and all(line in b.get('facts',[]) for line in lines) for k,lines in value.items())
                    b.setdefault('detail',{})[field]=copy.deepcopy(value)
                else:
                    assert b.get(field)==oldblocks[b['id']].get(field),(b['id'],field)
                    b[field]=value
            changed.append(b['id'])
        assert set(changed)==set(updates),'pure projection delta absent from captured complete IR'
        render=raw['extras']['render'];oldscope=render.pop('component_scope',None);oldids=render.pop('component_input_ids',None)
        if oldscope is not None:
            assert oldscope=='denoiser';entry=pure['new']['component_entry'];assert entry['root_id']=='denoiser' and entry['input_ids']==oldids
            actualids={b['id'] for b in render['loop_blocks']};assert set(oldids)<=actualids
            raw['component_entry']=entry
        else:assert oldids is None and 'component_entry' not in raw
        raw['construction_summary']=summary
        # Reverse only the explicitly transported deltas; whole complete IR must remain exact.
        reverse=copy.deepcopy(raw);reverse.pop('construction_summary');reverse.pop('component_entry',None)
        if oldscope is not None:reverse['extras']['render'].update(component_scope=oldscope,component_input_ids=oldids)
        for b in walk(reverse):
            if b['id'] not in updates:continue
            for field in updates[b['id']]:
                if field=='fact_display_lines':
                    b['detail'].pop(field)
                    if not b['detail']:b.pop('detail')
                else:b[field]=oldblocks[b['id']][field]
        assert reverse==read(base/'ir.json'),'non-enumerated complete IR mutation'
        sys.path.insert(0,str(root))
        from model_unfolder.ir import ModelIR,EvidenceWarning
        from model_unfolder.diagram import Diagram
        from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
        from model_unfolder.lint import lint_labels
        from model_unfolder.sable import _projection_audit_findings
        from model_unfolder.preview import svg_views
        import model_unfolder.diagram as diagram_module
        assert Path(diagram_module.__file__).resolve().is_relative_to(root.resolve())
        ir=ModelIR(**copy.deepcopy(raw));ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']]
        assert ir.to_dict()==raw
        d=Diagram(ir);d._mount_id=capture['mount_id'];assert canonical(d.param_count())==capture['parameters']
        page=d.to_html();(out/'page.html').write_text(page)
        events=canonical(d.render_events());(out/'render-events.json.gz').write_bytes(gzip.compress((json.dumps(events,sort_keys=True,separators=(',',':'))+'\n').encode(),mtime=0))
        dump(out/'ir.json',ir.to_dict());dump(out/'render-input.json',{'ir':d.to_ir(),'parameters':canonical(d.param_count()),'warnings':capture['warnings'],'mount_id':capture['mount_id']})
        checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)}
        checks['validate_block_tree']=canonical(validate_block_tree(ir));checks['label_lint']=lint_labels(d.to_ir());checks['projection_audit']=_projection_audit_findings(d.to_ir(),d.render_events());dump(out/'checks.json',checks)
        prior=read(base/'render-events.json.gz');event_key=lambda e:json.dumps(e,sort_keys=True,separators=(',',':'))
        from collections import Counter
        oldcounter=Counter(map(event_key,prior));newcounter=Counter(map(event_key,events))
        eventdelta={'before_count':len(prior),'after_count':len(events),'removed':[json.loads(k) for k,n in (oldcounter-newcounter).items() for _ in range(n)],'added':[json.loads(k) for k,n in (newcounter-oldcounter).items() for _ in range(n)]};dump(out/'event-deltas.json',eventdelta)
        oldsvgs=list(svg_views((base/'page.html').read_text()));newsvgs=list(svg_views(page));svgdelta=[]
        assert len(oldsvgs)==len(newsvgs)
        for number,(old,new) in enumerate(zip(oldsvgs,newsvgs)):
            if old!=new:svgdelta.append({'index':number,'before_sha256':hashlib.sha256(str(old).encode()).hexdigest(),'after_sha256':hashlib.sha256(str(new).encode()).hexdigest()})
        dump(out/'ir-deltas.json',{'block_fields':deltas,'block_occurrences_changed':changed,'construction_summary':summary,'component_entry':raw.get('component_entry'),'removed_legacy_component_scope':oldscope,'removed_legacy_component_input_ids':oldids})
        after=manifest(root);dump(out/'source-after.json',after);assert before==after
        assert pins['base']=={n:sha(base/n) for n in pins['base']};assert pins['facts']=={n:sha(factcase/n) for n in pins['facts']};assert pins['proofs']=={n:sha(proofdir/n) for n in pins['proofs']}
        result={'status':'PASS' if not any(checks.values()) else 'FAIL','case':base.name,'checkout':str(root),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'base_case':str(base),'fact_case':str(factcase),'summary_proof_dir':str(proofdir),'elapsed_seconds':round(time.monotonic()-started,3),'html_sha256':sha(out/'page.html'),'prior_html_sha256':sha(base/'page.html'),'checks':checks,'ir_deltas':'ir-deltas.json','event_deltas':'event-deltas.json','svg_deltas':svgdelta,'quantities_exact':True,'canonical_facts_unchanged':True,'historical_model_source_run_preserved':True,'no_model_or_execution':True,'blessed':False}
        dump(out/'result.json',result);print(json.dumps(result),flush=True);assert result['status']=='PASS',checks
    except BaseException as error:
        dump(out/'failure.json',{'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc(),'elapsed_seconds':round(time.monotonic()-started,3)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--old-checkout',default='/private/tmp/unfold-s8-context-final-5227e9c');p.add_argument('--base-case',required=True);p.add_argument('--fact-case',required=True);p.add_argument('--summary-proof-dir');p.add_argument('--output');p.add_argument('--project-only',action='store_true');a=p.parse_args()
    if a.project_only:print(json.dumps(project(Path(a.checkout),Path(a.fact_case),Path(a.base_case)),sort_keys=True))
    else:main(a)
