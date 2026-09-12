"""Saved-data-only recovered baseline join; every canonical hash must match."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
from capture_contract import HERE,validate
from capture_preservation import json_delta,json_bytes
p=argparse.ArgumentParser();p.add_argument('--source-manifest-sha256',required=True);a=p.parse_args()
# Exact source/script/input brackets remain valid before accepting saved packets.
plan=json.loads((HERE/'plan.json').read_bytes())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def bracket_origins():
    current=validate('current37',a.source_manifest_sha256)
    baseline_now=validate('baseline',a.source_manifest_sha256)
    origin=plan['baseline_recovery_origin']
    old_here=Path(origin['harness'])
    if old_here!=old_here.resolve() or old_here!=HERE.parent/'current31-preservation-capture':
        raise ValueError('Original31 baseline harness address differs')
    if sha((old_here/'manifest.json').read_bytes())!=origin['manifest_sha256']:
        raise ValueError('Original31 baseline manifest differs')
    if sha((old_here/'capture_contract.py').read_bytes())!=origin['capture_contract_sha256']:
        raise ValueError('Original31 baseline contract differs')
    old_plan=json.loads((old_here/'plan.json').read_bytes())
    if old_plan['lanes']['baseline']!=plan['lanes']['baseline']:
        raise ValueError('Original31 baseline source/input/output address differs')
    spec=importlib.util.spec_from_file_location('preservation31_baseline_origin_contract',old_here/'capture_contract.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    baseline=module.validate('baseline',origin['source_manifest_sha256'])
    if baseline['frozen_inputs']!=baseline_now['frozen_inputs'] or baseline['head']!=baseline_now['head']:
        raise ValueError('Original31 and current37 baseline source/input pins differ')
    return {'current37':current,'baseline':baseline}
before=bracket_origins()
def load_packet(name):
    root=Path(plan['lanes'][name]['output'])
    pins_before=json.loads((root/'pins-before.json').read_bytes())
    pins_after=json.loads((root/'pins-after.json').read_bytes())
    if pins_before!=pins_after or pins_before!=before[name]:raise ValueError('Saved lane bracket differs')
    lane=json.loads((root/'result.json').read_bytes())
    if not lane['pins_equal'] or lane['before']!=lane['after'] or lane['artifacts_before']!=lane['artifacts_after']:
        raise ValueError('Saved lane source/artifact bracket failed')
    worker=root/'capture/controller'
    result=json.loads((worker/'capture-result.json').read_bytes())
    if not result['capture_complete']:raise ValueError('Saved observation incomplete')
    expected_raw=(worker/'expected.executed.json').read_bytes()
    if sha(expected_raw)!=plan['lanes'][name]['expected_manifest_sha256']:raise ValueError('Saved expectation changed')
    expected=json.loads(expected_raw)
    cases={}
    for case in result['cases']:
        slug=case['slug']
        if slug in cases:raise ValueError('Repeated recovered witness')
        if case['capture_errors'] or case['original_exception'] is not None:raise ValueError('Witness capture failed')
        artifact=case['actual_surfaces'];path=worker/artifact['path']
        if not path.resolve().is_relative_to(worker.resolve()):raise ValueError('Packet address escape')
        raw=path.read_bytes()
        if sha(raw)!=artifact['sha256'] or len(raw)!=artifact['bytes']:raise ValueError('Saved canonical docs changed')
        # This is exactly P._canon_bytes captured by the original function, no reconstruction.
        docs=json.loads(raw)
        for key,value in docs.items():
            actual=sha(json.dumps(value,sort_keys=True,default=str).encode()) if value is not None else None
            if actual!=case['surface_sha256'].get(key):raise ValueError('Saved surface hash mismatch')
        if case['candidate']['input_sha256']!=expected['witnesses'][slug]['input_sha256']:
            raise ValueError('Saved witness executable input changed')
        cases[slug]=(case,docs)
    if set(cases)!=set(expected['witnesses']) or len(cases)!=29:raise ValueError('Saved29 membership differs')
    return root,expected,cases
current_root,expected,current=load_packet('current37')
_,baseline_expected,baseline=load_packet('baseline')
if baseline_expected!=expected:raise ValueError('Expected baseline manifests differ')
out=current_root/'recovered-deltas';out.mkdir(exist_ok=False)
summary={}
for slug,row in expected['witnesses'].items():
    target=out/slug;target.mkdir()
    old,old_docs=baseline[slug];new,new_docs=current[slug];surfaces={}
    for surface,expected_sha in row['surfaces'].items():
        actual_before=old['surface_sha256'].get(surface);actual_after=new['surface_sha256'].get(surface)
        item={'expected_sha256':expected_sha,'recovered_sha256':actual_before,'current_sha256':actual_after,
              'recovery_matches_expected':expected_sha is not None and actual_before==expected_sha,
              'current_matches_expected':expected_sha is not None and actual_after==expected_sha}
        if item['recovery_matches_expected']:
            changes=json_delta(old_docs[surface],new_docs[surface])
            raw=json_bytes(changes);(target/(surface+'.json')).write_bytes(raw)
            item.update(delta_count=len(changes),delta_sha256=sha(raw),delta_path=slug+'/'+surface+'.json')
        else:item['failure']='RECOVERED_SURFACE_DOES_NOT_MATCH_COMMITTED_EXPECTATION'
        surfaces[surface]=item
    view_changes=json_delta(row['views'],new['views'])
    (target/'views.json').write_bytes(json_bytes(view_changes))
    summary[slug]={'surfaces':surfaces,'view_delta_count':len(view_changes),
                   'recovered_views_match_expected':old['views']==row['views']}
after=bracket_origins()
if before!=after:raise ValueError('Saved-data comparison bracket changed')
record={'witnesses':summary,'all_surfaces_recovered':all(v['recovery_matches_expected'] for row in summary.values() for v in row['surfaces'].values()),
 'authority':'Exact recovered canonical expected documents only. Full raw JSON/view deltas retained. No acceptance, blessing or value-cause inference.'}
(out/'summary.json').write_bytes(json_bytes(record));print(json.dumps({'all_surfaces_recovered':record['all_surfaces_recovered'],'witnesses':len(summary)}))
