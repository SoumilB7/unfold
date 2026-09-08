"""Persist actual release generator candidates; never replace reviewed examples."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,difflib,gzip,hashlib,shutil,json
from capture_common import begin,finish,failure,write,sha,load_module,save_preservation_input
from graph_capture import GraphCapture,surfaces,old_parallel

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--diagnostic-slugs',required=True);a=p.parse_args()
import capture_common
assert Path(capture_common.__file__).resolve()==Path(__file__).with_name('capture_common.py').resolve()
root,out,before,start=begin(a)
used_paths=[*Path(__file__).parent.glob('*.py'),Path(a.diagnostic_slugs),root/'test_support/preservation.py',root/'tests/preservation_expected_manifest.json',*(root/'tests/sable_test_corpus').glob('*.json')]
used_pins={str(p):sha(p) for p in used_paths};write(out/'used-input-script-pins.json',used_pins)
try:
    generator=load_module('s8_actual_example_generator',root/'scripts/generate_examples.py')
    helpers=load_module('s8_example_capture_helpers',root/'scripts/demonstrate_s8_unet.py')
    from model_unfolder.preview import svg_views,_visual_hash
    reviewed=root/'examples';candidate=out/'candidate';original=out/'reviewed';original.mkdir()
    baseline_pins={str(p.relative_to(reviewed)):sha(p) for p in sorted(reviewed.rglob('*')) if p.is_file()};write(out/'reviewed-before.json',baseline_pins)
    fixture_pins={str(generator.CORPUS/(name+'.json')):sha(generator.CORPUS/(name+'.json')) for name in generator.EXAMPLES.values()};write(out/'fixture-pins.json',fixture_pins)
    captured=[];actual_unfold=generator.unfold;observer=GraphCapture();diagnostic_slugs=set(json.loads(Path(a.diagnostic_slugs).read_text()));diagnostic_results=[]
    def capture_unfold(config):
        diagram=actual_unfold(config);slug=list(generator.EXAMPLES.values())[len(captured)];observer.phase='actual:'+slug;captured.append((copy.deepcopy(config),diagram));return diagram
    with observer,patch.object(generator,'unfold',capture_unfold):rows=generator._render_rows(candidate,rasterize_hero=False)
    # This manifest deliberately has no new PNG seal: review candidates are not blessings.
    write(candidate/'manifest.json',{'schema':2,'examples':rows});write(out/'candidate-status.json',{'status':'UNREVIEWED_CANDIDATE','rasterized_hero':False,'reviewed_outputs_untouched':True,'all_eight_generator_rows':'Six previously stale examples plus two unchanged controls; full manifest denominator retained.'})
    write(out/'existing_checker_findings.json',generator._validation_errors(reviewed,candidate,rows))
    assert len(captured)==len(rows)==len(generator.EXAMPLES)
    # Retain every actual parsed object before any per-output diagnostic can stop.
    for row,(config,diagram) in zip(rows,captured):
        details=out/Path(row['file']).stem;details.mkdir();write(details/'input.json',config);write(details/'ir.json',diagram.ir.to_dict());write(details/'render-input.json',helpers.render_input_record(diagram))
    write(out/'graphs-and-regions.json',observer.records)
    diffs=[]
    for row,(config,diagram) in zip(rows,captured):
        name=row['file'];old=reviewed/name;new=candidate/name;shutil.copy2(old,original/name)
        details=out/Path(name).stem;details.mkdir(exist_ok=True);write(details/'input.json',config);write(details/'ir.json',diagram.ir.to_dict());write(details/'render-input.json',helpers.render_input_record(diagram));save_preservation_input(details,diagram,root,generator.CORPUS/(generator.EXAMPLES[Path(name).stem]+'.json'))
        # Comparator pages use a fresh renderer namespace before actual rendering;
        # the actual generator's fixed public mount and provenance bytes stay untouched.
        from model_unfolder.diagram import Diagram
        slug=generator.EXAMPLES[Path(name).stem];canonical_diagram=Diagram(copy.deepcopy(diagram.ir));canonical_diagram._mount_id='uf-0000000000';assert canonical_diagram.to_ir()==diagram.to_ir()
        with observer:
            observer.phase='actual:'+slug;canonical_page=canonical_diagram.to_html();(details/'canonical-page.html').write_text(canonical_page)
            surfaces(canonical_diagram,canonical_page,root,slug,details)
            if slug in diagnostic_slugs:
                diagnostic=old_parallel(canonical_diagram,canonical_page,root,slug,details/'old-parallel-only',observer);diagnostic_results.append(diagnostic);write(out/'graphs-and-regions.json',observer.records);write(out/'parallel-diagnostics.json',diagnostic_results)
                assert diagnostic['old_locked_views_recovered'],'Old-only renderer did not recover expected locked views for '+slug
        oldtext=old.read_text();newtext=new.read_text();changed=oldtext!=newtext
        (details/'html.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile='reviewed/'+name,tofile='candidate/'+name,n=2)).encode(),mtime=0))
        signatures={label:[{'label':v,'hash':_visual_hash(svg)} for v,svg in svg_views(text)] for label,text in [('reviewed',oldtext),('candidate',newtext)]};write(details/'svg-signatures.json',signatures)
        diffs.append({'file':name,'changed':changed,'reviewed_sha256':sha(old),'candidate_sha256':sha(new),'reviewed_bytes':old.stat().st_size,'candidate_bytes':new.stat().st_size,'full_diff':str((details/'html.diff.gz').relative_to(out)),'svg_signatures_equal':signatures['reviewed']==signatures['candidate'],'cause_disposition':'requires actual per-output review; no blanket baseline-drift acceptance'})
    hero=generator.HERO+'.html';oldhero=generator._hero_svg_input((reviewed/hero).read_text());newhero=generator._hero_svg_input((candidate/hero).read_text());(out/'hero-reviewed-input.svg').write_bytes(oldhero);(out/'hero-candidate-input.svg').write_bytes(newhero)
    shutil.copy2(reviewed/'manifest.json',original/'manifest.json');shutil.copy2(reviewed/'images'/f'{generator.HERO}.png',out/'reviewed-hero.png')
    write(out/'per-file-deltas.json',diffs);write(out/'graphs-and-regions.json',observer.records);write(out/'parallel-diagnostics.json',diagnostic_results)
    assert baseline_pins=={str(p.relative_to(reviewed)):sha(p) for p in sorted(reviewed.rglob('*')) if p.is_file()}
    assert fixture_pins=={p:sha(p) for p in fixture_pins}
    finish(root,out,before,start,{'status':'CAPTURED_REVIEW_REQUIRED','rows':len(rows),'changed_files':[r['file'] for r in diffs if r['changed']],'hero_svg_input_equal':oldhero==newhero,'reviewed_hero_input_sha256':hashlib.sha256(oldhero).hexdigest(),'candidate_hero_input_sha256':hashlib.sha256(newhero).hexdigest(),'actual_generator':'scripts.generate_examples._render_rows(rasterize_hero=False)','rasterizer_not_run':True,'sable_surface':'not rerun; original comparator remains for final gate','canonical_counterpart':'actual same-IR render with valid generated mount before existing comparator normalization; public generator pages unchanged','baselines_untouched':True})
except BaseException as error:
    failure(out,error);raise

finally:
    from graph_capture import manifest
    after=manifest(root);write(out/'source-after-finally.json',after);actual_pins={p:sha(p) if Path(p).is_file() else None for p in used_pins};write(out/'pin-check-finally.json',{'source_fingerprints_identical':before==after,'input_script_fingerprints_identical':used_pins==actual_pins,'actual_input_script_pins':actual_pins});assert before==after and used_pins==actual_pins,'Capture source/input/script changed during execution'
