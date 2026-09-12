"""Persist actual release generator candidates; never replace reviewed examples."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,difflib,gzip,hashlib,shutil
from capture_common import begin,finish,failure,write,sha,load_module,save_preservation_input

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root,out,before,start=begin(a)
try:
    generator=load_module('s8_actual_example_generator',root/'scripts/generate_examples.py')
    helpers=load_module('s8_example_capture_helpers',root/'scripts/demonstrate_s8_unet.py')
    from model_unfolder.preview import svg_views,_visual_hash
    reviewed=root/'examples';candidate=out/'candidate';original=out/'reviewed';original.mkdir()
    baseline_pins={str(p.relative_to(reviewed)):sha(p) for p in sorted(reviewed.rglob('*')) if p.is_file()};write(out/'reviewed-before.json',baseline_pins)
    fixture_pins={str(generator.CORPUS/(name+'.json')):sha(generator.CORPUS/(name+'.json')) for name in generator.EXAMPLES.values()};write(out/'fixture-pins.json',fixture_pins)
    captured=[];actual_unfold=generator.unfold
    def capture_unfold(config):
        diagram=actual_unfold(config);captured.append((copy.deepcopy(config),diagram));return diagram
    with patch.object(generator,'unfold',capture_unfold):rows=generator._render_rows(candidate,rasterize_hero=False)
    # This manifest deliberately has no new PNG seal: review candidates are not blessings.
    write(candidate/'manifest.json',{'schema':2,'examples':rows});write(out/'candidate-status.json',{'status':'UNREVIEWED_CANDIDATE','rasterized_hero':False,'reviewed_outputs_untouched':True,'all_eight_generator_rows':'Six previously stale examples plus two unchanged controls; full manifest denominator retained.'})
    write(out/'existing_checker_findings.json',generator._validation_errors(reviewed,candidate,rows))
    assert len(captured)==len(rows)==len(generator.EXAMPLES)
    diffs=[]
    for row,(config,diagram) in zip(rows,captured):
        name=row['file'];old=reviewed/name;new=candidate/name;shutil.copy2(old,original/name)
        details=out/Path(name).stem;details.mkdir();write(details/'input.json',config);write(details/'ir.json',diagram.ir.to_dict());write(details/'render-input.json',helpers.render_input_record(diagram));save_preservation_input(details,diagram,root,generator.CORPUS/(generator.EXAMPLES[Path(name).stem]+'.json'))
        oldtext=old.read_text();newtext=new.read_text();changed=oldtext!=newtext
        (details/'html.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile='reviewed/'+name,tofile='candidate/'+name,n=2)).encode(),mtime=0))
        signatures={label:[{'label':v,'hash':_visual_hash(svg)} for v,svg in svg_views(text)] for label,text in [('reviewed',oldtext),('candidate',newtext)]};write(details/'svg-signatures.json',signatures)
        diffs.append({'file':name,'changed':changed,'reviewed_sha256':sha(old),'candidate_sha256':sha(new),'reviewed_bytes':old.stat().st_size,'candidate_bytes':new.stat().st_size,'full_diff':str((details/'html.diff.gz').relative_to(out)),'svg_signatures_equal':signatures['reviewed']==signatures['candidate'],'cause_disposition':'requires actual per-output review; no blanket baseline-drift acceptance'})
    hero=generator.HERO+'.html';oldhero=generator._hero_svg_input((reviewed/hero).read_text());newhero=generator._hero_svg_input((candidate/hero).read_text());(out/'hero-reviewed-input.svg').write_bytes(oldhero);(out/'hero-candidate-input.svg').write_bytes(newhero)
    shutil.copy2(reviewed/'manifest.json',original/'manifest.json');shutil.copy2(reviewed/'images'/f'{generator.HERO}.png',out/'reviewed-hero.png')
    write(out/'per-file-deltas.json',diffs)
    assert baseline_pins=={str(p.relative_to(reviewed)):sha(p) for p in sorted(reviewed.rglob('*')) if p.is_file()}
    assert fixture_pins=={p:sha(p) for p in fixture_pins}
    finish(root,out,before,start,{'status':'CAPTURED_REVIEW_REQUIRED','rows':len(rows),'changed_files':[r['file'] for r in diffs if r['changed']],'hero_svg_input_equal':oldhero==newhero,'reviewed_hero_input_sha256':hashlib.sha256(oldhero).hexdigest(),'candidate_hero_input_sha256':hashlib.sha256(newhero).hexdigest(),'actual_generator':'scripts.generate_examples._render_rows(rasterize_hero=False)','rasterizer_not_run':True,'baselines_untouched':True})
except BaseException as error:
    failure(out,error);raise
