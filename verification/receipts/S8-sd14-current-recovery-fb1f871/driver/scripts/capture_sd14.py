"""One actual Sable parse for a selected frozen preservation input."""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,sys,time,traceback
from graph_capture import GraphCapture,write,manifest,sha,old_parallel
p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--slug',required=True);p.add_argument('--output',required=True);p.add_argument('--parallel-diagnostic',action='store_true');a=p.parse_args();root=Path(a.checkout).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);before=manifest(root);write(out/'source-before.json',before);assert before==json.loads(Path(a.source_manifest).read_text());sys.path.insert(0,str(root));start=time.monotonic();pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('graph_capture.py'),Path(__file__).with_name('capture_common.py'),root/'tests/unseen_model_configs'/f'{a.slug}.json',root/'test_support/preservation.py',root/'tests/preservation_expected_manifest.json')};write(out/'input-script-pins.json',pins)
try:
 import capture_common
 assert Path(capture_common.__file__).resolve()==Path(__file__).with_name('capture_common.py').resolve()
 from capture_common import load_module,save_diagram,save_preservation_input
 from model_unfolder.diagram import Diagram
 from model_unfolder.sable import sable
 from test_support import preservation
 helpers=load_module('preservation_capture_helpers',root/'scripts/demonstrate_s8_unet.py');fixture_path=root/'tests/unseen_model_configs'/f'{a.slug}.json';fixture=json.loads(fixture_path.read_text());assert a.slug=='sd-v1-4';expected=None;write(out/'fixture.json',fixture)
 captured=[];original=Diagram.to_html
 def capture_html(self,standalone=True):
  page=original(self,standalone=standalone)
  if standalone:captured.append((self,page))
  return page
 with GraphCapture() as observer:
  observer.phase='actual:'+a.slug
  with patch.object(Diagram,'to_html',capture_html):report=sable(fixture['config'],source=fixture.get('source','local'),render_images=False)
  assert captured and len({id(d.ir) for d,_ in captured})==1,'Expected one actual parsed IR'
  diagram,page=captured[-1];actual=out/'actual';actual.mkdir();saved,checks=save_diagram(actual,diagram,helpers);assert saved==page;save_preservation_input(actual,diagram,root,fixture_path);write(out/'sable-report.json',report)
  docs=out/'surface-docs';docs.mkdir();surface_hashes={}
  sable_doc={'mechanical_passed':report.mechanical_passed,'checks':[{'name':c.name,'blocking':c.blocking,'passed':c.passed,'findings':list(c.findings or [])} for c in report.checks]};write(docs/'sable.json',sable_doc);surface_hashes['sable']={'actual':hashlib.sha256(preservation._canon_bytes(sable_doc)).hexdigest(),'expected':None};write(docs/'surface-hashes.json',surface_hashes)
  diagnostic=old_parallel(diagram,page,root,a.slug,out/'old-parallel-only',observer) if a.parallel_diagnostic else None
  write(out/'graphs-and-regions.json',observer.records)
 after=manifest(root);write(out/'source-after.json',after);assert before==after;assert pins=={p:sha(p) for p in pins}
 result={'status':'CAPTURED_REVIEW_REQUIRED','slug':a.slug,'one_actual_parsed_ir':True,'checks':checks,'surface_hashes':surface_hashes,'old_parallel_result':diagnostic,'elapsed_seconds':round(time.monotonic()-start,3),'no_baseline_write':True,'blessed':False};write(out/'result.json',result);print(json.dumps({'slug':a.slug,'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
 if diagnostic:assert diagnostic['old_locked_views_recovered'],'Old-only renderer did not recover expected locked views; stop cause attribution'
except BaseException as e:
 write(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise

finally:
 after=manifest(root);write(out/'source-after-finally.json',after);actual_pins={p:sha(p) if Path(p).is_file() else None for p in pins};write(out/'pin-check-finally.json',{'source_fingerprints_identical':before==after,'input_script_fingerprints_identical':pins==actual_pins,'actual_input_script_pins':actual_pins});assert before==after and pins==actual_pins,'Capture source/input/script changed during execution'
