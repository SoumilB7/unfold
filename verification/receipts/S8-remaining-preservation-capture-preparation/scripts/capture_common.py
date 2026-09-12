"""Capture-only helpers; source pinning never grants permission to run a lane."""
from pathlib import Path
import dataclasses,gzip,hashlib,importlib.util,json,subprocess,sys,time,traceback


def canonical(value):
    if dataclasses.is_dataclass(value):return canonical(dataclasses.asdict(value))
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    if isinstance(value,(set,frozenset)):return sorted(canonical(v) for v in value)
    if isinstance(value,Path):return str(value)
    return value

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(canonical(value),indent=2,sort_keys=True)+'\n')
def source_manifest(root):
    return {str(p.relative_to(root)):sha(p) for d in ('model_unfolder','physics','scripts') for p in sorted((root/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}

def begin(args):
    root=Path(args.checkout).resolve();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before=source_manifest(root)
    assert before==json.loads(Path(args.source_manifest).read_text()),'candidate differs from separately pinned final source manifest'
    write(out/'source-before.json',before)
    write(out/'invocation.json',{'argv':sys.argv,'checkout':str(root),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'scripts':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'blessed':False})
    sys.path.insert(0,str(root));return root,out,before,time.monotonic()

def finish(root,out,before,start,result):
    after=source_manifest(root);write(out/'source-after.json',after);assert before==after
    result.update(elapsed_seconds=round(time.monotonic()-start,3),blessed=False)
    write(out/'result.json',result);print(json.dumps(canonical(result)),flush=True)

def failure(out,error):write(out/'failure.json',{'kind':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
def load_module(name,path):
    if str(Path(path).parent) not in sys.path:sys.path.insert(0,str(Path(path).parent))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def save_diagram(out,diagram,helpers):
    write(out/'ir.json',diagram.ir.to_dict());write(out/'render-input.json',helpers.render_input_record(diagram))
    page=diagram.to_html();(out/'page.html').write_text(page)
    (out/'render-events.json.gz').write_bytes(gzip.compress((json.dumps(canonical(diagram.render_events()),sort_keys=True,separators=(',',':'))+'\n').encode(),mtime=0))
    from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
    from model_unfolder.lint import lint_labels
    from model_unfolder.sable import _projection_audit_findings
    checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)}
    checks.update(validate_block_tree=canonical(validate_block_tree(diagram.ir)),label_lint=lint_labels(diagram.to_ir()),projection_audit=_projection_audit_findings(diagram.to_ir(),diagram.render_events()))
    write(out/'checks.json',checks);return page,checks


def save_preservation_input(out,diagram,root,fixture_path):
    """Retain the existing comparator's exact sidecar from this actual parse."""
    from test_support import preservation
    assert Path(preservation.__file__).resolve().is_relative_to(root)
    fixture=json.loads(fixture_path.read_text())
    display=diagram.to_ir();_,sidecar=preservation.split_structural_ir(display)
    payload=preservation._canon_bytes(sidecar)
    (out/'display-ir.json.gz').write_bytes(gzip.compress(preservation._canon_bytes(display),mtime=0))
    (out/'ledgers.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    expected=json.loads((root/'tests/preservation_expected_manifest.json').read_text())['witnesses'].get(fixture_path.stem)
    write(out/'preservation-input.json',{'fixture':str(fixture_path),'input_sha256':preservation.input_sha256(fixture_path),'fixture_source':fixture.get('source','local'),'comparator_sha256':sha(preservation.__file__),'comparator_module':preservation.__file__,'actual_ledger_sha256':hashlib.sha256(payload).hexdigest(),'expected':expected,'comparison_scope':'Exact actual Diagram.to_ir and existing comparator sidecar. Expected-hash recovery is separately required for historical cause attribution; this capture grants no baseline acceptance.'})
