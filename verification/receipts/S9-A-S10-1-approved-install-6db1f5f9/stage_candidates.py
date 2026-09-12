"""Stage the approved-delta candidates with their own generators, under source pins.

Writes only into the staging directory. Never touches the repository.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
STAGE = Path('/private/tmp/unfold-s9a-install-6db1f5f9')
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

def sha(b): return hashlib.sha256(b).hexdigest()
def shaf(p): return sha(Path(p).read_bytes())

def source_pins():
    out = {}
    for base in ('model_unfolder', 'physics', 'test_support', 'scripts'):
        for p in sorted((ROOT / base).rglob('*.py')):
            if '__pycache__' in p.parts: continue
            out[p.relative_to(ROOT).as_posix()] = shaf(p)
    for p in sorted((ROOT / 'model_unfolder').rglob('*.yaml')):
        out[p.relative_to(ROOT).as_posix()] = shaf(p)
    return out

def corpus_pins():
    out = {}
    for p in sorted((ROOT / 'tests/sable_test_corpus').rglob('*')):
        if p.is_file(): out[p.relative_to(ROOT).as_posix()] = shaf(p)
    for p in sorted((ROOT / 'tests/unseen_model_configs').rglob('*')):
        if p.is_file(): out[p.relative_to(ROOT).as_posix()] = shaf(p)
    return out

head = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
before = {'source': source_pins(), 'corpus': corpus_pins(), 'head': head}
(STAGE / 'pins-before.json').write_text(json.dumps(before, indent=1, sort_keys=True) + '\n')
print('pins before recorded:', len(before['source']), 'source files,', len(before['corpus']), 'corpus files, head', head, flush=True)

result = {'head': head, 'steps': {}}

# --- 1. preservation expected manifest, via its own generator -----------------
from test_support import preservation as P
t = time.time()
target = STAGE / 'preservation_expected_manifest.json'
manifest = P.build_expected_manifest(ROOT / 'tests/sable_test_corpus', target)
result['steps']['preservation_manifest'] = {
    'seconds': round(time.time() - t, 3), 'sha256': shaf(target),
    'witness_count': manifest['witness_count']}
print('preservation manifest staged:', result['steps']['preservation_manifest'], flush=True)

# --- 2. examples, via their own generator (deterministic HTML only) -----------
import scripts.generate_examples as GE
t = time.time()
cand = STAGE / 'examples'
if cand.exists():
    import shutil; shutil.rmtree(cand)
rows = GE._render_rows(cand, rasterize_hero=False)
result['steps']['examples'] = {'seconds': round(time.time() - t, 3), 'rows': rows}
print('examples staged:', len(rows), 'rows in', result['steps']['examples']['seconds'], 's', flush=True)

# --- 3. coverage, via its own generator ---------------------------------------
import scripts.coverage as CV
t = time.time()
doc = CV.generate()
rendered = CV._render(doc)
(STAGE / 'coverage.json').write_text(rendered, encoding='utf-8')
from model_unfolder.evidence.coverage import coverage_problems
problems = coverage_problems(doc)
result['steps']['coverage'] = {
    'seconds': round(time.time() - t, 3), 'sha256': shaf(STAGE / 'coverage.json'),
    'problems': problems,
    'proven': sum(r['proven'] for r in doc['models']),
    'flagged': sum(r['flagged'] for r in doc['models']),
    'silent': sum(r['silent'] for r in doc['models']),
    'models': len(doc['models'])}
print('coverage staged:', {k: v for k, v in result['steps']['coverage'].items() if k != 'problems'}, flush=True)
print('coverage problems:', problems, flush=True)

after = {'source': source_pins(), 'corpus': corpus_pins(),
         'head': subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}
(STAGE / 'pins-after.json').write_text(json.dumps(after, indent=1, sort_keys=True) + '\n')
result['pins_equal'] = (before == after)
tracked = subprocess.run(['git','-C',str(ROOT),'status','--porcelain'],capture_output=True,text=True).stdout
result['repo_status_after'] = tracked
(STAGE / 'stage-result.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
print('PINS EQUAL:', result['pins_equal'])
print('REPO STATUS AFTER:\n' + tracked)
