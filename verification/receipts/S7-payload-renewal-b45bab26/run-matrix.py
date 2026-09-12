from pathlib import Path
import dataclasses, importlib.util, json, os, subprocess, sys
root = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
out = Path('/private/tmp/unfold-s9a-s7-renewal-b45bab26/matrix-lane')
out.mkdir(exist_ok=False)
head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                      capture_output=True, text=True).stdout.strip()
assert head == 'b45bab26539eca773c2f03519ebfd72ca7719d93', head
assert not subprocess.run(['git', '-C', str(root), 'diff', '--name-only', 'HEAD'],
                          capture_output=True, text=True).stdout.strip(), 'tree must be clean'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  UNFOLD_EVIDENCE_CACHE_DIR=str(out / 'cache'))
spec = importlib.util.spec_from_file_location('s7_renewal_verify', root / 'scripts/verify_commit.py')
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module._run_lane(
    module.Lane('s7-matrix-renewal-candidate',
                (sys.executable, '/private/tmp/unfold-s9a-s7-renewal-b45bab26/generate-matrix.py')),
    root, out)
record = {**dataclasses.asdict(result), 'log_path': str(result.log_path),
          'passed': result.passed, 'head': head, 'active_matrix_renewed': False}
(out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
sys.exit(0 if result.passed else 1)
