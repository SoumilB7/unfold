from pathlib import Path
import ast,json,hashlib,subprocess
R=Path('/private/tmp/unfold-s8-final-verification-7168440');P=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');O=P/'verification/receipts/S8-caller-growth-refresh-7168440';O.mkdir(exist_ok=False)
old=json.loads((P/'verification/receipts/S8-reader-caller-final-independent/result.json').read_text());growth=json.loads((P/'verification/receipts/S8-final-growth-7168440/growth.json').read_text())
source={str(p.relative_to(R)):p.read_text() for d in ('model_unfolder','physics','scripts') for p in (R/d).rglob('*.py')};trees={p:ast.parse(s) for p,s in source.items()}
def refs(name,exclude):return [{'path':p,'line':n.lineno,'source':source[p].splitlines()[n.lineno-1].strip()} for p,t in trees.items() if p!=exclude for n in ast.walk(t) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id==name]
original=subprocess.check_output(['git','ls-tree','-r','--name-only','83140f1','model_unfolder/evidence'],cwd=R,text=True).splitlines();original=[p for p in original if Path(p).name.startswith('unet_') and p.endswith('.py')];assert len(original)==14
rows=[]
for prior in old['original_readers']:
 module=prior['module'];assert module in original;entries=[]
 for entry in prior['entries']:
  name=entry['name'];calls=[r for r in refs(name,module) if r['path'].startswith('model_unfolder/')];assert calls
  entries.append({'name':name,'definition_line':next(n.lineno for n in trees[module].body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),'production_references':calls})
 rows.append({'module':module,'entries':entries,'disposition':'retained with production caller; original module deletion not owed'})
new=[]
for prior in old['new_live_modules']+[{'module':'model_unfolder/evidence/construction_summary.py','entries':[{'name':'project_construction_summary'},{'name':'construction_summary_problems'}]}]:
 module=prior['module'];entries=[]
 for entry in prior['entries']:
  calls=[r for r in refs(entry['name'],module) if r['path'].startswith(('model_unfolder/','physics/'))];assert calls
  entries.append({'name':entry['name'],'production_references':calls})
 new.append({'module':module,'entries':entries})
assert len(new)==17
scoped=refs('legacy_unet_comparison','model_unfolder/adapters/diffusor/unet_differential.py');assert len(scoped)==1 and scoped[0]['path']=='scripts/demonstrate_s8_unet.py' and 'args.condition == "legacy"' in scoped[0]['source']
parser=source['model_unfolder/adapters/diffusor/parser.py'];legacy=source['model_unfolder/adapters/diffusor/unet_differential.py'];assert 'default=False' in legacy and 'finally:' in legacy and '_legacy_comparison.reset(token)' in legacy
helper=next(n for n in trees['model_unfolder/adapters/diffusor/parser.py'].body if isinstance(n,ast.FunctionDef) and n.name=='_parse_unet_model');assert 'if not legacy_comparison_enabled():' in ast.get_source_segment(parser,helper)
# Fixture extraction is a move: compare every named source fixture node, ignoring locations.
priorfixture=ast.parse(subprocess.check_output(['git','show','83140f1:tests/test_unet_stage_execution.py'],cwd=R,text=True));newfixture=ast.parse((R/'test_support/unet_stage_fixture.py').read_text())
def named(tree):
 out={}
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):out[n.name]=n
  elif isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name):out[target.id]=n
 return out
p,n=named(priorfixture),named(newfixture);fixture={key:ast.dump(p[key])==ast.dump(n[key]) for key in ('ROOT','FACTORY','_write','_bundle','_read')};assert all(fixture.values())
debtpath='model_unfolder/evidence/structural_debt.py';debtold=subprocess.check_output(['git','show','83140f1:'+debtpath],cwd=R);assert debtold==(R/debtpath).read_bytes()
fingerprints=['6857ba8217a73268','da27bb4cf97df230','07d25d5749cfe49b'];debtrows=[{'line':i+1,'source':s.strip()} for i,s in enumerate(source[debtpath].splitlines()) if any(v in s for v in fingerprints)];assert len(debtrows)==3
# Recompute the three growth buckets from exact committed numstat.
numstat=subprocess.check_output(['git','diff','--numstat','83140f1','7168440'],cwd=R,text=True).splitlines();buckets={k:{'files':0,'added':0,'deleted':0} for k in ('production','scripts','tests_and_support')}
for line in numstat:
 added,deleted,path=line.split('\t');bucket='production' if path.startswith(('model_unfolder/','physics/')) else 'scripts' if path.startswith('scripts/') else 'tests_and_support' if path.startswith(('tests/','test_support/')) else None
 if bucket:buckets[bucket]['files']+=1;buckets[bucket]['added']+=int(added);buckets[bucket]['deleted']+=int(deleted)
assert buckets==growth['summary']
result={'checkpoint':'7168440c81a3e79b1e220437e2f9fa77e25bc296','base':'83140f1','original_readers':rows,'original_live_count':14,'original_modules_deleted':0,'new_live_modules':new,'new_support_module_count':17,'recipe_extraction_separate':'execution_recipe.py is shared verification infrastructure moved in part from generate_s7_shadow; not an ordinary architecture parse probe.','legacy_enabling_callers':scoped,'legacy_isolation':'Source-reviewed ContextVar false default, scoped finally reset, independent helper guard, typed limited failure path; no fresh dynamic routing probes in this refresh.','legacy_comparison_bridge_added':1,'legacy_comparison_bridge_retired':0,'old_ordinary_author_entry_removed':1,'old_author_files_deleted':0,'growth':buckets,'fixture_source_nodes_ast_identical':fixture,'fixture_migration':'77 lines added in test_support/unet_stage_fixture.py; test_unet_stage_execution.py +1/-70. Counted tests/support, not architecture deletion.','restored_consumer_fingerprints':debtrows,'entire_structural_debt_register_exact_to_83140':True,'old_debt_eliminated':0,'scope_limit':'Static public-entry liveness and growth inventory; no claim every private helper necessary or every model covered. No models, pytest, production edits, new architecture grammar, commits or blessing.','source_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sorted(set(original)|{r['module'] for r in new}|{debtpath,'model_unfolder/adapters/diffusor/parser.py','model_unfolder/adapters/diffusor/unet_differential.py','model_unfolder/adapters/diffusor/unet_cutover.py','test_support/unet_stage_fixture.py','tests/test_unet_stage_execution.py'})}}
(O/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(O/'replay.py').write_text(Path(__file__).read_text());print(json.dumps({'original14_live':len(rows),'new17_live':len(new),'fixture_nodes_identical':fixture,'growth_exact':buckets,'restored_debt_rows':debtrows}))
