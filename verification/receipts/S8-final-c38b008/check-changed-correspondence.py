from pathlib import Path
import json,hashlib
root=Path(__file__).parent
load=lambda p:json.loads(p.read_text())
expected=load(root/'changed-expectation.json');a=root/'sdxl/ordinary';b=root/'sdxl/changed'
old={x['path']:x for x in load(a/'inventory.json')['modules']};new={x['path']:x for x in load(b/'inventory.json')['modules']}
added=set(new)-set(old);removed=set(old)-set(new);rows=[]
old_facts=load(a/'facts.json');new_facts=load(b/'facts.json');shape_key='root.denoiser.constructed_parameter_shapes'
old_shapes=old_facts[shape_key]['value'];new_shapes=new_facts[shape_key]['value']
for row in expected['rows']:
 prefix=row['expected_added_block'];paths={p for p in added if p==prefix or p.startswith(prefix+'.')};total=new_shapes['by_module'].get(prefix)
 rows.append({**row,'actual_added_occurrences':len(paths),'actual_added_paths':sorted(paths),'actual_added_parameters':total,'matches':len(paths)==row['expected_added_occurrences'] and total==row['expected_added_parameters']})
covered={p for row in rows for p in row['actual_added_paths']};old_result=load(a/'result.json');new_result=load(b/'result.json')
checks={'exact_added_paths':added==covered,'no_removed_occurrences':not removed,'exact_added_count':len(added)==expected['expected_added_occurrences'],'exact_parameter_delta':new_result['parameters']-old_result['parameters']==expected['expected_added_parameters'],'each_expected_block_matches':all(row['matches'] for row in rows),'unchanged_existing_parameter_shapes':all(new_shapes['parameters'].get(key)==value for key,value in old_shapes['parameters'].items())}
report={'scope':expected['scope'],'checks':checks,'rows':rows,'removed':sorted(removed),'unexpected_added':sorted(added-covered),'before_parameters':old_result['parameters'],'after_parameters':new_result['parameters'],'expected_sha256':hashlib.sha256((root/'changed-expectation.json').read_bytes()).hexdigest()}
(root/'changed-exact-correspondence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checks));assert all(checks.values())
