from pathlib import Path
import sys,importlib.util,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
source=ROOT/'scripts/report_s8_demonstration.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='5c696e8296f58b834d33204248f1610813a6709e4e0e9e6617a1bdb9b4c8e3cf'
spec=importlib.util.spec_from_file_location('independent_final_trace_reporter',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module.dump=lambda path,value:None
CASE=Path('/private/tmp/unfold-s8-final-f664467/sdxl')
traces=module.claim_traces(CASE)
(OUT/'claim-traces.json').write_text(json.dumps(traces,indent=2,sort_keys=True)+'\n')
compact=[]
for t in traces:
 compact.append({'claim':t['claim'],'occurrence':t['occurrence'],'fact':t['canonical_fact'],'stage':t['overview_stage'],'block':t['block'],'chain_gaps':t['chain_gaps'],'card':t['actual_card'],'numbers':t['numbers_on_actual_cards']})
(OUT/'trace-summary.json').write_text(json.dumps(compact,indent=2,sort_keys=True)+'\n')
print(json.dumps([{'occurrence':t['occurrence'],'stage':t['overview_stage'],'gaps':t['chain_gaps'],'card_fields':list(t['actual_card']),'card_facts':t['actual_card'].get('facts'),'drill_nodes':t['actual_card'].get('node_ids'),'numbers':t['numbers_on_actual_cards']} for t in traces],indent=2))
