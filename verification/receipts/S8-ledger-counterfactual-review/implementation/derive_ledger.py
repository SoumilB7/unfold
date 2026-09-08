"""One labelled derived ledger counterfactual; never changes actual output."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.util,json,shutil,traceback

CONTRACT_SHA256='bdd1fbf02418b265f7a0512120d35d595529b783f175dd3dd8543df55942485c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def read_gzip(p):return gzip.decompress(p.read_bytes())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checkout',type=Path,required=True);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--slug',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 contract_path=Path(__file__).resolve().parent.parent/'declaration-contract.json';assert sha(contract_path)==CONTRACT_SHA256,'Reviewed declaration contract changed';contract=json.loads(contract_path.read_text());root=args.checkout.resolve();expected_path=root/'tests/preservation_expected_manifest.json';comparator_path=root/'test_support/preservation.py';ledger_path=args.case/'ledgers.json.gz';record_path=args.case/'preservation-input.json';tracked=[Path(__file__).resolve(),contract_path,expected_path,comparator_path,ledger_path,record_path];before={str(p):sha(p) for p in tracked};write(out/'input-script-pins.json',before);shutil.copy2(contract_path,out/'declaration-contract.json')
 recovery=root/'verification/receipts/S8-preservation-ledger-recovery-preparation';fallback={'required_only_if_derived_check_fails':True,'argv':['python3',str(recovery/'capture_sidecar.py'),'--source','/private/tmp/unfold-s8-ledger-source-83140f1','--pins',str(recovery/'source-snapshot.json'),'--output',str(out/'actual-old-source-recovery'),args.slug],'script_sha256':contract['accepted_recovery_script_sha256'],'snapshot_pins_sha256':contract['accepted_snapshot_pins_sha256'],'execution':'Not executed by this read-only derivation. The serial executor must run this actual old-source recovery on any mismatch.'};write(out/'fallback-recovery-command.json',fallback)
 try:
  assert args.slug in contract['eligible_non_unet_witnesses'],'Witness is outside the approved non-UNet declaration-only scope'
  assert sha(expected_path)==contract['expected_manifest_sha256'],'Immutable expected manifest changed'
  assert sha(comparator_path)==contract['unchanged_comparator_sha256'],'Accepted canonicalizer source changed'
  expected=json.loads(expected_path.read_text())['witnesses'][args.slug];record=json.loads(record_path.read_text());assert Path(record['fixture']).stem==args.slug and record['input_sha256']==expected['input_sha256'],'Actual witness/input identity differs'
  spec=importlib.util.spec_from_file_location('_unchanged_s8_preservation',comparator_path);preservation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preservation)
  payload=read_gzip(ledger_path);actual=json.loads(payload);assert payload==preservation._canon_bytes(actual),'Actual ledger is not the unchanged comparator canonical payload';assert hashlib.sha256(payload).hexdigest()==record['actual_ledger_sha256'],'Actual sidecar hash differs from its capture record';shutil.copy2(ledger_path,out/'actual-ledger.json.gz')
  current=actual['config_access']['projection_coverage']['receipted_scopes'];old=contract['old_ordered_scopes'];additions=contract['reviewed_additions'];required=contract['required_current_ordered_scopes']
  assert len(old)==41 and len(additions)==12 and len(required)==53 and required==sorted(old+additions),'Reviewed ordered list contract is inconsistent'
  assert len({tuple(x) for x in required})==53,'Reviewed list has duplicate scopes'
  assert current==required,'Actual ordered scopes (including duplicates) differ from reviewed Bloom current list'
  derived=copy.deepcopy(actual);derived['config_access']['projection_coverage']['receipted_scopes']=copy.deepcopy(old);assert actual==json.loads(payload),'Actual ledger was mutated'
  restored=copy.deepcopy(derived);restored['config_access']['projection_coverage']['receipted_scopes']=copy.deepcopy(current);assert restored==actual,'Derived copy has an additional structural delta';write(out/'structural-delta.json',{'only_edit':{'path':contract['operation_path'],'before':current,'after':old},'restoring_only_this_path_recovers_actual_structure':True})
  derived_payload=preservation._canon_bytes(derived);(out/'DERIVED-ledger.canonical.json').write_bytes(derived_payload);derived_sha=hashlib.sha256(derived_payload).hexdigest();matched=derived_sha==expected['surfaces']['ledgers']
  operation={'classification':'DERIVED_CANONICAL_COUNTERFACTUAL','path':contract['operation_path'],'replace_exact_ordered_current_list':required,'with_exact_actual_old_bloom_list':old,'reviewed_twelve_additions':additions,'all_other_values_preserved_by_deep_copy':True,'current_canonical_sha256':hashlib.sha256(payload).hexdigest(),'derived_canonical_sha256':derived_sha,'immutable_expected_ledger_sha256':expected['surfaces']['ledgers'],'meaning':'A matching whole canonical hash proves all other serialized ledger values match this witness expected ledger. It is not an actual old run, mechanism proof, green preservation verdict, or baseline approval.'};write(out/'operation.json',operation)
  result={'status':'DERIVED_EXPECTED_HASH_MATCH' if matched else 'RED_ACTUAL_OLD_RECOVERY_REQUIRED','slug':args.slug,'expected_hash_matched':matched,'actual_old_run':False,'mechanism_proof':False,'preservation_gate_green':False,'blessed':False,'actual_current_bytes_unchanged':True,'operation':'operation.json','fallback_required':not matched};write(out/'result.json',result);print(json.dumps(result),flush=True);return 0 if matched else 2
 except (AssertionError,KeyError,TypeError,ValueError) as error:
  result={'status':'RED_ACTUAL_OLD_RECOVERY_REQUIRED','slug':args.slug,'error':str(error),'traceback':traceback.format_exc(),'expected_hash_matched':False,'actual_old_run':False,'mechanism_proof':False,'preservation_gate_green':False,'blessed':False,'fallback_required':True};write(out/'result.json',result);print(json.dumps(result),flush=True);return 2
 finally:
  after={str(p):sha(p) if p.is_file() else None for p in tracked};write(out/'pin-check-finally.json',{'all_actual_inputs_and_used_sources_unchanged':before==after,'after':after});assert before==after,'Actual inputs or used source changed during derivation'

if __name__=='__main__':raise SystemExit(main())
