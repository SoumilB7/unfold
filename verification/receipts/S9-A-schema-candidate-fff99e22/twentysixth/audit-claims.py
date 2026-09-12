from pathlib import Path
import json,gzip,collections,hashlib
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');out=Path('/private/tmp/unfold-s9a-twentysixth');folder=out/'matrix'
matrix=json.loads((folder/'matrix.json').read_text());rows=[];totals=collections.Counter();claim_counts=collections.Counter();projection=collections.Counter()
for target in matrix['models']:
 p=folder/'models'/(target['slug']+'.json.gz');payload=json.loads(gzip.decompress(p.read_bytes()));counts=collections.Counter();unstamped=[]
 for occurrence in payload['table']['occurrences']:
  projection[occurrence['projection']['kind']]+=1
  for finding in occurrence['projection'].get('fact_findings',[]):
   counts[finding['concrete_reason']]+=1
   if finding['concrete_reason']=='claim_proof_unstamped':
    claim_counts[finding['fact_key']]+=1;unstamped.append({'instance_path':occurrence['provenance']['instance_path'],**finding})
 totals.update(counts)
 rows.append({'slug':target['slug'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fact_finding_counts':dict(counts),'unstamped_fact_findings':unstamped})
result={'status':'unblessed actual matrix census; all findings retained','models':rows,'fact_finding_counts':dict(totals),'unstamped_by_fact':dict(claim_counts),'projection_counts':dict(projection)}
(out/'matrix-claim-census.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='models'},indent=2))
