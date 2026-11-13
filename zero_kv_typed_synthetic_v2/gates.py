import hashlib,json
from pathlib import Path
SOURCE=Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/v1/worker/utils.py')
def qualification(root):
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='85b23497017e5db07ea8f3fa56807c44dd7dd269f1977031bd86acdadfb0b5ec'
 return dict(private_synthetic=True,actual_model=False)
def call_counts_valid(rows):
 from .cases import CASES,intervals
 from zero_kv_phase2_typed_plan_v1.geometry import partition
 expected=[sum(k=='target'for k,a,b in partition(c['words']*4,intervals(c)))for c in CASES]
 if expected!=[1,2,0,2]:return False
 for index,c in enumerate(CASES):
  for arm in ('native','tile'):
   row=next((x for x in rows if x.get('case')==c['name'] and x.get('arm')==arm),None)
   if row is None:return False
   n=int(bool(c['ids']))if arm=='native'else 0;t=expected[index]if arm=='tile'else 0
   want=dict(native_attempted=n,native_returned=n,tile=dict(attempted=t,returned=t))
   if row.get('source_calls')!=want or type(row.get('candidate_launches'))is not int or row['candidate_launches']!=t:return False
   if any(type(v)is not int for v in [row['source_calls']['native_attempted'],row['source_calls']['native_returned'],row['source_calls']['tile']['attempted'],row['source_calls']['tile']['returned']]):return False
 return True

def audit(r,root=None):
 from .cases import CASES
 rows=r.get('records',[])
 ok=r.get('complete')is True and r.get('synchronized')is True and r.get('precision_initial')==r.get('precision_final')==dict(bf16=True,split_k=True) and [(x['case'],x['arm'])for x in rows]==[(c['name'],a)for c in CASES for a in ('native','tile')]
 ok=ok and all(x.get('expected_full_raw')is True and x.get('readonly_tables_raw')is True and x.get('same_current_stream')is True and x.get('cross_arm_equal')is True for x in rows)
 ok=ok and call_counts_valid(rows)
 return dict(passed=ok,private_synthetic=True,model_validated=False)
