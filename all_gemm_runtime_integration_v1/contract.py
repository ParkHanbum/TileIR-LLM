"""Pinned observed owners and thirteen-contract census; CPU only."""
from collections import Counter
from pathlib import Path
import json
R=Path(__file__).parent

def plan():
 p=json.loads((R/'PLAN-INVENTORY.json').read_text())
 assert p['owners_with_apply']==178 and p['selected_GEMM_owners']==153 and p['excluded_nonGEMM_owners']==25 and p['selected_calls_per_request']==2448
 assert len(p['selected'])==153 and len(p['excluded'])==25 and len(p['contracts'])==13
 assert len({x['path']for x in p['selected']+p['excluded']})==178
 return p

def phase_shape(spec,index):
 assert type(index)is int and 0<=index<16
 phase='prefill'if index==0 else'decode';n,k=spec['weight_shape'];m=spec['first_M']if index==0 else 32
 return phase,[m,n,k]

def coverage(rows,p):
 selected={x['path']:x for x in p['selected']}
 if len(rows)!=2448 or Counter(r['path']for r in rows)!=Counter({n:16 for n in selected}):return False
 if Counter((r['path'],r['call_index'])for r in rows)!=Counter({(n,i):1 for n in selected for i in range(16)}):return False
 for r in rows:
  phase,mnk=phase_shape(selected[r['path']],r['call_index'])
  if r['phase']!=phase or r['logical_MNK']!=mnk:return False
 return Counter(tuple(r['logical_MNK'])for r in rows)==Counter({tuple(c['logical_MNK']):c['calls']for c in p['contracts']})
