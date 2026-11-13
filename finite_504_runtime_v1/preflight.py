from pathlib import Path
def source_preflight(root):
 root=Path(root);P=root.parent
 for n in ('adapter.py','gates.py','run.py','launch.py','PREREQUISITE.json'):
  assert(root/n).is_file(),n
 for folder,names in {
  'floating_guard_504_design_v1':('factory.py','SOURCE.json','post.original.py.txt','post.candidate.py.txt','kv.original.py.txt','kv.candidate.py.txt','attention.original.py.txt','attention.candidate.py.txt'),
  'tileir_kv_strided_finite_v1':('api.py','contract.py','kernels.py'),
  'tileir_validation_primitives_v3':('api.py','contract.py','kernels.py'),
  'lean26_gdn_finite_runtime_v1':('adapter.py','cleanup.py','gates.py'),
 }.items():
  for n in names:assert(P/folder/n).is_file(),(folder,n)
