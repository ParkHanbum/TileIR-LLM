from pathlib import Path
def source_preflight(root):
 root=Path(root);parent=root.parent
 for n in ('adapter.py','gates.py','run.py','launch.py','PREREQUISITE.json'):
  assert (root/n).is_file(),n
 for group,names in {
  'embedding_int64_runtime_v2':('cast.py','forward.original.py.txt','forward.candidate.py.txt','gates.py'),
  'sampler_integer_runtime_v1':('integer.py','call.original.py.txt','call.candidate.py.txt','cleanup.py','gates.py'),
  'integer_pair_bridge_v1':('bridge.py',),
  'lean26_gdn_finite_runtime_v1':('adapter.py','gates.py'),
 }.items():
  for n in names:assert (parent/group/n).is_file(),(group,n)
