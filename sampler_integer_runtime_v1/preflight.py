from pathlib import Path
def source_preflight(root):
 for n in ('integer.py','adapter.py','gates.py','run.py','launch.py','cleanup.py','PREREQUISITE.json','edge_probe.py','call.original.py.txt','call.candidate.py.txt'):
  assert(Path(root)/n).is_file(),n
