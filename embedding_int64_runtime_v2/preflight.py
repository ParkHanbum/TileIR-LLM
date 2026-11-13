from pathlib import Path
def source_preflight(root):
 for name in ('adapter.py','cast.py','forward.original.py.txt','forward.candidate.py.txt','edge_probe.py','gates.py','cleanup.py','PREREQUISITE.json','run.py','launch.py'):
  assert (Path(root)/name).is_file(),name
