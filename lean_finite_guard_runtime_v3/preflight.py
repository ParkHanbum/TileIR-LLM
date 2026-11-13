from pathlib import Path
def source_preflight(root):
 for name in ('adapter.py','bindings.py','edge_probe.py','guarded_top5.py.txt','gates.py','cleanup.py','PREREQUISITE.json','PRIOR-FINITE-MODEL.json','run.py','launch.py'):
  assert (Path(root)/name).is_file(),('Missing source companion',name)
