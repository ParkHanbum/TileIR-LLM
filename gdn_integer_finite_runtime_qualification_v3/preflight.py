from pathlib import Path
def source_preflight(root):
 for name in ('adapter.py','core.py','gates.py','cleanup.py','PREREQUISITE.json','run.py','launch.py'):
  assert (Path(root)/name).is_file(),name
