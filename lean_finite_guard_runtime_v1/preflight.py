from pathlib import Path
def source_preflight(root):
 for name in ('adapter.py','sampler.py','guarded_top5.py.txt','gates.py','cleanup.py','PREREQUISITE.json','run.py','launch.py'):
  assert (Path(root)/name).is_file(),('Missing source companion',name)
