from pathlib import Path
def source_preflight(root):
 root=Path(root)
 for n in ('adapter.py','gates.py','run.py','launch.py','PREREQUISITE.json'):assert(root/n).is_file(),n
