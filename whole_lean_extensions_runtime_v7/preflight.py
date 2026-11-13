from pathlib import Path
import ast,json,hashlib
def source_preflight(root):
 root=Path(root)
 for p in root.glob('*.py'):ast.parse(p.read_text())
 for p,h in json.loads((root/'PINS.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 return dict(source_verified=True)
