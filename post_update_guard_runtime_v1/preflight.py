from pathlib import Path
import ast,json,hashlib
def source_preflight(root):
 root=Path(root)
 for p in root.glob('*.py'):ast.parse(p.read_text())
 for p,h in json.loads((root/'PINS.json').read_text()).items():
  with open(p,'rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,p
 return dict(source_verified=True)
