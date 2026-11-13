import hashlib,json
from pathlib import Path
def preflight(root):
 root=Path(root)
 for name in ('run.py','launch.py','observer.py','cleanup.py','result_gate.py','EXPECTED-TOKENS.json'):assert (root/name).is_file(),name
 path=root/'EXPECTED-TOKENS.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()=="1c5b9bd9fbcbd7bb14691a3da35698877a06a2ef6da7b79c1239ed5bdf6598b9"
 x=json.loads(path.read_text());assert len(x)==32 and all(len(t)==16 for t in x)
