"""CPU-only required companion validation before admission or GPU imports."""
import hashlib,json
from pathlib import Path
EXPECTED_SHA="1c5b9bd9fbcbd7bb14691a3da35698877a06a2ef6da7b79c1239ed5bdf6598b9"
def preflight(root):
 root=Path(root)
 for name in ('run.py','launch.py','adapter.py','gates.py','result_gate.py','precision.py','EXPECTED-TOKENS.json','SOURCE.json','prepare_inputs.original.py.txt','prepare_inputs.copied.py.txt'):
  assert (root/name).is_file(), ('Missing required companion',name)
 path=root/'EXPECTED-TOKENS.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_SHA,'Expected tokens changed'
 ids=json.loads(path.read_text())
 assert len(ids)==32 and all(len(row)==16 and all(type(x) is int for x in row) for row in ids)
 return dict(expected_tokens_sha256=EXPECTED_SHA,expected_tokens=512)

from attention_scoped_runtime_v2.preflight import verify_cache,sha
