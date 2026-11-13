"""Frozen512 token companion reused unchanged from actual native capture."""
import json,hashlib
from pathlib import Path
EXPECTED=Path('/home/n/work/TileIR/public-tileir-blas-20261111/gdn_decode_core_native_capture_v1/EXPECTED-TOKENS.json')
EXPECTED_SHA='1c5b9bd9fbcbd7bb14691a3da35698877a06a2ef6da7b79c1239ed5bdf6598b9'
def load_expected(path=EXPECTED):
 path=Path(path);assert path.resolve()==EXPECTED.resolve(),'Only pinned expected companion allowed'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_SHA
 x=json.loads(path.read_text());assert len(x)==32 and all(len(row)==16 and all(type(v)is int for v in row)for row in x)
 return x
def check_expected(actual,expected):
 assert actual==expected,'Frozen512 token mismatch; no replacement/timing expansion'
 return True
