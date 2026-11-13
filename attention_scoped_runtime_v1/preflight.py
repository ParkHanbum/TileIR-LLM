"""CPU filesystem checks before GPU admission/child. Parent creates fresh receipt.
Receipt schema: destination/source absolute dirs; files list of relative path, sha256.
Destination must contain exactly listed files, each detached from source inode.
No receipt is fabricated by this runner. Cache may change only after preflight.
"""
import json,hashlib
from pathlib import Path
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(root,cache=False):
 root=Path(root).resolve()
 tokens=root/'EXPECTED-TOKENS.json';assert sha(tokens)=='1c5b9bd9fbcbd7bb14691a3da35698877a06a2ef6da7b79c1239ed5bdf6598b9'
 ids=json.loads(tokens.read_text());assert len(ids)==32 and all(len(x)==16 for x in ids)
 pins=json.loads((root/'PINS.json').read_text());assert pins
 for p,h in pins.items():assert sha(p)==h,p
 if cache:
  r=json.loads((root/'CACHE-DESTINATION-RECEIPT.json').read_text());dest=root/'cache_v1';source=Path(r['source']).resolve()
  assert r['destination']==str(dest.resolve()) and source!=dest.resolve() and not dest.is_symlink()
  rows=r['files'];assert rows and len({x['path'] for x in rows})==len(rows)
  actual={str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()};assert actual=={x['path'] for x in rows}
  for x in rows:
   rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
   a=dest/rel;b=source/rel;assert not a.is_symlink() and a.resolve().is_relative_to(dest.resolve())
   assert sha(a)==sha(b)==x['sha256'];sa,sb=a.stat(),b.stat();assert (sa.st_dev,sa.st_ino)!=(sb.st_dev,sb.st_ino)
 return True
