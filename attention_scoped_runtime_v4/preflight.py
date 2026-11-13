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
 if cache:verify_cache(root)
 return True

def no_symlinks(path):
 assert path.is_absolute(), 'Cache path must be absolute'
 for part in (path,*path.parents):assert not part.is_symlink(), str(part)
 assert path.resolve()==path, 'Cache path must be canonical'

def tree_files(base):
 no_symlinks(base);assert base.is_dir()
 found=set()
 for p in base.rglob('*'):
  assert not p.is_symlink(), str(p)
  assert p.resolve().is_relative_to(base)
  assert p.is_file() or p.is_dir(), str(p)
  if p.is_file():found.add(str(p.relative_to(base)))
 return found

def verify_cache(root):
 root=Path(root);no_symlinks(root)
 r=json.loads((root/'CACHE-DESTINATION-RECEIPT.json').read_text())
 dest=Path(r['destination']);source=Path(r['source'])
 no_symlinks(source);no_symlinks(dest)
 assert dest==root/'cache_v1' and source!=dest
 assert not source.is_relative_to(dest) and not dest.is_relative_to(source)
 rows=r['files'];assert rows and len({x['path'] for x in rows})==len(rows)
 expected={x['path'] for x in rows}
 assert tree_files(source)==tree_files(dest)==expected, 'Cache file-set mismatch'
 for x in rows:
  rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==x['path']
  a=dest/rel;b=source/rel
  assert a.resolve().is_relative_to(dest) and b.resolve().is_relative_to(source)
  assert sha(a)==sha(b)==x['sha256'];sa,sb=a.stat(),b.stat()
  assert (sa.st_dev,sa.st_ino)!=(sb.st_dev,sb.st_ino)
 return True
