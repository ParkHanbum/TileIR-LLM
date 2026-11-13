"""Read-only fresh-cache provenance gate, before admission or model import."""
from pathlib import Path
import hashlib,json

def sha(p):
 with open(p,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def preflight(root):
 root=Path(root).resolve();receipt=root/'CACHE-PRESEED-01.json';assert receipt.is_file(),'Fresh cache preseed receipt missing'
 r=json.loads(receipt.read_text());expected=root/'cache_v1';dest=Path(r['destination']);src=Path(r['source'])
 assert dest.is_absolute()and dest==expected and not dest.is_symlink()and dest.resolve()==expected,'Preseed receipt destination is not this run cache'
 assert src.is_absolute()and src.is_dir()and src.resolve()!=dest.resolve()and dest.is_dir()
 assert r['same_bytes']and r['distinct_inodes']and r['source_unchanged']
 files=r['source_files'];assert isinstance(files,dict)and len(files)==r['count']and r['count']>0
 relative={};total=0
 for path,h in files.items():
  q=Path(path);rel=q.relative_to(src)if q.is_absolute()else q
  assert not rel.is_absolute()and '..'not in rel.parts and str(rel)not in relative
  a,b=src/rel,dest/rel;assert a.is_file()and b.is_file()and not a.is_symlink()and not b.is_symlink()
  assert a.resolve().is_relative_to(src.resolve())and b.resolve().is_relative_to(dest)
  assert sha(a)==sha(b)==h,('Cache copy bytes changed',str(rel))
  ast,bst=a.stat(),b.stat();assert (ast.st_dev,ast.st_ino)!=(bst.st_dev,bst.st_ino),'Cache hardlinks are not isolated copies'
  relative[str(rel)]=h;total+=ast.st_size
 assert {str(p.relative_to(src))for p in src.rglob('*')if p.is_file()}==set(relative),'Source cache file set changed'
 assert {str(p.relative_to(dest))for p in dest.rglob('*')if p.is_file()}==set(relative),'Destination cache is not the fresh declared copy'
 assert total==r['bytes']
 return dict(receipt_path=str(receipt),receipt_sha256=sha(receipt),source=str(src),destination=str(dest),file_count=len(relative),bytes=total,files_verified=True,separate_inodes_verified=True,scope='Fresh pre-model cache copy verified; runtime may subsequently add files')
