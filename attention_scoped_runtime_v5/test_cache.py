import unittest,tempfile,json,os
from pathlib import Path
from .preflight import verify_cache,sha
class CacheTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name).resolve();self.root=self.base/'run';self.root.mkdir();self.src=self.base/'source';self.src.mkdir();self.dest=self.root/'cache_v1';self.dest.mkdir()
  for d in (self.src,self.dest):(d/'a').write_bytes(b'abc')
  self.receipt=dict(source=str(self.src),destination=str(self.dest),files=[dict(path='a',sha256=sha(self.src/'a'))]);self.write()
 def write(self):(self.root/'CACHE-DESTINATION-RECEIPT.json').write_text(json.dumps(self.receipt))
 def test_valid(self):self.assertTrue(verify_cache(self.root))
 def test_source_extra(self):
  (self.src/'extra').write_bytes(b'x')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_dest_extra(self):
  (self.dest/'extra').write_bytes(b'x')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_relative_source(self):
  self.receipt['source']='source';self.write()
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_relative_dest(self):
  self.receipt['destination']='cache_v1';self.write()
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_source_symlink(self):
  alias=self.base/'alias';alias.symlink_to(self.src,target_is_directory=True);self.receipt['source']=str(alias);self.write()
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_source_child_symlink(self):
  (self.src/'link').symlink_to(self.src/'a')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_dest_symlink(self):
  (self.dest/'a').unlink();(self.dest/'a').symlink_to(self.src/'a')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_hardlink(self):
  (self.dest/'a').unlink();os.link(self.src/'a',self.dest/'a')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_hash(self):
  (self.src/'a').write_bytes(b'different')
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_traversal(self):
  self.receipt['files'][0]['path']='../a';self.write()
  with self.assertRaises(AssertionError):verify_cache(self.root)
 def test_ancestor_symlink(self):
  alias=self.base/'parentlink';alias.symlink_to(self.base,target_is_directory=True);self.receipt['source']=str(alias/'source');self.write()
  with self.assertRaises(AssertionError):verify_cache(self.root)
