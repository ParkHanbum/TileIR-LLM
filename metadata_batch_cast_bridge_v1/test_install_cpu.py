import ast,copy,__future__,unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from .bridge import Bridge
from runtime_metadata_five_v1 import binding as source

def originals():
 ns={}
 for key in ('conv','common','gdn'):
  e=source.EXPECTED[key];tree=ast.parse(e['source']);node=tree.body[0];ast.increment_lineno(tree,e['firstlineno']-1)
  exec(compile(tree,e['path'],'exec',flags=__future__.annotations.compiler_flag),ns)
 class Common:pass
 class Gdn:pass
 Common.compute_num_computed_tokens=ns[source.EXPECTED['common']['name']]
 Gdn.build=ns[source.EXPECTED['gdn']['name']]
 return Common,Gdn,ns[source.EXPECTED['conv']['name']]

def checked_without_hash(fn,key):
 e=source.EXPECTED[key]
 if fn.__code__.co_filename!=e['path'] or fn.__code__.co_firstlineno!=e['firstlineno']:raise ValueError('source owner')
 return e['source']

class Component:
 def __init__(self):self.failed=False;self.binding=None;self.count={};self.transport=SimpleNamespace(failed=False);self.closed=False
 def start(self,mode):self.count={}
 def hit(self,key,result):self.count[key]=self.count.get(key,0)+1;return result
 def subq(self,a,b):return self.hit('subq',[x-y for x,y in zip(a,b)])
 def subs(self,a,b):return self.hit('subs',[x-y for x,y in zip(a,b)])
 def full(self,*a,**k):return self.hit('full','full')
 def gt(self,a):return self.hit('gt',True)
 def call(self,dst,src,non_blocking=True):return self.hit('cast',dst)
 def finish(self):return dict(count=self.count)
 def close(self):self.closed=True

class Tests(unittest.TestCase):
 def setup_bridge(self):
  common,gdn,conv=originals();m=Component();c=Component();b=Bridge(m,c)
  with patch('runtime_metadata_five_lean_v1.gates.qualification',return_value=True),patch('batch_ptr_int_cast_lean_v1.gates.qualification',return_value=True),patch.object(source,'checked',checked_without_hash):b.install(common,gdn,conv)
  return b,common,gdn,conv,m,c
 def test_actual_source_install_start_finish_restore(self):
  b,common,gdn,conv,m,c=self.setup_bridge();originals_=[x[2]for x in b.installed];b.start()
  obj=common();obj.query_start_loc=list(range(33));obj.seq_lens=[53]*32;obj._num_computed_tokens_cache=None
  out=obj.compute_num_computed_tokens();self.assertEqual(out,[52]*32);self.assertIs(obj.compute_num_computed_tokens(),out)
  body,entry,owned=b.links;self.assertIs(body.__globals__['compute_causal_conv1d_metadata'],entry)
  self.assertIsNot(owned,conv)
  # Exercise installed owned globals helpers; full GDN body requires framework/runtime.
  owned.__globals__['_meta_full']((2048,),-1);owned.__globals__['_meta_full']((2048,),-1)
  dst=object();self.assertIs(owned.__globals__['_batch_ptr_cast'](dst,object()),dst);body.__globals__['_meta_gt'](object())
  r=b.finish();self.assertEqual(r['metadata']['count'],dict(subq=1,subs=1,full=2,gt=1));self.assertEqual(r['batch_cast']['count'],{'cast':1})
  b.close();self.assertIs(common.__dict__['compute_num_computed_tokens'],originals_[0]);self.assertIs(gdn.__dict__['build'],originals_[1]);self.assertTrue(m.closed and c.closed)
 def test_changed_active_class_rejected(self):
  b,common,gdn,conv,m,c=self.setup_bridge();gdn.build=lambda:None
  with self.assertRaises(RuntimeError):b.start()
  self.assertTrue(m.failed and c.failed)
 def test_changed_owned_global_rejected(self):
  b,common,gdn,conv,m,c=self.setup_bridge();b.links[0].__globals__['compute_causal_conv1d_metadata']=conv
  with self.assertRaises(RuntimeError):b.start()
  b.close()
 def test_joint_helper_failure_blocks_class_entry(self):
  b,common,gdn,conv,m,c=self.setup_bridge();b.start();c.failed=True
  with self.assertRaises(RuntimeError):common().compute_num_computed_tokens()
  b.close()
 def test_already_copied_gdn_rejected_before_install(self):
  common,gdn,conv=originals();gdn.build=lambda:None;b=Bridge(Component(),Component())
  with patch('runtime_metadata_five_lean_v1.gates.qualification',return_value=True),patch('batch_ptr_int_cast_lean_v1.gates.qualification',return_value=True):
   with self.assertRaises(ValueError):b.install(common,gdn,conv)
  self.assertFalse(b.installed)
