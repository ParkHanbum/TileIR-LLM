import unittest
from types import SimpleNamespace
from unittest.mock import patch
from . import adapter
from .cleanup import final_cleanup
from lean_public_composition_v1.support import Base
class Tests(unittest.TestCase):
 def fixture(self):
  s=object.__new__(adapter.Controller);s.active=True;s.failed=None;s.latch=lambda *a:None;s.parts=[]
  a=Base(s,'a',1);b=Base(s,'b',1);s.parts=[a,b]
  x=SimpleNamespace(f=lambda:1);y=SimpleNamespace(g=lambda:2);ox=x.f;oy=y.g
  a.bind(x,'f',lambda:3);b.bind(y,'g',lambda:4)
  return s,a,b,x,y,ox,oy
 def test_production_base_failure_permanent_hold(self):
  s,a,b,x,y,ox,oy=self.fixture();foreign=lambda:5;y.g=foreign;m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  with self.assertRaises(RuntimeError):adapter.close(m)
  self.assertEqual(b.bindings,[]);self.assertIs(x.f,ox);self.assertIs(y.g,foreign)
  self.assertEqual(len(s._binding_evidence[1][1]),1);self.assertEqual(s._lifetime_parts,[a,b])
  with self.assertRaises(RuntimeError):adapter.close(m)
  with self.assertRaises(AssertionError):adapter.release(m)
  self.assertIs(getattr(m,adapter.ATTR),s);self.assertIs(y.g,foreign)
 def test_partial_install_close_failure_owner_kept(self):
  s,a,b,x,y,ox,oy=self.fixture();y.g=lambda:9;m=SimpleNamespace()
  s.install=lambda:s.close()
  with patch.object(adapter,'Controller',return_value=s):
   with self.assertRaises(RuntimeError):adapter.install(m)
  self.assertIs(getattr(m,adapter.ATTR),s)
  with self.assertRaises(RuntimeError):adapter.close(m)
 def test_final_cleanup_cannot_promote(self):
  s,a,b,x,y,ox,oy=self.fixture();y.g=lambda:9;m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  with self.assertRaises(RuntimeError):s.close()
  engine=SimpleNamespace(apply_model=lambda fn:[fn(m)],engine_core=SimpleNamespace(shutdown=lambda:None));r={}
  final_cleanup(r,lambda:None,None,engine,adapter,True,lambda:None,lambda:None,propagating=True)
  self.assertFalse(r['complete']);self.assertFalse(r.get('adapter_released',False));self.assertIs(getattr(m,adapter.ATTR),s)
  self.assertTrue(any(e['stage']=='adapter_close'for e in r['cleanup_errors']))
 def test_production_success_restores_then_release(self):
  s,a,b,x,y,ox,oy=self.fixture();m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  receipt=adapter.close(m);self.assertTrue(receipt['restored']);self.assertIs(x.f,ox);self.assertIs(y.g,oy);adapter.release(m);self.assertFalse(hasattr(m,adapter.ATTR))
 def test_failed_sync_does_not_release(self):
  s,a,b,x,y,ox,oy=self.fixture();m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  engine=SimpleNamespace(apply_model=lambda fn:[fn(m)],engine_core=SimpleNamespace(shutdown=lambda:None));r={}
  def fail():raise RuntimeError('CPU injected sync')
  final_cleanup(r,lambda:None,None,engine,adapter,True,fail,lambda:None,propagating=True)
  self.assertIs(getattr(m,adapter.ATTR),s);self.assertFalse(r.get('adapter_released',False))
