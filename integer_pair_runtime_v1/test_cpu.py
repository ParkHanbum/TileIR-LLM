import unittest,types,sys,ast
from pathlib import Path
from unittest.mock import patch
from . import adapter as a
class T(unittest.TestCase):
 def test_actual_installers_delegated_and_reverse(self):
  import tempfile
  model=types.SimpleNamespace();events=[]
  def ci(s):events.append('cast_install');return {'ok':True}
  def ii(s):events.append('integer_install');return {'ok':True}
  def close(name):return lambda s:events.append(name)or dict(restored=True)
  with tempfile.TemporaryDirectory() as tmp,patch.object(a.Cast,'install',ci),patch.object(a.Integer,'install',ii),patch.object(a.Cast,'close',close('cast_close')),patch.object(a.Integer,'close',close('integer_close')),patch.object(a.lean,'close',side_effect=lambda m:events.append('lean_close')or dict(restored=True)):
   a.install_pair(model,Path(tmp)/'pair');a.close(model)
  self.assertEqual(events,['cast_install','integer_install','lean_close','integer_close','cast_close'])
 def test_cast_exception_latches_integer_and_lean(self):
  model=types.SimpleNamespace();calls=[];controller=types.SimpleNamespace(failed=None,latch=lambda *args:calls.append(args))
  setattr(model,a.lean.ATTR,controller)
  cast=a.Cast.__new__(a.Cast);cast.model=model;cast.failed=None
  integer=types.SimpleNamespace(failed=None)
  p=a.Bridge(cast,integer,lambda:controller);setattr(model,a.PAIR,p)
  cast.latch(ValueError('bad'));self.assertIsNotNone(integer.failed);self.assertEqual(len(calls),1)
 def test_pair_gate(self):
  from .gates import pair_gate
  def mk(n,mode,cast):
   rows=[dict(index=i,mode=mode,passed=True,fresh=True,metadata_preserved=True,raw_compared=False,**({'n':1696 if i==0 else 32}if cast else{'site':('positions','input_ids','ones')[i%3]}))for i in range(n)]
   return dict(calls=n,mode=mode,rows=rows,native_calls=n if mode=='shadow'else 0,tile_calls=n if mode=='replace'else 0,site_counts=dict(positions=16,input_ids=16,ones=16))
  p={'embedding_cast':mk(16,'replace',True),'sampler_integer':mk(48,'replace',False)};self.assertTrue(pair_gate(p,'replace'))
  p['sampler_integer']['rows'][0]['site']='ones'
  with self.assertRaises(AssertionError):pair_gate(p,'replace')
 def test_actual_copied_sampler_nested_live_helpers(self):
  from sampler_integer_runtime_v1 import test_cpu as prior
  with patch.object(prior,'Integer',a.Integer):
   prior.T('test_actual_owned_call16_live_helpers_nested_contexts').test_actual_owned_call16_live_helpers_nested_contexts()
 def test_actual_embedding_forward_current_delegate(self):
  from embedding_int64_runtime_v2 import test_cpu as prior
  with patch.object(prior,'Cast',a.Cast):
   prior.T('test_owned_source_current_delegate_and_restore').test_owned_source_current_delegate_and_restore()
 def test_source_only(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
  for n in ('torch','vllm','cuda.tile'):self.assertNotIn(n,sys.modules)
  self.assertFalse((Path(__file__).parent/'cache_v1').exists())
if __name__=='__main__':unittest.main()
