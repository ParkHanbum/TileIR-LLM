import unittest,struct,types
from staged_typed_inplace_runtime_v1.oracle import expected
class Tests(unittest.TestCase):
 def test_preserve_tail(self):
  raw=struct.pack('<8i',*range(8));out=expected(raw,(2,4),(4,1),'torch.int32',([1],[1],[99,88],[2]));self.assertEqual(struct.unpack('<8i',out),(0,1,2,3,4,99,88,7))
 def test_empty(self):self.assertEqual(expected(b'\x00'*8,(2,),(1,),'torch.int32',([],[],[],[])),b'\x00'*8)
 def test_non_racing(self):
  with self.assertRaises(AssertionError):expected(b'\x00'*8,(2,),(1,),'torch.int32',([0,0],[0,0],[1,2],[1,2]))
 def test_float_bits(self):
  raw=bytes.fromhex('0100c07f0000803f');out=expected(raw,(2,),(1,),'torch.float32',([1],[0],[-0.0],[1]));self.assertEqual(out,bytes.fromhex('0100c07f00000080'))
 def test_lifetime_rebinding(self):
  from staged_ownership_runtime_v1.test_observer import Tests as Mocks
  from .lifetime import Destination
  S,F,T,p,o=Mocks().make();S.clear_staged_writes=lambda self:None;a=S();d=Destination(a);self.assertFalse(d.check()['explicit_public_uva_owner']);a.gpu=S().gpu
  with self.assertRaises((ValueError,AssertionError)):d.check()
 def test_unsupported_native_candidate_zero(self):
  from staged_ownership_runtime_v1.test_observer import Tests as Mocks
  from .replacement import OwnershipObserver
  S,F,T,p,o=Mocks().make();S.clear_staged_writes=lambda self:None;o.__class__=OwnershipObserver;o.rows=[];o.destinations=[];o.controller=None
  with o:
   a=S();a._staged_write_indices=[2];a._staged_write_starts=[0];a._staged_write_contents=[1];a._staged_write_cu_lens=[1];p[0]='request'
   with self.assertRaises(RuntimeError):a.apply_write()
   self.assertEqual(a.n,0);self.assertEqual(o.rows[0]['candidate_returns'],0)
   with self.assertRaises(RuntimeError):a.apply_write()
   self.assertEqual(a.n,0)
 def test_original_address_mocked_candidate_no_native(self):
  import sys
  from unittest.mock import patch
  from staged_ownership_runtime_v1.test_observer import Tests as Mocks
  from staged_typed_shadow_runtime_v2.evidence import SCOPE
  from .replacement import OwnershipObserver
  S,F,T,p,o=Mocks().make();oldinit=S.__init__
  def init(owner):
   oldinit(owner);t=owner.gpu;t.s.n=32;t.numel=lambda:8;t.requires_grad=False;t.buffer=struct.pack('<8i',*range(8));t.view=lambda *a:t;t.cpu=lambda:t;t.numpy=lambda:t;t.tobytes=lambda:t.buffer
   owner._staged_write_starts=[];owner._staged_write_contents=[];owner._staged_write_cu_lens=[]
  S.__init__=init
  def clear(owner):
   for name in ('_staged_write_indices','_staged_write_starts','_staged_write_contents','_staged_write_cu_lens'):getattr(owner,name).clear()
  S.clear_staged_writes=clear;o.__class__=OwnershipObserver;o.rows=[];o.destinations=[];o.controller=None
  torch=types.ModuleType('torch');stream=object();torch.uint8='torch.uint8';torch.cuda=types.SimpleNamespace(current_stream=lambda:stream,synchronize=lambda:None)
  api=types.ModuleType('tileir_staged_typed_v2.api')
  def apply(t,*data,ledger):
   ledger.append(dict(nonempty=bool(data[0]),launch_attempts=int(bool(data[0])),launch_returns=int(bool(data[0])),scope=SCOPE));t.buffer=expected(t.buffer,t.shape,t.stride(),t.dtype,data)
  api.apply=apply
  with o:
   a=S();a._staged_write_indices=[1];a._staged_write_starts=[1];a._staged_write_contents=[77];a._staged_write_cu_lens=[1];p[0]='request';original=a.gpu
   import tileir_staged_typed_v2
   with patch.dict(sys.modules,{'torch':torch,'tileir_staged_typed_v2.api':api}),patch.object(tileir_staged_typed_v2,'api',api,create=True):self.assertIsNone(a.apply_write())
   self.assertEqual(a.n,0);self.assertIs(a.gpu,original);self.assertTrue(o.rows[0]['passed']);self.assertEqual(a._staged_write_indices,[]);self.assertEqual(struct.unpack('<8i',a.gpu.buffer),(0,1,2,3,4,77,6,7))
 def test_synthetic_evidence_gate_not_model_proof(self):
  import json,copy
  from pathlib import Path
  from . import gates
  root=Path(__file__).parent;prior=root.parent/'staged_typed_shadow_runtime_v2/stagedTypedShadow02/RESULT.json';r=json.loads(prior.read_text())
  r['scope']='native-to-lean-staged endpoint diagnostic';r['cache_preseed']['destination']=str(root/'cache_v1');o=r['ownership'];rows=o.pop('shadow_rows');o['replacement_rows']=rows;o['removed_diagnostics']=dict(full_backing_oracle_calls=0,full_raw_comparisons=0);o['retained_guards']=dict.fromkeys(('payload_bounds_disjoint','typed_owner_before_after','host_source_readonly','native_metadata_pools_untouched','native_list_semantics','candidate_stream_unchanged'),140)
  for x in rows:x.update(native_returns=0,host_source_readonly=True,native_metadata_pools_untouched=True,original_addresses_preserved=True,full_raw_oracle_performed=False)
  for label in ('staged','fused'):
   k='request.'+label
   for suffix in ('attempts','returns'):o['counts'][k+'.candidate_'+suffix]=o['counts'].pop(k+'.native_'+suffix)
  self.assertTrue(gates.audit(r,root)['passed'])
  r['ownership']['counts']['request.staged.native_returns']=1;self.assertFalse(gates.audit(r,root)['passed'])
  r['ownership']['counts']['request.staged.native_returns']=0;rows[0]['host_source_readonly']=False;self.assertFalse(gates.audit(r,root)['passed'])
 def test_shared_failure_blocks_and_local_propagates(self):
  from staged_ownership_runtime_v1.test_observer import Tests as Mocks
  from .replacement import OwnershipObserver
  S,F,T,p,o=Mocks().make();S.clear_staged_writes=lambda self:None;o.__class__=OwnershipObserver;o.rows=[];o.destinations=[];o.controller=None
  state=types.SimpleNamespace(active=True,failed=None)
  def latch(*args):state.failed=True
  state.latch=latch;o.bind_controller(state)
  with o:
   a=S();a._staged_write_indices=[2];a._staged_write_starts=[0];a._staged_write_contents=[1];a._staged_write_cu_lens=[1];p[0]='request'
   with self.assertRaises(RuntimeError):a.apply_write()
   self.assertTrue(state.failed);self.assertEqual(a.n,0)
   with self.assertRaises(AssertionError):a.apply_write()
   self.assertEqual(a.n,0)
 def test_no_oracle_source_retains_lifetime_sync(self):
  from pathlib import Path
  import ast
  root=Path(__file__).parent;s=(root/'replacement.py').read_text();ast.parse(s)
  for forbidden in ('def raw(','.cpu(','.numpy(','.tobytes(', 'from .oracle'):self.assertNotIn(forbidden,s)
  api=(root.parent/'tileir_staged_typed_v2/api.py').read_text();self.assertIn('stream.synchronize()',api)
