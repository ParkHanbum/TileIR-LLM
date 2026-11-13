import unittest,ast,json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from .adapter import Adapter,disjoint_intervals
R=Path(__file__).parent
class T(unittest.TestCase):
 def setUp(self):
  self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA'));self.g=self.guard.start();self.a=Adapter('/tmp/unwritten/shadow');self.native=[]
  def scatter(i,s,d):
   for idx,v in zip(i.tolist(),s.tolist()):
    if idx>=0:d[idx]=max(v,1)
   return None
  self.patch=patch('tileir_vllm_mamba_accepted_scatter_v1.api.scatter_num_accepted_',scatter);self.patch.start()
  def native(*a):self.native.append(1);return scatter(*a)
  self.a.native=native
  self.a.copied=lambda state,i,s,c:self.a.scatter(i,s,state.num_accepted_tokens_gpu)
 def tearDown(self):self.g.assert_not_called();self.guard.stop();self.patch.stop()
 def test_model_counts(self):
  for mode in ('shadow','replace'):
   self.a.start(mode)
   for j in range(16):
    state=NS(recoverssm=None,_align_mode=False,num_accepted_tokens_gpu=torch.full((32,),-99,dtype=torch.int32));self.assertIsNone(self.a.invoke(state,torch.arange(32,dtype=torch.int64),torch.arange(-2,30,dtype=torch.int32)))
   d=self.a.finish();self.assertEqual(d['native_scatter_calls'],16 if mode=='shadow' else 0)
  self.assertEqual(len(self.native),16)
 def test_partial_overlap(self):
  with self.assertRaises(AssertionError):disjoint_intervals([(100,128),(164,128),(400,256)])
  disjoint_intervals([(100,128),(228,128),(500,256)])
 def test_reject_optional_before_native(self):
  self.a.start('shadow');state=NS(recoverssm=object(),_align_mode=False,num_accepted_tokens_gpu=None)
  with self.assertRaises(AssertionError):self.a.invoke(state,None,None)
  with self.assertRaises(AssertionError):self.a.invoke(state,None,None)
  self.assertEqual(self.native,[])
 def test_oob_before_native(self):
  self.a.start('shadow');state=NS(recoverssm=None,_align_mode=False,num_accepted_tokens_gpu=torch.zeros(32,dtype=torch.int32));idx=torch.arange(1,33,dtype=torch.int64)
  with self.assertRaises(AssertionError):self.a.invoke(state,idx,torch.ones(32,dtype=torch.int32))
  self.assertEqual(self.native,[])
 def test_wrong_candidate_latch(self):
  self.a.start('shadow');state=NS(recoverssm=None,_align_mode=False,num_accepted_tokens_gpu=torch.zeros(32,dtype=torch.int32));idx=torch.arange(32,dtype=torch.int64);s=torch.ones(32,dtype=torch.int32)
  with patch('tileir_vllm_mamba_accepted_scatter_v1.api.scatter_num_accepted_',lambda *x:None),patch.object(Path,'mkdir'),patch('torch.save')as save:
   with self.assertRaises(AssertionError):self.a.invoke(state,idx,s)
   with self.assertRaises(AssertionError):self.a.invoke(state,idx,s)
   self.assertEqual(len(self.native),1);self.assertEqual(save.call_count,1)
 def test_actual_copied_path_rejects_scalar_and_empty_without_dispatch(self):
  class Kernel:
   def __init__(self):self.calls=0
   def __getitem__(self,grid):
    def invoke(*a):self.calls+=1
    return invoke
  for empty in (False,True):
   a=Adapter('/tmp/unwritten/shadow');fill=Kernel();native=Kernel();selected=[]
   ns=dict(torch=torch,_fill_num_accepted_kernel=fill,_scatter_num_accepted_kernel=native,_selected_scatter=lambda *x:selected.append(x))
   exec(compile('from __future__ import annotations\n'+(R/'postprocess.copied.py.txt').read_text(),'public-copy','exec'),ns)
   a.copied=ns['postprocess_state'];a.start('shadow');state=NS(recoverssm=None,_align_mode=False,num_accepted_tokens_gpu=torch.zeros(32,dtype=torch.int32))
   idx=torch.empty(0,dtype=torch.int64)if empty else torch.arange(32,dtype=torch.int64);sampled=torch.empty(0,dtype=torch.int32)if empty else 1
   with self.assertRaises(AssertionError):a.invoke(state,idx,sampled)
   self.assertEqual(fill.calls,0);self.assertEqual(native.calls,0);self.assertEqual(selected,[]);self.assertTrue(a.failed)
 def test_delta(self):
  old='_scatter_num_accepted_kernel[(num_reqs,)](\n                idx_mapping,\n                num_sampled,\n                self.num_accepted_tokens_gpu,\n            )'
  original=(R/'postprocess.original.py.txt').read_text();self.assertEqual(original.replace(old,'_selected_scatter(idx_mapping, num_sampled, self.num_accepted_tokens_gpu)'),(R/'postprocess.copied.py.txt').read_text())
 def test_ast(self):
  for f in R.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
