"""Actual Shadow.call path, stdlib fake tensors only; no framework imports."""
import copy,inspect,json,tempfile,types,unittest
from unittest.mock import patch,Mock
from . import api
from .shadow import Shadow
class Tensor:
 def __init__(self,x):self.x=copy.deepcopy(x)
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return copy.deepcopy(self.x)
 def clone(self,**kw):return Tensor(self.x)
 def contiguous(self):return self
 def view(self,*a):return self
 def numpy(self):return types.SimpleNamespace(tobytes=lambda:json.dumps(self.x).encode())
FAKE=types.SimpleNamespace(Tensor=Tensor,uint8=object(),preserve_format=object())
def bias_args():
 return dict(logits=Tensor([[1.]]),expanded_idx_mapping=Tensor(list(range(32))),pos=Tensor([67]*32),num_allowed_token_ids=Tensor([0]*32),allowed_token_ids=Tensor([]),num_logit_bias=Tensor([0]*32),logit_bias_token_ids=Tensor([]),logit_bias=Tensor([]),min_lens=Tensor([69]*32),num_stop_token_ids=Tensor([1]*32),restore_when_all_masked=Tensor([0]*32),stop_token_ids=Tensor([[42]+[-1]*127 for _ in range(32)]),check_all_masked_rows=False)
def combine_args():
 return dict(input_ids=Tensor([]),idx_mapping=Tensor(list(range(32))),last_sampled_tokens=Tensor([[42]for _ in range(32)]),query_start_loc=Tensor(list(range(33))),seq_lens=Tensor([54]*32),prefill_len=Tensor([53]*32),draft_tokens=Tensor([]),cu_num_logits=Tensor(list(range(33))),num_logits=32,num_new_sampled_tokens=1)
class Tests(unittest.TestCase):
 def rejected(self,label,args,signature):
  native=Mock(return_value=None);native.__signature__=signature;candidate=Mock(side_effect=AssertionError('candidate must not run'))
  with tempfile.TemporaryDirectory()as d,patch.dict('sys.modules',{'torch':FAKE}),patch.object(api,'validate_layout'),patch.object(api,'apply_logit_bias',candidate),patch.object(api,'combine_sampled_and_draft_tokens',candidate),patch.object(api,'prepare_rope_positions',candidate):
   s=Shadow(d);s.active=True;wrapped=s.wrap(native,label)
   with self.assertRaises(ValueError):wrapped(**args)
   self.assertTrue(s.failed);self.assertEqual(native.call_count,0);self.assertEqual(candidate.call_count,0)
   self.assertEqual(s.rows[0]['stage'],'preflight');self.assertEqual(s.rows[0]['native_calls'],0);self.assertEqual(s.rows[0]['candidate_calls'],0)
   with self.assertRaises(RuntimeError):wrapped(**args)
   self.assertEqual(native.call_count,0);self.assertEqual(candidate.call_count,0)
 def test_unsafe_index(self):
  a=combine_args();a['idx_mapping']=Tensor([32]*32);self.rejected('combine_sampled_and_draft_tokens',a,inspect.signature(api.combine_sampled_and_draft_tokens))
 def test_query_out_of_bounds(self):
  a=combine_args();a['query_start_loc']=Tensor(list(range(32))+[2049]);self.rejected('combine_sampled_and_draft_tokens',a,inspect.signature(api.combine_sampled_and_draft_tokens))
 def test_unsupported_no_new(self):
  a=combine_args();a['num_new_sampled_tokens']=0;self.rejected('combine_sampled_and_draft_tokens',a,inspect.signature(api.combine_sampled_and_draft_tokens))
 def test_unsafe_active_stop(self):
  a=bias_args();a['stop_token_ids']=Tensor([[-1]*128 for _ in range(32)]);self.rejected('bias',a,inspect.signature(api.apply_logit_bias))
 def test_unsupported_bias(self):
  a=bias_args();a['num_logit_bias']=Tensor([1]*32);self.rejected('bias',a,inspect.signature(api.apply_logit_bias))
 def test_unsafe_rope_extent(self):
  def original(self,idx_mapping,query_start_loc,prefill_lens,num_computed_tokens):pass
  owner=types.SimpleNamespace(num_dims=3,max_model_len=128,positions=Tensor([]),prefill_positions=types.SimpleNamespace(gpu=Tensor([])),prefill_delta=types.SimpleNamespace(gpu=Tensor([0]*32)))
  a=dict(self=owner,idx_mapping=Tensor(list(range(32))),query_start_loc=Tensor([53*i for i in range(33)]),prefill_lens=Tensor([200]*32),num_computed_tokens=Tensor([100]*32))
  self.rejected('prepare_rope_positions',a,inspect.signature(original))
 def test_valid_actual_call_once_and_original_return(self):
  a=bias_args();a['pos']=Tensor([68]*32) # Actual no-op source branch.
  native=Mock(return_value=None);native.__signature__=inspect.signature(api.apply_logit_bias);candidate=Mock(return_value=None)
  with tempfile.TemporaryDirectory()as d,patch.dict('sys.modules',{'torch':FAKE}),patch.object(api,'validate_layout'),patch.object(api,'apply_logit_bias',candidate):
   s=Shadow(d);s.active=True;out=s.wrap(native,'bias')(**a)
   self.assertIsNone(out);self.assertEqual(native.call_count,1);self.assertEqual(candidate.call_count,1)
   self.assertFalse(s.failed);self.assertTrue(s.rows[0]['passed']);self.assertTrue(s.rows[0]['preflight_passed']);self.assertTrue(s.rows[0]['original_return_preserved'])
if __name__=='__main__':unittest.main()
