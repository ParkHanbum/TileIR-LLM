import unittest,ast,sys,types
from pathlib import Path
from unittest.mock import patch
from .contract import gather_descriptor,validate_indices,ones_descriptor
from .api import gather_integer_rows
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_signed_index_duplicate_and_edges(self):
  for n in (32,1696):
   ids=[-n,-1,0,n-1]*8;self.assertEqual(validate_indices(ids,n),[0,n-1,0,n-1]*8)
   for bad in (-n-1,n):
    with self.assertRaises(AssertionError):validate_indices([bad]+[0]*31,n)
 def test_descriptor(self):
  for n in (32,1696):
   for d in ('torch.int32','torch.int64'):self.assertEqual(gather_descriptor((n,),(1,),d,0,(32,),(1,),'torch.int64',0),n)
  self.assertEqual(ones_descriptor((32,),'torch.int32',32),32)
  with self.assertRaises(AssertionError):ones_descriptor((32,),'torch.int32',True)
 def test_exact_three_expression_delta(self):
  s=(R/'call.original.py.txt').read_text()
  for a,b in [('input_batch.positions[input_batch.logits_indices]','_integer_gather(input_batch.positions,input_batch.logits_indices)'),('input_batch.input_ids[input_batch.logits_indices]','_integer_gather(input_batch.input_ids,input_batch.logits_indices)'),('input_batch.seq_lens.new_ones(input_batch.num_reqs)','_new_sampled_ones(input_batch.seq_lens,input_batch.num_reqs)')]:s=s.replace(a,b)
  self.assertEqual(s,(R/'call.candidate.py.txt').read_text())
 def test_oob_actual_api_before_launch(self):
  class Tensor:
   shape=(32,);dtype='torch.int64';device=types.SimpleNamespace(type='cuda');requires_grad=False
   def stride(self):return(1,)
   def storage_offset(self):return 0
   def is_neg(self):return False
   def is_conj(self):return False
   def cpu(self):return self
   def tolist(self):return[32]+[0]*31
  calls=[]
  with patch.dict(sys.modules,{'torch':types.SimpleNamespace(Tensor=Tensor,empty=lambda *a,**k:calls.append(1))}):
   with self.assertRaisesRegex(AssertionError,'Index outside'):gather_integer_rows(Tensor(),Tensor())
  self.assertEqual(calls,[])
 def test_ast_no_framework(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
  for n in ('torch','vllm','cuda.tile'):self.assertNotIn(n,sys.modules)
if __name__=='__main__':unittest.main()
