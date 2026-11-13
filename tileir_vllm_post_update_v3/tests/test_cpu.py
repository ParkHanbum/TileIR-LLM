import ast,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P))
from tileir_vllm_post_update_v3 import contract as c,shadow as s
class Tensor:
 def __init__(self,e,start):self.shape=e['shape'];self.dtype=e['dtype'];self.device=e['device'];self.layout='torch.strided';self.requires_grad=False;self.e=e;self.start=start
 def stride(self):return self.e['stride']
 def storage_offset(self):return self.e['offset']
 def is_conj(self):return False
 def is_neg(self):return False
 def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.start,nbytes=lambda:self.e['storage_bytes'])
 def clone(self):return self

def args():return tuple(Tensor(dict(c.EXPECTED[n]),(i+1)*100000000) for i,n in enumerate(c.NAMES))
def original(idx_mapping,num_computed_tokens,last_sampled_tokens,output_bin_counts,sampled_tokens,num_sampled,num_rejected,query_start_loc,all_token_ids,total_len):return None
class CPU(unittest.TestCase):
 def test_observed_metadata_accepts(self):c.validate_metadata(args())
 def test_negative_offset_shape_and_alias(self):
  for kind in ['offset','shape','alias','none']:
   aa=list(args())
   if kind=='offset':aa[0].e['offset']=1
   if kind=='shape':aa[0].shape=[31]
   if kind=='alias':aa[1].start=aa[0].start+4
   if kind=='none':aa[3]=None
   with self.assertRaises(ValueError):c.validate_metadata(aa)
 def test_limit_validation(self):
  for n in [0,17,True]:
   with self.assertRaises(ValueError):s.PostShadow(original,None,n)
 def test_native_once_passthrough_after_limit(self):
  torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda device:SimpleNamespace(synchronize=lambda:None)))
  w=s.PostShadow(original,torch,1);a=args()
  with patch.object(s,'clone_inputs',return_value=a) as clone,patch.object(s,'validate_values'),patch.object(s,'_launch_owned') as launch,patch.object(s,'_raw_equal',return_value=True):
   self.assertIsNone(w(*a));self.assertIsNone(w(*a));clone.assert_called_once();launch.assert_called_once()
  self.assertEqual((w.calls,w.native_attempts,w.native_returns,w.shadow_passed),(2,2,2,1))
 def test_uncertain_candidate_failure_is_failstop(self):
  w=s.PostShadow(original,None);a=args()
  with patch.object(s,'clone_inputs',return_value=a),patch.object(s,'validate_values'),patch.object(s,'_launch_owned',side_effect=RuntimeError('uncertain')) as launch:
   with self.assertRaises(RuntimeError):w(*a)
   with self.assertRaises(RuntimeError):w(*a)
   launch.assert_called_once()
  self.assertEqual((w.native_attempts,w.native_returns,w.candidate_attempts,w.candidate_enqueues),(1,1,1,0))
 def test_compile_ast_only(self):
  for p in (P/'tileir_vllm_post_update_v3').glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
