import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import observer as o
class Tests(unittest.TestCase):
 def setup_case(self,root,fail=False,mutate=False):
  x=NS(value=1);a=NS(value=2);b=NS(value=3);A=NS(value=4);dt=NS(value=5);ret=tuple(NS(value=i) for i in range(5));calls=[]
  def original(conv_output,a,b,A_log,dt_bias,num_k_heads,head_k_dim,head_v_dim,apply_l2norm=True,output_g_exp=False):
   calls.append(1)
   if fail:raise RuntimeError('native failure')
   if mutate:A_log.value+=1
   return ret
  q=NS(fused_post_conv_prep=original);ob=o.Observer(root);name='model.layers.0.linear_attn';m=NS(A_log=A,dt_bias=dt,prefix='layer0')
  ob.live={name:m};ob.modules={name:{}};ob.tensor=lambda t,label:{'storage_byte_sha256':str(t.value)}
  ctx=NS(attn_metadata={'layer0':NS(num_prefills=32,num_decodes=0,num_actual_tokens=1696,spec_sequence_masks=None)})
  ob.hook(q,lambda:ctx)
  return ob,q,original,(x,a,b,A,dt,16,128,128),ret,calls,ctx
 def test_exactly_once_return_inputs_phase(self):
  with tempfile.TemporaryDirectory() as t,patch.object(o,'digest',lambda v:str(v.value)),patch.object(o,'aliases',lambda v:{}):
   ob,q,orig,args,ret,calls,ctx=self.setup_case(Path(t)/'bank')
   self.assertIs(q.fused_post_conv_prep(*args),ret);self.assertEqual(len(calls),1)
   self.assertEqual(ob.samples[0]['phase']['num_prefills'],32);self.assertTrue(all(ob.samples[0]['input_storage_preserved'].values()))
   ob.close();self.assertIs(q.fused_post_conv_prep,orig)
 def test_native_failure_no_retry_and_restore(self):
  with tempfile.TemporaryDirectory() as t,patch.object(o,'digest',lambda v:str(v.value)),patch.object(o,'aliases',lambda v:{}):
   ob,q,orig,args,ret,calls,_=self.setup_case(Path(t)/'bank',fail=True)
   with self.assertRaises(RuntimeError):q.fused_post_conv_prep(*args)
   self.assertEqual(len(calls),1);self.assertTrue(ob.errors);ob.close();self.assertIs(q.fused_post_conv_prep,orig)
 def test_parameter_mutation_hold(self):
  with tempfile.TemporaryDirectory() as t,patch.object(o,'digest',lambda v:str(v.value)),patch.object(o,'aliases',lambda v:{}):
   ob,q,_,args,_,calls,_=self.setup_case(Path(t)/'bank',mutate=True)
   with self.assertRaises(AssertionError):q.fused_post_conv_prep(*args)
   self.assertEqual(len(calls),1);self.assertFalse(next(iter(ob.params.values()))['preserved']);ob.close()
 def test_decode_not_guessed_from_shape(self):
  with tempfile.TemporaryDirectory() as t:
   ob,q,_,args,_,calls,ctx=self.setup_case(Path(t)/'bank');ctx.attn_metadata['layer0'].num_prefills=0;ctx.attn_metadata['layer0'].num_decodes=32
   with self.assertRaises(AssertionError):q.fused_post_conv_prep(*args)
   self.assertEqual(calls,[]);ob.close()
 def test_foreign_binding_not_overwritten(self):
  with tempfile.TemporaryDirectory() as t:
   ob,q,_,_,_,_,_=self.setup_case(Path(t)/'bank');foreign=object();q.fused_post_conv_prep=foreign
   with self.assertRaises(RuntimeError):ob.close()
   self.assertIs(q.fused_post_conv_prep,foreign)
 def test_no_gpu_import(self):
  import sys
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
if __name__=='__main__':unittest.main()
