"""CPU mock contract tests only: no torch/CUDA import or numerical equivalence claim."""
import sys,types,unittest
from contextlib import contextmanager
from unittest.mock import patch
from .recurrent import PrefillConv,DecodeConv,PostConv,GDNDecode
class Controller:
 def __init__(self):self.active=True;self.failed=None
 def latch(self,key,error):
  if self.failed is None:self.failed=(key,str(error))
class Tensor:
 def __init__(self,shape=(),values=None,root=None,strides=()):
  self.shape=shape;self.values=values;self.root=self if root is None else root;self.strides=strides
 def detach(self):return Tensor(self.shape,self.values,self.root,self.strides)
 def data_ptr(self):return id(self.root)
 def stride(self):return self.strides
 def cpu(self):return self
 def tolist(self):return self.values
 def all(self):return True

def prefill(x,weight,conv_states,cache_indices,query_start_loc,has_initial_state,bias=None,activation='silu',null_block_id=0,pad_slot_id=-1,block_size_to_align=0,block_idx_first_scheduled_token=None,block_idx_last_scheduled_token=None,initial_state_idx=None,num_computed_tokens=None):raise AssertionError('native prefill forbidden')
def decode(x,conv_state,weight,conv_state_indices,bias=None,out=None,activation='silu',null_block_id=0,max_query_len=-1,num_accepted_tokens=None,query_start_loc=None,block_idx_last_scheduled_token=None,initial_state_idx=None):raise AssertionError('native decode forbidden')
def post(conv_output,a,b,A_log,dt_bias):raise AssertionError('native post forbidden')
def gdn(A_log,dt_bias,initial_state,out,ssm_state_indices,scale=128**-0.5,use_qk_l2norm_in_kernel=True):raise AssertionError('native GDN forbidden')
@contextmanager
def fixture(fail=None,bad_gdn=False):
 modules={}
 def module(name,**attrs):
  parts=name.split('.')
  for i in range(1,len(parts)+1):
   key='.'.join(parts[:i])
   if key not in modules:modules[key]=types.ModuleType(key);modules[key].__path__=[]
   if i>1:setattr(modules['.'.join(parts[:i-1])],parts[i-1],modules[key])
  modules[name].__dict__.update(attrs);return modules[name]
 q=module('vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn',causal_conv1d_fn=prefill,causal_conv1d_update=decode,fused_post_conv_prep=post,fused_recurrent_gated_delta_rule_packed_decode=gdn)
 owners=[(f'model.layers.{i}.linear_attn',types.SimpleNamespace(conv1d=types.SimpleNamespace(weight=Tensor((8192,3))),A_log=Tensor(),dt_bias=Tensor(),prefix=str(i)))for i in range(24)]
 model=types.SimpleNamespace(named_modules=lambda:iter(owners));events=[]
 def candidate(key):
  def fn(*args,**kwargs):
   events.append((key,args,kwargs))
   if fail==key:raise ValueError('candidate failure')
   if key=='gdn':return (object(),kwargs['initial_state'])if bad_gdn else(kwargs['out'],kwargs['initial_state'])
   if key=='decode':return args[0].detach()
   return object()
  return fn
 module('tileir_indexed_prefill_conv_v1',causal_conv1d_prefill_indexed=candidate('prefill'))
 module('tileir_indexed_decode_conv_v1',causal_conv1d_update_indexed=candidate('decode'))
 module('tileir_vllm_post_conv_v2.api',fused_post_conv_prep=candidate('post'))
 module('tileir_gdn_decode_core_v4',packed_decode=candidate('gdn'))
 module('vllm.forward_context',get_forward_context=lambda:types.SimpleNamespace(attn_metadata={str(i):'prefill'for i in range(24)}))
 module('gdn_post_conv_runtime_capture_v1.observer',phase_meta=lambda m:dict(phase=m,speculative=False))
 module('torch',isfinite=lambda x:Tensor())
 with patch.dict(sys.modules,modules):yield q,model,owners,events
class RecurrentTests(unittest.TestCase):
 def install(self,model):
  c=Controller();objects=[PrefillConv(c,'prefill',24),DecodeConv(c,'decode',360),PostConv(c,'post',24),GDNDecode(c,'gdn',360)]
  for o in objects:o.install(model);o.start()
  return c,objects
 def inputs(self,m):return Tensor((8192,1696)),Tensor((40,8192,3)),m.conv1d.weight,Tensor((32,)),Tensor((33,),list(range(0,1697,53))),Tensor((32,),[False]*32)
 def test_actual_install_chain_aliases_and_reverse_restore(self):
  with fixture()as(q,model,owners,events):
   old=dict(vars(q));c,objects=self.install(model)
   try:
    for name,m in owners:
     x,st,w,idx,starts,flags=self.inputs(m)
     q.causal_conv1d_fn(x,w,st,idx,starts,flags)
     key,args,kw=events[-1];self.assertEqual(key,'prefill')
     for alias,original in zip(args[:3],(x,w,st)):self.assertIsNot(alias,original);self.assertIs(alias.root,original.root)
     self.assertIs(args[3],starts);self.assertIs(args[4],idx);self.assertIs(args[5],flags)
     for _ in range(15):
      dx=Tensor((32,8192));out=q.causal_conv1d_update(dx,st,w,idx)
      self.assertIs(out.root,dx);self.assertIsNot(out,dx)
      for alias,original in zip(events[-1][1],(dx,st,w,idx)):self.assertIsNot(alias,original);self.assertIs(alias.root,original)
     vals=[Tensor(),Tensor(),Tensor(),m.A_log,m.dt_bias];q.fused_post_conv_prep(*vals)
     for key,original in zip(('conv_output','a','b','A_log','dt_bias'),vals):self.assertIs(events[-1][2][key].root,original)
     gst=Tensor((40,32,128,128),strides=(524288,16384,128,1));gout=Tensor();slots=Tensor((32,),list(range(1,33)))
     for _ in range(15):
      result=q.fused_recurrent_gated_delta_rule_packed_decode(m.A_log,m.dt_bias,gst,gout,slots)
      self.assertIs(result[0],gout);self.assertIs(result[1],gst)
      self.assertIs(events[-1][2]['initial_state'].root,gst);self.assertIs(events[-1][2]['out'].root,gout)
    for o in objects:o.finish()
    self.assertEqual(objects[1].delegated_prefill,24)
   finally:
    for o in reversed(objects):o.close()
   for key in('causal_conv1d_fn','causal_conv1d_update','fused_post_conv_prep','fused_recurrent_gated_delta_rule_packed_decode'):self.assertIs(getattr(q,key),old[key])
 def test_candidate_failure_latches_and_blocks_next_call(self):
  with fixture(fail='decode')as(q,model,owners,events):
   c,objects=self.install(model);m=owners[0][1];x,st,w,idx,starts,flags=self.inputs(m)
   try:
    with self.assertRaises(ValueError):q.causal_conv1d_update(Tensor((32,8192)),st,w,idx)
    n=len(events);self.assertIsNotNone(c.failed)
    with self.assertRaises(AssertionError):q.causal_conv1d_fn(x,w,st,idx,starts,flags)
    self.assertEqual(len(events),n)
   finally:
    for o in reversed(objects):o.close()
 def test_scope_guard_latches_without_candidate(self):
  with fixture()as(q,model,owners,events):
   c,objects=self.install(model);x,st,w,idx,starts,flags=self.inputs(owners[0][1]);flags.values[0]=True
   try:
    with self.assertRaises(AssertionError):q.causal_conv1d_fn(x,w,st,idx,starts,flags)
    self.assertFalse(events);self.assertIsNotNone(c.failed)
   finally:
    for o in reversed(objects):o.close()
 def test_gdn_wrong_return_identity_latches(self):
  with fixture(bad_gdn=True)as(q,model,owners,events):
   c,objects=self.install(model);m=owners[0][1]
   try:
    with self.assertRaises(AssertionError):q.fused_recurrent_gated_delta_rule_packed_decode(m.A_log,m.dt_bias,Tensor((40,32,128,128),strides=(524288,16384,128,1)),Tensor(),Tensor((32,),list(range(1,33))))
    self.assertIsNotNone(c.failed);self.assertEqual(len(events),1)
   finally:
    for o in reversed(objects):o.close()
 def test_inactive_controller_never_enters_candidate(self):
  with fixture()as(q,model,owners,events):
   c,objects=self.install(model);c.active=False;x,st,w,idx,starts,flags=self.inputs(owners[0][1])
   try:
    with self.assertRaises(AssertionError):q.causal_conv1d_fn(x,w,st,idx,starts,flags)
    self.assertFalse(events);self.assertIsNotNone(c.failed)
   finally:
    for o in reversed(objects):o.close()
if __name__=='__main__':unittest.main()
