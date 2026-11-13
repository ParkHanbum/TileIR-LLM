import unittest,struct,types,json
from .oracle import expected,read
from .ownership import Owners
from .cleanup import close_and_finish
def packed(code,values):return struct.pack('<'+code*len(values),*values)
class Tests(unittest.TestCase):
 def test_combine_prefill_decode_untouched(self):
  raw=dict(input_ids=packed('i',[-9]*40),idx_mapping=packed('q',range(32)),query_start_loc=packed('i',range(33)),seq_lens=packed('i',[53]+[54]*31),prefill_len=packed('i',[53]*32),last_sampled_tokens=packed('q',[42]*32))
  out,ret=expected('combine_sampled_and_draft_tokens',raw,{})
  self.assertEqual(read(out['input_ids'],'i'),[-9]+[42]*31+[-9]*8);self.assertEqual(read(ret,'q'),list(range(32)))
 def test_rope_integer_and_tail(self):
  raw={'self.positions':packed('q',[-1]*(3*2049)),'idx_mapping':packed('q',range(32)),'query_start_loc':packed('i',range(33)),'prefill_lens':packed('i',[1]*32),'num_computed_tokens':packed('i',[0]+[3]*31),'self.prefill_delta.gpu':packed('i',[0]+[2]*31),'self.prefill_positions.gpu':packed('i',range(96*128))}
  out,ret=expected('prepare_rope_positions',raw,{})
  p=read(out['self.positions'],'q');self.assertIsNone(ret);self.assertEqual([p[axis*2049]for axis in range(3)],[0,128,256]);self.assertEqual(p[1:32],[5]*31);self.assertEqual(p[32:2049],[-1]*2017)
 def test_bias_inf_payload_preserved(self):
  raw=dict(logits=bytes.fromhex('0000807f0100c07f'),expanded_idx_mapping=packed('q',[0]*32),pos=packed('q',[67]+[68]*31),min_lens=packed('i',[69]*32),num_stop_token_ids=packed('i',[1]*32),stop_token_ids=packed('i',[0]*(32*128)))
  out,_=expected('bias',raw,{});self.assertEqual(out['logits'],bytes.fromhex('000080ff0100c07f'))
 def test_post_cast_and_skip(self):
  raw=dict(idx_mapping=packed('q',[0]+[-1]*31),num_sampled=packed('i',[1]+[0]*31),num_rejected=packed('i',[0]*32),query_start_loc=packed('i',range(33)),sampled_tokens=packed('q',[0]*32),total_len=packed('i',[0]*32),num_computed_tokens=packed('i',[0]*32),last_sampled_tokens=packed('q',[-1]*32),output_bin_counts=packed('i',[0]),all_token_ids=packed('i',[-1]*(32*128)))
  out,_=expected('post_update',raw,{})
  self.assertEqual(read(out['num_computed_tokens'],'i'),[1]+[0]*31);self.assertEqual(read(out['output_bin_counts'],'i'),[1]);self.assertEqual(read(out['all_token_ids'],'i')[:2],[0,-1]);self.assertEqual(read(out['total_len'],'i'),[1]+[0]*31)
 def test_ownership_no_raw_address_log(self):
  def tensor(start,size):
   storage=types.SimpleNamespace(data_ptr=lambda:start,nbytes=lambda:size)
   return types.SimpleNamespace(untyped_storage=lambda:storage,numel=lambda:size//4,element_size=lambda:4,data_ptr=lambda:start,storage_offset=lambda:0,shape=(size//4,),stride=lambda:(1,),dtype='int32',device=types.SimpleNamespace(type='cuda'))
  o=Owners();cpu=tensor(123400,16);cpu.device=types.SimpleNamespace(type='cpu');cpu.is_pinned=lambda:True;owner=types.SimpleNamespace(cpu=cpu,_uva=tensor(123400,16));o.capture(owner)
  rows=o.retain({'v':tensor(123400,16)},('v',));self.assertEqual(rows['v']['public_uva_owner_matches'],1);self.assertNotIn('123400',json.dumps(rows));self.assertIs(o.owners[0],owner)
  with self.assertRaises(AssertionError):o.release(False)
  o.close();self.assertTrue(o.release(True)['released'])
 def test_missing_required_owner_holds(self):
  o=Owners();t=types.SimpleNamespace(untyped_storage=lambda:types.SimpleNamespace(data_ptr=lambda:100,nbytes=lambda:4))
  with self.assertRaises(AssertionError):o.retain({'x':t},('x',))
 def test_cleanup_attempts_final_after_helper_error(self):
  events=[];r={};replacement=types.SimpleNamespace(close=lambda:(_ for _ in ()).throw(ValueError('conflict')),summary=lambda:{},save=lambda:None)
  def final(propagating):events.append(propagating)
  with self.assertRaises(RuntimeError):close_and_finish(r,lambda:None,None,replacement,final,propagating=False)
  self.assertEqual(events,[True]);self.assertFalse(r['candidate_checks_passed'])
if __name__=='__main__':unittest.main()
class ReviewFixTests(unittest.TestCase):
 def controller(self):
  c=types.SimpleNamespace(active=True,failed=None)
  def latch(key,error):
   if c.failed is None:c.failed=dict(key=key,error=repr(error))
  c.latch=latch;return c
 def test_cap_before_any_call_and_shared_latch(self):
  import tempfile
  from .adapter import Replacement
  with tempfile.TemporaryDirectory()as d:
   a=Replacement(d,Owners());a.controller=self.controller();a.active=True;a.rows=[dict(label='bias')for _ in range(16)];calls=[];a.call=lambda *x:calls.append('candidate')
   f=a.wrap(lambda:calls.append('native'),'bias')
   with self.assertRaises(AssertionError):f()
   self.assertEqual(calls,[]);self.assertTrue(a.failed);self.assertIsNotNone(a.controller.failed)
 def test_lean_failure_blocks_replacement(self):
  import tempfile
  from .adapter import Replacement
  with tempfile.TemporaryDirectory()as d:
   a=Replacement(d,Owners());a.controller=self.controller();a.controller.failed={'original':'error'};a.active=True;calls=[];a.call=lambda *x:calls.append(1)
   with self.assertRaises(RuntimeError):a.wrap(lambda:calls.append(2),'bias')()
   self.assertEqual(calls,[])
 def test_owned_binding_drift_pre_call(self):
  import tempfile
  from .adapter import Replacement
  with tempfile.TemporaryDirectory()as d:
   a=Replacement(d,Owners());a.controller=self.controller();a.active=True;expected=object();a.owned_checks=[({'helper':object()},'helper',expected)];calls=[];a.call=lambda *x:calls.append(1)
   with self.assertRaises(AssertionError):a.wrap(lambda:calls.append(2),'bias')()
   self.assertEqual(calls,[]);self.assertIsNotNone(a.controller.failed)
 def test_cleanup_error_never_releases_owner(self):
  calls=[];r={}
  o=types.SimpleNamespace(restored=False)
  def close():o.restored=True
  o.close=close;o.release=lambda ok:calls.append('released')
  rep=types.SimpleNamespace(restored=False,close=lambda:(_ for _ in ()).throw(ValueError('failed')),summary=lambda:{},save=lambda:None)
  def final(p):r.update(synchronized_before_close=True,synchronized_after_close=True,engine_shutdown=True)
  with self.assertRaises(RuntimeError):close_and_finish(r,lambda:None,o,rep,final,propagating=False)
  self.assertEqual(calls,[])
class IdentityTests(unittest.TestCase):
 def tensor(self,start=100,size=16):
  st=types.SimpleNamespace(data_ptr=lambda:start,nbytes=lambda:size)
  return types.SimpleNamespace(untyped_storage=lambda:st,numel=lambda:size//4,element_size=lambda:4,data_ptr=lambda:start,storage_offset=lambda:0,shape=(size//4,),stride=lambda:(1,),dtype='int32',device=types.SimpleNamespace(type='cpu'),is_pinned=lambda:True)
 def test_owner_object_drift_rejected(self):
  o=Owners();owner=types.SimpleNamespace(cpu=self.tensor(),_uva=self.tensor());o.capture(owner);owner.cpu=self.tensor()
  with self.assertRaises(AssertionError):o.retain({'x':self.tensor()},('x',))
 def test_default_constructor_called_once_and_restored(self):
  from unittest.mock import patch
  calls=[];test=self
  class Uva:
   def __init__(self,*a,**kw):calls.append((a,kw));self.cpu=test.tensor();self._uva=test.tensor()
  original=Uva.__init__;mod=types.ModuleType('vllm.v1.worker.gpu.buffer_utils');mod.UvaBuffer=Uva
  with patch.dict('sys.modules',{'vllm.v1.worker.gpu.buffer_utils':mod}):
   o=Owners();o.install();instance=Uva(32,dtype='int32');self.assertEqual(calls,[((32,),{'dtype':'int32'})]);self.assertIs(o.owners[0],instance);o.close();self.assertIs(Uva.__init__,original)
 def test_exact_owned_globals_bind(self):
  import tempfile
  from unittest.mock import patch
  from .adapter import Replacement
  combine=lambda:None;bias=lambda:None;ns={'combine_sampled_and_draft_tokens':combine}
  exec(compile('def prepare_inputs(): return combine_sampled_and_draft_tokens()','/tmp/runner_padding_runtime_integration_v1/prepare_inputs.copied.py.txt','exec'),ns)
  copied=ns['prepare_inputs']
  def wrapper():return copied()
  module=types.ModuleType('vllm.v1.worker.gpu.sample.logit_bias');bn={'apply_logit_bias':bias};exec('def apply(self): return apply_logit_bias()',bn);module.LogitBiasState=type('LogitBiasState',(),{'apply':bn['apply']})
  obj=types.SimpleNamespace(combine_sampled_and_draft_tokens=combine,apply_logit_bias=bias);part=types.SimpleNamespace(key='runner_padding',bindings=[(None,None,wrapper)])
  with tempfile.TemporaryDirectory()as d,patch.dict('sys.modules',{'vllm.v1.worker.gpu.sample.logit_bias':module}):
   a=Replacement(d,Owners());a.bindings=[(obj,'combine_sampled_and_draft_tokens',None,combine),(obj,'apply_logit_bias',None,bias)]
   result=a.bind_controller(types.SimpleNamespace(parts=[part]));self.assertTrue(result['owned_copy_identity'])
   ns['combine_sampled_and_draft_tokens']=lambda:None
   with self.assertRaises(AssertionError):a.bind_controller(types.SimpleNamespace(parts=[part]))
