import ast,sys,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import observer as o
from unittest.mock import patch
class A(list):
 def tolist(self):return list(self)
def batch(flag=True):return types.SimpleNamespace(req_ids=['a'],is_prefilling_np=A([flag]),has_prefill=flag,num_tokens=1,idx_mapping_np=A([0]),num_scheduled_tokens=A([1]),prefill_len_np=A([53]),num_computed_prefill_tokens_np=A([0]))
class Tests(unittest.TestCase):
 def setup_observer(self):
  obs=o.Observer.__new__(o.Observer);obs.bindings=[];obs.pending=None;obs.samples={'prefill':{},'decode':{}};obs.errors=[];obs.seen={};obs.progress=lambda:None;obs.closed=False;return obs
 def test_once_return_restore(self):
  obs=self.setup_observer();calls=[]
  def helper(*args):calls.append(args);return None
  module=types.SimpleNamespace(prepare_pos_seq_lens=helper)
  class Runner:
   pcp_manager=None
   def __init__(self):self.input_buffers=types.SimpleNamespace(positions=object(),seq_lens=object());self.req_states=types.SimpleNamespace(num_computed_tokens=types.SimpleNamespace(gpu=object()))
   def prepare_inputs(self,s,b,d,l):
    ret=module.prepare_pos_seq_lens('idx','prefix',self.req_states.num_computed_tokens.gpu,self.input_buffers.positions,self.input_buffers.seq_lens);assert ret is None;return b
  original=Runner.prepare_inputs;runner=Runner();b=batch()
  obs.install_classes(Runner,module)
  with patch.object(o,'phase_metadata',return_value=dict(phase='prefill',idx_mapping=[0],req_ids=['a'])):self.assertIs(runner.prepare_inputs(None,b,None,0),b)
  self.assertEqual(len(calls),1);self.assertIsNone(obs.pending);obs.close();self.assertIs(Runner.prepare_inputs,original);self.assertIs(module.prepare_pos_seq_lens,helper)
 def test_phase(self):
  self.assertEqual(o.state_metadata(batch())['phase'],'prefill');self.assertEqual(o.state_metadata(batch(False))['phase'],'decode')
  b=batch();b.has_prefill=False
  with self.assertRaises(AssertionError):o.state_metadata(b)
 def test_exception_cleanup(self):
  obs=self.setup_observer();module=types.SimpleNamespace(prepare_pos_seq_lens=lambda *a:None)
  class Runner:
   pcp_manager=None
   def prepare_inputs(self,*args):raise RuntimeError('expected')
  original=Runner.prepare_inputs;obs.install_classes(Runner,module)
  try:
   with self.assertRaises(RuntimeError):Runner().prepare_inputs(None,batch(),None,0)
  finally:obs.close()
  self.assertIsNone(obs.pending);self.assertIs(Runner.prepare_inputs,original)
 def test_no_gpu(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
