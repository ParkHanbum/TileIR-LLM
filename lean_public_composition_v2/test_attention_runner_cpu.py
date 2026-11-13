import unittest,sys,types,ast,inspect
from collections import Counter
from unittest.mock import patch
from lean_public_composition_v2.attention_runner import RunnerPadding,HiddenGather,AcceptedScatter,AttentionCore,RunnerCensus,disjoint,owned_copy
class Controller:
 active=True;failed=None
 def latch(self,key,e):self.failed=(key,e)
class Tests(unittest.TestCase):
 def test_runner_census(self):
  a=RunnerCensus(Controller(),'x',16);a.counts=Counter({('runner','prefill'):1,('runner','decode'):15});self.assertEqual(a.finish()['calls'],16)
  a.counts=Counter({('runner','decode'):16})
  with self.assertRaises(AssertionError):a.finish()
 def test_padding_bridge_and_output_identity(self):
  a=RunnerPadding(Controller(),'runner_padding',16);a.positions=types.SimpleNamespace(pending=None);a.kept=[]
  def original(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras):pass
  a.original=original;b=types.SimpleNamespace(req_ids=list(range(32)));r=types.SimpleNamespace(pcp_manager=None,model_state=types.SimpleNamespace(num_new_sampled_tokens_per_step=1));s=types.SimpleNamespace(scheduled_spec_decode_tokens={})
  m=types.ModuleType('pos_seq_runtime_capture_v2.observer');m.state_metadata=lambda b:dict(phase='prefill')
  def copied(*args):
   self.assertIs(a.positions.pending,a.pending);a.positions.pending['calls']+=1;a.current=[object(),object()];a.fill_current=[False,True];idx=object()
   return types.SimpleNamespace(cu_num_logits=a.current[0],expanded_local_pos=a.current[1],req_ids=b.req_ids,idx_mapping=idx,expanded_idx_mapping=idx,num_reqs=32,num_draft_tokens=0)
  with patch.dict(sys.modules,{'pos_seq_runtime_capture_v2.observer':m}):
   out=a.invoke(copied,r,s,b,None,0);self.assertIs(out.cu_num_logits,a.current[0]);self.assertIsNone(a.pending);self.assertIsNone(a.positions.pending)
   with self.assertRaises(RuntimeError):a.invoke(lambda *x:(_ for _ in()).throw(RuntimeError('failure')),r,s,b,None,0)
   self.assertIsNone(a.pending);self.assertIsNone(a.positions.pending)
 def test_hidden_original_order_and_return(self):
  a=HiddenGather(Controller(),'hidden_gather',16);a.model=object();r=types.SimpleNamespace(model=a.model,batch_sharder=None);ret=object();events=[]
  def copy(*args):events.append('copied');a.current=object();return ret
  self.assertIs(a.invoke(copy,r),ret);self.assertEqual(events,['copied']);self.assertIsNone(a.pending)
  with self.assertRaises(RuntimeError):a.invoke(lambda *x:(_ for _ in()).throw(RuntimeError()),r)
  self.assertIsNone(a.pending)
 def test_alias_guard(self):
  def t(p,n):return types.SimpleNamespace(untyped_storage=lambda:types.SimpleNamespace(data_ptr=lambda:p,nbytes=lambda:n))
  disjoint(t(0,4),t(4,4))
  with self.assertRaises(AssertionError):disjoint(t(0,5),t(4,4))
 def test_exact_reverse_restore_and_latch(self):
  c=Controller();a=RunnerCensus(c,'x',1);obj=types.SimpleNamespace(dispatch=17,method=lambda:'old');old=obj.method
  a.bind(obj,'method',lambda:'new');self.assertEqual(obj.method(),'new');self.assertEqual(obj.dispatch,17);a.close();self.assertIs(obj.method,old)
  a.bind(obj,'method',lambda:(_ for _ in()).throw(ValueError('fail')))
  with self.assertRaises(ValueError):obj.method()
  self.assertIsNotNone(c.failed)
  with self.assertRaises(AssertionError):obj.method()
  a.close()
 def test_no_diagnostics_and_scope_guards(self):
  import lean_public_composition_v2.attention_runner as m
  s=inspect.getsource(m);tree=ast.parse(s)
  self.assertNotIn('torch.cuda.synchronize',s);self.assertNotIn('.clone()',s);self.assertNotIn('torch.save',s)
  self.assertIn('validate_metadata(meta(idx),meta(x),idx.detach().cpu().tolist())',s)
  self.assertIn('validate_indices(idx.cpu().tolist(),32)',s)
  self.assertIn("bool((t==1).all())",s)
  self.assertEqual([x.name for x in tree.body if isinstance(x,ast.ClassDef)],['AttentionCore','RunnerPadding','RunnerCensus','HiddenGather','AcceptedScatter'])
 def test_owned_copy_sources_have_expected_single_expressions(self):
  from pathlib import Path
  p=Path(__file__).resolve().parent.parent
  for folder,stem,needle in [('runner_hidden_gather_runtime_integration_v2','sample','_gather('),('mamba_accepted_runtime_integration_v2','postprocess','_selected_scatter(')]:
   s=(p/folder/(stem+'.copied.py.txt')).read_text();ast.parse(s);self.assertEqual(s.count(needle),1)
if __name__=='__main__':unittest.main()
