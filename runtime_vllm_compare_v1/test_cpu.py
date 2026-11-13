"""CPU-only static protocol tests; never import torch/vLLM or initialize CUDA."""
import ast,json,pathlib,subprocess,sys,unittest
R=pathlib.Path(__file__).resolve().parent
class Protocol(unittest.TestCase):
 def test_cpu_default(self):
  r=json.loads(subprocess.check_output([sys.executable,'-B',str(R/'run_vllm.py')],text=True));o=r['engine_options']
  self.assertEqual((len(r['input_ids']),r['warmup_requests'],r['timed_requests']),(53,2,6))
  self.assertFalse(o['enable_prefix_caching']);self.assertFalse(o['enforce_eager']);self.assertNotIn('async_scheduling',o)
  self.assertEqual((o['max_model_len'],o['max_num_seqs'],o['max_num_batched_tokens']),(128,32,2048))
 def test_primary_boundary(self):
  s=(R/'run_vllm.py').read_text();a=s.index('start=time.perf_counter_ns()',s.index('def request'))
  enqueue=s.index('engine.add_request',a);end=s.index('end=time.perf_counter_ns()',enqueue)
  self.assertLess(a,enqueue);self.assertIn('output=[list(final_outputs',s[enqueue:end]);self.assertIn('torch.cuda.synchronize()',s[enqueue:end])
  primary=s[s.index('if not a.sync_step_boundary:',enqueue):s.index('\n   else:',enqueue)]
  self.assertNotIn('first_public_output_ns=time',primary);self.assertNotIn('steps.append',primary)
 def test_repeat_and_diagnostic(self):
  s=(R/'run_vllm.py').read_text();self.assertIn("assert output==repeat_reference",s);self.assertNotIn('assert output==expected',s)
  self.assertIn('async_scheduling',s);self.assertIn('not puredecode',s)
 def test_bounded_cleanup(self):
  s=(R/'launch_vllm.py').read_text();ast.parse(s)
  for term in ['pass_fds','start_new_session=True','timeout=930','killpg','lock_reacquired','changed during run'] :self.assertIn(term,s)
  self.assertIn('engine.engine_core.shutdown()', (R/'run_vllm.py').read_text())
if __name__=='__main__':unittest.main()
