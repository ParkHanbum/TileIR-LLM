import ast,types,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v5.cleanup import final_cleanup
class Tests(unittest.TestCase):
 def exercise(self,drift=False,closefail=False,bodyfail=False):
  events=[];result={'complete':True}
  def event(name,fail=False):
   def f(*args):
    events.append(name)
    if fail:raise AssertionError(name)
    return ['restored']
   return f
  policy=types.SimpleNamespace(restore=event('policy',drift))
  engine=types.SimpleNamespace(apply_model=event('close',closefail),engine_core=types.SimpleNamespace(shutdown=event('shutdown')))
  # Execute the actual run.py finally AST, not a duplicate test cleanup block.
  tree=ast.parse(Path(__file__).with_name('run.py').read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');actual=next(x for x in main.body if isinstance(x,ast.Try))
  body=ast.parse("raise ValueError('original')" if bodyfail else 'pass').body
  module=ast.Module(body=[ast.Try(body=body,handlers=[],orelse=[],finalbody=actual.finalbody)],type_ignores=[]);ast.fix_missing_locations(module)
  fake=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=event('sync')))
  env=dict(final_cleanup=final_cleanup,result=result,save=event('save'),policy=policy,engine=engine,adapter=types.SimpleNamespace(close=object()),adapter_installed=True,close_attempted=False,signal=types.SimpleNamespace(alarm=event('alarm')),sys=sys,__import__=lambda name:fake)
  error=None
  try:exec(compile(module,'actual_run_finally','exec'),env)
  except BaseException as e:error=e
  self.assertEqual(events,['policy','sync','close','shutdown','alarm','save'])
  if drift or closefail:self.assertFalse(result['complete']);self.assertEqual(len(result['cleanup_errors']),int(drift)+int(closefail))
  if bodyfail:self.assertIsInstance(error,ValueError);self.assertEqual(str(error),'original')
  elif drift or closefail:self.assertIsInstance(error,RuntimeError)
  else:self.assertIsNone(error)
 def test_drift(self):self.exercise(drift=True)
 def test_close_failure(self):self.exercise(closefail=True)
 def test_both_and_original_error(self):self.exercise(drift=True,closefail=True,bodyfail=True)
 def test_success(self):self.exercise()
if __name__=='__main__':unittest.main()
