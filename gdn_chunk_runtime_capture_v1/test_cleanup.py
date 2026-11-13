import ast,unittest
from pathlib import Path
class CleanupTests(unittest.TestCase):
 def run_finally(self,fail=None,engine_present=True,propagating=False):
  import ast,sys
  from pathlib import Path
  from types import SimpleNamespace
  from cleanup import final_cleanup
  events=[];result={};observer=SimpleNamespace(close_only=object())
  def event(k):
   events.append(k)
   if k==fail:raise RuntimeError(k)
  engine=SimpleNamespace(apply_model=lambda _:event('restore'),engine_core=SimpleNamespace(shutdown=lambda:event('shutdown'))) if engine_present else None
  torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:event('sync')))
  class FakeSys:
   @staticmethod
   def exc_info():return (RuntimeError if propagating else None,None,None)
  scope=dict(final_cleanup=final_cleanup,result=result,save=lambda:event('save'),check_flags=lambda:event('flags'),engine=engine,observer=observer,observer_live=True,close_attempted=False,sys=FakeSys,signal=SimpleNamespace(alarm=lambda _:event('alarm')),__import__=lambda _:torch)
  main=next(n for n in ast.parse(Path(__file__).with_name('run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
  final=next(n for n in main.body if isinstance(n,ast.Try)).finalbody
  try:exec(compile(ast.fix_missing_locations(ast.Module(body=final,type_ignores=[])),'actual-finally','exec'),scope)
  except RuntimeError:
   self.assertIsNotNone(fail)
  return events,result
 def test_actual_finally_success(self):self.assertEqual(self.run_finally()[0],['flags','restore','sync','shutdown','alarm','save'])
 def test_actual_finally_none_engine(self):self.assertEqual(self.run_finally(engine_present=False)[0],['flags','alarm','save'])
 def test_actual_finally_each_failure_continues(self):
  for stage in ['flags','restore','sync','shutdown','alarm','save']:
   events,r=self.run_finally(stage,propagating=True);self.assertEqual(events,['flags','restore','sync','shutdown','alarm','save']);self.assertFalse(r['complete'])


if __name__=='__main__':unittest.main()
