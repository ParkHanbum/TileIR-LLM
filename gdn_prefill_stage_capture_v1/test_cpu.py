import ast,unittest,types
from pathlib import Path
from observer import Observer,STAGES
from run import result_gate
class Tests(unittest.TestCase):
 def fixture(self):
  module=types.SimpleNamespace();returns={}
  for name,labels in STAGES.items():
   value=object() if len(labels)==1 else tuple(object() for _ in labels);returns[name]=value;setattr(module,name,lambda *a,_v=value,**kw:_v)
  return module,returns
 def test_once_identity_restore(self):
  m,values=self.fixture();original={n:getattr(m,n) for n in STAGES};o=Observer(m,lambda t,l:{},lambda f:{});o.install()
  for n in STAGES:self.assertIs(getattr(m,n)(),values[n])
  o.close();self.assertTrue(o.complete());self.assertTrue(all(getattr(m,n) is f for n,f in original.items()))
 def test_duplicate(self):
  m,_=self.fixture();o=Observer(m,lambda t,l:{},lambda f:{});o.install();m.chunk_local_cumsum()
  with self.assertRaises(AssertionError):m.chunk_local_cumsum()
  o.close()
 def test_partial_failure_restore(self):
  m,_=self.fixture();del m.solve_tril;old=m.chunk_local_cumsum;o=Observer(m,lambda t,l:{},lambda f:{})
  with self.assertRaises(AttributeError):o.install()
  self.assertIs(m.chunk_local_cumsum,old)
 def test_gate_pending(self):self.assertFalse(result_gate({})['raw_pass'])
 def test_gate_requires_A(self):
  x=dict(complete=True,gpu_executed=True,full_final_raw=True,existing_A_raw_equal=False,existing_cumsum_raw_equal=True,observers_restored=True,stage_counts={n:1 for n in STAGES},initial_flags=[True,True],final_flags=[True,True],artifacts_written=True);self.assertFalse(result_gate(x)['raw_pass']);x['existing_A_raw_equal']=True;self.assertTrue(result_gate(x)['raw_pass'])
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
