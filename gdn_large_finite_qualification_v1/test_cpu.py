import unittest,ast,json,sys,copy
from pathlib import Path
from .cases import cases,expected,guarded_target
from .run import result_gate
R=Path(__file__).parent
class T(unittest.TestCase):
 def good(self):
  rows=[dict(dtype=d,n=n,variant=v,expected=expected(v),native=[expected(v)]*2,tile=[expected(v)]*2,host_target_calls=2*int(expected(v)),private_launches=4,input_bytes_preserved=True,metadata_preserved=True,same_current_stream=True,passed=True)for d,n,v in cases()]
  return dict(complete=True,gpu_executed=True,precision_before=[True,True],precision_after=[True,True],records=rows)
 def test_gate(self):
  r=self.good();self.assertTrue(result_gate(r)['raw_pass'])
  for kind in ('missing','duplicate','native','target','cleanup'):
   x=copy.deepcopy(r)
   if kind=='missing':x['records'].pop()
   elif kind=='duplicate':x['records'][1]=x['records'][0]
   elif kind=='native':x['records'][0]['native'][0]=False
   elif kind=='target':x['records'][4]['host_target_calls']=1
   else:x['cleanup_error']='failed'
   self.assertFalse(result_gate(x)['raw_pass'])
 def test_status_before_host_marker(self):
  events=[]
  def predicate(x):events.append('status_observed');return x
  self.assertTrue(guarded_target(True,predicate,lambda:events.append('target')))
  self.assertEqual(events,['status_observed','target']);events.clear()
  self.assertFalse(guarded_target(False,predicate,lambda:events.append('target')))
  self.assertEqual(events,['status_observed'])
  def error(x):raise RuntimeError('compile failed')
  with self.assertRaises(RuntimeError):guarded_target(True,error,lambda:events.append('target'))
  self.assertEqual(events,['status_observed'])
 def test_sizes_manifest(self):
  self.assertEqual(len(cases()),32);self.assertEqual(len(set(cases())),32)
  self.assertEqual(json.loads((R/'MANIFEST.json').read_text())['cases'],[list(x)for x in cases()])
  for d,n,v in cases():
   blocks=(n+1023)//1024;self.assertEqual(blocks,16384);self.assertLess(n,2**31)
   self.assertEqual(blocks*1024-n,0 if n==2**24 else 1)
 def test_ast_no_cuda(self):
  for p in [*R.glob('*.py'),*(R.parent/'tileir_validation_primitives_v3').glob('*.py')]:ast.parse(p.read_text())
  for name in ('torch','vllm','cuda.tile'):self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
