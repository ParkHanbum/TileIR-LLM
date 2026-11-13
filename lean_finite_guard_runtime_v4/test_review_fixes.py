import unittest,ast,json
from pathlib import Path
from unittest.mock import patch
from . import gates
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_edge_failure_records_gpu_before_call_and_never_engine(self):
  tree=ast.parse((R/'run.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  tr=next(n for n in main.body if isinstance(n,ast.Try));body=tr.body
  i=next(i for i,n in enumerate(body)if isinstance(n,ast.ImportFrom)and n.module=='lean_finite_guard_runtime_v4.edge_probe')
  fragment=ast.Module(body=body[i+1:i+5],type_ignores=[]);ast.fix_missing_locations(fragment)
  saved=[];r={'gpu_executed':False};calls=[]
  def probe():calls.append('edge');raise ValueError('synthetic edge failure')
  ns=dict(result=r,save=lambda:saved.append(dict(r)),edge_probe=probe)
  with self.assertRaisesRegex(ValueError,'synthetic edge failure'):exec(compile(fragment,'actual-run-fragment','exec'),ns)
  self.assertEqual(saved,[{'gpu_executed':True,'phase':'finite_edge_probe'}]);self.assertEqual(calls,['edge'])
 def test_prior_gate_requires_each_hash_audit_and_exits(self):
  refs=json.loads((R/'PRIOR-FINITE-MODEL.json').read_text())
  data={'result':{},'launch':{'returncode':0,'result_sha256':refs['result']['sha256']},'recovery':{'actual_returncode':0,'result_sha256':refs['result']['sha256'],'child_absent':True,'remaining_process_group':[],'lock_available':True}}
  def read(p,*a,**k):
   if p.name=='PRIOR-FINITE-MODEL.json':return json.dumps(refs)
   return json.dumps(next(data[k]for k,v in refs.items()if v['path']==str(p)))
  digest={v['path']:v['sha256']for v in refs.values()}
  with patch.object(Path,'read_text',read),patch.object(gates,'sha',lambda p:digest[str(p)]),patch('lean_finite_guard_runtime_v1.gates.audit',return_value={'passed':True})as audit:
   self.assertTrue(gates.prior_finite_model(R)['passed']);audit.assert_called_once()
   data['launch']['returncode']=1
   with self.assertRaises(AssertionError):gates.prior_finite_model(R)
   data['launch']['returncode']=0;digest[refs['recovery']['path']]='wrong'
   with self.assertRaises(AssertionError):gates.prior_finite_model(R)
  self.assertIn('prior=prior_finite_model(root)',(R/'gates.py').read_text())
if __name__=='__main__':unittest.main()
