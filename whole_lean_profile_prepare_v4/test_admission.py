import copy,json,unittest
from pathlib import Path
from .admission import verify_documents,R
class Tests(unittest.TestCase):
 def fixture(self):
  receipt=json.loads((R/'ACTUAL-ENDPOINT-PREREQUISITE.json').read_text());e=receipt['evidence'];docs=[json.loads(Path(e[k]['path']).read_text())for k in ('result','launch','recovery')]
  return docs,e['result']['sha256'],e['launch']['sha256']
 def test_actual_documents(self):
  docs,r,l=self.fixture();verify_documents(*docs,r,l)
 def test_exit_and_lifecycle_negative(self):
  for index,key,value in [(1,'returncode',1),(2,'actual_returncode',1),(1,'qualification_complete',False),(1,'child_absent',False),(2,'remaining_process_group',[123]),(2,'lock_available',False),(2,'source_cache_unchanged',False),(1,'timeout',True),(2,'launch_sha256','wrong')]:
   docs,r,l=self.fixture();docs[index][key]=value
   with self.assertRaises(AssertionError):verify_documents(*docs,r,l)
 def test_result_not_complete(self):
  docs,r,l=self.fixture();docs[0]['complete']=False
  with self.assertRaises(AssertionError):verify_documents(*docs,r,l)
if __name__=='__main__':unittest.main()

class RootTests(unittest.TestCase):
 def test_actual_audit_root_is_not_profile_default(self):
  from whole_lean_extensions_runtime_v7.gates import audit
  from .admission import V5
  receipt=json.loads((R/'ACTUAL-ENDPOINT-PREREQUISITE.json').read_text())
  data=json.loads(Path(receipt['evidence']['result']['path']).read_text())
  self.assertTrue(audit(data,V5)['passed'])
  self.assertFalse(audit(data,R)['passed'])
 def test_launcher_passes_explicit_profile_root(self):
  import ast
  tree=ast.parse((R/'launch.py').read_text())
  calls=[x for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='audit']
  self.assertEqual(len(calls),1)
  self.assertTrue(any(k.arg=='root'and isinstance(k.value,ast.Name)and k.value.id=='R'for k in calls[0].keywords))
 def test_generic_source_preflight_uses_argument_root(self):
  import ast
  from whole_lean_extensions_runtime_v7.preflight import source_preflight
  import inspect
  body=inspect.getsource(source_preflight)
  self.assertIn("root/'PINS.json'",body)
  run=ast.parse((R/'run.py').read_text())
  self.assertTrue(any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='preflight'and ast.unparse(x.args[0])=='R'for x in ast.walk(run)))
