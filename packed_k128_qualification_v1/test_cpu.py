import ast,copy,json,sys,unittest
from pathlib import Path
from .run import result_gate,validate_manifest,LABELS,ARMS,ROLES,expected_path
R=Path(__file__).parent
GOOD=dict(raw_equal=True,strict_failures=0,reference_nonfinite=0,candidate_nonfinite=0)
REF=dict(math_mode=0,pointer_mode=0,algo=99,compute_type=68,alpha=1,beta=0)
def fixture():
 cases=[]
 for label in LABELS:
  cases.append(dict(label=label,role=ROLES[label],expected_candidate_path=expected_path(label),controls_pass=True,inputs_unchanged=True,input_backing_unchanged=True,math_mode=0,algo=99,native_repeat=GOOD.copy(),original_checks=[GOOD.copy(),GOOD.copy()],reference_calls=[REF.copy(),REF.copy()],captured_comparison=GOOD.copy(),candidate_records=[dict(arm=arm,repeat=rep,candidate_path=expected_path(label),comparison=GOOD.copy(),inputs_unchanged=True,input_backing_unchanged=True,output_contract_pass=True) for arm in ARMS for rep in range(2)]))
 return dict(finished=True,gpu_executed=True,timing=False,adopted=False,initial_flags=[True,True],final_flags=[True,True],source_pins_unchanged=True,reference_closed=True,cases=cases,arms={a:dict(state='five_matrix_pass',passed_cases=LABELS.copy()) for a in ARMS})
class Tests(unittest.TestCase):
 def test_no_backend_imports(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_manifest(self):
  m=json.loads((R/'MANIFEST.json').read_text());self.assertEqual(len(validate_manifest(m)),5)
  m['reference']['math_mode']=16
  with self.assertRaises(AssertionError):validate_manifest(m)
 def test_actual_holdout_and_roles(self):
  m=json.loads((R/'MANIFEST.json').read_text());rows=m['cases'];self.assertNotEqual(rows[0]['X']['sha256'],rows[1]['X']['sha256']);self.assertNotEqual(rows[0]['W']['sha256'],rows[1]['W']['sha256'])
  self.assertTrue(rows[0]['require_default_matches_captured']);self.assertFalse(rows[1]['require_default_matches_captured'])
  self.assertEqual([r['expected_candidate_path']for r in rows],['bf16_k128']*2+['unchanged_base']*3)
 def test_full_pass(self):self.assertEqual(result_gate(fixture())['qualified_arms'],['bf16_k128'])
 def test_empty_no_pass(self):self.assertFalse(result_gate({})['raw_pass'])
 def test_selected_first_hold(self):
  r=fixture();r['cases']=r['cases'][:1];a=ARMS[0];r['arms'][a]=dict(state='HOLD',passed_cases=[],halted_at=dict(case=LABELS[0],repeat=0));r['cases'][0]['candidate_records']=r['cases'][0]['candidate_records'][:1];r['cases'][0]['candidate_records'][0]['comparison']=dict(GOOD,raw_equal=False,strict_failures=32)
  self.assertTrue(result_gate(r)['raw_difference_hold'])
 def test_holdout_second_hold(self):
  r=fixture();r['cases']=r['cases'][:2];a=ARMS[0];r['arms'][a]=dict(state='HOLD',passed_cases=LABELS[:1],halted_at=dict(case=LABELS[1],repeat=0));r['cases'][1]['candidate_records']=r['cases'][1]['candidate_records'][:1];r['cases'][1]['candidate_records'][0]['comparison']=dict(GOOD,raw_equal=False,strict_failures=32)
  self.assertTrue(result_gate(r)['raw_difference_hold']);self.assertFalse(result_gate(r)['raw_pass'])
 def test_F_capture_not_reference(self):
  r=fixture();r['cases'][1]['captured_comparison']=dict(GOOD,raw_equal=False,strict_failures=32);self.assertTrue(result_gate(r)['raw_pass'])
  r['cases'][1]['candidate_records'][0]['comparison']=dict(GOOD,raw_equal=False,strict_failures=32);self.assertFalse(result_gate(r)['raw_pass'])
 def test_reject_wrong_path_or_role(self):
  r=fixture();r['cases'][0]['candidate_records'][0]['candidate_path']='unchanged_base';self.assertFalse(result_gate(r)['raw_pass'])
  r=fixture();r['cases'][1]['role']='selection_set_one_unique_vector';self.assertFalse(result_gate(r)['raw_pass'])
 def test_reject_missing_controls(self):
  for change in ['flags','repeat','math','original','captured','cleanup','close','count','strict','layout','input']:
   r=fixture()
   if change=='flags':r['final_flags']=[False,True]
   if change=='repeat':r['cases'][0]['native_repeat']['raw_equal']=False
   if change=='math':r['cases'][0]['reference_calls'][0]['math_mode']=16
   if change=='original':r['cases'][0]['original_checks'][0]['raw_equal']=False
   if change=='captured':r['cases'][0]['captured_comparison']['raw_equal']=False
   if change=='cleanup':r['cleanup_errors']=['x']
   if change=='close':r['reference_closed']=False
   if change=='count':r['cases'].pop()
   if change=='strict':r['cases'][0]['candidate_records'][0]['comparison']['strict_failures']=1
   if change=='layout':r['cases'][0]['candidate_records'][0]['output_contract_pass']=False
   if change=='input':r['cases'][0]['candidate_records'][0]['input_backing_unchanged']=False
   self.assertFalse(result_gate(r)['raw_pass'],change)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
