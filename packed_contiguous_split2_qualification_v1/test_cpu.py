import ast,copy,json,sys,unittest
from pathlib import Path
from .run import result_gate,validate_manifest,LABELS,ARMS
R=Path(__file__).parent
GOOD=dict(raw_equal=True,strict_failures=0,reference_nonfinite=0,candidate_nonfinite=0)
REF=dict(math_mode=0,pointer_mode=0,algo=99,compute_type=68,alpha=1,beta=0)
def fixture():
    cases=[]
    for label in LABELS:
        cases.append(dict(label=label,controls_pass=True,inputs_unchanged=True,input_backing_unchanged=True,math_mode=0,algo=99,native_repeat=GOOD.copy(),original_checks=[GOOD.copy(),GOOD.copy()],reference_calls=[REF.copy(),REF.copy()],captured_comparison=GOOD.copy(),candidate_records=[dict(arm=arm,repeat=rep,comparison=GOOD.copy(),inputs_unchanged=True,input_backing_unchanged=True,output_contract_pass=True) for arm in ARMS for rep in range(2)]))
    return dict(finished=True,gpu_executed=True,timing=False,adopted=False,initial_flags=[True,True],final_flags=[True,True],source_pins_unchanged=True,reference_closed=True,cases=cases,arms={a:dict(state='five_matrix_pass',passed_cases=LABELS.copy()) for a in ARMS})
class Tests(unittest.TestCase):
    def test_no_backend_imports(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
    def test_manifest(self):
        m=json.loads((R/'MANIFEST.json').read_text());self.assertEqual(len(validate_manifest(m)),5)
        m['reference']['math_mode']=16
        with self.assertRaises(AssertionError):validate_manifest(m)
    def test_full_pass(self):
        g=result_gate(fixture());self.assertTrue(g['raw_pass']);self.assertEqual(g['qualified_arms'],list(ARMS))
    def test_empty_not_pass(self):
        self.assertFalse(result_gate({})['raw_pass'])
    def test_independent_arm_holds(self):
        r=fixture();r['cases']=r['cases'][:1]
        for a in ARMS:
            r['arms'][a]=dict(state='HOLD',passed_cases=[],halted_at=dict(case=LABELS[0],repeat=0))
        r['cases'][0]['candidate_records']=[dict(arm=a,repeat=0,comparison=dict(GOOD,raw_equal=False,strict_failures=32)) for a in ARMS]
        g=result_gate(r);self.assertTrue(g['raw_difference_hold']);self.assertFalse(g['raw_pass'])
    def test_mixed_one_hold_one_qualified(self):
        r=fixture();arm=ARMS[0];r['arms'][arm]=dict(state='HOLD',passed_cases=[],halted_at=dict(case=LABELS[0],repeat=0))
        for i,c in enumerate(r['cases']):
            c['candidate_records']=[z for z in c['candidate_records'] if z['arm']!=arm]
            if i==0:c['candidate_records'].insert(0,dict(arm=arm,repeat=0,comparison=dict(GOOD,raw_equal=False,strict_failures=32)))
        g=result_gate(r);self.assertTrue(g['raw_pass']);self.assertEqual(g['qualified_arms'],[ARMS[1]])
    def test_incomplete_reference_or_flags_rejected(self):
        for change in ('flags','repeat','mode','original','captured','cleanup','close','incomplete','numerical','layout','input'):
            r=fixture()
            if change=='flags':r['final_flags']=[False,True]
            if change=='repeat':r['cases'][0]['native_repeat']['raw_equal']=False
            if change=='mode':r['cases'][0]['reference_calls'][0]['math_mode']=16
            if change=='original':r['cases'][0]['original_checks'][0]['raw_equal']=False
            if change=='captured':r['cases'][0]['captured_comparison']['raw_equal']=False
            if change=='cleanup':r['cleanup_errors']=['x']
            if change=='close':r['reference_closed']=False
            if change=='incomplete':r['cases'].pop()
            if change=='numerical':r['cases'][0]['candidate_records'][0]['comparison']['strict_failures']=1
            if change=='layout':r['cases'][0]['candidate_records'][0]['output_contract_pass']=False
            if change=='input':r['cases'][0]['candidate_records'][0]['inputs_unchanged']=False
            self.assertFalse(result_gate(r)['raw_pass'],change)
    def test_ast(self):
        for f in R.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
