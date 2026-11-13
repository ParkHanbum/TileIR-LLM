"""CPU-only gate/fixture regressions; no framework or GPU import."""
import copy
import unittest
from . import modules_gemm as m

GOOD = dict(raw_equal=True, raw_mismatches=0, strict_failures=0,
            reference_nonfinite=0, candidate_nonfinite=0)

def result_fixture():
    result=dict(complete=True,context_restored=True,reference_closed=True,timing=False,reference_version=130101,
                reference='fresh public GemmEx D99/math0/compute68; never historical F',precision_flags=[True,True],gpu_candidate_calls=6,gpu_reference_calls=6,cases=[])
    for i,(name,shape) in enumerate(zip(m.CASES,m.SHAPES)):
        good=dict(GOOD,elements=shape[0]*shape[1],atol=1e-5,rtol=1e-5)
        negative=dict(good,raw_equal=False,raw_mismatches=1,strict_failures=1)
        ref=dict(math_mode=0,pointer_mode=0,algo=99,compute_type=68,alpha=1,beta=0)
        result['cases'].append(dict(label=name,mnk=list(shape),backend='packed_k128_partial_v1' if i==0 else 'tileirblas_once_occ2',base_path=None if i==0 else 'blocked' if i==1 else 'prefill128',native_calls=[ref.copy(),ref.copy()],native_repeat=good.copy(),candidate_comparisons=[good.copy(),good.copy()],input_full_backing_readonly=True,output_nonalias=True,nondefault_current_stream=True,raw_negative_detected=True,strict_finite_negative_detected=True,raw_negative=negative.copy(),strict_negative=negative.copy()))
    return result

class Tests(unittest.TestCase):
    def test_result_audit_good(self): self.assertTrue(m.audit(result_fixture()))
    def test_result_audit_missing_duplicate_cases(self):
        for kind in ('missing','duplicate','order'):
            r=result_fixture()
            if kind=='missing': r['cases'].pop()
            elif kind=='duplicate': r['cases'][1]=r['cases'][0]
            else: r['cases'].reverse()
            with self.assertRaises(AssertionError): m.audit(r)
    def test_result_audit_raw_strict_backend_context(self):
        for kind in ('raw','strict','backend','context','nativecount','candidatecount','shape','readonly','negative','reference','tolerance'):
            r=result_fixture();c=r['cases'][0]
            if kind=='raw': c['candidate_comparisons'][0]['raw_equal']=False
            elif kind=='strict': c['candidate_comparisons'][1]['strict_failures']=1
            elif kind=='backend': c['backend']='tileirblas_once_occ2'
            elif kind=='context': r['context_restored']=False
            elif kind=='nativecount': c['native_calls'].pop()
            elif kind=='candidatecount': c['candidate_comparisons'].pop()
            elif kind=='shape': c['mnk'][0]=1
            elif kind=='readonly': c['input_full_backing_readonly']=False
            elif kind=='negative': c['strict_negative']['strict_failures']=0
            elif kind=='reference': c['native_calls'][0]['algo']=-1
            else: c['native_repeat']['rtol']=0.1
            with self.assertRaises(AssertionError): m.audit(r)
    def test_good(self): m.require_comparison(GOOD)
    def test_one_bit_regression(self):
        bad = m.compare_small_words([0x3f80], [0x3f80 ^ 1])
        with self.assertRaises(AssertionError): m.require_comparison(bad)
    def test_large_finite_regression(self):
        bad = m.compare_small_words([0x3f80], [0x4700])
        self.assertEqual(bad['candidate_nonfinite'], 0)
        self.assertEqual(bad['strict_failures'], 1)
        with self.assertRaises(AssertionError): m.require_comparison(bad)
    def test_nonfinite(self):
        for field in ('reference_nonfinite','candidate_nonfinite'):
            with self.assertRaises(AssertionError): m.require_comparison(dict(GOOD, **{field:1}))
    def test_missing_fields_fail_closed(self):
        for field in GOOD:
            bad = copy.copy(GOOD); del bad[field]
            with self.assertRaises(AssertionError): m.require_comparison(bad)
    def test_saved_scope(self):
        self.assertEqual(tuple(r['label'] for r in m.rows()), m.CASES)
        self.assertEqual(m.SHAPES, ((32,64,2560),(32,12288,2560),(1696,18432,2560)))
        self.assertFalse(m.rows()[0]['require_default_matches_captured'])
    def test_external_inputs_only(self):
        deps = m.dependencies()
        self.assertEqual(len(deps),9)
        self.assertFalse(any('nativeY.pt' in p or 'original_0.pt' in p for p in deps))
    def test_bad_word_fixtures(self):
        for a,b in (([],[]),([True],[0]),([65536],[0]),([1],[1,2])):
            with self.assertRaises(ValueError): m.compare_small_words(a,b)
    def test_signed_zero_raw_not_tolerance(self):
        result=m.compare_small_words([0],[0x8000])
        self.assertEqual(result['strict_failures'],0)
        with self.assertRaises(AssertionError): m.require_comparison(result)
    def test_cpu_catalog(self):
        self.assertEqual(len(m.cpu_tests()),7)
        self.assertEqual(len(set(m.cpu_tests())),7)

if __name__ == '__main__': unittest.main()
