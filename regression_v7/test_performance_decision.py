import math
import unittest
try:
    from .performance_decision import decide
except ImportError:
    from performance_decision import decide

class PerformanceDecisionTests(unittest.TestCase):
    def test_exact_five_percent(self):
        r = decide([100]*6, [105]*6)
        self.assertEqual(r['verdict'], 'WITHIN_BUDGET')
        self.assertEqual(r['ratio_interval95'], [1.05, 1.05])
        self.assertFalse(r['execution_authorized'])
    def test_boundary_neighbors(self):
        below, above = math.nextafter(1.05, 0), math.nextafter(1.05, math.inf)
        self.assertEqual(decide([1]*6, [below]*6)['verdict'], 'WITHIN_BUDGET')
        self.assertEqual(decide([1]*6, [above]*6)['verdict'], 'REGRESSION')
    def test_known_ratios_and_pairing(self):
        r = decide([100,200,300,400,500,600], [110,220,330,440,550,660])
        self.assertEqual(r['verdict'], 'REGRESSION')
        self.assertAlmostEqual(r['geometric_ratio'], 1.1)
        self.assertEqual(decide([100]*6, [90]*6)['verdict'], 'WITHIN_BUDGET')
    def test_screening_always_inconclusive(self):
        for n in range(2,6):
            for c in (90,100,105,120):
                self.assertEqual(decide([100]*n,[c]*n)['verdict'], 'INCONCLUSIVE')
    def test_noisy_not_pass(self):
        self.assertEqual(decide([100]*6,[80,130,90,125,100,115])['verdict'],'INCONCLUSIVE')
    def test_every_process_direction_required(self):
        # High mean/lower CI alone is insufficient when one process is at budget.
        r = decide([100]*10,[105]+[200]*9)
        self.assertGreater(r['ratio_interval95'][0],1.05)
        self.assertEqual(r['verdict'],'INCONCLUSIVE')
    def test_invalid_values_both_sides(self):
        for bad in (True,False,0,-1,float('nan'),float('inf'),-float('inf'),'100',None):
            for left,right in (([bad]*6,[100]*6),([100]*6,[bad]*6)):
                with self.assertRaises(ValueError): decide(left,right)
    def test_invalid_counts(self):
        for left,right in (([],[]),([1],[1]),([1]*6,[1]*5),([1]*11,[1]*11)):
            with self.assertRaises(ValueError): decide(left,right)
    def test_unrepresentable_ratio(self):
        for left,right in (([1e-300]*6,[1e300]*6),([1e300]*6,[1e-300]*6)):
            with self.assertRaises(ValueError): decide(left,right)

if __name__ == '__main__': unittest.main()
