"""Standard-library only. No torch, cuda.tile, compilation, or GPU queries."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import api


class Tests(unittest.TestCase):
    def test_import_has_no_backend(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('cuda.tile', sys.modules)
        self.assertNotIn(__package__ + '.kernels', sys.modules)

    def test_contiguous_partition(self):
        for k in (64, 128, 2560, 4096, 9216):
            first, second = api.tile_ranges(k)
            self.assertEqual(list(first) + list(second), list(range(k//32)))
            self.assertEqual(len(first), len(second))
            self.assertEqual((first.stop, second.start), (k//64, k//64))
        self.assertNotEqual(tuple(api.tile_ranges(2560)[0]), tuple(range(0, 80, 2)))

    def test_logical_coverage_and_boundary(self):
        first, second = api.tile_ranges(2560)
        ks = [32*t + 16*h + q for part in (first, second)
              for t in part for h in (0, 1) for q in range(16)]
        self.assertEqual(ks, list(range(2560)))
        # Integer only: a logical coverage test, NOT a floating/MMA emulator.
        x = [(i % 7)-3 for i in ks]
        w = [(i % 11)-5 for i in ks]
        self.assertEqual(sum(a*b for a,b in zip(x,w)),
                         sum(x[i]*w[i] for i in range(1280)) +
                         sum(x[i]*w[i] for i in range(1280,2560)))

    def test_geometry(self):
        self.assertEqual(api.contract(32,64,2560,'fp32_partial'), (32,128))
        self.assertEqual(api.contract(1696,64,2560,'bf16_partial'), (256,128))
        self.assertEqual(api.contract(1696,18432,2560,'bf16_partial'), (128,128))

    def test_bad_scope(self):
        for args in ((32,64,2559,'fp32_partial'), (32,64,0,'fp32_partial'),
                     (31,64,2560,'fp32_partial'), (32,0,2560,'fp32_partial'),
                     (32,64,2560,True), (32,64,2560,'auto'),
                     (True,64,2560,'fp32_partial'), (32,64,2**31,'fp32_partial')):
            with self.assertRaises((TypeError,ValueError)):
                api.contract(*args)

    def test_bad_scope_never_allocates_or_enqueues(self):
        with patch.object(api,'_validate',return_value=SimpleNamespace(m=32,n=64,k=2559)), \
             patch.object(api,'_allocate') as alloc, patch.object(api,'_enqueue') as enqueue:
            with self.assertRaises(ValueError):
                api.linear_candidate(object(),object(),mode='bf16_partial')
            alloc.assert_not_called(); enqueue.assert_not_called()

    def test_two_arms_differ_only_in_constant(self):
        x,w,y = object(),object(),object()
        calls=[]
        with patch.object(api,'_validate',return_value=SimpleNamespace(m=32,n=64,k=2560)), \
             patch.object(api,'_allocate',return_value=y), \
             patch.object(api,'_enqueue',side_effect=lambda *args:calls.append(args)):
            self.assertIs(api.linear_candidate(x,w,mode='fp32_partial'), y)
            self.assertIs(api.linear_candidate(x,w,mode='bf16_partial'), y)
        self.assertEqual(calls[0][:-1],calls[1][:-1])
        self.assertEqual((calls[0][-1],calls[1][-1]), (False,True))

    def test_source_structure(self):
        root=Path(__file__).parent
        for p in root.glob('*.py'): ast.parse(p.read_text())
        tree=ast.parse((root/'kernels.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='mma' for n in calls),4)
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='store' for n in calls),1)
        branches=[n for n in ast.walk(tree) if isinstance(n,ast.If)]
        self.assertEqual(len(branches),1)
        self.assertEqual(ast.unparse(branches[0].test),'ROUND_PARTIALS')


if __name__ == '__main__': unittest.main()
