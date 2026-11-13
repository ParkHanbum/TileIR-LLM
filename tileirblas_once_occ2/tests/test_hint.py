import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from tileirblas_once_occ2 import _enqueue as backend
ROOT=Path(__file__).resolve().parents[1]
class Hint(unittest.TestCase):
    def tearDown(self):backend._prefill_kernel_occ2.cache_clear()
    def test_public_hint_cached(self):
        baseline=Mock();hinted=object();baseline.replace_hints.return_value=hinted
        backend._prefill_kernel_occ2.cache_clear()
        with patch.object(backend,'_resolve_kernel',return_value=baseline) as resolve:
            self.assertIs(backend._prefill_kernel_occ2(),hinted);self.assertIs(backend._prefill_kernel_occ2(),hinted)
            resolve.assert_called_once_with('tileirblas_prefill.kernels','prefill_kernel')
            baseline.replace_hints.assert_called_once_with(occupancy=2)
    def test_prefill_launch_same_args(self):
        a,w,y,kernel=object(),object(),object(),object()
        with patch.object(backend,'_prefill_kernel_occ2',return_value=kernel),patch.object(backend,'_resolve_kernel') as resolve,patch.object(backend,'_launch_kernel') as launch:
            self.assertIs(backend.enqueue(a,w,y,SimpleNamespace(m=1696,n=8192,k=2560),'prefill128'),y)
            resolve.assert_not_called();launch.assert_called_once_with(a,kernel,(14,64,1),(a,w,y,2560,32))
    def test_public_api_identical(self):
        self.assertEqual((ROOT/'api.py').read_bytes(),(ROOT.parent/'tileirblas_once/api.py').read_bytes())
    def test_enqueue_only_kernel_resolution_changes(self):
        a=ast.parse((ROOT/'_enqueue.py').read_text());b=ast.parse((ROOT.parent/'tileirblas_once/_enqueue.py').read_text())
        fa=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='enqueue');fb=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='enqueue')
        def normalize(f):
            for n in f.body:
                if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='kernel' for x in n.targets):n.value=ast.Constant('kernel_resolution')
            return ast.dump(f,include_attributes=False)
        self.assertEqual(normalize(fa),normalize(fb))
    def test_hint_error_propagates(self):
        backend._prefill_kernel_occ2.cache_clear();baseline=Mock();baseline.replace_hints.side_effect=ValueError('unsupported')
        with patch.object(backend,'_resolve_kernel',return_value=baseline),patch.object(backend,'_launch_kernel') as launch:
            with self.assertRaises(ValueError):backend.enqueue(None,None,object(),SimpleNamespace(m=1696,n=8192,k=2560),'prefill128')
            launch.assert_not_called()
if __name__=='__main__':unittest.main()
