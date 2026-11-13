"""CPU regression for undelimited bytecode source-string tables."""
import unittest
from .whole import REPO,pinned_source_paths
class SourcePaths(unittest.TestCase):
    def test_malformed_directory_symbol_prefix_then_exact_path(self):
        root=str(REPO)+'/'
        path=root+'public_sampling_guard_v1/kernel.py'
        data=(root+'public_sampling_guard_v1gather_columnsgather_columns_Kt2_A2u16_I32'+path+'default\0').encode()
        self.assertEqual(pinned_source_paths(data,{path:'pinned'}),[path])
    def test_unpinned_actual_source_is_rejected(self):
        root=str(REPO)+'/'
        good=root+'public_sampling_guard_v1/kernel.py'
        bad=root+'untracked/kernel.py'
        for data in ((bad+'\0'+good).encode(),(root+'directory_symbol'+bad+'default').encode()):
            with self.assertRaises(AssertionError):pinned_source_paths(data,{good:'pinned'})
    def test_exact_known_bytes_and_multiple_sources(self):
        a=str(REPO)+'/a/kernel.py';b=str(REPO)+'/b/kernel.py'
        self.assertEqual(pinned_source_paths((a+'default\0'+b+'default\0'+a).encode(),{a:'a',b:'b'}),[a,b])
        with self.assertRaises(AssertionError):pinned_source_paths((str(REPO)+'/a/wrongkernel.py').encode(),{a:'a'})
    def test_directory_only_not_promoted_to_source(self):
        self.assertEqual(pinned_source_paths((str(REPO)+'/directory_symbol').encode(),{}),[])
if __name__=='__main__':unittest.main()
