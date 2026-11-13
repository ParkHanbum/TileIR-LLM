import ast
import hashlib
import json
from pathlib import Path
import unittest
from tileirblas_blocked import select_tile, k_tiles

ROOT = Path(__file__).resolve().parents[1]

class BlockMapping(unittest.TestCase):
    def test_policy(self):
        self.assertEqual(select_tile(32,248320),(32,128))
        self.assertEqual(select_tile(1696,32),(32,32))
        self.assertEqual(select_tile(1696,8192),(256,128))
        self.assertEqual(select_tile(32,248320,'fixed'),(256,128))
        with self.assertRaises(ValueError): select_tile(32,32,'autotune')
    def test_aligned_k_order(self):
        for k in (0,32,64,2560,4096,9216):
            old=[start+group*16+lane for start,valid in k_tiles(k) for group in range(2) for lane in range(16) if group*16+lane<valid]
            new=[t*32+g*16+l for t in range(k//32) for g in range(2) for l in range(16)]
            self.assertEqual(old,new)
    def test_subtile_address_equivalence(self):
        # Compare loaded/extracted logical coordinates with v1 addresses.
        for m,n,k,sa in [(32,257,64,135680),(259,32,64,64),(259,129,64,80)]:
            bm,bn=select_tile(m,n)
            for bidm in range((m+bm-1)//bm):
                for bidn in range((n+bn-1)//bn):
                    self.assertLess(bidm*bm,m);self.assertLess(bidn*bn,n)
                    for ti in range(k//32):
                        for g in range(2):
                            for lane in (0,15):
                                ki=ti*32+g*16+lane
                                for row in (0,bm-1):
                                    mi=bidm*bm+row
                                    old=mi*sa+ki if mi<m else None
                                    block=bidm*bm*sa+row*sa+ti*32+g*16+lane if mi<m else None
                                    self.assertEqual(old,block)
                                for col in (0,bn-1):
                                    ni=bidn*bn+col
                                    self.assertEqual(ni*k+ki if ni<n else None, bidn*bn*k+col*k+ti*32+g*16+lane if ni<n else None)
    def test_fallback_identical(self):
        self.assertEqual((ROOT/'fallback.py').read_bytes(),(ROOT.parent/'tileirblas/kernels.py').read_bytes())
    def test_original_files_unmodified(self):
        pins=json.loads((ROOT/'ORIGINAL-SOURCE-PINS.json').read_text())
        for path,digest in pins.items():
            self.assertEqual(hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest(),digest,path)
    def test_parse_no_compilation(self):
        for p in ROOT.glob('*.py'): ast.parse(p.read_text())
if __name__=='__main__': unittest.main()
