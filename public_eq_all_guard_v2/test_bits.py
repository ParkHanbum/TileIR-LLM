import unittest,struct,json,random
from pathlib import Path
class Tests(unittest.TestCase):
 def test_binary32_predicate(self):
  words=[int(s,16)for s in json.loads(Path(__file__).with_name('SYNTHETIC-CASES.json').read_text())['input_words_hex']]
  rng=random.Random(20261113);words += [rng.getrandbits(32)for _ in range(512)]
  for bits in words:
   value=struct.unpack('<f',struct.pack('<I',bits))[0]
   self.assertEqual((bits&0x7fffffff)==0,value==0)
   self.assertEqual(bits==0x3f800000,value==1)
 def test_no_fp_comparison_in_kernel(self):
  s=Path(__file__).with_name('kernel.py').read_text()
  self.assertIn('ct.bitcast(a,ct.uint32)',s);self.assertNotIn('dtype=ct.float32',s)
