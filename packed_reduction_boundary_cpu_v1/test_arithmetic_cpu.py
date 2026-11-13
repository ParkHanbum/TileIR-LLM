import importlib.util,struct,unittest
from pathlib import Path
p=Path(__file__).parent/'analyze.py';s=importlib.util.spec_from_file_location('arithmetic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_integer_ties(self):
  one=1<<266;half=1<<(266-24);ulp=half*2
  self.assertEqual(m.round_fp32_units(one+half),one)
  self.assertEqual(m.round_fp32_units(one+3*half),one+2*ulp)
  self.assertEqual(m.round_fp32_units(-one-half),-one)
 def test_subnormal_ties(self):
  sub=1<<(266-149)
  self.assertEqual(m.round_fp32_units(sub//2),0)
  self.assertEqual(m.round_fp32_units(3*sub//2),2*sub)
  self.assertEqual(m.round_fp32_units(sub),sub)
 def test_exact_finite_fp32(self):
  for bits in [0,1,2,0x007fffff,0x00800000,0x3f000000,0x3f800000,0x3f800001,0x7f7fffff,0x80000001,0xbf800000]:
   value=struct.unpack('<f',struct.pack('<I',bits))[0];num,den=value.as_integer_ratio();n=num*((1<<266)//den)
   self.assertEqual(m.round_fp32_units(n),n)
 def test_new_grains_partition(self):
  for parts in (5,10,20,40,80):
   self.assertEqual(2560%parts,0);self.assertIn(2560//parts,(512,256,128,64,32))
if __name__=='__main__':unittest.main()
