import unittest,ast
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
from .proof import launch_owned
class Tests(unittest.TestCase):
 def test_launch_only_and_counts(self):
  a=N(size=64);reg=N(check=lambda:None,allocations={0:a},segments={0:N(allocation=a,offset=8,stride=16,page=8,blocks=3)})
  calls=[];r={}
  def zero(a,iv,ledger):
   for x in iv:calls.append(x);ledger['attempted']+=1;ledger['returned']+=1
  with patch(__package__+'.proof.validate_scope',lambda *x:None):launch_owned(reg,[0,2],zero=zero,synchronize=lambda:None,record=r)
  self.assertEqual(calls,[(8,16),(40,48)]);self.assertEqual(r['owner_payload_read_bytes'],0);self.assertEqual(r['zero_returned'],2);self.assertFalse(r['target_oracle_executed'])
 def test_invalid_scope_before_launch(self):
  reg=N(check=lambda:None,allocations={},segments={})
  with self.assertRaises(Exception):launch_owned(reg,[],zero=lambda *x:self.fail('zero'),synchronize=lambda:self.fail('sync'),record={})
 def test_no_payload_read_api_in_source(self):
  src=(Path(__file__).parent/'proof.py').read_text();tree=ast.parse(src)
  self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr in('cpu','numpy','count_nonzero','digest','sha256')for n in ast.walk(tree)))
 def test_source_only(self):
  root=Path(__file__).parent;self.assertFalse((root/'cache_v1').exists());self.assertEqual(list(root.glob('CACHE-PRESEED*')),[])
  for p in root.glob('*.py'):ast.parse(p.read_text())
