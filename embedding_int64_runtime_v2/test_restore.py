import unittest,types
from unittest.mock import patch
from . import adapter,cast
class T(unittest.TestCase):
 def test_lean_failure_actual_cast_forward_restore_attempted(self):
  original=lambda x:x;wrapper=lambda x:None;owner=types.SimpleNamespace(forward=wrapper)
  state=cast.Cast.__new__(cast.Cast);state.owner=owner;state.wrapped=wrapper;state.had=True;state.prior=original;state.old=original;state.active=True
  model=types.SimpleNamespace();setattr(model,cast.ATTR,state)
  with patch.object(adapter.lean,'close',side_effect=RuntimeError('lean restore failed')):
   with self.assertRaisesRegex(RuntimeError,'Independent adapter restore'):adapter.close(model)
  self.assertIs(owner.forward,original);self.assertIs(getattr(model,cast.ATTR),state)
  with patch.object(adapter.lean,'release')as lr,patch.object(adapter.cast,'release')as cr:
   with self.assertRaises(AssertionError):adapter.release(model)
  lr.assert_not_called();cr.assert_not_called()
 def test_both_errors_aggregated(self):
  model=types.SimpleNamespace();calls=[]
  def fail(which):
   calls.append(which);raise RuntimeError(which)
  with patch.object(adapter.lean,'close',side_effect=lambda m:fail('lean')),patch.object(adapter.cast,'close',side_effect=lambda m:fail('cast')):
   with self.assertRaises(RuntimeError):adapter.close(model)
  self.assertEqual(calls,['lean','cast']);self.assertEqual(len(getattr(model,adapter.CLOSE_STATE)['errors']),2)
 def test_success_restore_release(self):
  model=types.SimpleNamespace();calls=[]
  with patch.object(adapter.lean,'close',return_value={'restored':True}),patch.object(adapter.cast,'close',return_value={'restored':True}):self.assertTrue(adapter.close(model)['embedding_cast_restore']['restored'])
  with patch.object(adapter.lean,'release',side_effect=lambda m:calls.append('lean')),patch.object(adapter.cast,'release',side_effect=lambda m:calls.append('cast')):adapter.release(model)
  self.assertEqual(calls,['lean','cast']);self.assertFalse(hasattr(model,adapter.CLOSE_STATE))
if __name__=='__main__':unittest.main()
