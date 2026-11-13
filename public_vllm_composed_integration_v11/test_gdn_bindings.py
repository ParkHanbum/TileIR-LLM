import ast,copy,types,tempfile,unittest
from pathlib import Path
from gdn_decode_core_runtime_v4 import adapter as core
class T(unittest.TestCase):
 def exercise(self,drift=False):
  # Execute the real install body with only its import replaced by a fake public module.
  tree=ast.parse(Path(core.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='install'))
  assert isinstance(fn.body[0],ast.Import);fn.body=fn.body[1:];module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
  def helper(*args,**kwargs):return args
  q=types.SimpleNamespace(causal_conv1d_fn=object(),causal_conv1d_update=object(),fused_post_conv_prep=object(),fused_recurrent_gated_delta_rule_packed_decode=helper)
  namespace=dict(core.__dict__,q=q);exec(compile(module,'<actual-core-install>','exec'),namespace)
  layers=[]
  for i in range(32):
   if i%4!=3:layers.append((f'model.layers.{i}.linear_attn',types.SimpleNamespace(A_log=types.SimpleNamespace(data_ptr=lambda i=i:i*2+1),dt_bias=types.SimpleNamespace(data_ptr=lambda i=i:i*2+2))))
  model=types.SimpleNamespace(named_modules=lambda:layers)
  with tempfile.TemporaryDirectory() as d:
   a=core.Adapter(Path(d)/'core');namespace['install'](a,model)
   expected=dict(a.untouched);self.assertEqual(len(a.owners),24);self.assertIsNot(q.fused_recurrent_gated_delta_rule_packed_decode,helper)
   if drift:
    q.causal_conv1d_update=object()
    with self.assertRaises(AssertionError):a.close()
    # Original core binding was restored before untouched-binding HOLD.
    self.assertIs(q.fused_recurrent_gated_delta_rule_packed_decode,helper)
    q.causal_conv1d_update=expected['causal_conv1d_update']
   else:self.assertTrue(a.close()['restored'])
   for k,v in expected.items():self.assertIs(getattr(q,k),v)
   self.assertIs(q.fused_recurrent_gated_delta_rule_packed_decode,helper)
 def test_existing_conv_wrappers_retained(self):self.exercise()
 def test_other_binding_drift_holds(self):self.exercise(True)
