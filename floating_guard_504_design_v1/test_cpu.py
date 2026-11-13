import unittest,json,ast,types,sys
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_exact_delta(self):
  specs=json.loads((R/'SOURCE.json').read_text())
  for tag,s in specs.items():
   original=(R/(tag+'.original.py.txt')).read_text();new=(R/(tag+'.candidate.py.txt')).read_text()
   self.assertEqual(new,original.replace(s['old'],s['new']));ast.parse(new)
 def test_actual_kv_body_shortcircuit(self):
  class Tensor:
   def __init__(self,n):self.n=n
   def detach(self):events.append('detach:'+self.n);return self
  def target(*a,**k):events.append('target')
  v={k:Tensor(k)for k in ('key','value','slot_mapping','k_scale','v_scale')}
  for failed,checks in [('key',['key']),('value',['key','value']),(None,['key','value'])]:
   events=[]
   def finite(site,x):events.append('finite:'+x.n);return x.n!=failed
   ns={'_finite':finite};exec((R/'kv.candidate.py.txt').read_text(),ns)
   me=types.SimpleNamespace(call=lambda n,p,fn,**kw:fn(**kw))
   with patch.dict(sys.modules,{'torch':types.SimpleNamespace(),'tileir_vllm_kv_scatter_v1.api':types.SimpleNamespace(reshape_and_cache_flash=target)}):
    if failed:
     with self.assertRaises(AssertionError)as ctx:ns['apply'](me,{'name':'n','phase':'prefill'},v)
     self.assertEqual(str(ctx.exception),'')
    else:ns['apply'](me,{'name':'n','phase':'prefill'},v)
   self.assertEqual(events[:5],['detach:'+k for k in v]);self.assertEqual([e for e in events if e.startswith('finite:')],['finite:'+k for k in checks]);self.assertEqual('target'in events,failed is None)
 def test_attention_identity_shortcircuit_ast(self):
  tree=ast.parse((R/'attention.candidate.py.txt').read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assert) and 'result is out' in ast.unparse(n.test))
  self.assertIsInstance(node.test,ast.BoolOp);self.assertIsInstance(node.test.op,ast.And)
  self.assertEqual(ast.unparse(node.test.values[0]),'result is out')
  self.assertEqual(ast.unparse(node.test.values[1]),"_finite('attention.out', out)")
if __name__=='__main__':unittest.main()
