import ast,unittest
from pathlib import Path
class T(unittest.TestCase):
 def test_only_rank_statements(self):
  root=Path(__file__).parent
  a=ast.parse((root/'compute_topk_scores.original.py.txt').read_text()).body[0]
  b=ast.parse((root/'copied_scores.py.txt').read_text()).body[0]
  class Normalize(ast.NodeTransformer):
   def visit_Name(self,node):
    if node.id=='_tileir_selected_logprobs':node.id='compute_token_logprobs'
    return node
  Normalize().visit(b)
  pos=next(i for i,n in enumerate(a.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='token_ranks' for x in n.targets))
  self.assertIsInstance(a.body[pos+1],ast.Expr)
  self.assertEqual(ast.dump(ast.Module(body=a.body[:pos]+a.body[pos+2:],type_ignores=[])),ast.dump(ast.Module(body=b.body[:pos]+b.body[pos+1:],type_ignores=[])))
  self.assertEqual(ast.unparse(b.body[pos]),'token_ranks = _tileir_selected_ranks(logits, sampled_token_ids)')
  self.assertEqual(ast.dump(a.args),ast.dump(b.args))
