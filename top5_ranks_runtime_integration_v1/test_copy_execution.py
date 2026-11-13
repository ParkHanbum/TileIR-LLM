import types,unittest
from .adapter import build_copy
class T(unittest.TestCase):
 def test_copy_uses_two_tile_ops_native_scores_cat(self):
  events=[]
  class Tensor:
   shape=(32,248320)
   def unsqueeze(self,dim):events.append(('unsqueeze',dim));return self
  logits=Tensor();ids=Tensor();packed=Tensor();scores=Tensor();ranks=Tensor()
  class Output:
   def __init__(self,**kw):self.__dict__.update(kw)
  def top5(x):self.assertIs(x,logits);events.append('tile-top5');return Tensor()
  def cat(xs,dim):self.assertEqual(dim,1);events.append('native-cat');return packed
  def rank(x,i):self.assertIs(x,logits);self.assertIs(i,ids);events.append('tile-rank');return ranks
  def score(x,i):self.assertIs(x,logits);self.assertIs(i,packed);events.append('native-logprob');return scores
  def forbidden(*a,**kw):raise AssertionError('Native replaced suboperation executed')
  module=types.ModuleType('fake_public_logprob');module.torch=types.SimpleNamespace(Tensor=Tensor,topk=forbidden,cat=cat,is_tensor=lambda _:False);module.LogprobsTensors=Output;module.compute_token_logprobs=score;module._ranks_kernel=forbidden
  copy=build_copy(module,rank,top5);out=copy(logits,5,ids)
  self.assertIs(out.logprob_token_ids,packed);self.assertIs(out.logprobs,scores);self.assertIs(out.selected_token_ranks,ranks)
  self.assertEqual(events,[('unsqueeze',-1),'tile-top5','native-cat','native-logprob','tile-rank'])
