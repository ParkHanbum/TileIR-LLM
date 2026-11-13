import types,unittest
from .adapter import build_copy
class T(unittest.TestCase):
 def test_copy_uses_native_topk_and_two_tile_ops(self):
  events=[]
  class Tensor:
   shape=(32,248320)
   def unsqueeze(self,dim):events.append(('unsqueeze',dim));return self
  logits=Tensor();ids=Tensor();packed=Tensor();scores=Tensor();ranks=Tensor()
  class Output:
   def __init__(self,**kw):self.__dict__.update(kw)
  def topk(x,n,dim):self.assertIs(x,logits);self.assertEqual((n,dim),(5,-1));events.append('native-topk');return types.SimpleNamespace(indices=Tensor())
  def cat(xs,dim):self.assertEqual(dim,1);events.append('native-cat');return packed
  def rank(x,i):self.assertIs(x,logits);self.assertIs(i,ids);events.append('tile-rank');return ranks
  def score(x,i):self.assertIs(x,logits);self.assertIs(i,packed);events.append('tile-logprob');return scores
  def forbidden(*a,**kw):raise AssertionError('Native replaced suboperation executed')
  module=types.ModuleType('fake_public_logprob');module.torch=types.SimpleNamespace(Tensor=Tensor,topk=topk,cat=cat,is_tensor=lambda _:False);module.LogprobsTensors=Output;module.compute_token_logprobs=forbidden;module._ranks_kernel=forbidden
  copy=build_copy(module,rank,score);out=copy(logits,5,ids)
  self.assertIs(out.logprob_token_ids,packed);self.assertIs(out.logprobs,scores);self.assertIs(out.selected_token_ranks,ranks)
  self.assertEqual(events,[('unsqueeze',-1),'native-topk','native-cat','tile-logprob','tile-rank'])
