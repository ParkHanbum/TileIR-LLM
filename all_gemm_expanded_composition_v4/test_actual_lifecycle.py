"""Actual adapter classes and public wrapper lifecycle, CPU mock tensors/APIs only.
These test data/descriptor handling and latch/restore, never numerical correctness.
"""
import ast,contextlib,hashlib,inspect,json,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
from .adapter import PARTS
from .entry_guards import EntryGuards,METHODS
from .test_entry_guards import Controller,fixture
from .test_extensions import fake_modules
NS=types.SimpleNamespace;R=pathlib.Path(__file__).resolve().parent

def source_audit():
 rows=[]
 for key,module,count in PARTS:
  cls=module.padding.Adapter if key=='runner_padding' else module.Adapter
  obj=object.__new__(cls);methods=[];writes=[];sources={}
  for name in METHODS:
   desc=inspect.getattr_static(cls,name,None)
   if isinstance(desc,types.FunctionType):
    value=getattr(obj,name)
    assert isinstance(value,types.MethodType) and value.__self__ is obj and value.__func__ is desc
    methods.append(name)
  for base in cls.__mro__:
   if base is object:continue
   path=pathlib.Path(inspect.getsourcefile(base));sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   for node in ast.walk(ast.parse(inspect.getsource(base))):
    if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
     targets=node.targets if isinstance(node,ast.Assign) else [node.target]
     for target in targets:
      for t in ast.walk(target):
       if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr in METHODS:writes.append(dict(name=t.attr,statement=ast.unparse(node)))
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='setattr' and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
     assert isinstance(node.args[1],ast.Constant) and node.args[1].value not in METHODS,('dynamic self-write',key,ast.unparse(node))
  assert not (set(methods)&{x['name']for x in writes}),('Entry method overwritten',key)
  rows.append(dict(component=key,adapter_class=cls.__module__+'.'+cls.__qualname__,mro=[c.__module__+'.'+c.__qualname__ for c in cls.__mro__ if c is not object],methods=methods,data_writes=writes,sources=sources))
 return rows

class SourceAuditTests(unittest.TestCase):
 def test_actual_25_mro_26_methods_and_three_data_fields(self):
  rows=source_audit();self.assertEqual(len(rows),25);self.assertEqual(sum(len(r['methods'])for r in rows),26)
  self.assertEqual({(r['component'],w['name'])for r in rows for w in r['data_writes']},{('norm','dispatch'),('gated_norm','dispatch'),('accepted_scatter','dispatch')})
 def test_all_real_class_methods_globally_blocked_and_restore(self):
  for key,module,count in PARTS:
   cls=module.padding.Adapter if key=='runner_padding' else module.Adapter;obj=object.__new__(cls)
   # Actual class/MRO descriptors retained; only install binding/state schema mocked.
   fake,owner,name=fixture(key,lambda:None)
   for k,v in vars(fake).items():
    if k!='body':setattr(obj,k,v)
   c=Controller();c.attach(key,obj);g=EntryGuards(c);g.install(key,obj)
   real=[n for n in METHODS if isinstance(inspect.getattr_static(cls,n,None),types.FunctionType)]
   self.assertEqual({r['attribute']for r in g.map[key]if r['kind']=='component_method'},set(real))
   c.latch('other',RuntimeError('upstream failure'))
   for entry in [getattr(owner,name),*[getattr(obj,n)for n in real]]:
    with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):entry()
   g.restore(key)
   for n in real:self.assertIs(getattr(obj,n).__func__,inspect.getattr_static(cls,n))
 def test_callable_data_not_replaced_and_identity_preserved(self):
  class Data:
   def __call__(self):raise AssertionError('Not an entry')
  class A:pass
  for datum in (None,0,Data(),lambda:None):
   a=A();a.dispatch=datum;owner=NS(entry=lambda:None);a.bindings=[(owner,'entry',None,owner.entry)]
   c=Controller();c.attach('pos_seq',a);g=EntryGuards(c);g.install('pos_seq',a);self.assertIs(a.dispatch,datum)
   g.restore('pos_seq');self.assertIs(a.dispatch,datum)
 def test_foreign_or_instance_override_method_rejected_before_mutation(self):
  class A:
   def apply(self):pass
  for foreign in (A().apply,lambda:None):
   a=A();a.apply=foreign;owner=NS(entry=lambda:None);before=owner.entry;a.bindings=[(owner,'entry',None,before)]
   c=Controller();c.attach('pos_seq',a);g=EntryGuards(c)
   with self.assertRaisesRegex(AssertionError,'Unexpected component method identity'):g.install('pos_seq',a)
   self.assertIs(owner.entry,before);self.assertIs(a.apply,foreign);self.assertFalse(g.records)

class NormLifecycleTests(unittest.TestCase):
 def test_actual_norm_install_forward_observer_start_and_close(self):
  from norm_runtime_integration_v4 import adapter as norm
  from norm_runtime_contract_capture_v2.dispatch_observer import DispatchObserver
  calls=[]
  class Op:
   def dispatch(self):return NS(provider='native',inplace=False,supported=True,impl_fn=lambda:None,func_impl_fn=lambda:None)
  op=Op();original_dispatch=op.dispatch.__func__;observer=DispatchObserver({'rms_norm':op},lambda f:{})
  class Tensor:
   shape=(32,2560)
   def detach(self):return self
  class GemmaRMSNorm:
   weight=Tensor();weight.shape=(2560,);variance_epsilon=1e-6
   def forward(self,x,residual=None):calls.append('native');op.dispatch();return x
  modules=[(f'layers.{i}.{k}',GemmaRMSNorm())for i in range(32)for k in ('input_layernorm','post_attention_layernorm')]+[('norm',GemmaRMSNorm())]
  model=NS(named_modules=lambda:modules);tensor=Tensor();tile=types.ModuleType('tileir_vllm_gemma_norm_v3');tile.validate=lambda *a:None
  def numeric(x,*args):calls.append('candidate-mock');return x
  tile.gemma_rms_norm=numeric;torch=types.ModuleType('torch');torch.no_grad=contextlib.nullcontext
  desc=lambda t:dict(shape=list(t.shape),stride=[2560,1],storage_offset=0,dtype='mock')
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,{'torch':torch,tile.__name__:tile}),patch.multiple(norm,identity=lambda f:{},operand_metadata=lambda t:{},verify_detach=lambda *a:{},logical=lambda t,torch:id(t),descriptor=desc,aliases=lambda d:{},metrics=lambda *a:dict(raw_equal=True,candidate_nonfinite=0,reference_nonfinite=0),new_dispatch_observer=lambda:observer.install()):
   a=norm.Adapter(pathlib.Path(d)/'norm');a.install(model);c=Controller();c.attach('norm',a);g=EntryGuards(c);g.install('norm',a)
   self.assertIs(a.dispatch,observer);self.assertFalse(any(r['attribute']=='dispatch'for r in g.map['norm']))
   a.start('shadow');self.assertIs(modules[0][1].forward(tensor),tensor);self.assertTrue(a.rows[0]['passed']);self.assertEqual(len(a.rows[0]['provider_events']),1)
   # Synthetic phase admission only; this is not all1040 numerical proof.
   a.shadow_pass=True;a.start('replace');self.assertIs(a.dispatch,observer);self.assertIs(modules[0][1].forward(tensor),tensor)
   self.assertEqual(calls,['native','candidate-mock','candidate-mock']);c.latch('other',ValueError('stop'))
   with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):modules[0][1].forward(tensor)
   g.restore('norm');self.assertTrue(a.close()['restored']);self.assertIsNone(a.dispatch);self.assertIs(op.dispatch.__func__,original_dispatch)
   self.assertTrue(all('forward'not in m.__dict__ for n,m in modules))
 def test_actual_gated_install_none_start_and_close(self):
  from gated_norm_runtime_integration_v2 import adapter as gated
  class RMSNormGated:
   weight=NS(shape=(128,));group_size=None;norm_before_gate=True;bias=None;activation='silu'
   def forward_cuda(self,*a):return None
   forward=forward_cuda
  modules=[(f'layers.{i}.linear_attn.norm',RMSNormGated())for i in range(32)if i%4!=3];model=NS(named_modules=lambda:modules)
  with tempfile.TemporaryDirectory()as d,patch.multiple(gated,identity=lambda f:{},operand_metadata=lambda t:{}):
   a=gated.Adapter(pathlib.Path(d)/'g');a.install(model);c=Controller();c.attach('gated_norm',a);g=EntryGuards(c);g.install('gated_norm',a)
   self.assertIsNone(a.dispatch);a.start('shadow');self.assertIsNone(a.dispatch);a.shadow_pass=True;a.start('replace');self.assertIsNone(a.dispatch)
   g.restore('gated_norm');self.assertTrue(a.close()['restored']);self.assertIsNone(a.dispatch)
   self.assertTrue(all('forward'not in m.__dict__ for n,m in modules))

class AcceptedLifecycleTests(unittest.TestCase):
 def test_actual_install_start_invoke_finish_reverse_close_counter_reset(self):
  from mamba_accepted_runtime_integration_v2 import adapter as accepted
  mod=types.ModuleType('vllm.v1.worker.gpu.model_states.mamba_hybrid')
  def original(self,*a):return None
  State=type('MambaHybridModelState',(),dict(postprocess_state=original));mod.MambaHybridModelState=State;mod._scatter_num_accepted_kernel=object()
  class Tensor:shape=(32,)
  torch=types.ModuleType('torch');torch.Tensor=Tensor;idx=Tensor();sampled=Tensor();state=State();state.recoverssm=None;state._align_mode=False;state.num_accepted_tokens_gpu=object();calls=[]
  def numeric_mock(self,i,s,d):
   assert self.pending and self.dispatch==self.calls-1;assert isinstance(self.dispatch,int)
   calls.append(self.mode);self.dispatch+=1;self.rows.append(dict(passed=True,native_scatter_calls=int(self.mode=='shadow')))
  model=NS();mods=fake_modules(mod.__name__,mod);mods['torch']=torch
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,mods),patch.object(accepted.inspect,'getsource',lambda f:(accepted.R/'postprocess.original.py.txt').read_text()),patch.object(accepted.Adapter,'scatter',numeric_mock):
   accepted.install(model,str(pathlib.Path(d)/'a'));a=getattr(model,accepted.ATTR);c=Controller();c.model=model;c.installed=[('accepted_scatter',accepted)];g=EntryGuards(c);g.install('accepted_scatter',a)
   self.assertEqual(type(a.dispatch),int);self.assertEqual(a.dispatch,0);self.assertNotIn('dispatch',{r['attribute']for r in g.map['accepted_scatter']})
   for mode in ('shadow','replace'):
    accepted.start(model,mode);self.assertEqual(type(a.dispatch),int);self.assertEqual(a.dispatch,0)
    for _ in range(16):self.assertIsNone(state.postprocess_state(idx,sampled))
    self.assertTrue(accepted.finish(model)['shadow_pass']);self.assertEqual(a.dispatch,16)
   c.latch('other',ValueError('upstream'))
   with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):state.postprocess_state(idx,sampled)
   self.assertEqual(len(calls),32);g.restore('accepted_scatter');self.assertTrue(accepted.close(model)['restored']);self.assertIs(State.postprocess_state,original);self.assertFalse(g.records)
if __name__=='__main__':unittest.main()
