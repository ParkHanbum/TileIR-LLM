import unittest,copy
from .modules_attention_kv import audit,dependencies
class Tests(unittest.TestCase):
 def fixture(self):
  import math
  from .modules_attention_kv import ATT_SHAPES,KV_SHAPES
  def metric(shape):return dict(raw_equal=True,raw_different_elements=0,strict_failures=0,candidate_nonfinite=0,reference_nonfinite=0,strict_atol=1e-5,strict_rtol=1e-5,shape=shape,dtype='torch.bfloat16',elements=math.prod(shape))
  a=[dict(bank_index=i,kind=arm,repeat_index=j,passed=True,same_stream=True,output_disjoint=True,inputs_preserved=True,return_out_alias=True,repeat_raw=True,metrics=metric(ATT_SHAPES[i]))for i in range(6)for arm in ('native','tile')for j in range(2)]
  k=[dict(phase=p,arm=arm,repeat=j,passed=True,same_stream=True,return_none=True,full_owned_storage_raw_equal=True,input_full_owned_storage_preserved=True,repeat_raw_equal=True,shared_backing=True,selected_metrics={'key':metric(KV_SHAPES[p]),'value':metric(KV_SHAPES[p])})for p in ('prefill','decode')for arm in ('native','tile')for j in range(2)]
  return dict(complete=True,attention=dict(complete=True,records=a),kv=dict(complete=True,records=k))
 def test_metric_schema_tamper(self):
  for domain in ('attention','kv'):
   for key,value in [('raw_different_elements',1),('raw_different_elements',False),('strict_failures',True),('candidate_nonfinite',1),('reference_nonfinite',1),('shape',[1]),('elements',1),('dtype','torch.float32'),('strict_atol',1e-3),('strict_rtol',1e-3)]:
    r=self.fixture();row=r[domain]['records'][0];m=row['metrics']if domain=='attention'else row['selected_metrics']['key'];m[key]=value
    self.assertFalse(audit(r),(domain,key,value))
 def test_fixture_dependencies(self):self.assertGreater(len(dependencies()),10)
 def test_identity_and_corruption(self):
  r=self.fixture();self.assertTrue(audit(r))
  for domain in ('attention','kv'):
   for action in ('duplicate','missing','reverse','raw','alias','readonly'):
    x=copy.deepcopy(r);rows=x[domain]['records']
    if action=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif action=='missing':rows.pop()
    elif action=='reverse':rows.reverse()
    elif action=='raw':
     if domain=='attention':rows[0]['metrics']['raw_equal']=False
     else:rows[0]['selected_metrics']['key']['raw_equal']=False
    elif action=='alias':rows[0]['return_out_alias' if domain=='attention' else 'shared_backing']=False
    else:rows[0]['inputs_preserved' if domain=='attention' else 'input_full_owned_storage_preserved']=False
    self.assertFalse(audit(x),(domain,action))
 def helper(self,target,name,env):
  import ast,inspect
  from . import modules_attention_kv as m
  tree=ast.parse(inspect.getsource(getattr(m,target)))
  node=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name==name)
  ns=dict(env);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-helper>','exec'),ns)
  return ns[name]
 def test_helpers_ast_original(self):
  import ast,inspect
  from pathlib import Path
  from . import modules_attention_kv as m
  for target,path,names in [('_attention','paged_attention_qualification_v10/run.py',('digest','load')),('_kv','attention_kv_qualification_v2/run.py',('load','restore','views'))]:
   old=ast.parse((m.P/path).read_text());new=ast.parse(inspect.getsource(getattr(m,target)))
   for name in names:
    a=next(x for x in ast.walk(old)if isinstance(x,ast.FunctionDef)and x.name==name)
    b=next(x for x in ast.walk(new)if isinstance(x,ast.FunctionDef)and x.name==name)
    self.assertEqual(ast.dump(a),ast.dump(b),(target,name))
 def test_actual_digest_and_load(self):
  import types,hashlib
  class T:
   def detach(self):return self
   def cpu(self):return self
   def contiguous(self):return self
   def view(self,d):return self
   def numpy(self):return self
   def tobytes(self):return b'actual-helper-bytes'
  x=T();torch=types.SimpleNamespace(uint8='u8',load=lambda *a,**k:x)
  digest=self.helper('_attention','digest',dict(torch=torch,hashlib=hashlib))
  expected=hashlib.sha256(x.tobytes()).hexdigest();self.assertEqual(digest(x),expected)
  entry=dict(path='mock-only',sha256='hash',logical_sha256=expected)
  load=self.helper('_attention','load',dict(torch=torch,sha=lambda p:'hash',digest=digest));self.assertIs(load(entry),x)
  bad=dict(entry,logical_sha256='bad')
  with self.assertRaises(AssertionError):load(bad)
  self.assertIs(self.helper('_kv','load',dict(torch=torch))(entry),x)
 def test_actual_restore_and_views(self):
  import types
  class CPU:
   dtype='bf16'
   def element_size(self):return 2
  cpu=CPU();base=object();copies=[];calls=[]
  class View:
   def copy_(self,x):copies.append(x)
  def as_strided(*args):calls.append(args);return View()
  torch=types.SimpleNamespace(full=lambda *a,**k:base,as_strided=as_strided)
  restore=self.helper('_kv','restore',dict(torch=torch,load=lambda e:cpu))
  md=dict(storage_nbytes=64,shape=[2,2],stride=[4,1],storage_offset=1)
  b,v=restore(dict(metadata=md));self.assertIs(b,base);self.assertEqual(copies,[cpu]);self.assertEqual(calls[-1],(base,[2,2],[4,1],1))
  kd=dict(stride=[4,1],storage_offset=1);vd=dict(stride=[4,1],storage_offset=3)
  views=self.helper('_kv','views',dict(torch=torch,shape=[2,2],kd=kd,vd=vd))
  result=views(base);self.assertEqual(len(result),2);self.assertEqual([x[-1]for x in calls[-2:]],[1,3])
if __name__=='__main__':unittest.main()
