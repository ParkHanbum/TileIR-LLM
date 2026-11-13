import ast,copy,json,os,re,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).parent;P=R.parent;sys.path[:0]=[str(R),str(P)]
import contract
from result_gate import audit

class Tag:
 def __init__(self,name):self.name=name
 def __str__(self):return self.name
DT=Tag('torch.bfloat16');DEV=Tag('cuda:0')
class FakeTensor:
 def __init__(self,shape):self.shape=shape;self.dtype=DT;self.device=DEV;self.ndim=2;self._version=1
 def stride(self,i=None):s=(self.shape[1],1);return s if i is None else s[i]
 def storage_offset(self):return 0
 def data_ptr(self):return id(self)
 def untyped_storage(self):return types.SimpleNamespace(nbytes=lambda:self.shape[0]*self.shape[1]*2,data_ptr=lambda:id(self))
class LinearMethod:
 def apply(self,layer,x,bias=None):
  layer.calls+=1
  if getattr(layer,'fail',False):raise RuntimeError('native error')
  layer.last=FakeTensor((x.shape[0],layer.weight.shape[0]));return layer.last
class EmbeddingMethod(LinearMethod):pass

def model_fixture():
 rows=[]
 for i in range(32):
  kind='attn_out' if i%4==3 else 'gdn_out';suffix=contract.GROUPS[kind][0]
  rows.append(('language_model.model.layers.%d.%s'%(i,suffix),types.SimpleNamespace(weight=FakeTensor((2560,4096)),quant_method=LinearMethod(),bias=None,calls=0)))
  rows.append(('language_model.model.layers.%d.mlp.down_proj'%i,types.SimpleNamespace(weight=FakeTensor((2560,9216)),quant_method=LinearMethod(),bias=None,calls=0)))
 rows.append(('language_model.lm_head',types.SimpleNamespace(weight=FakeTensor((248320,2560)),quant_method=EmbeddingMethod(),bias=None,calls=0)))
 return types.SimpleNamespace(named_modules=lambda:rows),rows

class Tests(unittest.TestCase):
 def test_no_backend_import(self):self.assertNotIn('torch',sys.modules)
 def test_actual_nondry_entry_import_fresh_process(self):
  python='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
  env={k:v for k,v in os.environ.items() if k!='PYTHONPATH' and k not in ('LD_PRELOAD','LD_DEBUG') and not k.startswith(('TILEIR_','CUDA_TILE_','VLLM_','TRITON_','TORCHINDUCTOR_'))}
  env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  with tempfile.TemporaryDirectory() as d:
   for name,expect in [('unpacked_gemm_native_capture_v1','missing'),(R.name,'resolved')]:
    r=subprocess.run([python,'-B',str(R/'test_entry_import.py'),str(P/name),str(Path(d)/name),expect],env=env,cwd=d,capture_output=True,text=True,timeout=30)
    self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    got=json.loads(r.stdout);self.assertEqual(got['backend_modules_loaded'],0);self.assertEqual(got['real_lock_operations'],0)
    self.assertIn('before_backend' if expect=='resolved' else 'original_missing',got['status'])
 def test_scope_counts(self):
  self.assertEqual(sum(x[1] for x in contract.GROUPS.values()),65)
  self.assertEqual(sum(x[1]*16 for x in contract.GROUPS.values()),1040)
  shapes={(1696 if k!='head' else 32,v[2],v[3]) for k,v in contract.GROUPS.items()}|{(32,v[2],v[3]) for v in contract.GROUPS.values()}
  self.assertEqual(len(shapes),5)
 def test_phase(self):
  self.assertEqual(contract.phase('head',0,32,2560),'prefill')
  self.assertEqual(contract.phase('head',1,32,2560),'decode')
  for args in [('down',0,32,9216),('head',16,32,2560),('gdn_out',1,32,2560)]:
   with self.assertRaises(AssertionError):contract.phase(*args)
 def test_artifact_cap(self):
  size=0
  for k,v in contract.GROUPS.items():
   for phase in ('prefill','decode'):
    m=32 if k=='head' or phase=='decode' else 1696;size+=2*m*(v[2]+v[3])
  self.assertLess(size,contract.LOGICAL_CAP);self.assertLess(contract.CHUNK_BYTES,32*1024*1024)
 def test_checkpoint_keys(self):
  for k in contract.GROUPS:
   s=contract.checkpoint(k);self.assertTrue(Path(s['path']).is_file());self.assertEqual(s['key'],contract.WEIGHT_KEYS[k])
  self.assertTrue(contract.checkpoint('head')['key'].endswith('embed_tokens.weight'))
 def test_public_log_parser(self):
  source=P/'packed_vllm_runtime_compare_v4/packedPrecision01/cublas-D.log';text=source.read_text();starts=list(re.finditer(r'^I! cuBLAS',text,re.M));block=None
  for i,m in enumerate(starts):
   b=text[m.start():starts[i+1].start() if i+1<len(starts) else len(text)]
   if ' cublasGemmEx('in b:block=b;break
  self.assertIsNotNone(block)
  args={k:v for k,_,v in re.findall(r'^i!\s{2}(\w+): type=(.*?); val=(.*)$',block,re.M)}
  ptr=lambda key:int(re.search(r'0x(?:0x)?([0-9a-fA-F]+)',args[key])[1],16)
  x=dict(shape=[int(args['n']),int(args['k'])],stride=[int(args['ldb']),1],data_ptr=ptr('B'));w=dict(shape=[int(args['m']),int(args['k'])],stride=[int(args['lda']),1],data_ptr=ptr('A'));y=dict(shape=[int(args['n']),int(args['m'])],stride=[int(args['ldc']),1],data_ptr=ptr('C'))
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'log';p.write_text(block);r=contract.parse_one(p,x,w,y);self.assertIsNone(r['arguments']['alpha']['value'])
   with self.assertRaises(AssertionError):contract.parse_one(p,dict(x,data_ptr=x['data_ptr']+2),w,y)
   p.write_text(block+block)
   with self.assertRaises(AssertionError):contract.parse_one(p,x,w,y)
   p.write_text(block.replace('MathMode=CUBLAS_DEFAULT_MATH','MathMode=CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION'))
   with self.assertRaises(AssertionError):contract.parse_one(p,x,w,y)
 def test_native_once_same_object_restore(self):
  import observer
  model,rows=model_fixture();mods={}
  for name in ('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.linear','vllm.model_executor.layers.vocab_parallel_embedding'):
   mods[name]=types.ModuleType(name);mods[name].__path__=[]
  mods['vllm.model_executor.layers.linear'].UnquantizedLinearMethod=LinearMethod
  mods['vllm.model_executor.layers.vocab_parallel_embedding'].UnquantizedEmbeddingMethod=EmbeddingMethod
  fake_t=types.SimpleNamespace(Tensor=FakeTensor,bfloat16=DT);mods['torch']=fake_t
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,mods):
   o=observer.Observer(Path(d)/'bank')
   with patch.object(o,'save',return_value={}):
    o.install(model);name,m=next((n,m)for n,m in rows if n.endswith('layers.1.linear_attn.out_proj'))
    out=m.quant_method.apply(m,FakeTensor((1696,4096)),None);self.assertIs(out,m.last);self.assertEqual(m.calls,1)
    self.assertFalse(o.pending);self.assertEqual(o.rows[0]['original_calls'],1)
    o.close();self.assertNotIn('apply',m.quant_method.__dict__)
 def test_result_gate_rejects_incomplete(self):self.assertFalse(audit({})['passed'])
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
