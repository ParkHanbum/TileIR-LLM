from pathlib import Path
import hashlib,json
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111');B=P/'norm_runtime_integration_v4';R=P/'public_vllm_composed_integration_v1';R.mkdir(exist_ok=False)
for n in ['__init__.py','launch.py','run.py','precision.py']:
 s=(B/n).read_text().replace('norm_runtime_integration_v4','public_vllm_composed_integration_v1').replace('normIntegration04','composedIntegration01').replace('all width2560 GemmaRMSNorm native shadow','GemmaRMSNorm, attention gate and block gather combined native shadow').replace('norm_native_shadow','composed_native_shadow').replace('norm_replacement','composed_replacement').replace('norm_correctness_complete','composed_correctness_complete').replace('Norm shadow raw/contract HOLD','Composed shadow raw/contract HOLD')
 (R/n).write_text(s)
(R/'adapter.py').write_text('''"""Compose three independently qualified, public-source functional adapters."""
from pathlib import Path
from norm_runtime_integration_v4 import adapter as norm
from attention_gate_runtime_integration_v1 import adapter as attention
from block_gather_runtime_integration_v1 import adapter as blocks
ATTR='_tileir_public_composed_v1'
PARTS=(('norm',norm,1040),('attention_gate',attention,128),('block_gather',blocks,16))
def install(model,output):
 assert not hasattr(model,ATTR)
 root=Path(output);root.mkdir(exist_ok=False);installed=[];results={}
 setattr(model,ATTR,installed)
 try:
  for key,module,count in PARTS:
   results[key]=module.install(model,str(root/key));installed.append((key,module))
  return dict(components=results,logical_calls_expected=1184,scope='Three functional public-source adapters; no combined optimized graph or ABI claim')
 except BaseException:
  close(model);raise
def start(model,mode):
 assert len(getattr(model,ATTR))==3
 return {key:module.start(model,mode) for key,module,count in PARTS}
def finish(model):
 components={key:module.finish(model) for key,module,count in PARTS}
 for key,module,count in PARTS:assert components[key]['supported_calls']==count
 return dict(components=components,shadow_pass=all(v['shadow_pass'] for v in components.values()),supported_calls=sum(v['supported_calls'] for v in components.values()),supported_vendor_fallback_calls=sum(v.get('supported_vendor_fallback_calls',0) for v in components.values()))
def close(model):
 installed=getattr(model,ATTR,None)
 if installed is None:return dict(restored=True)
 results={};errors=[]
 for key,module in reversed(installed):
  try:results[key]=module.close(model)
  except BaseException as e:errors.append((key,repr(e)))
 delattr(model,ATTR)
 assert not errors,errors
 assert all(v['restored'] for v in results.values())
 return dict(restored=True,components=results)
''')
(R/'gates.py').write_text('''import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
PROOFS=(('norm_runtime_integration_v4','normIntegration04','c095d7e62a33d0507e3d0a190e38d948ff1bd6abbe747b2bbcae1f5d2951894d'),('attention_gate_runtime_integration_v1','attentionGateIntegration01','b6121375798381b9655917c8a30c16a672d238aed7c2d52e23809721c64ad45d'),('block_gather_runtime_integration_v1','blockGatherIntegration01','5b25f22ff2bd88599941f592e97c6f023a8ab8dfe975f0a8cbc3af81575b873d'))
def qualification(root):
 evidence=[]
 for folder,name,expected in PROOFS:
  base=P/folder;r=base/name/'RESULT.json';l=base/(name+'-launch.json');c=base/(name+'-recovery.json')
  assert hashlib.sha256(r.read_bytes()).hexdigest()==expected
  result=json.loads(r.read_text());launch=json.loads(l.read_text());recovery=json.loads(c.read_text())
  assert result['complete'] and result['public_logprob_gate']['passed'] and launch['returncode']==0 and launch['result_gate']['passed']
  assert recovery['child_pid_absent'] and recovery['lock_available']
  evidence.append(dict(result=str(r),result_sha=expected,launch=str(l),recovery=str(c)))
 return dict(components=evidence,scope='Separate prior integrations, not evidence of composition; new1184shadow must pass')
''')
(R/'result_gate.py').write_text('''def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  a,b=r['shadow_request'],r['replacement_request']
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for arm in (a,b):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==1184 and d['supported_vendor_fallback_calls']==0
   assert {k:v['supported_calls'] for k,v in d['components'].items()}=={'norm':1040,'attention_gate':128,'block_gather':16}
   assert arm['allow_bf16_reduced_precision_reduction'] is False and arm['allow_bf16_reduced_precision_reduction_split_k'] is True
  for key,n in [('attention_gate',128),('block_gather',16)]:
   assert a['selected_dispatch']['components'][key]['native_gate_calls']==n
   assert b['selected_dispatch']['components'][key]['native_gate_calls']==0
   assert b['selected_dispatch']['components'][key]['tileir_gate_calls']==n
  assert not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='Three-adapter functional composition only; no performance or ABI claim')
 except (KeyError,AssertionError,TypeError):return dict(passed=False,errors=['Composed functional contract HOLD'])
''')
(R/'PLAN.md').write_text('''Three independently successful public-source adapters composed in one eager B32/P53/G16 request. F precision(false,true) held identically in both arms. Norm65 modules1040 calls + attention8 modules128 gates + block gather16 =1184 logical calls, not GPU kernel events or global coverage. All shadow subgates must pass before combined real replacement. Other operations native. 512 IDs and returned public top5/chosen logprobs checked. No timing/performance, no optimized graphs or ABI claim. Existing individual proof files/caches unchanged. Source-only reuse of bounded930s norm runner; fresh composedIntegration01. New directory only. Two requests produce side effects only in their ordinary inference caches; hooks and policy restored, engine shutdown verified. Failure closes all installed adapters in reverse order.
''')
(R/'test_cpu.py').write_text('''import unittest,types,tempfile
from pathlib import Path
from unittest.mock import patch
from public_vllm_composed_integration_v1 import adapter
from public_vllm_composed_integration_v1.gates import qualification
class Tests(unittest.TestCase):
 def parts(self,fail=None):
  events=[];parts=[]
  for key,n in [('norm',1040),('attention_gate',128),('block_gather',16)]:
   def install(m,p,key=key):
    events.append(('install',key))
    if key==fail:raise RuntimeError('injected')
    return {}
   def close(m,key=key):events.append(('close',key));return {'restored':True}
   module=types.SimpleNamespace(install=install,close=close,start=lambda m,mode:{},finish=lambda m,n=n:{'shadow_pass':True,'supported_calls':n})
   parts.append((key,module,n))
  return parts,events
 def test_compose(self):
  parts,e=self.parts();m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   adapter.install(m,str(Path(d)/'new'));adapter.start(m,'shadow');r=adapter.finish(m);self.assertEqual(r['supported_calls'],1184);self.assertTrue(r['shadow_pass']);self.assertTrue(adapter.close(m)['restored']);self.assertEqual(e[-3:],[('close','block_gather'),('close','attention_gate'),('close','norm')])
 def test_partial_install_cleanup(self):
  parts,e=self.parts('attention_gate');m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'new'))
  self.assertFalse(hasattr(m,adapter.ATTR));self.assertEqual(e[-1],('close','norm'))
 def test_qualification(self):self.assertEqual(len(qualification(Path(__file__).parent)['components']),3)
if __name__=='__main__':unittest.main()
''')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={}
for folder,name,_ in __import__('ast').literal_eval("(('norm_runtime_integration_v4','normIntegration04',''),('attention_gate_runtime_integration_v1','attentionGateIntegration01',''),('block_gather_runtime_integration_v1','blockGatherIntegration01',''))"):
 base=P/folder
 for path,digest in json.loads((base/'PINS.json').read_text()).items():assert sha(path)==digest;pins[path]=digest
 for f in [base/'PINS.json',base/name/'RESULT.json',base/(name+'-launch.json'),base/(name+'-recovery.json')]:pins[str(f)]=sha(f)
for f in R.iterdir():
 if f.is_file():pins[str(f)]=sha(f)
(R/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n');print('pins',len(pins),sha(R/'PINS.json'))
