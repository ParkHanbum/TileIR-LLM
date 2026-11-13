import ast,copy,importlib.util,json,os,subprocess,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(R),str(P)]
from contract import GROUPS,SHAPES,REFERENCE,validate,schedule,result_gate

def fixture(fail=None):
 r=dict(finished=True,gpu_executed=True,adopted=False,timing=False,initial_flags=[True,True],final_flags=[True,True],source_pins_unchanged=True,reference_closed=True,logical_calls=0,contracts=[])
 for name,labels in GROUPS.items():
  group=dict(contract=name,state='sample_PASS',banks=[],skipped=[]);r['contracts'].append(group)
  for i,label in enumerate(labels):
   bank=dict(label=label,state='PASS',weight_checkpoint_equal=True,inputs_unchanged=True,backing_unchanged=True,records=[]);group['banks'].append(bank)
   for stage,repeat in [('native',0),('native',1),('tileir',0),('tileir',1)]:
    failed=(label,stage,repeat)==fail
    rec=dict(stage=stage,repeat=repeat,api=dict(REFERENCE),input_unchanged=True,backing_unchanged=True,output_contract_pass=True,comparison=dict(raw_equal=not failed,strict_failures=int(failed),candidate_nonfinite=0,reference_nonfinite=0))
    bank['records'].append(rec);r['logical_calls']+=1
    if failed:bank['state']='HOLD';group.update(state='HOLD',halted_bank=label,skipped=labels[i+1:]);break
   if bank['state']=='HOLD':break
 return r

class Tests(unittest.TestCase):
 def test_imports_no_backend(self):
  spec=importlib.util.spec_from_file_location('cpu_entry',R/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_manifest_capture_binding(self):
  m=json.loads((R/'MANIFEST.json').read_text());groups=validate(m)
  cap=json.loads(Path(m['capture_manifest']['path']).read_text());samples={s['kind']+'_'+s['phase']:s for s in cap['samples']}
  self.assertEqual(len(groups),5);self.assertEqual(sum(len(g['banks'])for g in groups),8)
  for g in groups:
   for row in g['banks']:
    self.assertEqual(row['sample'],samples[row['label']]);self.assertEqual(row['weight'],cap['weights'][row['sample']['weight_key']])
 def test_scope_not_collapsed(self):
  self.assertEqual(len(GROUPS['out_prefill']),2);self.assertEqual(len(GROUPS['head']),2)
  self.assertEqual(SHAPES['head'],[32,248320,2560])
 def test_exact_pass(self):
  r=fixture();self.assertEqual(r['logical_calls'],32);self.assertTrue(result_gate(r)['raw_pass']);self.assertEqual(len(result_gate(r)['qualified_sample_contracts']),5)
 def test_first_fail_native_or_tile_continues_other_contracts(self):
  for stage,rep in [('native',0),('native',1),('tileir',0),('tileir',1)]:
   r=fixture(('gdn_out_prefill',stage,rep));gate=result_gate(r)
   self.assertTrue(gate['complete']);self.assertTrue(gate['raw_difference_hold']);self.assertEqual(gate['qualified_sample_contracts'],list(GROUPS)[1:])
   self.assertEqual(r['contracts'][0]['skipped'],['attn_out_prefill'])
 def test_fail_second_same_contract_bank(self):
  r=fixture(('attn_out_prefill','tileir',1));self.assertTrue(result_gate(r)['complete']);self.assertEqual(r['contracts'][0]['skipped'],[])
 def test_reject_bad_receipts(self):
  for modify in [lambda r:r['contracts'].pop(),lambda r:r.update(final_flags=[False,True]),lambda r:r.update(logical_calls=31),lambda r:r['contracts'][0]['banks'][0].update(backing_unchanged=False),lambda r:r['contracts'][0]['banks'][0]['records'][0]['api'].update(algo=-1),lambda r:r['contracts'][0]['banks'][0]['records'].pop(),lambda r:r.update(error='error')]:
   r=fixture();modify(r);self.assertFalse(result_gate(r)['complete'])
 def test_reject_false_hold(self):
  r=fixture(('head_prefill','tileir',0));r['contracts'][-1]['banks'][-1]['records'][-1]['comparison']['raw_equal']=True;r['contracts'][-1]['banks'][-1]['records'][-1]['comparison']['strict_failures']=0
  self.assertFalse(result_gate(r)['complete'])
 def test_schedule_independent_first_failure(self):
  groups=validate(json.loads((R/'MANIFEST.json').read_text()));calls=[];records=[]
  def execute(row,bank):calls.append(row['label']);bank['state']='HOLD' if row['label']=='gdn_out_prefill' else 'PASS'
  schedule(groups,execute,records,lambda:None)
  self.assertNotIn('attn_out_prefill',calls);self.assertEqual(len(calls),7);self.assertEqual(len(records),5);self.assertEqual(records[-1]['state'],'sample_PASS')
 def test_fresh_entry_import(self):
  env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'};env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  code="import runpy,sys;sys.path.insert(0,%r);runpy.run_path(%r,run_name='cpu_only_entry');assert 'torch' not in sys.modules and 'cuda.tile' not in sys.modules"%(str(R),str(R/'run.py'))
  r=subprocess.run([sys.executable,'-B','-c',code],cwd='/tmp',env=env,capture_output=True,text=True,timeout=15);self.assertEqual(r.returncode,0,r.stderr)
 def test_storage_caps(self):
  maximum=max(2*(M*K+N*K+M*N)for M,N,K in SHAPES.values());self.assertLess(maximum,2*1024**3)
  first_failure_max=sum(2*M*N for M,N,K in SHAPES.values());self.assertLess(first_failure_max,64*1024**2)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
