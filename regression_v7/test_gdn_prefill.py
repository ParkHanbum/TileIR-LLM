import copy,unittest
if __package__:
 from . import modules_gdn_prefill as m
else:
 import modules_gdn_prefill as m

def metric(desc):
 return dict(reference_descriptor=copy.deepcopy(desc),candidate_descriptor=copy.deepcopy(desc),bytes_compared=desc['owner_bytes'],raw_byte_mismatches=0,raw_equal=True,strict_failures=0,reference_nonfinite=0,candidate_nonfinite=0)

def fixture():
 descriptors=m.expected_descriptors()
 r=dict(complete=True,solver=m.SOLVER,timing=False,scope=dict(prefill_actual_banks=1,parent_synthetic_fixtures=3,live_shared_owner=False,all_24_layers=False),records=[],raw_negative_detected=True)
 for case,arm,repeat in m.expected_order():
  row=dict(case=case,arm=arm,repeat=repeat,passed=True,readonly=True,repeat_raw=True,stream_unchanged=True)
  if case=='core_layer0':row.update(stage_metrics=[dict(label=k,comparison=metric(descriptors[k]))for k in m.LABELS],output_metrics={k:metric(descriptors[k])for k in ('o','final_state')},stages=list(m.LABELS),raw_all_stages=True,raw_output_state=True,finite=True,fresh_nonalias=True,observer_restored=True)
  else:row.update(dict.fromkeys(('full_private_backing_raw','gather_raw','invert_raw','mask_raw','scatter_return_identity','mask_return_identity','fresh_nonalias','false_positive_zero','special_bits'),True))
  r['records'].append(row)
 r['negative_probes']=[]
 for label in ('o','final_state'):
  negatives=[]
  for case in ('bitflip','finite_large'):
   bad=metric(descriptors[label]);bad.update(raw_equal=False,raw_byte_mismatches=1,strict_failures=1)
   negatives.append(dict(label=label,case=case,rejected=True,comparison=bad))
  r['negative_probes'].append(dict(label=label,positive=metric(descriptors[label]),negative=negatives))
 return r
class Tests(unittest.TestCase):
 def test_valid_exact_scope(self):self.assertTrue(m.audit(fixture())['passed']);self.assertEqual(len(m.expected_order()),16)
 def test_missing_duplicate_order(self):
  for kind in ('missing','duplicate','order'):
   r=fixture()
   if kind=='missing':r['records'].pop()
   elif kind=='duplicate':r['records'][1]=copy.deepcopy(r['records'][0])
   else:r['records'].reverse()
   self.assertFalse(m.audit(r)['passed'])
 def test_old_solver_or_scope_promotion(self):
  r=fixture();r['solver']='tileir_gdn_solve_tril_v5';self.assertFalse(m.audit(r)['passed'])
  r=fixture();r['scope']['live_shared_owner']=True;self.assertFalse(m.audit(r)['passed'])
 def test_core_and_parent_corruption(self):
  for index,field in ((0,'raw_output_state'),(1,'finite'),(2,'observer_restored'),(3,'fresh_nonalias'),(4,'full_private_backing_raw'),(5,'invert_raw'),(6,'scatter_return_identity'),(7,'false_positive_zero'),(8,'readonly'),(9,'special_bits'),(10,'stream_unchanged'),(11,'repeat_raw')):
   r=fixture();r['records'][index][field]=False;self.assertFalse(m.audit(r)['passed'])
 def test_stage_omission(self):
  r=fixture();r['records'][0]['stages'].pop();self.assertFalse(m.audit(r)['passed'])
 def test_manifest_identity(self):
  d=m.dependencies();self.assertEqual(d[str(m.MANIFEST)],m.FIXED[str(m.MANIFEST)]);self.assertEqual(set(m.FIXED)-set(d),set())
 def test_stage_metric_tampering(self):
  for field in ('shape','dtype','stride','numel','owner_bytes','typed_count','raw_count','finite_count'):
   r=fixture();c=r['records'][0]['stage_metrics'][0]['comparison']
   if field=='typed_count':c['raw_byte_mismatches']=False
   elif field=='raw_count':c['raw_byte_mismatches']=1
   elif field=='finite_count':c['candidate_nonfinite']=1
   else:c['candidate_descriptor'][field]=None
   self.assertFalse(m.audit(r)['passed'])
 def test_negative_oracle_omission(self):
  for change in ('missing','accepted','wrongstate','positivebad'):
   r=fixture()
   if change=='missing':r['negative_probes'].pop()
   elif change=='accepted':r['negative_probes'][0]['negative'][0]['rejected']=False
   elif change=='wrongstate':r['negative_probes'][1]['label']='o'
   else:r['negative_probes'][0]['positive']['raw_byte_mismatches']=1
   self.assertFalse(m.audit(r)['passed'])
 def test_actual_cpu_snapshot_checker_and_corruption(self):
  import torch
  for dtype in (torch.bfloat16,torch.float32):
   reference=torch.tensor([[0.,1.,-2.],[3.,4.,5.]],dtype=dtype)
   positive=m.snapshot_check(torch,reference,reference.clone(),m.describe(reference))
   self.assertEqual(positive['bytes_compared'],reference.numel()*reference.element_size())
   probe=m.negative_probe(torch,reference,'cpu_output_or_state')
   self.assertEqual([n['case']for n in probe['negative']],['bitflip','finite_large'])
   self.assertTrue(all(n['rejected']for n in probe['negative']))
 def test_always_pass_checker_is_detected(self):
  import torch
  reference=torch.tensor([1.,2.],dtype=torch.float32)
  def always_pass(*args):return metric(m.describe(reference))
  with self.assertRaises(RuntimeError):m.negative_probe(torch,reference,'state',checker=always_pass)
if __name__=='__main__':unittest.main()
