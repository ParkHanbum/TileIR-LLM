"""CPU-only plan and fail-closed result audit. No backend imports."""
from pathlib import Path
import json
GROUPS={'out_prefill':['gdn_out_prefill','attn_out_prefill'],'out_decode':['gdn_out_decode','attn_out_decode'],'down_prefill':['down_prefill'],'down_decode':['down_decode'],'head':['head_prefill','head_decode']}
SHAPES={'out_prefill':[1696,2560,4096],'out_decode':[32,2560,4096],'down_prefill':[1696,2560,9216],'down_decode':[32,2560,9216],'head':[32,248320,2560]}
REFERENCE=dict(math_mode=0,algo=99,compute_type=68,pointer_mode=0,alpha=1,beta=0)
CHUNK_BYTES=16*1024*1024
FAILURE_CAP=64*1024*1024

def validate(m):
 assert m['schema']=='unpacked-five-contract-eight-bank-qualification-v1'
 assert m['torch_flags_required']==[True,True] and m['reference']==REFERENCE
 assert m['backend']=='tileirblas_once_occ2.linear' and m['timing'] is False and m['model_replacement'] is False
 assert [g['contract'] for g in m['groups']]==list(GROUPS)
 for g in m['groups']:
  name=g['contract'];assert g['logical_MNK']==SHAPES[name]
  assert [x['label']for x in g['banks']]==GROUPS[name]
  for row in g['banks']:
   s=row['sample'];w=row['weight'];M,N,K=SHAPES[name]
   for meta,shape in [(s['X']['metadata'],[M,K]),(s['Y']['metadata'],[M,N]),(w['runtime_metadata'],[N,K])]:
    assert meta['shape']==shape and meta['stride']==[shape[1],1] and meta['storage_offset']==0 and meta['storage_nbytes']==2*shape[0]*shape[1]
   assert s['flags_before']==s['flags_after']==[True,True] and s['X_preserved'] and s['returned_native_object']
   assert w['logical_sha256']==w['checkpoint_named_tensor_sha256']==w['final_logical_sha256'] and w['full_named_tensor_byte_equal']
 return m['groups']

def passing(c):
 return c.get('raw_equal') is True and c.get('strict_failures')==0 and c.get('candidate_nonfinite')==0 and c.get('reference_nonfinite')==0

def schedule(groups,execute_bank,records,save):
 for group in groups:
  entry=dict(contract=group['contract'],state='active',banks=[],skipped=[]);records.append(entry);save()
  for i,row in enumerate(group['banks']):
   bank=dict(label=row['label'],records=[],state='active');entry['banks'].append(bank);save()
   execute_bank(row,bank)
   assert bank['state'] in ('PASS','HOLD')
   if bank['state']=='HOLD':
    entry.update(state='HOLD',halted_bank=row['label'],skipped=[x['label']for x in group['banks'][i+1:]]);save();break
   save()
  if entry['state']=='active':entry['state']='sample_PASS'
  save()

def result_gate(r):
 qualified=[]
 try:
  assert r['finished'] and r['gpu_executed'] and r['adopted'] is False and r['timing'] is False
  assert r['initial_flags']==r['final_flags']==[True,True]
  assert r['source_pins_unchanged'] and r['reference_closed'] and not r.get('error') and not r.get('cleanup_errors')
  assert [g['contract']for g in r['contracts']]==list(GROUPS)
  calls=0
  for g in r['contracts']:
   names=GROUPS[g['contract']];banks=g['banks'];assert banks and [b['label']for b in banks]==names[:len(banks)]
   for i,b in enumerate(banks):
    assert b['weight_checkpoint_equal'] and b['inputs_unchanged'] and b['backing_unchanged']
    seq=[('native',0),('native',1),('tileir',0),('tileir',1)];records=b['records'];assert records and [(q['stage'],q['repeat'])for q in records]==seq[:len(records)]
    for j,q in enumerate(records):
     assert q['input_unchanged'] and q['backing_unchanged'] and q['output_contract_pass']
     if q['stage']=='native':assert all(q['api'][k]==v for k,v in REFERENCE.items())
     if not passing(q['comparison']):assert b['state']=='HOLD' and j==len(records)-1
     else:assert b['state']=='PASS' or j<len(records)-1
    calls+=len(records)
    if b['state']=='PASS':assert len(records)==4 and all(passing(q['comparison'])for q in records)
    else:assert b['state']=='HOLD' and i==len(banks)-1 and not passing(records[-1]['comparison'])
   if g['state']=='sample_PASS':
    assert len(banks)==len(names) and all(b['state']=='PASS'for b in banks) and not g['skipped'];qualified.append(g['contract'])
   else:
    assert g['state']=='HOLD' and banks[-1]['state']=='HOLD' and g['halted_bank']==banks[-1]['label'] and g['skipped']==names[len(banks):]
  assert r['logical_calls']==calls
  return dict(complete=True,raw_pass=len(qualified)==5,qualified_sample_contracts=qualified,raw_difference_hold=len(qualified)<5,model_qualified=False)
 except (KeyError,AssertionError,TypeError,IndexError):
  return dict(complete=False,raw_pass=False,qualified_sample_contracts=[],raw_difference_hold=False,model_qualified=False)
