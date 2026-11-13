import unittest,ast
from pathlib import Path
from .chain import execute,LABELS
from .run import result_gate
class Tests(unittest.TestCase):
 def test_chain_edges(self):
  x={k:object() for k in ('q','k','v','g','beta','initial_state','cu_seqlens','chunk_indices','chunk_offsets')};y={k:object() for k in LABELS};calls=[]
  def op(n,args,kw,result):
   def f(*a,**k):self.assertEqual(a,args);self.assertEqual(k,kw);calls.append(n);return result
   return f
  cu=x['cu_seqlens'];ci=x['chunk_indices'];ops={
   'cumsum':op('cumsum',(x['g'],),dict(cu_seqlens=cu,chunk_indices=ci),y['g_cumsum']),
   'kkt':op('kkt',(),dict(k=x['k'],g=y['g_cumsum'],beta=x['beta'],cu_seqlens=cu,chunk_indices=ci,bk=64),y['A_kkt']),
   'solve':op('solve',(y['A_kkt'],cu,ci),{},y['A_inverse']),
   'wy':op('wy',(x['k'],x['v'],x['beta'],y['g_cumsum'],y['A_inverse'],cu,ci),{},(y['w'],y['u'])),
   'state':op('state',(x['k'],y['w'],y['u'],y['g_cumsum'],x['initial_state'],cu,ci,x['chunk_offsets']),{},(y['h'],y['v_new'],y['final_state'])),
   'output':op('output',(x['q'],x['k'],y['v_new'],y['h'],y['g_cumsum'],cu,ci),{},y['o'])}
  seen=[];out=execute(x,ops,lambda d:seen.extend(d));self.assertEqual(seen,list(LABELS));self.assertEqual(out,(y['o'],y['final_state']));self.assertEqual(calls,list(ops))
 def test_first_stage_stop(self):
  calls=[];ops={'cumsum':lambda *a,**k:object()};x={k:object() for k in ('g','cu_seqlens','chunk_indices')}
  def fail(d):calls.append(d);raise RuntimeError('latch')
  with self.assertRaises(RuntimeError):execute(x,ops,fail)
  self.assertEqual(len(calls),1)
 def test_gate(self):
  r=dict(complete=True,initial_flags=[True,True],final_flags=[True,True],records=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in (0,1)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][2]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_partial(self):self.assertFalse(result_gate({})['raw_pass'])
 def test_source(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
