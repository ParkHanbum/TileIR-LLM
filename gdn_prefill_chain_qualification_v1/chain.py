"""Six TileIR stages only. No native intermediate inputs or native fallback."""
LABELS=('g_cumsum','A_kkt','A_inverse','w','u','h','v_new','final_state','o')
def execute(inputs,ops,observe):
 x=inputs;cu=x['cu_seqlens'];ci=x['chunk_indices']
 g=ops['cumsum'](x['g'],cu_seqlens=cu,chunk_indices=ci);observe({'g_cumsum':g})
 a=ops['kkt'](k=x['k'],g=g,beta=x['beta'],cu_seqlens=cu,chunk_indices=ci,bk=64);observe({'A_kkt':a})
 ai=ops['solve'](a,cu,ci);observe({'A_inverse':ai})
 w,u=ops['wy'](x['k'],x['v'],x['beta'],g,ai,cu,ci);observe({'w':w,'u':u})
 h,vn,final=ops['state'](x['k'],w,u,g,x['initial_state'],cu,ci,x['chunk_offsets']);observe({'h':h,'v_new':vn,'final_state':final})
 out=ops['output'](x['q'],x['k'],vn,h,g,cu,ci);observe({'o':out})
 return out,final
