"""Process-local public Python helper observation; exact call/return preserved."""
import functools
STAGES={'chunk_local_cumsum':('g_cumsum',),'chunk_scaled_dot_kkt_fwd':('A_kkt',),'solve_tril':('A_inverse',),'recompute_w_u_fwd':('w','u'),'chunk_gated_delta_rule_fwd_h':('h','v_new','final_state'),'chunk_fwd_o':('o',)}
class Observer:
 def __init__(self,module,snapshot,identity):self.module=module;self.snapshot=snapshot;self.identity=identity;self.bindings=[];self.rows=[];self.counts={};self.closed=False
 def install(self):
  try:
   for name,labels in STAGES.items():
    original=getattr(self.module,name)
    @functools.wraps(original)
    def wrapped(*args,_name=name,_labels=labels,_original=original,**kwargs):
     self.counts[_name]=self.counts.get(_name,0)+1;assert self.counts[_name]==1
     value=_original(*args,**kwargs)
     values=value if isinstance(value,tuple) else (value,)
     assert len(values)==len(_labels)
     row=dict(name=_name,original_calls=1,return_identity_preserved=True,helper=self.identity(_original),outputs={label:self.snapshot(t,label) for label,t in zip(_labels,values)})
     self.rows.append(row)
     return value
    self.bindings.append((name,original,wrapped));setattr(self.module,name,wrapped)
  except BaseException:self.close();raise
 def close(self):
  conflicts=[]
  for name,original,wrapped in reversed(self.bindings):
   if getattr(self.module,name) is not wrapped:conflicts.append(name)
   else:setattr(self.module,name,original)
  self.bindings=[];self.closed=not conflicts;assert not conflicts
 def complete(self):return self.closed and self.counts=={n:1 for n in STAGES} and [r['name'] for r in self.rows]==list(STAGES)
