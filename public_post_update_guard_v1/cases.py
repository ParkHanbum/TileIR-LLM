from copy import deepcopy

def base():
    return dict(idx=list(range(32)),computed=[0]*32,sampled=[5]*32,ns=[1]*32,nr=[0]*32,q=list(range(33)),total=[0]*32,hist={})
def cases():
    rows=[]
    def add(name,changes,error=None):
        d=base()
        for key,index,value in changes:d[key][index]=value
        rows.append((name,d,error))
    add('valid',[])
    add('state_bounds',[('idx',0,32)],'state index bounds')
    add('state_before_duplicate',[('idx',0,32),('idx',1,32)],'state index bounds')
    add('duplicate',[('idx',0,1)],'conflicting active state indices')
    add('duplicate_before_ns',[('idx',0,1),('ns',1,2)],'conflicting active state indices')
    for n in (-2147483648,-1,2,2147483647):add('bad_ns_'+str(n),[('ns',0,n)],'sample count exceeds observed row')
    add('ns_before_append',[('ns',0,2),('total',0,-1)],'sample count exceeds observed row')
    add('negative_total',[('total',0,-1)],'append bounds')
    add('append_overflow',[('total',0,128)],'append bounds')
    add('append_exact',[('total',0,127)])
    add('no_append_at_end',[('total',0,128),('ns',0,0)])
    for t in (-9223372036854775808,-1,248320,9223372036854775807):add('bad_token_'+str(t),[('sampled',0,t)],'histogram/token bounds')
    add('hist_overflow',[('hist',(0,5),2147483647)],'histogram integer overflow')
    add('negative_hist_is_valid',[('hist',(0,5),-2147483648)])
    add('query_overflow',[('q',0,-2147483648),('q',1,2147483647)],'query/rejection arithmetic overflow')
    add('delta_overflow',[('nr',0,-2147483648)],'query/rejection arithmetic overflow')
    add('computed_overflow',[('computed',0,2147483647)],'computed integer overflow')
    add('computed_underflow',[('computed',0,-2147483648),('q',1,-1)],'computed integer overflow')
    add('inactive_extrema',[('idx',0,-9223372036854775808),('ns',0,-2147483648),('sampled',0,9223372036854775807),('q',0,-2147483648)])
    d=base();d.update(idx=[-1]*32,ns=[2147483647]*32,sampled=[-1]*32,total=[-1]*32,computed=[2147483647]*32,q=[-2147483648 if i%2 else 2147483647 for i in range(33)])
    rows.append(('all_inactive',d,None))
    return rows

def arguments(d):
    return d['idx'],d['computed'],lambda s,t:d['hist'].get((s,t),0),d['sampled'],d['ns'],d['nr'],d['q'],d['total']
