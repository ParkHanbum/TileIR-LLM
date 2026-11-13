from .contract import MIN,MAX

def cases():
 raw=[([],0,1,0),([],1,0,3),([0],0,1,0),([-1],0,1,3),([1],0,1,1),([MIN,0,MAX-1],MIN,MAX,2),([MAX],MIN,MAX,0),([MIN],MIN,MIN,1),([0]*32,0,248320,0),([248319]*192,0,248320,3),([0]*191+[248320],0,248320,2),([0]*1024,0,1,1),([0],1,0,0),([MIN],MIN,MIN+1,1),([MAX-1],MAX-1,MAX,2),([0]*1022+[1],0,1,3)]
 return [dict(values=v,lower=l,upper=h,offset=o,shape=([32,6]if len(v)==192 else []if i==2 else[len(v)]))for i,(v,l,h,o)in enumerate(raw)]
