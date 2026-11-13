"""Original error order, independent of Torch/CUDA imports."""
MESSAGES=('state index bounds','conflicting active state indices','sample count exceeds observed row','append bounds','histogram/token bounds','histogram integer overflow','query/rejection arithmetic overflow','query/rejection arithmetic overflow','computed integer overflow')

def run_stages(evaluate):
    """Do not enqueue stage i+1 until stage i's host status is accepted."""
    for stage,message in enumerate(MESSAGES):
        bad=evaluate(stage)
        if type(bad)is not int or not 0<=bad<=32:raise RuntimeError('Invalid guard status')
        if bad:raise ValueError(message)

def reference(idx,computed,hist,sampled,ns,nr,q,total):
    """Pure integer oracle; sparse histogram accessor supports bounded CPU cases."""
    active=[i for i,s in enumerate(idx)if s>=0]
    states=[idx[i]for i in active]
    def need(v,k):
        if not v:raise ValueError(MESSAGES[k])
    need(all(s<32 for s in states),0)
    need(len(set(states))==len(states),1)
    need(all(0<=ns[i]<=1 for i in active),2)
    need(all(total[idx[i]]>=0 and total[idx[i]]+ns[i]<=128 for i in active),3)
    accepted=[i for i in active if ns[i]>0]
    need(all(0<=sampled[i]<248320 for i in accepted),4)
    need(all(hist(idx[i],sampled[i])<2147483647 for i in accepted),5)
    qd=[q[i+1]-q[i]for i in range(32)];delta=[qd[i]-nr[i]for i in range(32)]
    fits=lambda x:-2147483648<=x<=2147483647
    need(all(fits(qd[i])for i in active),6)
    need(all(fits(delta[i])for i in active),7)
    need(all(fits(computed[idx[i]]+delta[i])for i in active),8)
