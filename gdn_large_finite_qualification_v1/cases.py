DTYPES=('torch.bfloat16','torch.float32')
SIZES=(16777216,16777215)
VARIANTS=('finite','negative_zero','subnormal','maxfinite','nan','negative_nan','inf','negative_inf')
def cases():return [(d,n,v)for d in DTYPES for n in SIZES for v in VARIANTS]
def expected(v):return v in VARIANTS[:4]
def guarded_target(x,predicate,target):
 ok=predicate(x)
 assert type(ok)is bool
 if not ok:return False
 target();return True
