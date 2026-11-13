"""Shared first-failure and exact existing-owned-global guards; no numerical changes."""
def connect(ops,controller,four):
    if four.controller is not controller:raise RuntimeError('four/base controller mismatch')
    for name in ('zeros','contiguous_b','contiguous_a'):
        original=getattr(ops,name)
        def guarded(*args,_fn=original,_label=name,**kwargs):
            try:
                if controller.failed is not None or not controller.active or four.failed or not four.active or ops.failed:
                    raise RuntimeError('combined first-failure/active guard')
                for namespace,key,expected in four.owned_checks:
                    if namespace[key] is not expected:raise RuntimeError('four owned global changed')
                return _fn(*args,**kwargs)
            except BaseException as error:
                ops.failed=True
                controller.latch('gdn_owned/'+_label,error)
                four.failed=True
                raise
        setattr(ops,name,guarded)
