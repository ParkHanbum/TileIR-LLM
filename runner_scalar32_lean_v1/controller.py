# Public scalar32 lean wrapper. Numeric API/kernel and source binding reused unchanged.
from runner_scalar32_qualification_v3.binding import Binding
from runner_scalar32_qualification_v3.qualifier import owner_signature
from runner_scalar32_qualification_v3.api import validate,fill_request_slot
class Controller:
    def __init__(self,owner,candidate=fill_request_slot,check_global=lambda:None):
        self.owner=owner;self.signature=owner_signature(owner);self.candidate=candidate
        self.check_global=check_global;self.failed=None;self.active=False;self.indices=[];self.attempts=0;self.returns=0
        original=type(owner).__dict__['add_request']
        if 'add_request' in owner.__dict__ or owner.add_request.__func__ is not original:raise ValueError('instance override')
        self.binding=Binding(type(owner),self.call)
        body=self.binding.replacement
        def guarded(actual,index,request):
            try:
                self.check_entry()
                if actual is not self.owner or owner_signature(actual)!=self.signature:raise ValueError('owner changed')
                self.check_slot(index)
                return body(actual,index,request)
            except BaseException as e:self.failed=type(e).__name__;raise
        self.binding.replacement=guarded
    def check_entry(self):
        if self.failed is not None or not self.active:raise RuntimeError('failed or inactive')
        self.check_global()
    def check_slot(self,index):
        validate(self.owner.num_accepted_tokens_gpu,index)
        if self.attempts>=32 or index in self.indices:raise ValueError('duplicate or excess')
    def install(self):self.binding.install()
    def start(self):
        if self.failed is not None:raise RuntimeError('latched')
        if self.active:raise RuntimeError('phase already active')
        self.indices=[];self.attempts=0;self.returns=0;self.active=True
    def call(self,out,index):
        try:
            self.check_entry();self.check_slot(index)
            if out is not self.owner.num_accepted_tokens_gpu or owner_signature(self.owner)!=self.signature:raise ValueError('owner changed')
            self.indices.append(index);self.attempts+=1
            self.candidate(out,index)
            self.returns+=1
        except BaseException as e:self.failed=type(e).__name__;raise
    def finish(self):
        if self.failed is not None or not self.active or self.attempts!=32 or self.returns!=32 or sorted(self.indices)!=list(range(32)):
            self.failed='census';raise RuntimeError('census')
        self.active=False
        return self.receipt()
    def receipt(self):
        return dict(calls=self.returns,attempts=self.attempts,indices=sorted(self.indices),failed=self.failed,selected_native_calls=0,selected_fallback_calls=0,logical_operations=self.returns)
    def close(self):
        self.active=False;self.binding.close()
