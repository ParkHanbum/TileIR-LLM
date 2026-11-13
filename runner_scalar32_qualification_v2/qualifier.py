# SPDX-License-Identifier: Apache-2.0
"""CPU-prepared callable qualifier; no automatic runtime/GPU entry."""
from runner_scalar32_qualification_v2.binding import Binding
from runner_scalar32_qualification_v2.api import validate

def owner_signature(owner):
    out=owner.num_accepted_tokens_gpu
    validate(out,0)
    if owner.max_num_reqs != 32 or owner._align_mode is not False:
        raise ValueError("only max32/align=false")
    if getattr(owner, "recoverssm", None) is not None:
        raise ValueError("recoverssm unsupported")
    storage=out.untyped_storage()
    if out.element_size()!=4 or storage.nbytes()!=128 or out.data_ptr()!=storage.data_ptr() or out.data_ptr()%4:
        raise ValueError("typed exact128B backing/alignment")
    return (id(owner),id(out),out.data_ptr(),storage.data_ptr(),storage.nbytes(),out.device)

class ScalarQualifier:
    def __init__(self,owner,candidate,raw):
        # raw is a coordinator-supplied full typed backing byte snapshot; never addresses→tensors.
        self.owner=owner; self.sig=owner_signature(owner); self.candidate=candidate;self.raw=raw
        self.failed=False;self.mode=None;self.rows=[];self.reference=None
        original=type(owner).__dict__["add_request"]
        if "add_request" in owner.__dict__ or owner.add_request.__func__ is not original:
            raise ValueError("instance cached binding unsupported")
        self.binding=Binding(type(owner),self.call)
        body=self.binding.replacement
        def guarded(actual,index,request):
            if self.failed or actual is not self.owner:
                self.failed=True;raise RuntimeError("foreign owner or latched")
            try:
                if owner_signature(actual)!=self.sig:raise ValueError("owner changed before super")
                return body(actual,index,request)
            except BaseException:
                self.failed=True;raise
        self.binding.replacement=guarded
    def start(self,mode):
        if self.failed or mode not in ("shadow","replace"):raise RuntimeError("latched or bad phase")
        if mode=="replace" and self.reference is None:raise RuntimeError("shadow prerequisite")
        self.mode=mode;self.rows=[]
    def call(self,out,index):
        if self.failed:raise RuntimeError("latched")
        try:
            if self.mode is None or owner_signature(self.owner)!=self.sig or out is not self.owner.num_accepted_tokens_gpu:
                raise ValueError("owner/phase changed")
            validate(out,index)
            before=self.raw(out)
            if len(before)!=128:raise ValueError("raw extent")
            expected=bytearray(before);expected[4*index:4*index+4]=(1).to_bytes(4,"little",signed=True)
            if self.mode=="shadow":
                out[index].fill_(1)
                native=self.raw(out)
                # Candidate runs on independent typed clone preserving full preimage.
                scratch=out.clone(); scratch[index].fill_(int.from_bytes(before[4*index:4*index+4],"little",signed=True))
                self.candidate(scratch,index)
                if native!=bytes(expected) or self.raw(scratch)!=native:raise ValueError("shadow raw")
            else:
                n=len(self.rows)
                if n>=32 or self.reference[n]!=(index,before,bytes(expected)):raise ValueError("same-input mismatch")
                self.candidate(out,index)
                if self.raw(out)!=bytes(expected):raise ValueError("replace raw")
            self.rows.append((index,before,bytes(expected)))
        except BaseException:
            self.failed=True;raise
    def finish(self):
        if self.failed or len(self.rows)!=32 or sorted(r[0] for r in self.rows)!=list(range(32)):
            self.failed=True;raise RuntimeError("census or failure")
        if self.mode=="shadow":self.reference=list(self.rows)
        return {"calls":32,"indices":sorted(r[0] for r in self.rows),"raw_backing_bytes_per_call":128,"mode":self.mode}
    def close(self):self.binding.close()

def run_phases(q,request,endpoint,expected):
    """request resets same runtime's requests; endpoint returns canonical (512IDs,2560public).
    Caller supplies frozen expected, exact runtime and cleanup/admission, not optional flags.
    """
    if len(expected[0])!=512 or len(expected[1])!=2560:raise ValueError("frozen endpoint extent")
    result={"complete":False}
    q.binding.install()
    try:
        for phase in ("shadow","replace"):
            q.start(phase);value=endpoint(request(phase))
            if value!=expected:q.failed=True;raise ValueError("endpoint mismatch")
            result[phase]=q.finish()
        result["complete"]=True
        return result
    finally:q.close()

def synthetic(candidate,raw,make):
    # make returns new typed cuda int32 vector32 with nonconstant sentinel values.
    for i in range(32):
        original=make();reference=original.clone();trial=original.clone()
        before=raw(original)
        reference[i].fill_(1);candidate(trial,i)
        if raw(reference)!=raw(trial) or raw(original)!=before:raise ValueError("synthetic full backing")
    return {"cases":32,"passed":True}
