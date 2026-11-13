from .contract import run_stages

class Guard:
    """One validation lifecycle; permanent failure and retained uncertain owners."""
    def __init__(self):self.failed=False;self.pending=[];self.calls=0;self.stages=[]
    def validate(self,torch,args):
        if self.failed:raise RuntimeError('Post-update guard failure latched')
        try:
            from tileir_vllm_post_update_v3.contract import validate_metadata
            validate_metadata(args)
            import cuda.tile as ct
            from .kernel import stage_guard
            # Typed contiguous views, no pointer-to-tensor reconstruction.
            idx,computed,last,hist,sampled,ns,nr,q,tokens,total=(x.view(-1)for x in args)
            stream=torch.cuda.current_stream(idx.device);handle=stream.cuda_stream
            status=torch.empty((1,),dtype=torch.int32,device=idx.device)
            packed=(idx,computed,hist,sampled,ns,nr,q,total,status)
            self.calls+=1
            def evaluate(stage):
                owners=(packed,stream);self.pending.append(owners)
                ct.launch(stream,(1,),stage_guard,packed+(stage,))
                bad=status.item() # completes the stage before a dependent launch
                if torch.cuda.current_stream(idx.device).cuda_stream!=handle:raise RuntimeError('Current stream changed')
                self.pending.pop();self.stages.append(stage)
                return bad
            run_stages(evaluate)
            return None
        except BaseException:
            self.failed=True
            raise # No fallback or retry after an uncertain enqueue.
