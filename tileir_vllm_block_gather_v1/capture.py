"""Opt-in one-call adapter for an existing approved eager capture runner.

Not installed automatically. No engine construction, launch, lock or GPU query.
Caller supplies the original bound method and an unused per-call directory.
Call only outside graph capture, with no concurrent source/out mutation.
"""
def capture_call(owner, original, indices, padded, directory, *, phase_label):
    from pathlib import Path
    import json,hashlib,torch
    root=Path(directory); root.mkdir(parents=True,exist_ok=False)
    manifest={'status':'started','phase_label':phase_label,'num_reqs_padded':padded,'artifacts':{}}
    def save(name,t):
        s=t.untyped_storage()
        meta={'shape':list(t.shape),'stride':list(t.stride()),'storage_offset':t.storage_offset(),'dtype':str(t.dtype),'device':str(t.device),'data_ptr':t.data_ptr(),'storage_ptr':s.data_ptr(),'storage_bytes':s.nbytes()}
        path=root/(name+'.pt');torch.save(t.detach().to('cpu').clone(),path)
        manifest['artifacts'][name]={'metadata':meta,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def progress(): (root/'PROGRESS.json').write_text(json.dumps(manifest,indent=2))
    progress()
    try:
        save('indices',indices);save('counts',owner.num_blocks.gpu)
        for g,t in enumerate(owner.block_tables):save('src_'+str(g),t.gpu)
        for g,t in enumerate(owner.input_block_tables):save('before_'+str(g),t)
        progress()
        # Exactly the original public call, no substitution or extra repeat.
        result=original(indices,num_reqs_padded=padded)
        for g,t in enumerate(owner.input_block_tables):save('after_'+str(g),t)
        manifest['status']='complete';progress()
        return result
    except BaseException as e:
        manifest['status']='failed';manifest['error']=repr(e);progress();raise
