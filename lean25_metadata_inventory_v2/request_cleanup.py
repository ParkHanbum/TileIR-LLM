"""Commit only after numerical cleanup and helper restoration, even on failure."""
import json,hashlib
def finish_observation(result,save,cleanup,helpers,ledger,output):
 errors=[]
 try:cleanup()
 except BaseException as e:errors.append(e)
 try:
  if helpers is not None:helpers.__exit__(None,None,None)
 except BaseException as e:errors.append(e)
 try:
  if ledger is not None:
   data=ledger.snapshot();text=json.dumps(data,indent=2)+'\n'
   assert len(text.encode())<=64*1024*1024
   path=output/'METADATA.json';path.write_text(text)
   if 'metadata_inventory' in result:
    result['metadata_inventory'].update(metadata_path=str(path),metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   result['request_helpers_restored']=data['helpers_restored']
   assert data['helpers_restored'] and not data['failed']
 except BaseException as e:errors.append(e)
 if errors:
  result['complete']=False;result['observation_cleanup_errors']=[repr(e)for e in errors]
 save()
 if errors:raise RuntimeError('Observation cleanup HOLD: '+repr(errors))
