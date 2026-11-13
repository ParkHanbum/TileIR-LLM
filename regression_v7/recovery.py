"""Separate, read-mostly post-run recovery proof. Never retries work or signals processes."""
import argparse, fcntl, importlib.util, json, os, time
from pathlib import Path
from . import common
from .whole import group_members
ADMISSION=Path('/home/n/work/TileIR/cublas-gemm-analysis-20261111/runtime-cubin-probe/launch_probe.py')
ADMISSION_SHA='bc956326f1c9f80c1ead01ab071ab18a48ae6a2b4e8c5ee26171789a4b931c0d'

def validate_receipts(outer,launch,result,result_sha):
    assert outer=='EXIT=0\n','Actual outer exit receipt is missing or nonzero'
    assert type(launch.get('returncode'))is int and launch['returncode']==0
    assert launch.get('qualification_complete')is True and launch.get('result_gate',{}).get('passed')is True
    assert launch.get('child_absent')is True and launch.get('remaining_process_group')==[]
    assert launch.get('lock_reacquired')is True and launch.get('after')
    assert 'error'not in launch and not launch.get('timeout') and launch.get('finished')
    assert launch.get('result_sha256')==result_sha and result.get('complete')is True
    assert type(launch.get('pid'))is int and launch['pid']>1
    return launch['pid']

def recover(root,head):
    common.verify(head,external=True)
    root=common.external_root(root)
    assert not(root/'RECOVERY.json').exists(),'Recovery receipt already exists'
    lp=root/'regression01-launch.json';rp=root/'regression01/RESULT.json';ep=root/'OUTER-EXIT.txt'
    launch=json.loads(lp.read_text());result=json.loads(rp.read_text())
    pid=validate_receipts(ep.read_text(),launch,result,common.sha(rp))
    assert not Path('/proc',str(pid)).exists() and not group_members(pid),'Owned process still live'
    assert common.sha(ADMISSION)==ADMISSION_SHA
    spec=importlib.util.spec_from_file_location('regression_admission_only',ADMISSION)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with open(helper.LOCK,'r')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        after=helper.admit()
    common.verify(head)
    record=dict(complete=True,head=head,baseline=common.BASE,actual_outer_exit=0,actual_child_exit=0,
        child_absent=True,remaining_process_group=[],lock_available=True,post=after,
        launch_sha256=common.sha(lp),result_sha256=common.sha(rp),outer_exit_sha256=common.sha(ep),
        timestamp=time.time(),automatic_retry=False,signals_sent=0,scope='Independent after actual observed outer completion; no new model execution')
    with(root/'RECOVERY.json').open('x')as f:json.dump(record,f,indent=2);f.write('\n')
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head',required=True);p.add_argument('--artifact-root',required=True,type=Path)
    a=p.parse_args();print(json.dumps(recover(a.artifact_root,a.head),indent=2))
if __name__=='__main__':main()
