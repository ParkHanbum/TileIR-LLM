import json,os
class DurableLedger:
    def __init__(self,path):
        self.path=path
        with open(path,'x') as f:f.flush();os.fsync(f.fileno())
    def __call__(self,event):
        with open(self.path,'a') as f:
            f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
