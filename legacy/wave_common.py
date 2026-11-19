import hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
PARENT=ROOT.parent
FIRST=PARENT/'first4-1a34c7c8'
KV=PARENT/'kv-37f19abe'
DATA=pathlib.Path('/data/a100-ultrachat5000-vanilla-20261111-7858dc6a')
TRAIN=pathlib.Path('/data/a100-repair-raw5000-20261113-v1')
SHORT=pathlib.Path('/data/task59-a100-closedloop-20261114-v1')
LOCK='/data/a100-first-rejection-20261111-9b21f68f/a1_gpu.lock'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def lines(p):return [json.loads(s) for s in pathlib.Path(p).read_text().splitlines() if s]
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def command(args):return subprocess.check_output(args,text=True).strip()
def event(kind,**kw):
 x=dict(kw);x.update(kind=kind,pid=os.getpid(),unix=time.time());write(ROOT/'status.json',x)
 with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(x),flush=True)
