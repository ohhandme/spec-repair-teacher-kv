"""Standalone path adapter, with no predecessor queue or synthetic receipts."""
import hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
_ASSETS=json.loads((ROOT/'assets_runtime.json').read_text())
PARENT=FIRST=KV=ROOT
TRAIN=pathlib.Path(_ASSETS['train_root'])
DATA=pathlib.Path(_ASSETS['data_root'])
SHORT=pathlib.Path(_ASSETS.get('short_root',ROOT/'optional_short_oracles'))
LOCK=_ASSETS['gpu_lock']
def remap(path):
 value=str(path).replace('\\','/')
 for old,new in sorted(_ASSETS.get('path_map',{}).items(),key=lambda x:-len(x[0])):
  old=old.rstrip('/')
  if value==old or value.startswith(old+'/'):return str(pathlib.Path(new)/value[len(old):].lstrip('/'))
 return str(path)
def sha(path):
 with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):
 path=pathlib.Path(path);value=json.loads(path.read_bytes())
 if path.resolve()==(ROOT/'ack_training/selected_arms.json').resolve():
  return [dict(name='RAW',kind='RAW',checkpoint=None,sha256=None)]+value
 return value
def lines(path):return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s]
def write(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def command(args):return subprocess.check_output(args,text=True).strip()
def event(kind,**kwargs):
 value=dict(kwargs,kind=kind,pid=os.getpid(),unix=time.time());write(ROOT/'status.json',value)
 with (ROOT/'events.jsonl').open('a',encoding='utf-8') as stream:
  stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())
 print(json.dumps(value),flush=True)


def validate_cursor(state, arm, freeze_hash, registration_hash, rows, batch_size, epochs):
    if (state.get('arm'), state.get('freeze_sha256'), state.get('registration_sha256')) != (arm, freeze_hash, registration_hash):
        raise ValueError('checkpoint arm, freeze or registration differs')
    epoch, position, updates = (state[k] for k in ('epoch', 'position', 'updates'))
    if any(type(v) is not int for v in (epoch, position, updates)):
        raise ValueError('invalid training cursor type')
    if not (1 <= epoch <= epochs + 1 and 0 <= position <= rows):
        raise ValueError('invalid training cursor')
    if (epoch == epochs + 1 and position != 0) or (position != rows and position % batch_size):
        raise ValueError('invalid batch boundary')
    expected = (epoch - 1) * ((rows + batch_size - 1) // batch_size) + (position + batch_size - 1) // batch_size
    if updates != expected or len(state['history']) != epoch - 1:
        raise ValueError('updates or epoch history differs from training cursor')


def publish_manifest(out, roster, arms, records):
    # All registered jobs have passed audit, including receipts sealed before a crash.
    path = out / 'manifest.jsonl'
    pending = out / 'manifest.pending'
    with pending.open('w', encoding='utf-8', newline='\n') as stream:
        for req in roster:
            for arm in arms:
                rid, name = req['request_id'], arm['name']
                folder = out / 'requests' / rid / name
                receipt = records[(rid, name)]
                entry = dict(request_id=rid, arm=name, path=str(folder),
                             receipt_sha256=sha(folder / 'receipt.json'), unix=receipt['unix'])
                stream.write(json.dumps(entry) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
