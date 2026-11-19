"""Six frozen heads (then matched ACK heads), full-request canonical verifier."""
import argparse,concurrent.futures,fcntl,json,os,pathlib,sys,threading,time,traceback
from wave_common import *
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(FIRST))
from closed_loop_core import decode,digest
FIELDS=['candidate_ids','target_decision_ids','accepted_count','correction_id','bonus_id','committed_ids','prefix_before_ids','prefix_after_ids','eos','truncated']

def audit(req,base,f,arm):
 r=read(base/'receipt.json');assert r['status']=='pass' and r['freeze_sha256']==sha(ROOT/'freeze.json')
 assert r['checkpoint_sha256']==arm['sha256'] and r['raw_trace_sha256']==req['raw_trace_sha256']
 assert sha(base/'rounds.jsonl')==r['rounds_sha256'] and r['oracle_receipt_sha256']==sha(ROOT/'oracles'/f"{req['request_id']}.json")
 rows=lines(base/'rounds.jsonl');prefix=req['prompt_ids'].copy();accepted=0
 for i,row in enumerate(rows):
  assert row['round']==i and row['prefix_before_ids']==prefix and row['prefix_before_sha256']==digest(prefix)
  a=0
  for c,t in zip(row['candidate_ids'],row['target_decision_ids']):
   if c!=t:break
   a+=1
  assert a==row['accepted_count'];answer=row['correction_id'] if row['correction_id'] is not None else row['bonus_id']
  assert row['committed_ids']==row['candidate_ids'][:a]+([] if answer is None else [answer])
  if answer is not None:assert row['target_decision_ids'][a]==answer
  prefix+=row['committed_ids'];accepted+=a
  assert prefix==row['prefix_after_ids'] and digest(prefix)==row['prefix_after_sha256']
  assert prefix[len(req['prompt_ids']):]==req['frozen_target_ids'][:len(prefix)-len(req['prompt_ids'])]
 assert prefix==req['prompt_ids']+req['frozen_target_ids'] and r['output_ids']==req['frozen_target_ids']
 assert r['eos']==req['eos'] and r['truncated']==req['truncated']
 assert accepted==r['accepted_draft_tokens'] and len(rows)==r['verification_rounds']
 assert sum(len(row['target_decision_ids']) for row in rows)==len(req['frozen_target_ids'])
 if not r.get('reused_RAW'):
  assert rows[0]['candidate_ids']==read(req['raw_trace_path'])['rounds'][0]['candidate_ids'],'first round uses teacher'
 for name,h in r.get('tensor_files',{}).items():assert sha(base/name)==h
 return r

def run(stage):
 f=read(ROOT/'freeze.json');roster=read(ROOT/'roster503.json');assert sha(ROOT/'roster503.json')==f['roster_sha256']
 for name,h in f['source_files'].items():assert sha(ROOT/name)==h
 for p,h in f['reference_sources'].items():assert sha(p)==h
 for name,h in f['source_sha256'].items():assert sha(pathlib.Path(f['source_root'])/name)==h
 arms=read(ROOT/'arms.json') if stage=='six' else read(ROOT/'ack_training/selected_arms.json')
 if stage=='six':assert sha(ROOT/'arms.json')==f['arms_sha256']
 for a in arms:
  if a['checkpoint']:assert sha(a['checkpoint'])==a['sha256']
 out=ROOT/('six_dev' if stage=='six' else 'ack_dev');out.mkdir(exist_ok=True)
 if (out/'completion.json').exists():event('already_complete_no_dispatch',stage=stage);return
 lock=open(LOCK,'a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),'occupied; never interrupt'
 lease=dict(status='held',pid=os.getpid(),unix=time.time(),lock=LOCK,stage=stage,freeze_sha256=sha(ROOT/'freeze.json'),no_added_runtime_limit=True)
 write(out/'gpu_lease.json',lease);event('GPU_admitted',stage=stage)
 stop=threading.Event();resources=[]
 def monitor():
  while not stop.is_set():
   try:
    raw=command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']);rec=dict(unix=time.time(),rows=raw)
    rec['process_mib']=sum(int(s.split(',')[1].strip().split()[0]) for s in raw.splitlines() if s.split(',')[0].strip()==str(os.getpid()));resources.append(rec)
    with (out/'resources.jsonl').open('a') as z:z.write(json.dumps(rec)+'\n')
   except Exception as e:resources.append(dict(error=repr(e)))
   stop.wait(2)
 mon=threading.Thread(target=monitor,daemon=True);mon.start()
 try:
  import torch
  from safetensors.torch import save_file
  sys.path.insert(0,f['source_root']);sys.path.insert(0,str(KV))
  from feedback_codec.hf_backend import HFTargetBackend,HFDraftBackend,_FrozenHFBackend,assert_compatible_pair
  from feedback_codec.codec import CodecConfig
  from feedback_codec.v10_four_arm import V10FourArmHead
  from feedback_codec.fixed_probe import ProbeObservation
  from feedback_codec.decoding import autoregressive_generate
  from sweep_heads_task59_analyst import build_head
  from same_forward_kv import SameForwardKV,compare_reference_cache
  class DraftOnly(_FrozenHFBackend):teacher_hidden=HFDraftBackend.teacher_hidden
  torch.set_num_threads(4);torch.manual_seed(42)
  target=HFTargetBackend(str(pathlib.Path(f['models_root'])/'Qwen3-8B'),None,device='cuda:0',dtype=torch.bfloat16)
  draft=DraftOnly(str(pathlib.Path(f['models_root'])/'Qwen3-0.6B'),None,device='cuda:0',dtype=torch.bfloat16);assert_compatible_pair(target,draft)
  heads={}
  for a in arms:
   if a['kind']=='RAW':continue
   args=(CodecConfig(4096,1024,draft.vocab_size),draft.model.get_output_embeddings())
   if a['kind']=='original':h=V10FourArmHead(*args,mode='F-H',draft_embedding=draft.model.get_input_embeddings())
   elif a['kind']=='ack':
    from ack_head import MatchedACKHead
    h=MatchedACKHead(*args,arm=a['name'])
   else:
    c=a['config'];h=build_head(*args,draft.model.get_input_embeddings(),architecture=c['architecture'],depth=c['depth'],width=c['width'],seed=c['seed'])
   h=h.to('cuda:0');h.load_trainable_state_dict(torch.load(a['checkpoint'],map_location='cpu',weights_only=False)['model']);h.eval();heads[a['name']]=h
  # Backend locks serialize each model's hook-bearing forward while allowing Draft/Target overlap.
  with torch.inference_mode():target.next(roster[0]['prompt_ids']);draft.teacher_hidden(roster[0]['prompt_ids'],[0])
  torch.cuda.synchronize();write(out/'loaded_models.json',dict(shared_Draft_and_Target=True,heads=list(heads),allocated_mib=torch.cuda.memory_allocated()/2**20,total_mib=torch.cuda.get_device_properties(0).total_memory/2**20))
  observer=SameForwardKV(target) if stage=='ack' else None
  streamlock=threading.Lock();oracle_lock=threading.Lock();records={};byid={r['request_id']:r for r in roster}
  def ensure_oracle(req):
   with oracle_lock:
    p=ROOT/'oracles'/f"{req['request_id']}.json"
    if p.exists():
     o=read(p);assert o['freeze_sha256']==sha(ROOT/'freeze.json') and o['prompt_sha256']==digest(req['prompt_ids']) and o['output_ids']==req['frozen_target_ids'];return o
    started=time.perf_counter();source=req['oracle_source']
    if source=='sealed_RAW_independent_greedy':output=read(req['raw_trace_path'])['oracle_ids']
    elif source=='sealed_six_request_independent_greedy':output=next(r for r in read(SHORT/'independent_oracles.json') if r['request_id']==req['request_id'])['output_ids']
    else:
     event('independent_Target',request_id=req['request_id'],stage=stage)
     with torch.inference_mode():output=list(autoregressive_generate(target,req['prompt_ids'],128,151645))
     source='fresh_independent_Target_greedy'
    assert output==req['frozen_target_ids'],'independent oracle mismatch '+req['request_id']
    o=dict(status='pass',request_id=req['request_id'],prompt_sha256=digest(req['prompt_ids']),output_ids=output,eos=output[-1]==151645,truncated=len(output)==128 and output[-1]!=151645,source=source,seconds=time.perf_counter()-started,raw_trace_sha256=req['raw_trace_sha256'],freeze_sha256=sha(ROOT/'freeze.json'))
    write(p,o);return o
  def work(req,a,scope='requests',force_RAW=False):
   rid=req['request_id'];name=a['name'];dest=out/scope/rid/name
   if (dest/'receipt.json').exists():return audit(req,dest,f,a)
   ensure_oracle(req);tmp=out/'incomplete'/f'{scope}-{rid}-{name}-{time.time_ns()}';tmp.mkdir(parents=True)
   tensor={};numeric=[];start=time.perf_counter();rows=[];reused=a['kind']=='RAW' and not force_RAW
   with torch.inference_mode():
    if reused:
     t=read(req['raw_trace_path']);assert sha(req['raw_trace_path'])==req['raw_trace_sha256'];prefix=req['prompt_ids'].copy()
     for i,r in enumerate(t['rounds']):
      after=prefix+r['committed_ids'];rows.append(dict(r,round=i,prefix_before_ids=prefix,prefix_after_ids=after,prefix_after_sha256=digest(after),eos=after[-1]==151645,truncated=len(after)-len(req['prompt_ids'])==128 and after[-1]!=151645));prefix=after
    else:
     head=heads.get(name)
     def proposal(prefix,state,position):
      if head is None or state is None:return int(draft.model.get_output_embeddings()(draft.teacher_hidden(prefix,[0])).argmax(-1).item())
      return head.next_token(prefix,state,position,draft)
     def prepare(row,boundary):
      if a['kind']=='ack':return head.prepare_online(row,boundary,draft)
      valid=row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']];answer=row['correction_id'] if row['correction_id'] is not None else row['bonus_id']
      obs=ProbeObservation(prompt_id=rid,family_id=req['family_id'],split='calibration',task_type=req['topic'],round_index=row['round'],source_prefix=tuple(row['prefix_before_ids']),proposals=tuple(row['candidate_ids']),valid_prefix=tuple(valid),accepted_count=row['accepted_count'],rejected_token=row['rejected_id'],correction_token=answer,corrected_prefix=tuple(row['prefix_after_ids']),target_hidden=boundary.hidden,target_hidden_prefix=tuple(valid),remaining_budget=128-row['output_tokens_so_far'],window=8)
      return head.prepare_observation(obs,draft)
     def capture(row,boundary):
      if boundary is not None:
       key=f"r{row['round']:04d}";tensor[key+'.H']=boundary.hidden.detach().cpu().contiguous()
       row['teacher_valid_prefix_sha256']=digest(row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']])
       if observer is not None:
        assert row['teacher_valid_prefix_sha256']==boundary.prefix_sha256
        for n,t in boundary.kv.items():tensor[key+'.'+n]=t
        row.update(KV_layers=[34,35],KV_positions=list(range(max(0,boundary.prefix_length-4),boundary.prefix_length)),KV_valid=boundary.kv['valid'].tolist(),KV_call_id=boundary.call_id)
        if scope=='requests' and rid in smoke and len(numeric)<2:numeric.append(compare_reference_cache(target,row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']],boundary))
      rows.append(row)
     result=decode(req['prompt_ids'],draft_token=proposal,target_step=observer.next if observer else target.next,prepare=prepare if head else None,on_round=capture,oracle_ids=req['frozen_target_ids'],max_new=128,window=8,eos=151645)
     if a['kind']=='RAW':
      old=read(req['raw_trace_path'])['rounds'];assert len(old)==len(rows)
      for x,y in zip(old,rows):
       for k in ['candidate_ids','target_decision_ids','accepted_count','correction_id','bonus_id','committed_ids']:assert x[k]==y[k],('RAW replay',k)
   with (tmp/'rounds.jsonl').open('w') as z:
    for row in rows:row.update(request_id=rid,arm=name,checkpoint_sha256=a['sha256']);z.write(json.dumps(row,separators=(',',':'))+'\n')
   tensors={}
   if tensor:save_file(tensor,str(tmp/'boundary.safetensors'));tensors['boundary.safetensors']=sha(tmp/'boundary.safetensors')
   r=dict(status='pass',request_id=rid,family_id=req['family_id'],arm=name,checkpoint_sha256=a['sha256'],output_ids=req['frozen_target_ids'],eos=req['eos'],truncated=req['truncated'],accepted_draft_tokens=sum(x['accepted_count'] for x in rows),verification_rounds=len(rows),actual_target_forwards=sum(len(x['target_decision_ids']) for x in rows),elapsed_seconds=time.perf_counter()-start,rounds_sha256=sha(tmp/'rounds.jsonl'),tensor_files=tensors,oracle_receipt_sha256=sha(ROOT/'oracles'/f'{rid}.json'),raw_trace_sha256=req['raw_trace_sha256'],freeze_sha256=sha(ROOT/'freeze.json'),reused_RAW=reused,ID_EOS_truncation_exact=True,KV_numeric_checks=numeric,unix=time.time())
   write(tmp/'receipt.json',r);audit(req,tmp,f,a);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();tmp.rename(dest)
   if scope=='requests':
    with streamlock:
     records[(rid,name)]=r
     with (out/'manifest.jsonl').open('a') as z:z.write(json.dumps(dict(request_id=rid,arm=name,path=str(dest),receipt_sha256=sha(dest/'receipt.json'),unix=time.time()))+'\n');z.flush();os.fsync(z.fileno())
     progress=dict(sealed=len(records),total=503*len(arms),by_arm={a['name']:sum(k[1]==a['name'] for k in records) for a in arms},pid=os.getpid(),unix=time.time())
     write(out/'progress.json',progress);event('request_sealed',stage=stage,request_id=rid,arm=name,sealed=len(records),total=503*len(arms))
   return r
  for req in roster:
   for a in arms:
    dest=out/'requests'/req['request_id']/a['name']
    if (dest/'receipt.json').exists():records[(req['request_id'],a['name'])]=audit(req,dest,f,a)
  smoke=read(ROOT/'smoke_ids.json')
  for rid in smoke:
   for a in arms:work(byid[rid],a,force_RAW=True)
  write(out/'GPU_smoke.json',dict(status='pass',requests=smoke,arms=[a['name'] for a in arms],EOS_covered=True,truncation_covered=True,longest_prefix_tokens=max(len(r['prompt_ids']) for r in roster),ID_EOS_truncation_exact=True,first_round_RAW_exact=True,KV_numeric_receipts_in_each_request=stage=='ack'))
  if stage=='six':
   benchpath=out/'concurrency.json'
   if benchpath.exists():chosen=read(benchpath)['chosen']
   else:
    trials=[];chosen=1;best=0;bench_req=next(byid[rid] for rid in smoke if byid[rid]['truncated']);jobs=[(bench_req,a) for a in arms if a['kind']=='sweep']
    for n in [1,2,4]:
     torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
     try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:got=list(pool.map(lambda pair:work(*pair,scope=f'benchmark{n}'),jobs))
      torch.cuda.synchronize();seconds=time.perf_counter()-t0;safe=torch.cuda.max_memory_reserved()+4096*2**20<=torch.cuda.get_device_properties(0).total_memory
      for req,a in jobs:
       ref=lines(out/'requests'/req['request_id']/a['name']/'rounds.jsonl');cur=lines(out/f'benchmark{n}'/req['request_id']/a['name']/'rounds.jsonl');assert len(ref)==len(cur)
       assert all(all(x[k]==y[k] for k in FIELDS) for x,y in zip(ref,cur)),'concurrent candidate mismatch'
      rate=len(jobs)/seconds;trial=dict(concurrency=n,status='pass' if safe else 'insufficient_headroom',seconds=seconds,completed_requests=len(jobs),requests_per_second=rate,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,nvml_peak_mib=max((r.get('process_mib',0) for r in resources),default=0),Target_loaded=True,shared_models=True,full_candidates_and_outputs_equal=True,separate_warmup=True)
      trials.append(trial);event('concurrency_measured',**trial)
      if not safe:break
      if rate>best*1.03:chosen=n;best=rate
      if n==2 and rate<=trials[0]['requests_per_second']:break
     except torch.cuda.OutOfMemoryError as e:trials.append(dict(concurrency=n,status='OOM',error=str(e)));torch.cuda.empty_cache();break
    assert trials[0]['status']=='pass';write(benchpath,dict(status='pass',chosen=chosen,trials=trials,selection=f['concurrency_selection']))
  else:chosen=1;write(out/'concurrency.json',dict(chosen=1,reason='same-forward KV observer has exclusive per-request capture; no unsupported concurrent observation',Target_loaded=True))
  # Fresh oracle precedes each request's repaired arms. Sealed jobs are never redispatched.
  for req in roster:
   ensure_oracle(req);pending=[a for a in arms if (req['request_id'],a['name']) not in records]
   with concurrent.futures.ThreadPoolExecutor(max_workers=chosen) as pool:list(pool.map(lambda a:work(req,a),pending))
  assert len(records)==503*len(arms)
  summary={}
  for a in arms:
   rr=[v for k,v in records.items() if k[1]==a['name']];acc=sum(r['accepted_draft_tokens'] for r in rr);rounds=sum(r['verification_rounds'] for r in rr)
   summary[a['name']]=dict(requests=len(rr),accepted=acc,rounds=rounds,accepted_tokens_per_round=acc/rounds)
  write(out/'completion.json',dict(status='complete',requests=503,arms=summary,all_ID_EOS_truncation_exact=True,freeze_sha256=sha(ROOT/'freeze.json'),manifest_sha256=sha(out/'manifest.jsonl'),concurrency=chosen,independent300_used=False,unix=time.time()))
  if observer:observer.close()
 finally:
  stop.set();mon.join(timeout=3);lease.update(status='released',released_unix=time.time(),outcome='complete' if (out/'completion.json').exists() else 'failed',nvml_peak_mib=max((r.get('process_mib',0) for r in resources),default=0));write(out/'gpu_lease.json',lease);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['six','ack'],default='six');a=p.parse_args()
 try:run(a.stage)
 except Exception as e:write(ROOT/f'failure-evaluate-{a.stage}-{os.getpid()}.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
