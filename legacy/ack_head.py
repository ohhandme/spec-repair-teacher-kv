"""Matched active ACK/H/KV graph; teacher data is previous legal boundary only."""
import torch
from torch import nn
from feedback_codec.codec import MatchedP1Codec
from teacher_kv_task59_analyst import KVControl

class MatchedACKHead(nn.Module):
 def __init__(self,config,output_head,*,arm):
  super().__init__();assert arm in ('ACK','ACK_H','ACK_KV');self.arm=arm;self.config=config
  self.core=MatchedP1Codec(config,output_head,mode='protocol_only');self.signal=KVControl()
  nn.init.zeros_(self.core.repair.U.weight)
 def signal_inputs(self,x):
  local=x['local'];batch=local.shape[0]
  k=local[:,None,None,:].expand(batch,2,4,1024);v=k
  if self.arm=='ACK_H':
   h=x['H'].reshape(batch,4,1024);k=k+h[:,:2,None,:];v=v+h[:,2:,None,:]
  elif self.arm=='ACK_KV':k=k+x['K'].flatten(-2);v=v+x['V'].flatten(-2)
  return k.reshape(batch,2,4,8,128),v.reshape(batch,2,4,8,128),x['boundary_valid']
 def control(self,x,generator=None):
  args=(x['local'],x['rejected_embedding'],x['correction_embedding'],x['accepted'])
  if generator is None:codes=self.core.local_encode(*args)
  else:
   scores=self.core._local_scores(*args);noise=-torch.empty_like(scores).exponential_(generator=generator).log()
   soft=(scores+noise).softmax(-1);hard=torch.zeros_like(soft).scatter_(-1,soft.argmax(-1,keepdim=True),1.);codes=hard-soft.detach()+soft
  n=x['local'].shape[0];control=self.core.repair._lookup(codes[:,0],n)+self.core.repair._lookup(codes[:,1],n)
  return control+self.signal(*self.signal_inputs(x))
 def forward_batch(self,x,generator=None):
  return self.core.repair.repair_from_control(x['hidden'],self.control(x,generator),torch.arange(1,9,device=x['local'].device))
 def prepare_online(self,row,boundary,draft):
  prefix=row['prefix_after_ids'];valid=row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']]
  answer=row['correction_id'] if row['correction_id'] is not None else row['bonus_id'];assert prefix==valid+[answer]
  assert boundary.prefix_length==len(valid)
  from same_forward_kv import digest
  assert boundary.prefix_sha256==digest(valid) and boundary.token_id==answer
  local=draft.teacher_hidden(prefix,[0]);table=draft.model.get_input_embeddings();device=local.device
  x=dict(local=local,accepted=torch.tensor([row['accepted_count']],device=device),rejected_embedding=table(torch.tensor([0 if row['rejected_id'] is None else row['rejected_id']],device=device)),correction_embedding=table(torch.tensor([answer],device=device)),boundary_valid=boundary.kv['valid'][None].to(device))
  if self.arm=='ACK_H':x['H']=boundary.hidden[None].to(device)
  if self.arm=='ACK_KV':x.update(K=boundary.kv['K'][None].to(device),V=boundary.kv['V'][None].to(device))
  return dict(prefix=tuple(prefix),control=self.control(x))
 def next_token(self,prefix,state,position,draft):
  assert tuple(prefix[:len(state['prefix'])])==state['prefix']
  hidden=draft.teacher_hidden(prefix,[0]);logits=self.core.repair.repair_from_control(hidden,state['control'],position)
  assert torch.isfinite(logits).all();return int(logits.argmax(-1).item())
 def trainable_state_dict(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
 def load_trainable_state_dict(self,state):
  current={n:p for n,p in self.named_parameters() if p.requires_grad};assert set(state)==set(current)
  with torch.no_grad():
   for n,p in current.items():assert p.shape==state[n].shape and torch.isfinite(state[n]).all();p.copy_(state[n].to(p))
