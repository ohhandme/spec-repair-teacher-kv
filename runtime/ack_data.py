"""Identical legal row mapping. Labels and loss mask never enter signal adapters."""
from functools import lru_cache
import json,pathlib
from wave_common import remap
import torch
from safetensors import safe_open
from cache_adapter import FrozenWindows,attach_embeddings
class MatchedWindows(FrozenWindows):
 def __init__(self,records,cache_root,split,common_path,packed_root=None):
  super().__init__(records,cache_root,split)
  rows=[json.loads(s) for s in pathlib.Path(common_path).read_text().splitlines() if s];lookup={(r['request_id'],r['t_round']):r for r in rows}
  self.common=[]
  for ri,wi in self.windows:
   r=self.records[ri];w=r['windows'][wi];c=lookup.pop((r['request_id'],w['round']))
   assert c['target_ids']==w['labels'] and c['observation_key']==w['observation_key']
   assert c['accepted_count']==w['accepted'] and c['answer_id']==w['correction'] and c['rejected_id']==w['rejected']
   assert c['target_valid_mask']==[True]*w['length']+[False]*(8-w['length']);self.common.append(c)
  assert not lookup
  self.packed=None
  if packed_root is not None:
   import numpy as np
   self.packed={k:np.load(pathlib.Path(packed_root)/f'{split}_{k}.npy',mmap_mode='r') for k in ['K','V','mask']}
 @lru_cache(maxsize=48)
 def features(self,path):
  with safe_open(remap(path),framework='pt',device='cpu') as f:
   return {k:f.get_tensor(k) for k in f.keys() if k.endswith(('.K','.V','.valid','.hidden'))}
 def batch(self,indices):
  x=super().batch(indices);ks=[];vs=[];ms=[]
  if self.packed is not None:
   x.update(H=x.pop('target'),K=torch.from_numpy(self.packed['K'][indices].copy()).view(torch.bfloat16),V=torch.from_numpy(self.packed['V'][indices].copy()).view(torch.bfloat16),boundary_valid=torch.from_numpy(self.packed['mask'][indices].copy()));return x
  for j,idx in enumerate(indices):
   c=self.common[int(idx)];z=self.features(c['features_path']);p=c['tensor_prefix'];h=z[p+'.hidden']
   assert torch.equal(h,x['target'][j]);ks.append(z[p+'.K']);vs.append(z[p+'.V']);ms.append(z[p+'.valid'])
  x.update(H=x.pop('target'),K=torch.stack(ks),V=torch.stack(vs),boundary_valid=torch.stack(ms));return x
def select_view(x,arm):
 keys=['local','hidden','accepted','rejected_embedding','correction_embedding','boundary_valid']
 if arm=='ACK_H':keys+=['H']
 if arm=='ACK_KV':keys+=['K','V']
 return {k:x[k] for k in keys}
