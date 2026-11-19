"""Actual head/optimizer/data IO and tiny Qwen3 KV tests; CPU evidence only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as entry


def main():
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(2); torch.manual_seed(42)
    checks = []
    with tempfile.TemporaryDirectory(prefix='teacher-kv-cpu-') as tmp:
        root = Path(tmp); out = root/'run'; train = root/'train'; train.mkdir()
        assets = dict(models_root=str(root/'models'), train_root=str(train), data_root=str(root/'raw'),
                      gpu_lock=str(root/'gpu.lock'), path_map={'/data': str(root/'data')})
        asset_file = root/'assets.json'; asset_file.write_text(json.dumps(assets))
        entry.configure(asset_file, out, list(entry.ARMS), 'equal')
        entry.configure(asset_file, out, list(entry.ARMS), 'equal')
        try:
            entry.configure(asset_file, out, list(entry.ARMS), 'front')
        except ValueError:
            checks.append('immutable configured loss rejects overwrite')
        else:
            raise AssertionError('different loss overwrote the same run')
        assert entry.remap('/data2/a', {'/data': '/mount'}) == '/data2/a'
        assert entry.remap('/data/raw/a', {'/data': '/one', '/data/raw': '/two'}).replace('\\','/') == '/two/a'
        sys.path[:0] = [str(out), str(HERE/'vendor')]
        from ack_head import MatchedACKHead
        from ack_data import MatchedWindows, select_view
        from cache_adapter import attach_embeddings
        from sweep_heads_task59_analyst import window_ce
        from teacher_kv_task59_analyst import read_boundary, _prefix_sha
        from closed_loop_core import cpu_tests
        from feedback_codec.codec import CodecConfig
        table = torch.nn.Embedding(97, 1024).to(torch.bfloat16).requires_grad_(False)
        output = torch.nn.Linear(1024, 97, bias=False).to(torch.bfloat16).requires_grad_(False)
        cfg = CodecConfig(4096, 1024, 97)
        heads = {}
        for arm in entry.ARMS:
            torch.manual_seed(42); heads[arm] = MatchedACKHead(cfg, output, arm=arm)
        initial = heads['ACK'].trainable_state_dict()
        assert all(all(torch.equal(h.trainable_state_dict()[k], v) for k,v in initial.items()) for h in heads.values())
        checks.append('all three arms have identical initialization and active matched capacity')
        cache = train/'cache'; cache.mkdir(); records = []; common = []
        for j, length in enumerate((8, 3)):
            rid = f'fixture{j}'; hidden = torch.randn(length,1024).to(torch.bfloat16)
            teacher = torch.randn(4096).to(torch.bfloat16); valid = torch.tensor([True,True,False,False])
            k = torch.randn(2,4,8,128).to(torch.bfloat16); v = torch.randn_like(k)
            k[:,~valid] = 0; v[:,~valid] = 0
            path = root/'data'/f'{rid}.safetensors'; path.parent.mkdir(exist_ok=True)
            save_file({'r.hidden': teacher,'r.K':k,'r.V':v,'r.valid':valid},str(path))
            save_file({'draft_sequence':hidden,'target':teacher[None]},str(cache/f'{rid}.safetensors'))
            labels = list(range(10,10+length))
            w = dict(round=0, labels=labels, observation_key=f'key{j}', accepted=1,
                     correction=4,rejected=3,offset=0,length=length)
            records.append(dict(request_id=rid,split='train',windows=[w]))
            common.append(dict(request_id=rid,t_round=0,target_ids=labels,observation_key=f'key{j}',
                               accepted_count=1,answer_id=4,rejected_id=3,
                               target_valid_mask=[True]*length+[False]*(8-length),
                               features_path=f'/data/{rid}.safetensors',tensor_prefix='r'))
        common_file = root/'common.jsonl';common_file.write_text(''.join(json.dumps(r)+'\n' for r in common))
        ds = MatchedWindows(records, cache, 'train', common_file)
        x = attach_embeddings(ds.batch([0,1]),table,'cpu')
        checks.append('real safetensors common-row mapping and relocated feature paths')
        sizes = []
        for arm, h in heads.items():
            view = select_view(x,arm); y = h.forward_batch(view)
            assert torch.equal(y,output(x['hidden']))
            with torch.no_grad():h.core.repair.U.weight.normal_(std=.01)
            ref = h.forward_batch(view); changed = copy.deepcopy(x)
            changed['labels'].fill_(96);changed['valid'].fill_(False)
            assert torch.equal(ref,h.forward_batch(select_view(changed,arm)))
            if arm == 'ACK':
                changed=copy.deepcopy(x);changed['K']*=100;changed['V']*=100;changed['H']*=100
                assert torch.equal(ref,h.forward_batch(select_view(changed,arm)))
            elif arm == 'ACK_KV':
                changed=copy.deepcopy(x);changed['K'][:,:,2:]=1e4;changed['V'][:,:,2:]=-1e4
                assert torch.equal(ref,h.forward_batch(select_view(changed,arm)))
                changed=copy.deepcopy(x);changed['K'][:,0,0]+=1
                assert not torch.equal(ref,h.forward_batch(select_view(changed,arm)))
            else:
                changed=copy.deepcopy(x);changed['H']+=1
                assert not torch.equal(ref,h.forward_batch(select_view(changed,arm)))
            h.zero_grad(set_to_none=True)
            loss=window_ce(h.forward_batch(view,torch.Generator().manual_seed(42)),x['labels'],[1]*8).mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters() if p.requires_grad)
            assert all(p.grad.abs().sum()>0 for p in h.signal.parameters())
            sizes.append(sum(p.numel() for p in h.parameters() if p.requires_grad))
        assert len(set(sizes))==1
        checks.append('zero residual RAW equality, future-label isolation, mask and legal H/KV sensitivity, finite gradients')
        logits=torch.randn(2,8,97);labels=x['labels'];weights=[8,7,6,5,4,3,2,1]
        token=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape(2,8)
        weighted=(labels!=-100)*torch.tensor(weights)
        torch.testing.assert_close(window_ce(logits,labels,weights),(token*weighted).sum(1)/weighted.sum(1))
        checks.append('padding excluded and front weights normalized within each window')
        # Save after one real optimizer update, then compare uninterrupted vs restored next update.
        h=heads['ACK_KV'];h.load_trainable_state_dict(initial);opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4)
        rng=torch.Generator().manual_seed(42)
        def step(model,optimizer,generator):
            optimizer.zero_grad(set_to_none=True)
            loss=window_ce(model.forward_batch(select_view(x,'ACK_KV'),generator),x['labels'],[1]*8).mean()
            loss.backward();optimizer.step()
        step(h,opt,rng)
        checkpoint=root/'resume.pt';torch.save(dict(model=h.trainable_state_dict(),optimizer=opt.state_dict(),rng=rng.get_state()),checkpoint)
        step(h,opt,rng);expected=h.trainable_state_dict()
        restored=MatchedACKHead(cfg,output,arm='ACK_KV');payload=torch.load(checkpoint,weights_only=False)
        restored.load_trainable_state_dict(payload['model'])
        opt2=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4);opt2.load_state_dict(payload['optimizer'])
        rng2=torch.Generator();rng2.set_state(payload['rng']);step(restored,opt2,rng2)
        assert all(torch.equal(expected[k],v) for k,v in restored.trainable_state_dict().items())
        checks.append('actual model+AdamW+private Gumbel RNG checkpoint restores identical next update')
        # Reject a future/misaligned teacher boundary using the original validator.
        valid_prefix=[1,2,10];valid=torch.tensor([True,True,True,False]);k=torch.zeros(2,4,8,128,dtype=torch.bfloat16)
        row=dict(candidate_ids=[10,11],accepted_count=1,correction_id=12,bonus_id=None,
                 stop_reason='continue',committed_ids=[10,12],prefix_before_sha256=_prefix_sha([1,2]),
                 prefix_after_commit_sha256=_prefix_sha(valid_prefix+[12]),tensor_prefix='r',
                 hidden=dict(source_prefix_sha256=_prefix_sha(valid_prefix),predicted_token_id=12),
                 kv=dict(layer_ids=[34,35],absolute_positions=[0,1,2],cache_length=3,valid_mask=valid.tolist(),
                         K_RoPE='post_RoPE',dtype='BF16',available_before_receipt=True,excludes_unprocessed_answer=True))
        reader=SimpleNamespace(get_tensor=lambda name:{'r.K':k,'r.V':k,'r.valid':valid}[name])
        read_boundary(row,[1,2],reader);bad=copy.deepcopy(row);bad['kv']['cache_length']=4
        try:read_boundary(bad,[1,2],reader)
        except ValueError:pass
        else:raise AssertionError('future cache boundary accepted')
        checks.append('future/unprocessed-answer KV boundary rejected')
        logic=cpu_tests();assert logic['status']=='pass'
        checks.append('canonical SD loop: seven actual accept/reject/EOS/bonus/truncation cases')
        # Actual tiny Qwen3 with real post-RoPE attention K/V observation.
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from feedback_codec.hf_backend import HFTargetBackend
        from same_forward_kv import SameForwardKV
        tc=Qwen3Config(vocab_size=97,hidden_size=1024,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=8,num_key_value_heads=8,head_dim=128,max_position_embeddings=32,
                      pad_token_id=0,bos_token_id=1,eos_token_id=96,attention_dropout=0.)
        tc._attn_implementation='eager'
        target=HFTargetBackend(Qwen3ForCausalLM(tc),object(),device='cpu',dtype=torch.bfloat16)
        prefix=[1,2,3,4,5,6];baseline=target.next(prefix)
        with SameForwardKV(target,layer_ids=(0,1),check_source=True) as observer:
            observed=observer.next(prefix)
            assert observed.token_id==baseline.token_id and torch.equal(observed.hidden,baseline.hidden)
            with torch.inference_mode():full=target.model(input_ids=target._ids(prefix),use_cache=True,return_dict=True)
            for i in range(2):
                for key,name in [('keys','K'),('values','V')]:
                    ref=getattr(full.past_key_values.layers[i],key)[0,:,-4:].permute(1,0,2)
                    assert torch.equal(ref,observed.kv[name][i])
        checks.append('actual tiny Qwen3 post-RoPE K/V matches independent cache; Target token/hidden unchanged')
        assert not torch.cuda.is_initialized()
        result=dict(status='pass',GPU_used=False,checks=checks,trainable_parameters_each=sizes[0],
                    projection_parameters=98304,source='real frozen historical components; synthetic small CPU fixtures',
                    scope='not a full production train/eval, 8B Target run, or GPU/latency result')
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
