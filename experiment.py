"""Standalone historical ACK/H/KV repair-head experiment; no teacher collection."""
import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ARMS = ('ACK', 'ACK_H', 'ACK_KV')


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def value_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sealed(path, value=None, *, data=None):
    data = data if data is not None else (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable output differs; choose a new run directory: ' + str(path))
    else:
        fd, temporary = tempfile.mkstemp(prefix=path.name+'.pending-', dir=path.parent)
        pending = Path(temporary)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(pending, path)  # Publish complete bytes, without replacing another writer.
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError('concurrent immutable output differs: ' + str(path))
        finally:
            pending.unlink(missing_ok=True)


def check_repo():
    manifest = read(HERE / 'SOURCE_MANIFEST.json')
    for name, expected in manifest.items():
        if digest(HERE / name) != expected:
            raise ValueError('repository source differs: ' + name)
    return manifest


def remap(path, mapping):
    text = str(path).replace('\\', '/')
    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
        old = old.rstrip('/')
        if text == old or text.startswith(old + '/'):
            return str(Path(new) / text[len(old):].lstrip('/'))
    return str(path)


def configure(assets_path, out, arms, loss):
    sources = check_repo(); out = out.resolve(); assets = read(assets_path)
    if out == HERE or HERE in out.parents:
        raise ValueError('output must be outside the checkout')
    for key in ('models_root', 'train_root', 'data_root', 'gpu_lock'):
        if not isinstance(assets.get(key), str) or not assets[key]:
            raise ValueError('missing asset path: ' + key)
    mapping = assets.get('path_map', {})
    if any(not old.startswith('/') or not new for old, new in mapping.items()):
        raise ValueError('path_map requires explicit absolute historical roots')
    historical = read(HERE / 'inputs/historical_registration.json')
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ARMS):
        raise ValueError('unknown or duplicate arm')
    selected = [a for a in ARMS if a in arms]
    registration = {k: v for k, v in historical.items() if k != 'GPU_queue'}
    registration.update(arms=selected, position_weights=[1]*8 if loss == 'equal' else list(range(8, 0, -1)),
                        position_weighting=loss, standalone=True,
                        calibration='equal-position window CE, request-equal mean; earliest minimum')
    for split in ('train', 'calibration'):
        entry = historical['common_windows'][split]
        path = assets.get('common_windows', {}).get(split, remap(entry['path'], mapping))
        registration['common_windows'][split] = {**entry, 'path': str(path)}
    runtime = {p.name: p.read_bytes() for p in sorted((HERE / 'runtime').glob('*.py'))}
    for name, data in runtime.items():
        sealed(out / name, data=data)
    sealed(out / 'assets_runtime.json', assets); sealed(out / 'ACK_registration.json', registration)
    roster = read(HERE / 'inputs/development503.json')
    for row in roster:
        row['raw_trace_path'] = remap(row['raw_trace_path'], mapping)
    if len(roster) != 503 or len({r['request_id'] for r in roster}) != 503:
        raise ValueError('not the frozen 503 requests')
    sealed(out / 'roster503.json', roster)
    smoke = [r['request_id'] for r in roster if r['eos']][:1] + [r['request_id'] for r in roster if r['truncated']][:2]
    if len(smoke) != 3:
        raise ValueError('smoke requires EOS and truncated cases')
    sealed(out / 'smoke_ids.json', smoke)
    original = read(HERE / 'inputs/historical_freeze.json')
    manifest = Path(assets.get('models_manifest', HERE / 'inputs/models_manifest.json')).resolve()
    if digest(manifest) != original['models_manifest_sha256']:
        raise ValueError('model manifest differs from the historical experiment')
    freeze = dict(models_root=assets['models_root'], models_manifest_path=str(manifest),
                  models_manifest_sha256=digest(manifest), source_root=str(HERE / 'vendor'),
                  source_sha256=original['source_sha256'],
                  source_files={n: sha256(b).hexdigest() for n, b in runtime.items()}, reference_sources={},
                  roster_sha256=digest(out / 'roster503.json'), original_roster_sha256=original['roster_sha256'],
                  ACK_registration_sha256=digest(out / 'ACK_registration.json'),
                  repository_source_binding=value_hash(sources), assets_sha256=digest(out / 'assets_runtime.json'),
                  window=8, max_new=128, eos=151645, dtype='BF16', attention='eager',
                  independent300_sealed=True, development_only=True,
                  migration='standalone orchestration and path relocation; no predecessor receipts')
    sealed(out / 'freeze.json', freeze)
    return out


def contract_inputs(out):
    freeze = read(out / 'freeze.json')
    if freeze['repository_source_binding'] != value_hash(check_repo()):
        raise ValueError('source changed after configure')
    for name, key in [('ACK_registration.json', 'ACK_registration_sha256'),
                      ('roster503.json', 'roster_sha256'), ('assets_runtime.json', 'assets_sha256')]:
        if digest(out / name) != freeze[key]:
            raise ValueError('configured input changed: ' + name)
    return freeze, read(out / 'ACK_registration.json')


def preflight(out, target=False):
    freeze, reg = contract_inputs(out); assets = read(out / 'assets_runtime.json')
    for name, expected in freeze['source_files'].items():
        if digest(out / name) != expected:
            raise ValueError('configured source differs: ' + name)
    for name, expected in freeze['source_sha256'].items():
        if digest(Path(freeze['source_root']) / name) != expected:
            raise ValueError('vendor source differs: ' + name)
    if not target:
        if digest(Path(assets['train_root']) / 'index.jsonl') != reg['train_index_sha256']:
            raise ValueError('RAW5000 index differs')
        for entry in reg['common_windows'].values():
            if digest(entry['path']) != entry['sha256']:
                raise ValueError('teacher boundary index differs: ' + entry['path'])
    if digest(freeze['models_manifest_path']) != freeze['models_manifest_sha256']:
        raise ValueError('model manifest changed')
    for model in (('Qwen3-0.6B', 'Qwen3-8B') if target else ('Qwen3-0.6B',)):
        for name, entry in read(freeze['models_manifest_path'])[model].items():
            if digest(Path(assets['models_root']) / model / name) != entry['sha256']:
                raise ValueError('model file differs: ' + model + '/' + name)
    return dict(status='pass', model_scope='Draft+Target' if target else 'Draft', GPU_launched=False,
                tensor_cache_scope='original lazy readers; whole tensor corpus not hashed here')


def historical_selection(chosen, cfg):
    catalog = {r['name']: r for r in read(HERE / 'inputs/historical_selected_arms.json')}
    if cfg['position_weighting'] != 'equal':
        raise ValueError('historical weights were trained with equal position weights')
    for entry in chosen:
        original = catalog[entry['name']]
        if (entry['sha256'], entry['selected_epoch']) != (original['sha256'], original['selected_epoch']):
            raise ValueError('historical selection differs for ' + entry['name'])


def reserve_origin(out, kind):
    sealed(out / 'weights_origin.json', dict(kind=kind, freeze_sha256=digest(out / 'freeze.json')))


def import_checkpoints(out, mapping_path):
    _, cfg = contract_inputs(out)
    entries = read(mapping_path); chosen = []
    if set(entries) != set(cfg['arms']):
        raise ValueError('checkpoint arms must match configured arms')
    for arm in cfg['arms']:
        entry = entries[arm]; path = Path(entry['path']).resolve()
        if digest(path) != entry['sha256']:
            raise ValueError('checkpoint differs: ' + arm)
        chosen.append(dict(name=arm, kind='ack', checkpoint=str(path), sha256=entry['sha256'],
                           selected_epoch=entry.get('epoch'), external_checkpoint=True))
    historical_selection(chosen, cfg)
    if (out / 'ack_training/completion.json').exists():
        raise ValueError('run already contains completed training')
    reserve_origin(out, 'historical_import')
    sealed(out / 'ack_training/selected_arms.json', chosen)
    sealed(out / 'external_checkpoint_import.json', dict(status='evaluation_only', no_new_training=True,
           mapping_sha256=digest(mapping_path), freeze_sha256=digest(out / 'freeze.json'), arms=chosen))


def selected_weights(out, cfg):
    selected = out / 'ack_training/selected_arms.json'
    imported = out / 'external_checkpoint_import.json'
    trained = out / 'ack_training/completion.json'
    if imported.is_file() == trained.is_file() or not selected.is_file():
        raise ValueError('one completed weight source is required')
    entries = read(selected)
    receipt = read(imported if imported.is_file() else trained)
    fid = digest(out / 'freeze.json')
    if ([r['name'] for r in entries] != cfg['arms'] or receipt['arms'] != entries
            or receipt.get('freeze_sha256') != fid):
        raise ValueError('selected arms or frozen training contract differ')
    if imported.is_file():
        if receipt.get('status') != 'evaluation_only' or receipt.get('no_new_training') is not True:
            raise ValueError('invalid historical import receipt')
        historical_selection(entries, cfg)
        origin = 'historical_import'
    else:
        if (receipt.get('status') != 'complete' or receipt.get('registration_sha256') != digest(out / 'ACK_registration.json')
                or receipt.get('updates_each') != cfg['total_updates']):
            raise ValueError('training completion does not match registration')
        origin = 'trained'
    for entry in entries:
        if entry['kind'] != 'ack' or digest(entry['checkpoint']) != entry['sha256']:
            raise ValueError('selected checkpoint differs: ' + entry['name'])
    return origin, {r['name']: r['sha256'] for r in entries}


def compare(out):
    freeze, cfg = contract_inputs(out)
    fid = digest(out / 'freeze.json')
    origin, checkpoint = selected_weights(out, cfg)
    checkpoint['RAW'] = None
    completion = read(out / 'ack_dev/completion.json')
    if completion['status'] != 'complete' or completion['freeze_sha256'] != fid:
        raise ValueError('matching completed evaluation required')
    if digest(out / 'ack_dev/manifest.jsonl') != completion['manifest_sha256']:
        raise ValueError('result manifest differs')
    manifest = [json.loads(s) for s in (out/'ack_dev/manifest.jsonl').read_text().splitlines() if s]
    manifest_map = {(r['request_id'], r['arm']): r for r in manifest}
    roster = read(out/'roster503.json')
    expected_pairs = {(r['request_id'], arm) for r in roster for arm in ['RAW']+cfg['arms']}
    if (len(manifest_map) != len(manifest) or set(manifest_map) != expected_pairs
            or len({r['request_id'] for r in roster}) != len(roster) or completion['requests'] != len(roster)):
        raise ValueError('duplicate or incomplete request-arm manifest')
    raw_rounds = {}
    raw_fields = ('candidate_ids', 'target_decision_ids', 'accepted_count', 'correction_id', 'bonus_id', 'committed_ids')
    def raw_signature(rows):
        return value_hash([[row[key] for key in raw_fields] for row in rows])
    for request in roster:
        trace = Path(request['raw_trace_path'])
        if digest(trace) != request['raw_trace_sha256']:
            raise ValueError('archived RAW trace differs')
        archived = read(trace)['rounds']
        if not archived:
            raise ValueError('empty archived RAW trace')
        raw_rounds[request['request_id']] = archived[0]['candidate_ids'], raw_signature(archived)
    result = {}
    for arm in ['RAW'] + cfg['arms']:
        receipts = []
        for request in roster:
            folder = out / 'ack_dev/requests' / request['request_id'] / arm
            receipt = read(folder / 'receipt.json'); blob = (folder / 'rounds.jsonl').read_bytes()
            if (digest(folder/'receipt.json') != manifest_map[(request['request_id'], arm)]['receipt_sha256']
                    or receipt['request_id'] != request['request_id'] or receipt['arm'] != arm
                    or receipt['status'] != 'pass' or receipt['family_id'] != request['family_id']
                    or receipt['raw_trace_sha256'] != request['raw_trace_sha256'] or receipt['freeze_sha256'] != fid or receipt['checkpoint_sha256'] != checkpoint[arm]
                    or sha256(blob).hexdigest() != receipt['rounds_sha256']):
                raise ValueError('mixed contract/checkpoint or changed rounds')
            for name, expected in receipt['tensor_files'].items():
                if Path(name).name != name or digest(folder / name) != expected:
                    raise ValueError('saved boundary tensor differs')
            oracle = out / 'oracles' / (request['request_id'] + '.json')
            if digest(oracle) != receipt['oracle_receipt_sha256']:
                raise ValueError('oracle receipt differs')
            oracle_row = read(oracle)
            if (oracle_row['request_id'] != request['request_id'] or oracle_row['output_ids'] != request['frozen_target_ids']
                    or oracle_row['freeze_sha256'] != fid or oracle_row['prompt_sha256'] != value_hash(request['prompt_ids'])):
                raise ValueError('oracle input or output differs')
            rows = [json.loads(line) for line in blob.splitlines() if line]; generated = []; accepted = 0
            initial, signature = raw_rounds[request['request_id']]
            if not rows or rows[0]['candidate_ids'] != initial:
                raise ValueError('first draft differs from registered RAW trace')
            if arm == 'RAW' and raw_signature(rows) != signature:
                raise ValueError('RAW candidates or verification differ from registered trace')
            prefix = request['prompt_ids'].copy()
            for i, row in enumerate(rows):
                if row['round'] != i or row['prefix_before_ids'] != prefix:
                    raise ValueError('round order or causal prefix differs')
                proposals = row['candidate_ids']
                budget = freeze['max_new'] - len(generated)
                eos = freeze['eos']
                if (not proposals or len(proposals) > min(freeze['window'], budget) or eos in proposals[:-1]
                        or (proposals[-1] != eos and len(proposals) != min(freeze['window'], budget))
                        or (generated and generated[-1] == eos)
                        or (row['arm'], row['request_id'], row['checkpoint_sha256']) != (arm, request['request_id'], checkpoint[arm])):
                    raise ValueError('invalid proposal window or row identity')
                target = request['frozen_target_ids'][len(generated):]
                n = 0
                for a, b in zip(proposals, target):
                    if a != b:
                        break
                    n += 1
                correction = bonus = rejected = None
                if n < len(proposals):
                    if n >= len(target):
                        raise ValueError('missing target decision for rejection')
                    rejected, correction = proposals[n], target[n]
                elif proposals[-1] != eos and n < budget:
                    if n >= len(target):
                        raise ValueError('missing target bonus decision')
                    bonus = target[n]
                if (n, correction, bonus, rejected) != (row['accepted_count'], row['correction_id'], row['bonus_id'], row['rejected_id']):
                    raise ValueError('verification decision differs')
                committed = proposals[:n] + [v for v in (correction, bonus) if v is not None]
                if committed != row['committed_ids'] or committed != row['target_decision_ids']:
                    raise ValueError('verification and commit differ')
                accepted += n; generated.extend(committed); prefix.extend(committed)
                if (row['prefix_after_ids'] != prefix or generated != request['frozen_target_ids'][:len(generated)]
                        or row['eos'] != (generated[-1] == eos) or row['truncated'] != (generated[-1] != eos and len(generated) == freeze['max_new'])):
                    raise ValueError('committed output prefix differs')
            if generated != request['frozen_target_ids'] or receipt['output_ids'] != generated or accepted != receipt['accepted_draft_tokens'] or len(rows) != receipt['verification_rounds']:
                raise ValueError('inexact output or counts')
            if (not rows or not (rows[-1]['eos'] or rows[-1]['truncated'])
                    or any(rows[-1][k] != receipt[k] or receipt[k] != request[k] for k in ('eos', 'truncated'))
                    or receipt['actual_target_forwards'] != len(generated)):
                raise ValueError('terminal state or target decision count differs')
            receipts.append(receipt)
        accepted = sum(r['accepted_draft_tokens'] for r in receipts); rounds = sum(r['verification_rounds'] for r in receipts)
        result[arm] = dict(requests=len(receipts), accepted=accepted, rounds=rounds, accepted_tokens_per_round=accepted/rounds)
    if result != completion['arms']:
        raise ValueError('completion metrics differ from sealed rounds')
    sealed(out / 'comparison.json', dict(status='complete', arms=result, contract_sha256=fid,
           weights_origin=origin, position_weighting=cfg['position_weighting'],
           scope='503 development only; reused RAW times are not fresh latency; no E2E speed claim'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('check-repo','cpu-smoke','configure','preflight','train','eval','compare','import-checkpoints'))
    p.add_argument('--assets', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--arms', default=','.join(ARMS)); p.add_argument('--loss', choices=('equal','front'), default='equal')
    p.add_argument('--checkpoints', type=Path); args = p.parse_args()
    if args.phase == 'check-repo':
        print(json.dumps(dict(status='pass', bound_sources=len(check_repo())))); return
    if args.phase == 'cpu-smoke':
        check_repo(); subprocess.run([sys.executable, str(HERE/'cpu_smoke.py')], check=True); return
    if args.out is None:
        p.error('--out is required')
    out = args.out.resolve()
    if args.phase == 'configure':
        if args.assets is None:
            p.error('--assets is required')
        configure(args.assets, out, args.arms.split(','), args.loss)
        print(json.dumps(dict(status='configured', out=str(out), GPU_launched=False))); return
    if args.phase == 'import-checkpoints':
        check_repo()
        if args.checkpoints is None:
            p.error('--checkpoints is required')
        import_checkpoints(out, args.checkpoints); return
    if args.phase == 'compare':
        print(json.dumps(compare(out))); return
    result = preflight(out, target=args.phase == 'eval')
    if args.phase == 'preflight':
        print(json.dumps(result)); return
    if os.name != 'posix':
        raise ValueError('original GPU runtime requires Linux fcntl; CPU smoke works on Windows')
    if args.phase == 'train':
        if (out / 'external_checkpoint_import.json').exists():
            raise ValueError('imported weights are evaluation only')
        reserve_origin(out, 'trained')
        if (out / 'ack_training/completion.json').exists():
            selected_weights(out, read(out / 'ACK_registration.json'))
    else:
        selected_weights(out, read(out / 'ACK_registration.json'))
        if (out / 'ack_dev/completion.json').exists():
            compare(out)
            return
    command = [sys.executable, str(out/('train_ack.py' if args.phase == 'train' else 'evaluate_dev.py'))]
    subprocess.run(command + ([] if args.phase == 'train' else ['--stage','ack']), cwd=out, check=True)


if __name__ == '__main__':
    main()
