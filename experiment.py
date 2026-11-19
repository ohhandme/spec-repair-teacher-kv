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


def preflight(out, target=False):
    sources = check_repo(); freeze = read(out / 'freeze.json'); assets = read(out / 'assets_runtime.json')
    if value_hash(sources) != freeze['repository_source_binding'] or digest(out / 'assets_runtime.json') != freeze['assets_sha256']:
        raise ValueError('source or assets changed after configure')
    for name, expected in freeze['source_files'].items():
        if digest(out / name) != expected:
            raise ValueError('configured source differs: ' + name)
    for name, expected in freeze['source_sha256'].items():
        if digest(Path(freeze['source_root']) / name) != expected:
            raise ValueError('vendor source differs: ' + name)
    reg = read(out / 'ACK_registration.json')
    if digest(out / 'ACK_registration.json') != freeze['ACK_registration_sha256'] or digest(out / 'roster503.json') != freeze['roster_sha256']:
        raise ValueError('registration or roster changed')
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


def import_checkpoints(out, mapping_path):
    entries = read(mapping_path); cfg = read(out / 'ACK_registration.json'); chosen = []
    if set(entries) != set(cfg['arms']):
        raise ValueError('checkpoint arms must match configured arms')
    for arm in cfg['arms']:
        entry = entries[arm]; path = Path(entry['path']).resolve()
        if digest(path) != entry['sha256']:
            raise ValueError('checkpoint differs: ' + arm)
        chosen.append(dict(name=arm, kind='ack', checkpoint=str(path), sha256=entry['sha256'],
                           selected_epoch=entry.get('epoch'), external_checkpoint=True))
    sealed(out / 'ack_training/selected_arms.json', chosen)
    sealed(out / 'external_checkpoint_import.json', dict(status='evaluation_only', no_new_training=True,
           mapping_sha256=digest(mapping_path), arms=chosen))


def compare(out):
    check_repo(); cfg = read(out / 'ACK_registration.json'); fid = digest(out / 'freeze.json')
    completion = read(out / 'ack_dev/completion.json')
    if completion['status'] != 'complete' or completion['freeze_sha256'] != fid:
        raise ValueError('matching completed evaluation required')
    if digest(out / 'ack_dev/manifest.jsonl') != completion['manifest_sha256']:
        raise ValueError('result manifest differs')
    checkpoint = {r['name']: r['sha256'] for r in read(out / 'ack_training/selected_arms.json')}; checkpoint['RAW'] = None
    manifest = [json.loads(s) for s in (out/'ack_dev/manifest.jsonl').read_text().splitlines() if s]
    manifest_map = {(r['request_id'], r['arm']): r for r in manifest}
    roster = read(out/'roster503.json')
    expected_pairs = {(r['request_id'], arm) for r in roster for arm in ['RAW']+cfg['arms']}
    if len(manifest_map) != len(manifest) or set(manifest_map) != expected_pairs:
        raise ValueError('duplicate or incomplete request-arm manifest')
    result = {}
    for arm in ['RAW'] + cfg['arms']:
        receipts = []
        for request in roster:
            folder = out / 'ack_dev/requests' / request['request_id'] / arm
            receipt = read(folder / 'receipt.json'); blob = (folder / 'rounds.jsonl').read_bytes()
            if (digest(folder/'receipt.json') != manifest_map[(request['request_id'], arm)]['receipt_sha256']
                    or receipt['request_id'] != request['request_id'] or receipt['arm'] != arm
                    or receipt['freeze_sha256'] != fid or receipt['checkpoint_sha256'] != checkpoint[arm]
                    or sha256(blob).hexdigest() != receipt['rounds_sha256']):
                raise ValueError('mixed contract/checkpoint or changed rounds')
            rows = [json.loads(line) for line in blob.splitlines() if line]; generated = []; accepted = 0
            prefix = request['prompt_ids'].copy()
            for i, row in enumerate(rows):
                if row['round'] != i or row['prefix_before_ids'] != prefix:
                    raise ValueError('round order or causal prefix differs')
                n = next((i for i, (a, b) in enumerate(zip(row['candidate_ids'], row['target_decision_ids'])) if a != b), len(row['candidate_ids']))
                if n != row['accepted_count']:
                    raise ValueError('acceptance count differs')
                answer = row['correction_id'] if row['correction_id'] is not None else row['bonus_id']
                committed = row['candidate_ids'][:n] + ([] if answer is None else [answer])
                if row['committed_ids'] != committed or row['target_decision_ids'] != committed:
                    raise ValueError('verification and commit differ')
                accepted += n; generated.extend(committed); prefix.extend(committed)
                if row['prefix_after_ids'] != prefix or generated != request['frozen_target_ids'][:len(generated)]:
                    raise ValueError('committed output prefix differs')
            if generated != request['frozen_target_ids'] or receipt['output_ids'] != generated or accepted != receipt['accepted_draft_tokens'] or len(rows) != receipt['verification_rounds']:
                raise ValueError('inexact output or counts')
            receipts.append(receipt)
        accepted = sum(r['accepted_draft_tokens'] for r in receipts); rounds = sum(r['verification_rounds'] for r in receipts)
        result[arm] = dict(requests=len(receipts), accepted=accepted, rounds=rounds, accepted_tokens_per_round=accepted/rounds)
    sealed(out / 'comparison.json', dict(status='complete', arms=result, contract_sha256=fid,
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
    command = [sys.executable, str(out/('train_ack.py' if args.phase == 'train' else 'evaluate_dev.py'))]
    subprocess.run(command + ([] if args.phase == 'train' else ['--stage','ack']), cwd=out, check=True)


if __name__ == '__main__':
    main()
