"""Fetch pinned public activations, or hard-link an existing local acquisition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/glossary_2026-10-23'
REPO = 'NahuelRoldan/qwen3-8b-glossary-full-hidden-states-2026-10-23'
REVISION = '54a33c4b473df6b65538ef0b7d89a72324fc56a1'
PREFIX = 'cloud_glossary_2026-10-23'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='Download all 100 rollouts (~21 GiB).')
    group.add_argument('--rollouts', nargs='+', help='Default: r0001 r0026.')
    parser.add_argument('--source', type=Path, help='Hard-link an existing local raw-data folder; same volume required.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    names = [f'r{i:04d}' for i in range(1,101)] if args.all else (args.rollouts or ['r0001','r0026'])
    if len(set(names)) != len(names) or any(n not in {f'r{i:04d}' for i in range(1,101)} for n in names):
        parser.error('Rollout identifiers must be unique and in r0001..r0100.')
    reference = json.loads((DATA/'r0001/result.json').read_text(encoding='utf-8'))
    jobs = [('prompt_h.npy', reference['prompt_h_sha256'])]
    for name in names:
        record = json.loads((DATA/name/'result.json').read_text(encoding='utf-8'))
        jobs.append((f'{name}/generated_h.npy', record['files_sha256']['generated_h.npy']))
    print('Pinned dataset:', REPO, REVISION)
    print('Arrays:', len(jobs), '| rollout activation upper bound:', round(len(names)*216/1024,2), 'GiB + prompt')
    if args.dry_run:
        for relative,_ in jobs:
            print(relative)
        return
    if args.source is None:
        missing = sum(not (DATA / relative).exists() for relative,_ in jobs)
        if shutil.disk_usage(DATA).free < missing*216*1024**2 + 2*1024**3:
            raise RuntimeError('Insufficient disk space including a 2 GiB reserve.')
    for relative, expected in jobs:
        destination = DATA/relative
        if destination.exists():
            if digest(destination) != expected:
                raise RuntimeError(f'Existing array has an unexpected hash: {relative}')
            print('Verified existing:', relative, flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.source is not None:
            source = args.source.resolve()/relative
            if digest(source) != expected:
                raise RuntimeError(f'Local source hash mismatch: {relative}')
            os.link(source, destination)
            print('Linked:', relative, flush=True)
            continue
        temporary = destination.with_suffix(destination.suffix+'.partial')
        size = temporary.stat().st_size if temporary.exists() else 0
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PREFIX}/{relative}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={size}-'} if size else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            append = size > 0 and response.status == 206
            with temporary.open('ab' if append else 'wb') as f:
                shutil.copyfileobj(response, f, length=8*1024**2)
        if digest(temporary) != expected:
            raise RuntimeError(f'Download hash mismatch; partial file preserved: {relative}')
        temporary.replace(destination)
        print('Downloaded and verified:', relative, flush=True)


if __name__ == '__main__':
    main()
