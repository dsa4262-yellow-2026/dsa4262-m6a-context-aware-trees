"""Download the SG-NEx JSON/index pairs in a samples.csv inventory."""
import argparse
import csv
from pathlib import Path
import subprocess
import urllib.request


def download(url, target):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        expected = int(response.headers['Content-Length'])
    if target.exists():
        if target.stat().st_size != expected:
            raise ValueError(f'Existing file has an incomplete or different size: {target}')
        print(f'Already downloaded: {target.name}', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    if not partial.exists() or partial.stat().st_size != expected:
        subprocess.run([
            'curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
            '--continue-at', '-', '--output', str(partial), url,
        ], check=True)
    if partial.stat().st_size != expected:
        raise ValueError(f'Incomplete download: {partial}. Rerun the download command.')
    partial.rename(target)
    print(f'Downloaded: {target.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=Path(__file__).resolve().with_name('samples.csv'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--sample', nargs='+', help='Inventory sample IDs; omit for all samples.')
    args = parser.parse_args()
    with args.samples.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.sample:
        unknown = set(args.sample) - {row['sample_id'] for row in rows}
        if unknown:
            parser.error(f'Sample IDs absent from inventory: {sorted(unknown)}')
        rows = [row for row in rows if row['sample_id'] in args.sample]
    root = args.data_root.resolve()
    for row in rows:
        print(row['sample_id'], flush=True)
        for kind in ['signals', 'info']:
            target = (root / row[kind]).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Inventory path leaves the data folder: {row[kind]}')
            url = row['url'] if kind == 'signals' else row['url'].rsplit('/', 1)[0] + '/data.info'
            download(url, target)
    print(f'Complete: {len(rows)} samples, {2 * len(rows)} files.', flush=True)


if __name__ == '__main__':
    main()
