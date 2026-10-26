"""Benchmark the official m6Anet checkpoint on a fixed set of development genes."""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_trees.features import KEY, iter_sites
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


CHECKPOINT = 'HCT116_RNA002'


def prepare(signals, selected, folder):
    wanted = set(selected[KEY].itertuples(index=False, name=None))
    seen, eligible = set(), []
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'data.json').open('wb') as target, (folder / 'data.info').open('w', newline='') as handle:
        info = csv.writer(handle)
        info.writerow(KEY + ['start', 'end', 'n_reads'])
        for transcript, position, context, reads in iter_sites(signals):
            key = (transcript, int(position))
            if key not in wanted:
                continue
            seen.add(key)
            if len(reads) < 20:
                continue
            # Native format needs an index column. The course input has no molecule IDs.
            packed = [row.tolist() + [index] for index, row in enumerate(reads)]
            payload = json.dumps({transcript: {str(int(position)): {context: packed}}},
                                 ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii') + b'\n'
            start = target.tell()
            target.write(payload)
            info.writerow([transcript, int(position), start, target.tell(), len(reads)])
            eligible.append(key)
    if seen != wanted or not eligible:
        raise ValueError('Benchmark sites must all exist in the raw input, with at least one eligible site.')
    return pd.DataFrame(eligible, columns=KEY), {'selected_sites': len(wanted),
                                               'eligible_sites': len(eligible),
                                               'below_20_reads': len(wanted) - len(eligible)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', nargs=2, metavar=('NAME', 'SIGNALS'), required=True)
    parser.add_argument('--sites', type=Path, help='Held-out fold NPZ; defaults to the selected bundled fold.')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--output', type=Path, required=True, help='New directory for benchmark predictions and settings.')
    parser.add_argument('--work-dir', type=Path, required=True, help='New scratch directory for native inputs and per-read outputs.')
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8) or not os.environ.get('CONDA_PREFIX'):
        raise RuntimeError('Run this script inside the supplied Conda Python 3.8 environment.')
    if version('m6anet') != '2.1.0':
        raise RuntimeError('This adapter requires m6anet 2.1.0.')
    if args.processes < 1 or args.batch_size < 1:
        parser.error('Process count and batch size must be positive.')
    for folder in (args.output, args.work_dir):
        if folder.exists() and any(folder.iterdir()):
            parser.error('Output and work directories must be new or empty.')
        folder.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.work_dir.resolve():
        parser.error('Keep scratch files separate from benchmark results.')
    names = [name for name, _ in args.dataset]
    if len(set(names)) != len(names):
        parser.error('Dataset names must be distinct.')
    archive = args.sites or Path(__file__).resolve().parents[1] / 'results' / f'fold{args.fold}.npz'
    with np.load(archive, allow_pickle=False) as saved:
        sites = pd.DataFrame({name: saved[name] for name in KEY + ['label']})
        sites['dataset'] = ['data' + str(x) for x in saved['source']]
        sites['fold'] = args.fold
        sites['context_aware_trees'] = saved['prediction'].mean(axis=1)
    if sites.duplicated(['dataset', *KEY]).any():
        raise ValueError('Benchmark selection contains duplicate site identifiers.')
    sites = sites[sites.fold.eq(args.fold) & sites.dataset.isin(names)]
    if set(sites.dataset) != set(names):
        raise ValueError('Each supplied dataset needs selected sites.')

    executable = shutil.which('m6anet')
    if not executable:
        raise RuntimeError('m6anet executable is missing; activate the Conda environment.')
    metadata = {
        'checkpoint': CHECKPOINT,
        'python': sys.version.split()[0],
        'versions': {name: version(name) for name in ['m6anet', 'torch', 'numpy', 'pandas', 'scikit-learn']},
        'fold': args.fold, 'seed': 4262, 'num_iterations': 1000, 'sampled_reads_per_iteration': 20,
        'processes': args.processes, 'batch_size': args.batch_size,
        'sampling': 'Native m6anet 2.1.0 Monte Carlo sampling with replacement; worker randomness can vary by platform.',
        'adaptation': 'Nine signals unchanged; original numeric positions; appended synthetic within-site read indices.',
        'normalization': 'Official bundled motif-specific normalization; no fitting on evaluated sites.',
        'training_overlap': 'Unknown: the official HCT116 checkpoint may overlap the course data; this is a reference comparison.',
        'datasets': [],
    }
    results = []
    started = time.monotonic()
    for index, (name, signals) in enumerate(args.dataset):
        work = args.work_dir.resolve() / str(index)
        expected, details = prepare(signals, sites[sites.dataset.eq(name)], work / 'input')
        print(f'{name}: running native m6Anet on {len(expected):,} sites.', flush=True)
        # Version 2.1.0 flushes when (batch + 1) % save_per_batch is nonzero.
        # A value above the number of batches flushes every batch, including the last.
        command = [executable, 'inference', '--input_dir', str(work / 'input'),
                   '--out_dir', str(work / 'native'), '--pretrained_model', CHECKPOINT,
                   '--device', 'cpu', '--n_processes', str(args.processes),
                   '--batch_size', str(args.batch_size), '--num_iterations', '1000',
                   '--seed', '4262', '--save_per_batch', str(len(expected) + 1)]
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        subprocess.run(command, env=environment, check=True)
        native = pd.read_csv(work / 'native/data.site_proba.csv', dtype={'transcript_id': str})
        joined = expected.merge(native, on=KEY, how='outer', validate='one_to_one', indicator=True)
        if not joined._merge.eq('both').all():
            raise ValueError('Native output does not cover exactly the eligible site set.')
        scores = joined.probability_modified.to_numpy()
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Native m6Anet produced invalid probabilities.')
        result = joined[KEY].copy()
        result.insert(0, 'dataset', name)
        result['score'] = scores
        results.append(result)
        metadata['datasets'].append({'dataset': name, **details, 'predicted_sites': len(result)})
        print(f'{name}: verified all {len(result):,} native predictions.', flush=True)
    output = args.output / 'predictions.csv.gz'
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(output, index=False, float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    comparison = sites.merge(predictions, on=['dataset', *KEY], validate='one_to_one')
    if len(comparison) != len(sites):
        raise ValueError('Model comparison must cover every selected held-out site.')
    metrics = []
    for dataset in ['all'] + names:
        rows = comparison if dataset == 'all' else comparison[comparison.dataset.eq(dataset)]
        for model, column in [('context_aware_trees', 'context_aware_trees'), ('m6anet_native', 'score')]:
            precision, recall, _ = precision_recall_curve(rows.label, rows[column])
            metrics.append(dict(dataset=dataset, fold=args.fold, model=model, sites=len(rows),
                                roc_auc=roc_auc_score(rows.label, rows[column]), pr_auc=auc(recall, precision)))
    pd.DataFrame(metrics).to_csv(args.output / 'comparison.csv', index=False)
    metadata.update(status='complete', seconds=time.monotonic() - started)
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
