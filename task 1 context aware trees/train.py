"""Train the shared encoder, local XGBoost and full CatBoost."""
import argparse
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import torch

from context_trees.data import prepare_training, close_reads
from context_trees.model import fit_encoder, site_matrix, fit_trees, save_model, feature_names, RECIPE

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', type=Path, help='Manifest for named binary and optional ordered datasets.')
    source.add_argument('--data-dir', type=Path, help='Folder containing course data0, data1 and data2; defaults to inputs/training.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=HERE / 'model/reference.joblib')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=4262)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--rank-weight', type=float, default=1.)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    if min(args.threads, args.epochs, args.width) < 1 or args.rank_weight < 0:
        parser.error('Use positive threads/epochs/width and nonnegative rank weight.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory.')
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        base = args.config.resolve().parent
    else:
        base = (args.data_dir or HERE/'inputs/training').resolve()
        config = dict(datasets=[
            dict(name='data0',kind='binary',signals='data0/dataset0.json.gz',labels='data0/data.info.labelled'),
            dict(name='data1',kind='binary',signals='data1/dataset1.json.gz',labels='data1/data.info'),
            dict(name='data2',kind='ordered',signals='data2/dataset2.json.gz',labels='data2/data.info',
                 order_column='label',group_columns=['transcript_position'])])
    reference = joblib.load(args.reference)
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='shared_trees_train_', dir=args.work_dir) as scratch:
        prepared, ordered = prepare_training(config, base, scratch, reference)
        try:
            network, scalers, history = fit_encoder(prepared, ordered, device=args.device,
                threads=args.threads, seed=args.seed, epochs=args.epochs,
                rank_weight=args.rank_weight, width=args.width)
            values = site_matrix(network, prepared, scalers, device=args.device)
            ordered_values = None if ordered is None else site_matrix(network, ordered, scalers, device=args.device)
            tree, cat = fit_trees(values, prepared['sites'].label.to_numpy(),
                                  ordered_values, None if ordered is None else ordered['sites'],
                                  device=args.device, threads=args.threads, seed=args.seed)
            save_model(args.output, network, scalers, tree, cat)
            shutil.copyfile(args.reference, args.output / 'reference.joblib')
            metadata = dict(RECIPE, seed=args.seed, rank_weight=args.rank_weight,
                feature_names=feature_names(args.width), encoder_epochs=args.epochs,
                full_predictor='CatBoost', local_predictor='XGBoost',
                full_feature_count=args.width+33, local_feature_count=args.width+12,
                local_feature_names=feature_names(args.width)[:args.width+12],
                binary_sites=len(prepared['sites']), ordered_sites=0 if ordered is None else len(ordered['sites']),
                ordered_groups=0 if ordered is None else int(ordered['sites'].order_group.nunique()),
                supervision='Shared encoder and both final predictors use binary labels and optional within-group ordering. No origin classifier.',
                dependencies={name: version(name) for name in
                    ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'xgboost', 'catboost', 'joblib']})
            (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        finally:
            close_reads(prepared['reads'])
            if ordered is not None:
                close_reads(ordered['reads'])
    print('Saved the shared encoder and the 45/55 full/local tree pair.', flush=True)


if __name__ == '__main__':
    main()
