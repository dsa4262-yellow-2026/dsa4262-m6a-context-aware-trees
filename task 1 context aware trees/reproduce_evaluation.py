"""Refit the selected recipe for grouped-gene or complete-sample evaluation."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from context_trees.data import prepare_sites, attach_order, close_reads
from context_trees.features import transcript_signal_context
from context_trees.model import (
    fit_encoder,
    site_matrix,
    fit_trees,
    tree_predictions,
    RECIPE,
)

HERE = Path(__file__).resolve().parent


def grouped_folds(sites, ordered, gene_map):
    mapping = pd.read_csv(gene_map).set_index("gene_id", verify_integrity=True).fold
    folds = sites.gene_id.map(mapping)
    if folds.isna().any() or not folds.isin(range(5)).all():
        raise ValueError(
            "The supplied gene map must cover every gene with a fold from 0 to 4."
        )
    positions = np.sort(ordered.transcript_position.unique())
    assigned = {}
    for fold, (_, held) in enumerate(
        KFold(5, shuffle=True, random_state=4262).split(positions)
    ):
        assigned.update({position: fold for position in positions[held]})
    return folds.to_numpy(int), ordered.transcript_position.map(assigned).to_numpy(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["genes", "samples"], default="genes")
    parser.add_argument(
        "--reference", type=Path, default=HERE / "model/reference.joblib"
    )
    parser.add_argument(
        "--gene-folds", type=Path, default=HERE / "results/gene_folds.csv"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory.")
    if args.threads < 1 or (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("Use positive threads and an available device.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    reference = joblib.load(args.reference)
    runs = []
    with tempfile.TemporaryDirectory(
        prefix="evaluation_", dir=args.work_dir
    ) as scratch:
        prepared = prepare_sites(
            [
                (
                    0,
                    args.data_dir / "data0/dataset0.json.gz",
                    args.data_dir / "data0/data.info.labelled",
                ),
                (
                    1,
                    args.data_dir / "data1/dataset1.json.gz",
                    args.data_dir / "data1/data.info",
                ),
            ],
            Path(scratch) / "binary",
            reference,
        )
        ordered = None
        try:
            ordered = prepare_sites(
                [(2, args.data_dir / "data2/dataset2.json.gz", None)],
                Path(scratch) / "ordered",
                reference,
            )
            ordered["sites"] = attach_order(
                ordered["sites"],
                args.data_dir / "data2/data.info",
                "label",
                ["transcript_position"],
                2,
            )
            ordered["sites"]["order_group"] = ordered["sites"].transcript_position
            sites, mixtures = prepared["sites"], ordered["sites"]
            folds, mix_folds = grouped_folds(sites, mixtures, args.gene_folds)
            if args.mode == "genes":
                shutil.copyfile(args.gene_folds, args.output / "gene_folds.csv")
            for fold in range(5 if args.mode == "genes" else 2):
                if args.mode == "genes":
                    train, held = np.flatnonzero(folds != fold), np.flatnonzero(
                        folds == fold
                    )
                    mix_train, mix_held = np.flatnonzero(
                        mix_folds != fold
                    ), np.flatnonzero(mix_folds == fold)
                    if np.intersect1d(
                        sites.gene_id.iloc[train], sites.gene_id.iloc[held]
                    ).size:
                        raise ValueError(
                            "A gene crosses the training/evaluation boundary."
                        )
                else:
                    train = np.flatnonzero(sites.source.to_numpy() != fold)
                    held = np.flatnonzero(sites.source.to_numpy() == fold)
                    mix_train = np.arange(len(mixtures))
                    mix_held = np.array([], dtype=int)
                seed = 4262 + fold
                print(
                    f"Fitting {args.mode} {fold}: {len(train):,} binary sites; {len(mix_train):,} ordered sites.",
                    flush=True,
                )
                network, scalers, _ = fit_encoder(
                    prepared,
                    ordered,
                    train,
                    mix_train,
                    device=args.device,
                    threads=args.threads,
                    seed=seed,
                )
                values = site_matrix(network, prepared, scalers, device=args.device)
                mix_values = site_matrix(network, ordered, scalers, device=args.device)
                tree, cat = fit_trees(
                    values[train],
                    sites.label.to_numpy()[train],
                    mix_values[mix_train],
                    mixtures.iloc[mix_train].reset_index(drop=True),
                    device=args.device,
                    threads=args.threads,
                    seed=seed,
                )
                arrays = {
                    name: sites[name].to_numpy()[held]
                    for name in ["source", "transcript_position", "label", "n_reads"]
                }
                arrays.update(
                    {
                        name: sites[name].to_numpy(dtype=str)[held]
                        for name in ["transcript_id", "gene_id", "kmer"]
                    }
                )
                arrays.update(
                    index=held, prediction=tree_predictions(tree, cat, values[held])
                )
                run = dict(
                    fold=fold,
                    seed=seed,
                    training_sites=len(train),
                    held_out_sites=len(held),
                    ordered_training_sites=len(mix_train),
                )
                if args.mode == "genes":
                    arrays.update(
                        mixture_index=mix_held,
                        mixture_position=mixtures.transcript_position.to_numpy()[
                            mix_held
                        ],
                        mixture_nominal=mixtures.order_level.to_numpy()[mix_held],
                        mixture_prediction=tree_predictions(
                            tree, cat, mix_values[mix_held]
                        ),
                    )
                else:
                    retained = np.sort(
                        np.random.default_rng(seed).choice(
                            held, size=len(held) // 2, replace=False
                        )
                    )
                    cutoff = float(np.median(sites.n_reads.to_numpy()[held]))
                    covered = held[sites.n_reads.to_numpy()[held] >= cutoff]
                    for name, rows in [("retained", retained), ("covered", covered)]:
                        subset = values[rows].copy()
                        subset[:, -5:] = transcript_signal_context(
                            sites.iloc[rows],
                            prepared["summary"][rows],
                            scalers,
                            prepared["reference_matched"][rows],
                        )
                        arrays[name + "_index"] = rows
                        arrays[name + "_prediction"] = tree_predictions(
                            tree, cat, subset
                        )
                    arrays["coverage_threshold"] = cutoff
                    run.update(
                        held_out_sample=fold,
                        training_binary_sample=1 - fold,
                        retained_sites=len(retained),
                        coverage_retained_sites=len(covered),
                        coverage_threshold=cutoff,
                    )
                np.savez_compressed(
                    args.output
                    / f'{"fold" if args.mode=="genes" else "sample"}{fold}.npz',
                    **arrays,
                )
                runs.append(run)
                del network, tree, cat, values, mix_values
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                print(f"Completed {args.mode} {fold}.", flush=True)
        finally:
            close_reads(prepared["reads"])
            if ordered is not None:
                close_reads(ordered["reads"])
    protocol = dict(
        RECIPE,
        mode=args.mode,
        runs=runs,
        component_order=["local_xgboost", "full_catboost"],
        scope=(
            "Five gene-grouped and ordered-position-grouped development folds."
            if args.mode == "genes"
            else "Train on the other complete binary sample and Dataset 2. Shared genes and coordinates are allowed; held-out sample labels are excluded from all fitting."
        ),
        context_checks="Keep half the held-out sites, or sites at or above median coverage; recompute only observed other-site context. Compare the same retained sites with their full-context scores.",
        selection="Architecture and blend weight were selected using internal and external development comparisons; fitted parameters use Datasets 0–2.",
        ordering="Dataset 2 levels specify within-position order, not calibrated site probabilities.",
    )
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
