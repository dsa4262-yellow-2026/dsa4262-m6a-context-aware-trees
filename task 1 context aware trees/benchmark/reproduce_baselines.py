"""Reproduce the three all-site baselines on the final model's gene folds."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from context_trees.data import attach_labels
from evaluate import KEYS, load_results
from expected_m6anet import predict as released_prediction
import retrain_m6anet


def prepare(data, results, work, output, threads):
    sites, _ = load_results(results)
    sites = sites[KEYS + ["fold"]]
    reference = joblib.load(HERE.parent / "model/reference.joblib")
    for source, label_name in [(0, "data.info.labelled"), (1, "data.info")]:
        selected = sites.loc[sites.source.eq(source)].reset_index(drop=True)
        actual = attach_labels(
            selected[["transcript_id", "transcript_position", "n_reads"]],
            data / f"data{source}" / label_name,
            reference,
        )
        for name in ["gene_id", "label"]:
            if not np.array_equal(selected[name], actual[name]):
                raise ValueError(
                    f"Dataset {source}: original {name} differs from the held-out records."
                )
    if (work / "raw.npy").exists():
        raise ValueError("Choose a new work directory for the raw-read preparation.")
    sites.to_csv(work / "sites.csv", index=False)
    retrain_m6anet.prepare(data, work / "sites.csv", work)
    with np.load(work / "moments.npz") as moments:
        # The original baseline stores the raw channel means in float32.
        means = (moments["total"] / sites.n_reads.to_numpy()[:, None]).astype(
            np.float32
        )
    scores = np.empty(len(sites))
    labels, folds = sites.label.to_numpy(), sites.fold.to_numpy()
    for fold in range(5):
        fitting, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        baseline = make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            LogisticRegression(C=1, max_iter=3000, random_state=4262),
        )
        with threadpool_limits(limits=threads):
            baseline.fit(means[fitting], labels[fitting])
            scores[held] = baseline.predict_proba(means[held])[:, 1]
    np.savez_compressed(output / "logistic.npz", prediction=scores)
    print(
        f"Prepared {len(sites):,} original sites and fitted the logistic baseline.",
        flush=True,
    )


def collect(work, output):
    sites = pd.read_csv(work / "sites.csv")
    with np.load(output / "logistic.npz") as saved:
        logistic = saved["prediction"]
    with np.load(output / "released.npz") as saved:
        for name in [
            "source",
            "transcript_id",
            "transcript_position",
            "kmer",
            "n_reads",
        ]:
            if not np.array_equal(sites[name], saved[name]):
                raise ValueError(f"Released m6Anet site alignment differs: {name}.")
        released = saved["prediction"]
        released_protocol = json.loads(saved["details"].item())
    retrained = np.full(len(sites), np.nan)
    protocols = []
    for fold in range(5):
        with np.load(output / f"retrained_{fold}.npz") as saved:
            held = np.flatnonzero(sites.fold.to_numpy() == fold)
            if not np.array_equal(held, saved["test_idx"]):
                raise ValueError(
                    f"Retrained m6Anet fold {fold} has different held-out rows."
                )
            protocol = json.loads(saved["protocol"].item())
            if any(
                not np.array_equal(saved[name], values)
                for name, values in retrain_m6anet.site_arrays(sites.iloc[held]).items()
            ):
                raise ValueError("Retrained m6Anet used a different site/fold table.")
            retrained[held] = saved["prediction"]
            protocols.append(protocol)
    scores = np.column_stack([logistic, released, retrained])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(
            "Every baseline must score every site with a finite value in [0, 1]."
        )
    arrays = {
        name: sites[name].to_numpy(
            dtype=str if name in ["transcript_id", "gene_id", "kmer"] else None
        )
        for name in KEYS + ["fold"]
    }
    np.savez_compressed(
        output / "baselines.npz",
        **arrays,
        prediction=scores,
        model_names=np.array(["logistic", "m6anet_released", "m6anet_retrained"]),
    )
    protocol = dict(
        observations=len(sites),
        genes=int(sites.gene_id.nunique()),
        alignment=KEYS + ["fold"],
        logistic=dict(
            features="Nine raw channel means, stored as float32.",
            preprocessing="Training-fold constant removal and standardization.",
            C=1,
            max_iter=3000,
            seed=4262,
        ),
        m6anet_released=released_protocol,
        m6anet_retrained=protocols,
        inference="1 - (1 - mean_read_probability)**20; exact expectation of 20 draws with replacement.",
    )
    (output / "baseline_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(f'Wrote {output / "baselines.npz"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Original data0/ and data1/ inputs; needed by prepare/released.",
    )
    parser.add_argument("--results", type=Path, default=HERE.parent / "results")
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Raw-read scratch; allow 10 GB free.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Baseline scores, checkpoints and protocol.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "released", "retrain", "collect"],
        default="all",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="For --stage retrain; omit to fit all five folds.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.stage in ["all", "prepare", "released"] and args.data_dir is None:
        parser.error(
            "--data-dir is required for original-signal preparation or released inference."
        )
    if args.fold is not None and args.stage != "retrain":
        parser.error(
            "--fold applies only to --stage retrain; collect requires all five folds."
        )
    if args.threads < 1:
        parser.error("--threads must be positive.")
    for folder in [args.work_dir, args.output]:
        folder.mkdir(parents=True, exist_ok=True)
    if args.work_dir.resolve() == args.output.resolve():
        parser.error("Keep raw-read scratch separate from baseline output.")
    if args.stage in ["all", "prepare"] and any(args.output.iterdir()):
        parser.error("Start with a new output directory.")
    torch.set_num_threads(args.threads)
    if args.stage in ["all", "prepare"]:
        prepare(args.data_dir, args.results, args.work_dir, args.output, args.threads)
    if args.stage in ["all", "released"]:
        prediction = released_prediction(
            args.data_dir, HERE / "m6anet_reference", args.device
        )
        np.savez_compressed(args.output / "released.npz", **prediction)
    if args.stage in ["all", "retrain"]:
        for fold in range(5) if args.fold is None else [args.fold]:
            print(
                f"Retraining m6Anet architecture for held-out fold {fold}.", flush=True
            )
            retrain_m6anet.run(
                SimpleNamespace(
                    sites=args.work_dir / "sites.csv",
                    cache=args.work_dir,
                    output=args.output,
                    fold=fold,
                    seed=4262,
                    epochs=60,
                    batch_sites=512,
                    lr=0.0004,
                    amsgrad=True,
                    device=args.device,
                    threads=args.threads,
                )
            )
    if args.stage in ["all", "collect"]:
        collect(args.work_dir, args.output)


if __name__ == "__main__":
    main()
