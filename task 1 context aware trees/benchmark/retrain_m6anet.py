"""Same-gene-fold m6Anet architecture, fitted afresh on lecturer data0 + data1.

Prepare raw float64 reads once; no logged features or pretrained assets enter.
Defaults use the paper's five-fold-CV recipe: inner 25% gene validation,
60 epochs, batch 512, Adam AMSGrad and learning rate 0.0004. The v2.1.0 CLI
instead defaults to 50 epochs, ordinary Adam, batch 256. Here epoch selection
and inference use the exact expectation of deployed 20-read sampling with
replacement. The paper instead averages five 20-read validation bags sampled
without replacement. We freshly refit on outer training genes after selection;
the paper retains the selected inner-fit checkpoint. This is a same-fold
architecture comparison, not an exact reproduction of that training run.
Dependencies: numpy, pandas, scikit-learn, torch. Derived from GoekeLab/m6anet
v2.1.0 (MIT, Copyright (c) 2021 Goke Lab); retain its LICENSE when distributing.
"""
import gzip
from itertools import product
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

VERSION = "2.1.0"
SITE_COLUMNS = [
    "source",
    "transcript_id",
    "transcript_position",
    "gene_id",
    "label",
    "n_reads",
    "kmer",
    "fold",
]
CONTEXTS = ["".join(s) for s in product("ACGT", "AGT", "AG", "A", "C", "ACT", "ACGT")]
KMERS = sorted({s[j : j + 5] for s in CONTEXTS for j in range(3)})
ENCODING = {s: [KMERS.index(s[j : j + 5]) for j in range(3)] for s in CONTEXTS}


def site_arrays(sites):
    return {
        name: sites[name].to_numpy(
            dtype=str if name in ["transcript_id", "gene_id", "kmer"] else None
        )
        for name in SITE_COLUMNS
    }


def prepare(data, sites_path, cache):
    """Keep original channel units and precision, with per-site raw moments."""
    sites = pd.read_csv(sites_path)
    cache.mkdir(parents=True, exist_ok=True)
    counts = sites.n_reads.to_numpy(dtype=np.int64)
    offsets = np.r_[0, counts.cumsum()]
    raw = np.lib.format.open_memmap(
        cache / "raw.npy", mode="w+", dtype=np.float64, shape=(int(offsets[-1]), 9)
    )
    total, square = np.empty((len(sites), 9)), np.empty((len(sites), 9))
    expected = iter(
        sites[
            ["source", "transcript_id", "transcript_position", "kmer", "n_reads"]
        ].itertuples(index=False, name=None)
    )
    row = 0
    for source in (0, 1):
        path = data / f"data{source}" / f"dataset{source}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                tx, positions = next(iter(json.loads(line).items()))
                position, contexts = next(iter(positions.items()))
                context, values = next(iter(contexts.items()))
                if (source, tx, int(position), context, len(values)) != next(expected):
                    raise ValueError(
                        "Raw site identity differs from the supplied fold table."
                    )
                values = np.asarray(values, dtype=np.float64)
                raw[offsets[row] : offsets[row + 1]] = values
                total[row], square[row] = values.sum(0), np.square(values).sum(0)
                row += 1
        print(json.dumps(dict(prepared_source=source, sites_written=row)), flush=True)
    if row != len(sites):
        raise ValueError("Raw input ended before all declared sites were read.")
    raw.flush()
    np.savez_compressed(
        cache / "moments.npz", total=total, square=square, **site_arrays(sites)
    )


def physical_scaler(total, square, counts, codes, rows):
    """Pool each physical 5-mer across its occurrences in all three windows."""
    n, s, q = np.zeros(66), np.zeros((66, 3)), np.zeros((66, 3))
    for window in range(3):
        ids, columns = codes[rows, window], slice(3 * window, 3 * window + 3)
        np.add.at(n, ids, counts[rows])
        np.add.at(s, ids, total[rows, columns])
        np.add.at(q, ids, square[rows, columns])
    if np.any(n == 0):
        raise ValueError("Training genes do not represent all 66 physical 5-mers.")
    mean = s / n[:, None]
    scale = np.sqrt(q / n[:, None] - mean**2)
    return mean, scale


class ReadModel(nn.Module):
    """Exact official 66x2 embedding and 15 -> 150/BN/ReLU -> 32/ReLU -> 1."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(66, 2)
        self.layers = nn.Sequential(
            nn.Linear(15, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values, codes):
        encoded = self.embedding(codes).reshape(-1, 6)
        return self.layers(torch.cat((values.reshape(-1, 9), encoded), dim=1)).flatten()


def noisy_or_loss(logits, labels):
    # H = -log survival. Float64 avoids float32 product/sigmoid saturation.
    hazard = F.softplus(logits.double()).reshape(-1, 20).sum(1)
    log_positive = torch.log(-torch.expm1(-hazard))
    return torch.where(labels.bool(), -log_positive, hazard).mean()


class Bags:
    def __init__(self, raw, sites, scaler):
        self.raw = raw
        self.counts = sites.n_reads.to_numpy(dtype=np.int64)
        self.offsets = np.r_[0, self.counts.cumsum()]
        self.codes = np.asarray([ENCODING[s] for s in sites.kmer], dtype=np.int64)
        self.labels = sites.label.to_numpy(dtype=np.float32)
        self.mean, self.scale = scaler
        if self.counts.min() < 20:
            raise ValueError(
                "This official 20-read training objective needs >=20 reads per site."
            )

    def normalized(self, indices, codes, device):
        values = self.raw[indices].reshape(-1, 3, 3)
        values = ((values - self.mean[codes]) / self.scale[codes]).reshape(-1, 9)
        return (
            torch.as_tensor(values, dtype=torch.float32, device=device),
            torch.as_tensor(codes, dtype=torch.long, device=device),
        )

    def sampled(self, rows, rng, device):
        indices = np.concatenate(
            [
                self.offsets[i] + rng.choice(self.counts[i], 20, replace=False)
                for i in rows
            ]
        )
        codes = np.repeat(self.codes[rows], 20, axis=0)
        x, k = self.normalized(indices, codes, device)
        return x, k, torch.as_tensor(self.labels[rows], device=device)


@torch.inference_mode()
def predict(model, data, rows, device, batch_sites=128):
    model.eval()
    means = np.empty(len(rows))
    for start in range(0, len(rows), batch_sites):
        batch = rows[start : start + batch_sites]
        counts = data.counts[batch]
        indices = np.concatenate(
            [np.arange(data.offsets[i], data.offsets[i + 1]) for i in batch]
        )
        codes = np.repeat(data.codes[batch], counts, axis=0)
        x, k = data.normalized(indices, codes, device)
        probability = model(x, k).sigmoid().cpu().numpy().astype(np.float64)
        means[start : start + len(batch)] = (
            np.add.reduceat(probability, np.r_[0, counts.cumsum()[:-1]]) / counts
        )
    with np.errstate(divide="ignore"):
        log_survival = 20 * np.log1p(-means)
    return means, -np.expm1(log_survival), log_survival


def train(args, data, rows, validation=None, epochs=None, stage="select"):
    # Fresh model and independent sampling stream for inner training and refit.
    torch.manual_seed(args.seed + args.fold)
    rng = np.random.default_rng(args.seed + args.fold)
    model = ReadModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    negative, positive = rows[data.labels[rows] == 0], rows[data.labels[rows] == 1]
    history, best_score, best_epoch = [], -np.inf, 0
    for epoch in range(1, (epochs or args.epochs) + 1):
        started = time.perf_counter()
        order = np.r_[negative, rng.choice(positive, len(negative), replace=True)]
        rng.shuffle(order)
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_sites):
            batch = order[start : start + args.batch_sites]
            x, k, y = data.sampled(batch, rng, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = noisy_or_loss(model(x, k), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        record = dict(stage=stage, epoch=epoch, loss=loss_sum / len(order))
        if validation is not None:
            mean, _, _ = predict(model, data, validation, args.device)
            precision, recall, _ = precision_recall_curve(data.labels[validation], mean)
            score = float(auc(recall, precision))
            record["validation_pr_auc"] = score
            if score > best_score:
                best_score, best_epoch = score, epoch
        record["seconds"] = time.perf_counter() - started
        history.append(record)
        print(json.dumps(record), flush=True)
    return model, best_epoch, history


def run(args):
    torch.set_num_threads(args.threads)
    sites = pd.read_csv(args.sites)
    raw = np.load(args.cache / "raw.npy", mmap_mode="r")
    with np.load(args.cache / "moments.npz") as moments:
        if any(
            not np.array_equal(moments[name], values)
            for name, values in site_arrays(sites).items()
        ):
            raise ValueError("Raw cache and current site/fold table differ.")
        total, square = moments["total"], moments["square"]
    counts = sites.n_reads.to_numpy(dtype=np.int64)
    codes = np.asarray([ENCODING[s] for s in sites.kmer], dtype=np.int64)
    outer_train = np.flatnonzero(sites.fold.to_numpy() != args.fold)
    outer_test = np.flatnonzero(sites.fold.to_numpy() == args.fold)
    genes = sites.gene_id.to_numpy()
    if np.intersect1d(genes[outer_train], genes[outer_test]).size:
        raise ValueError("Outer training and test genes overlap.")
    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=args.seed + args.fold
    )
    fit, val = next(inner.split(outer_train, groups=genes[outer_train]))
    fit, val = outer_train[fit], outer_train[val]
    scaler = physical_scaler(total, square, counts, codes, fit)
    _, selected_epoch, history = train(args, Bags(raw, sites, scaler), fit, val)
    scaler = physical_scaler(total, square, counts, codes, outer_train)
    data = Bags(raw, sites, scaler)
    model, _, refit_history = train(
        args, data, outer_train, epochs=selected_epoch, stage="refit"
    )
    means, prediction, log_survival = predict(model, data, outer_test, args.device)
    protocol = dict(
        official_version=VERSION,
        seed=args.seed,
        fold=args.fold,
        train_sites=len(outer_train),
        test_sites=len(outer_test),
        inner_train_sites=len(fit),
        inner_validation_sites=len(val),
        selected_epoch=selected_epoch,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        amsgrad=args.amsgrad,
        batch_sites=args.batch_sites,
        normalization="Raw units, read-weighted physical 5-mer mean/SD, training genes only",
        training="Balanced site oversampling; 20 reads without replacement; noisy-OR BCE",
        recipe="Paper five-fold-CV defaults: 25% inner gene validation, 60 epochs, batch512, Adam AMSGrad lr0.0004",
        selection="Inner gene PR-AUC on exact expectation of deployed 20-read sampling with replacement",
        paper_validation="Five 20-read bags without replacement, averaged; selected inner-fit checkpoint retained",
        refit="Fresh random initialization and scaling on all outer training genes, for the selected epoch count",
        scope="Same-fold official-architecture retraining, not exact reproduction of the paper training run",
        torch=torch.__version__,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"retrained_{args.fold}.npz",
        test_idx=outer_test,
        prediction=prediction,
        mean_read_probability=means,
        log_survival_20=log_survival,
        **site_arrays(sites.iloc[outer_test]),
        protocol=np.asarray(json.dumps(protocol)),
        history=np.asarray(json.dumps(history + refit_history)),
    )
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            mean=scaler[0],
            scale=scaler[1],
            kmers=KMERS,
            protocol=protocol,
        ),
        args.output / f"retrained_{args.fold}.pt",
    )
