"""Official HCT116_RNA002 weights, exact expected 20-read inference.

Reads original lecturer JSON: dwell/std/current at -1, 0, +1, with no log.
Release identities remain separate. mean_read_probability preserves ranking
when the equivalent site probability rounds to one. Uses the released assets in benchmark/m6anet_reference/.
Derived from GoekeLab/m6anet (MIT); see m6anet_reference/LICENSE.txt.
"""
import gzip
from itertools import product
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F

VERSION = "2.1.0"
SEVENMERS = ["".join(s) for s in product("ACGT", "AGT", "AG", "A", "C", "ACT", "ACGT")]
KMERS = sorted({s[j : j + 5] for s in SEVENMERS for j in range(3)})
KMER_TO_INT = {s: i for i, s in enumerate(KMERS)}


class PretrainedReadModel:
    def __init__(self, official, device="cpu"):
        root = Path(official)
        self.device = device
        self.state = torch.load(
            root / "rna002_hct116.pt", map_location=device, weights_only=True
        )
        norm = joblib.load(root / "rna002_hct116.joblib")
        self.preparation = {}
        for context in SEVENMERS:
            kmers = [context[j : j + 5] for j in range(3)]
            self.preparation[context] = (
                np.concatenate([norm[k][0] for k in kmers]),
                np.concatenate([norm[k][1] for k in kmers]),
                np.asarray([KMER_TO_INT[k] for k in kmers], dtype=np.int64),
            )

    def prepare(self, context, reads):
        mean, std, indices = self.preparation[context]
        # Native NanopolishDS normalizes float64 JSON numbers before float32.
        features = ((np.asarray(reads, dtype=np.float64) - mean) / std).astype(
            np.float32
        )
        return features, np.broadcast_to(indices, (len(features), 3)).copy()

    @torch.inference_mode()
    def read_probability(self, features, kmers):
        s = self.state
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(kmers, dtype=torch.long, device=self.device)
        embedding = F.embedding(
            ids, s["read_level_encoder.1.embedding_layer.weight"]
        ).reshape(-1, 6)
        x = torch.cat((x, embedding), dim=1)
        p = "read_level_encoder.3.layers."
        x = F.linear(x, s[p + "0.weight"], s[p + "0.bias"])
        x = F.batch_norm(
            x,
            s[p + "1.running_mean"],
            s[p + "1.running_var"],
            s[p + "1.weight"],
            s[p + "1.bias"],
            training=False,
            eps=1e-5,
        )
        x = F.relu(x)
        p = "read_level_encoder.4.layers.0."
        x = F.relu(F.linear(x, s[p + "weight"], s[p + "bias"]))
        p = "pooling_filter.probability_layer.0."
        return (
            torch.sigmoid(F.linear(x, s[p + "weight"], s[p + "bias"]))
            .flatten()
            .cpu()
            .numpy()
        )


def site_probability(mean_read_probability):
    mean = np.asarray(mean_read_probability, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_survival = 20 * np.log1p(-mean)
    return -np.expm1(log_survival), log_survival


def predict(data, official, device="cpu", batch_reads=32768, limit_sites=0):
    model = PretrainedReadModel(official, device)
    rows, means, xbuf, kbuf, lengths = [], [], [], [], []
    timing = []

    def flush():
        if not lengths:
            return
        p = model.read_probability(np.concatenate(xbuf), np.concatenate(kbuf)).astype(
            np.float64
        )
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        means.extend(np.add.reduceat(p, starts) / lengths)
        xbuf.clear()
        kbuf.clear()
        lengths.clear()

    for source in (0, 1):
        started, count, n_reads = time.perf_counter(), 0, 0
        path = (
            Path(data) / ("data" + str(source)) / ("dataset" + str(source) + ".json.gz")
        )
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                tx, positions = next(iter(record.items()))
                position, contexts = next(iter(positions.items()))
                context, reads = next(iter(contexts.items()))
                if sum(lengths) + len(reads) > batch_reads:
                    flush()
                x, k = model.prepare(context, reads)
                xbuf.append(x)
                kbuf.append(k)
                lengths.append(len(x))
                rows.append((source, tx, int(position), context, len(x)))
                count += 1
                n_reads += len(x)
                if limit_sites and count >= limit_sites:
                    break
        flush()
        timing.append(
            dict(
                source=source,
                sites=count,
                reads=n_reads,
                seconds=time.perf_counter() - started,
            )
        )
        print(json.dumps(timing[-1]), flush=True)
    prediction, log_survival = site_probability(means)
    result = {
        "source": np.asarray([r[0] for r in rows], dtype=np.int8),
        "transcript_id": np.asarray([r[1] for r in rows]),
        "transcript_position": np.asarray([r[2] for r in rows], dtype=np.int64),
        "kmer": np.asarray([r[3] for r in rows]),
        "n_reads": np.asarray([r[4] for r in rows], dtype=np.int32),
        "prediction": prediction,
        "mean_read_probability": np.asarray(means),
        "log_survival_20": log_survival,
    }
    details = dict(
        checkpoint="HCT116_RNA002",
        version=VERSION,
        inference="exact expectation of native sampling: 20 draws with replacement",
        ranking="mean_read_probability",
        device=device,
        timing=timing,
        saturated_sites=int(np.sum(prediction == 1)),
        unique_mean_scores=len(np.unique(means)),
        unique_site_scores=len(np.unique(prediction)),
        torch=torch.__version__,
        numpy=np.__version__,
    )
    result["details"] = np.asarray(json.dumps(details))
    return result
