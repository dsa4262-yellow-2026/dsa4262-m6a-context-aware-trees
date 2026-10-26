"""Score complete SG-NEx runs with context-aware trees and align their site metadata."""
import argparse
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
KEY = ["transcript_id", "transcript_position"]
MODEL = "context_aware_trees_v1"
BATCH = 100000


def prediction_settings(model, device, threads):
    manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
    required = {"encoder.pt", "xgboost.ubj", "catboost.cbm", "reference.joblib"}
    if manifest["name"] != "Context-aware trees" or not all(
        (model / name).is_file() for name in required
    ):
        raise ValueError("Select the saved context-aware trees model directory.")
    if (manifest.get("full_weight"), manifest.get("local_weight")) != (0.45, 0.55):
        raise ValueError("Use the selected 45/55 model for this release.")
    return dict(
        device=device,
        threads=threads,
        batch_size=128,
        full_weight=0.45,
        local_weight=0.55,
        model_version="1",
    )


def check_completed(directory, signals, info, saved):
    index = pd.read_csv(info, dtype={key: str for key in KEY})
    stored = pq.read_table(
        directory / "sites.parquet", columns=KEY + ["n_reads", "score", "model"]
    ).to_pandas()
    if (
        len(index) != saved["sites"]
        or index.empty
        or int(index.end.iloc[-1]) != signals.stat().st_size
    ):
        raise ValueError("Input index does not cover the complete recorded sample.")
    for name in KEY + ["n_reads"]:
        if not np.array_equal(index[name], stored[name]):
            raise ValueError(f"Completed results and input index differ: {name}.")
    if (
        not np.isfinite(stored.score).all()
        or not stored.score.between(0, 1).all()
        or not stored.model.eq(MODEL).all()
    ):
        raise ValueError(
            "Completed predictions have invalid scores or a different model."
        )


def export_sites(signals, info, predictions, output, reference, iter_sites):
    """Align original JSON, byte index and CSV before attaching reference metadata."""
    sites = iter(iter_sites(signals))
    indices = iter(pd.read_csv(info, dtype={k: str for k in KEY}, chunksize=BATCH))
    schema = pa.schema(
        [
            (KEY[0], pa.string()),
            (KEY[1], pa.string()),
            ("kmer", pa.string()),
            ("n_reads", pa.int64()),
            ("gene_id", pa.string()),
            ("reference_matched", pa.bool_()),
            ("score", pa.float64()),
            ("model", pa.string()),
        ]
    )
    counts = dict(
        sites=0, sites_at_least_20_reads=0, total_reads=0, reference_unmatched_sites=0
    )
    last_end = 0
    temporary = output.with_suffix(".tmp")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for scores in pd.read_csv(
            predictions,
            dtype={k: str for k in KEY},
            float_precision="round_trip",
            chunksize=BATCH,
        ):
            if list(scores) != KEY + ["score"]:
                raise ValueError(
                    "Prediction CSV must have exactly the three required columns."
                )
            index = next(indices, None)
            raw = list(islice(sites, len(scores)))
            if index is None or len(raw) != len(scores) or len(index) != len(scores):
                raise ValueError("JSON, index and prediction row counts differ.")
            rows = []
            for transcript, position, kmer, reads in raw:
                key, p = transcript.split(".")[0], int(position)
                sequence = reference["sequences"].get(key, "")
                matched = 3 <= p < len(sequence) - 3 and sequence[p - 3 : p + 4] == kmer
                rows.append(
                    (
                        transcript,
                        position,
                        kmer,
                        len(reads),
                        reference["transcript_gene"].get(key, ""),
                        matched,
                    )
                )
            frame = pd.DataFrame(
                rows, columns=KEY + ["kmer", "n_reads", "gene_id", "reference_matched"]
            )
            for column in KEY:
                if not np.array_equal(
                    frame[column], scores[column]
                ) or not np.array_equal(frame[column], index[column]):
                    raise ValueError(
                        f"Original {column} keys/order differ across JSON, index and CSV."
                    )
            if not np.array_equal(frame.n_reads, index.n_reads):
                raise ValueError("Original read counts differ from the index.")
            starts, ends = index.start.to_numpy(), index.end.to_numpy()
            if (
                starts[0] != last_end
                or np.any(ends <= starts)
                or not np.array_equal(starts[1:], ends[:-1])
            ):
                raise ValueError("Index byte offsets are not contiguous and ordered.")
            last_end = int(ends[-1])
            values = scores.score.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Predictions must be finite and in [0,1].")
            frame["score"], frame["model"] = values, MODEL
            writer.write_table(
                pa.Table.from_pandas(
                    frame, schema=schema, preserve_index=False
                ).replace_schema_metadata(None)
            )
            counts["sites"] += len(frame)
            counts["sites_at_least_20_reads"] += int(frame.n_reads.ge(20).sum())
            counts["total_reads"] += int(frame.n_reads.sum())
            counts["reference_unmatched_sites"] += int((~frame.reference_matched).sum())
    if (
        next(indices, None) is not None
        or next(sites, None) is not None
        or last_end != signals.stat().st_size
    ):
        raise ValueError("Index and predictions do not cover the complete JSON.")
    parquet = iter(
        pq.ParquetFile(temporary).iter_batches(
            batch_size=BATCH, columns=KEY + ["score"]
        )
    )
    for scores in pd.read_csv(
        predictions,
        dtype={k: str for k in KEY},
        float_precision="round_trip",
        chunksize=BATCH,
    ):
        stored = next(parquet).to_pandas()
        if not np.array_equal(
            stored[KEY].to_numpy(), scores[KEY].to_numpy()
        ) or not np.array_equal(stored.score, scores.score):
            raise ValueError("CSV and Parquet keys/scores differ.")
    if next(parquet, None) is not None:
        raise ValueError("Parquet has extra rows.")
    temporary.replace(output)
    counts["sites_below_20_reads"] = counts["sites"] - counts["sites_at_least_20_reads"]
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing the inventory signal/index paths.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Saved context-aware trees model; defaults to the bundled final fit.",
    )
    parser.add_argument("--samples", type=Path, default=HERE / "samples.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results_context_trees")
    parser.add_argument(
        "--sample", nargs="+", help="Selected inventory sample IDs; omit for all runs."
    )
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs when using the same inputs and fitted model.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    predictor = HERE.parent / "task 1 context aware trees"
    model = (args.model or predictor / "model").resolve()
    inventory = pd.read_csv(args.samples).fillna("")
    required = [
        "sample_id",
        "cell_line",
        "replicate",
        "run",
        "replicate_id",
        "signals",
        "info",
        "sites",
        "sites_at_least_20_reads",
    ]
    if (
        not set(required) <= set(inventory)
        or inventory.empty
        or inventory.sample_id.duplicated().any()
    ):
        raise ValueError(
            "Supply a nonempty inventory with the documented columns and unique sample IDs."
        )
    if inventory.groupby("replicate_id").cell_line.nunique().gt(1).any():
        raise ValueError("A biological replicate belongs to more than one cell line.")
    if args.sample:
        missing = set(args.sample) - set(inventory.sample_id)
        if missing:
            raise ValueError(f"Samples absent from inventory: {sorted(missing)}")
        inventory = inventory.loc[inventory.sample_id.isin(args.sample)]
    settings = prediction_settings(model, args.device, args.threads)
    sys.path.insert(0, str(predictor))
    from context_trees.features import iter_sites

    reference = joblib.load(model / "reference.joblib")
    for sample in inventory.itertuples():
        signals, info = args.data_root / sample.signals, args.data_root / sample.info
        sample_identity = dict(
            sample_id=sample.sample_id,
            cell_line=sample.cell_line,
            replicate=int(sample.replicate),
            run=int(sample.run),
            replicate_id=sample.replicate_id,
        )
        directory = args.output / sample.sample_id
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "run.json"
        files = ["predictions.csv.gz", "sites.parquet"]
        if marker.exists():
            if not args.resume:
                raise ValueError(
                    f"{sample.sample_id}: results already exist. Use --resume for the same inputs/model, or choose a new output folder."
                )
            saved = json.loads(marker.read_text(encoding="utf-8"))
            same = saved.get("model") == MODEL and all(
                saved.get(k) == v for k, v in {**sample_identity, **settings}.items()
            )
            if not same:
                raise ValueError(
                    f"{sample.sample_id}: sample or execution settings differ; choose a new output folder."
                )
            check_completed(directory, signals, info, saved)
            print(f"{sample.sample_id}: complete, skipped.", flush=True)
            continue
        command = [
            sys.executable,
            "-B",
            str(predictor / "predict.py"),
            "--data",
            str(signals.resolve()),
            "--model",
            str(model),
            "--output",
            str((directory / files[0]).resolve()),
            "--threads",
            str(args.threads),
            "--device",
            args.device,
            "--batch-size",
            "128",
        ]
        scratch = args.work_dir / sample.sample_id if args.work_dir else None
        if scratch is not None:
            scratch.mkdir(parents=True, exist_ok=True)
            command += ["--work-dir", str(scratch.resolve())]
        print(
            f"{sample.sample_id}: predicting {sample.sites:,} complete-sample sites.",
            flush=True,
        )
        started, peak, scratch_peak = time.monotonic(), 0, 0
        process = subprocess.Popen(command)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if scratch is not None:
                scratch_peak = max(
                    scratch_peak,
                    sum(
                        p.stat().st_size
                        for p in scratch.glob("context_trees_*/reads.bin")
                        if p.exists()
                    ),
                )
            time.sleep(2)
        if process.wait() != 0:
            raise RuntimeError(
                f"{sample.sample_id}: prediction failed; no completion record written."
            )
        counts = export_sites(
            signals,
            info,
            directory / files[0],
            directory / files[1],
            reference,
            iter_sites,
        )
        if (
            counts["sites"] != sample.sites
            or counts["sites_at_least_20_reads"] != sample.sites_at_least_20_reads
        ):
            raise ValueError(
                f"{sample.sample_id}: output counts differ from the inventory."
            )
        saved = dict(
            **sample_identity,
            model=MODEL,
            **settings,
            **counts,
            elapsed_seconds=time.monotonic() - started,
            peak_process_rss_bytes=peak,
            scratch_reads_peak_bytes=scratch_peak,
            prediction_bytes=(directory / files[0]).stat().st_size,
            analysis_bytes=(directory / files[1]).stat().st_size,
            overlaps_data0=sample.sample_id == "SGNex_Hct116_directRNA_replicate3_run1",
        )
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker)
        print(
            f'{sample.sample_id}: validated {counts["sites"]:,} rows in {saved["elapsed_seconds"]:.1f}s; peak prediction memory {peak/2**30:.2f} GiB.',
            flush=True,
        )


if __name__ == "__main__":
    main()
