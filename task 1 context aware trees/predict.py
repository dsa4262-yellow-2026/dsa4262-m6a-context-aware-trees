"""Score a complete unlabelled JSON or JSON.gz file with the fitted two-stage model."""
import argparse
import gzip
import io
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import torch

from context_trees.data import close_reads, prepare_sites
from context_trees.model import (
    load_model,
    site_matrix,
    tree_predictions,
    combine_predictions,
)


def predict_file(
    signal_path,
    output,
    model_directory,
    device="cpu",
    threads=3,
    batch_size=128,
    work_dir=None,
    reference_path=None,
):
    signal_path, output, model_directory = (
        Path(signal_path),
        Path(output),
        Path(model_directory),
    )
    if signal_path.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ.")
    if output.resolve().is_relative_to(model_directory.resolve()):
        raise ValueError("Write predictions outside the fitted model directory.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use CPU execution.")
    network, scalers, tree, cat = load_model(model_directory, threads)
    reference = joblib.load(
        reference_path
        if reference_path is not None
        else model_directory / "reference.joblib"
    )
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="context_trees_", dir=work_dir) as scratch:
        prepared = prepare_sites([(0, signal_path, None)], scratch, reference)
        try:
            values = site_matrix(
                network, prepared, scalers, device=device, batch_size=batch_size
            )
            probability = combine_predictions(tree_predictions(tree, cat, values))
            sites = prepared["sites"]
            result = pd.DataFrame(
                dict(
                    transcript_id=sites.transcript_id,
                    transcript_position=sites.position_key,
                    score=probability,
                )
            )
        finally:
            close_reads(prepared["reads"])
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="prediction_", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        if output.suffix == ".gz":
            with open(temporary, "wb") as stream:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=stream, mtime=0
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed, encoding="utf-8", newline=""
                    ) as writer:
                        result.to_csv(writer, index=False, lineterminator="\n")
        else:
            result.to_csv(temporary, index=False, lineterminator="\n")
        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Complete sample in the JSON Lines input format described in README.md; gzip accepted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV or CSV.gz with transcript_id,transcript_position,score.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
        help="Saved model folder; defaults to the bundled fit.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--threads", type=int, default=3, help="CPU worker threads (default: 3)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Maximum sites per neural batch (default: 128). All reads are retained.",
    )
    parser.add_argument(
        "--work-dir", type=Path, help="Scratch directory for temporary read arrays."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Override the included human annotation for another species.",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Threads and batch size must be positive.")
    result = predict_file(
        args.data,
        args.output,
        args.model,
        args.device,
        args.threads,
        args.batch_size,
        args.work_dir,
        args.reference,
    )
    print(f"Wrote {len(result):,} scores to {args.output.name}.")


if __name__ == "__main__":
    main()
