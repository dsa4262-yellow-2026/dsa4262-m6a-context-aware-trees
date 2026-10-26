"""Package complete, unfiltered per-run predictions with portable reference tables."""
import argparse
import gzip
import io
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def export_csv(sites, output):
    """Export original identifiers, row order and float64 scores from Parquet."""
    with output.open("wb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as writer:
                writer.write("transcript_id,transcript_position,score\n")
                for batch in pq.ParquetFile(sites).iter_batches(
                    batch_size=100000,
                    columns=["transcript_id", "transcript_position", "score"],
                ):
                    batch.to_pandas().to_csv(
                        writer, index=False, header=False, lineterminator="\n"
                    )


def export_reference(reference, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for tid in sorted(
        set(reference["sequences"])
        | set(reference["geometry"])
        | set(reference["transcript_gene"])
    ):
        sequence = reference["sequences"].get(tid)
        geometry = reference["geometry"].get(tid, {})
        records.append(
            dict(
                transcript_id=tid,
                gene_id=reference["transcript_gene"].get(tid),
                sequence=sequence,
                transcript_length=None if sequence is None else len(sequence),
                exon_boundaries=None
                if "boundaries" not in geometry
                else np.asarray(geometry["boundaries"]).tolist(),
                **{
                    key: None
                    if key not in geometry
                    else [
                        dict(start=int(a), end=int(b))
                        for a, b in np.asarray(geometry[key], dtype=int).reshape(-1, 2)
                    ]
                    for key in ["CDS", "UTR", "stop_codon"]
                },
            )
        )
    schema = pa.schema(
        [
            ("transcript_id", pa.string()),
            ("gene_id", pa.string()),
            ("sequence", pa.string()),
            ("transcript_length", pa.int64()),
            ("exon_boundaries", pa.list_(pa.int64())),
            *[
                (
                    name,
                    pa.list_(pa.struct([("start", pa.int64()), ("end", pa.int64())])),
                )
                for name in ["CDS", "UTR", "stop_codon"]
            ],
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(records, schema=schema),
        output / "transcripts.parquet",
        compression="zstd",
    )
    for name in ["genes.csv", "transcript_aliases.csv"]:
        shutil.copyfile(HERE / name, output / name)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results_context_trees")
    parser.add_argument("--samples", type=Path, default=HERE / "samples.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=HERE.parent / "task 1 context aware trees/model/reference.joblib",
    )
    args = parser.parse_args()
    if (
        args.output.resolve() == args.results.resolve()
        or args.results.resolve().is_relative_to(args.output.resolve())
    ):
        parser.error(
            "The output must not contain or replace the source results directory."
        )
    inventory = pd.read_csv(args.samples, keep_default_na=False)
    completed = []
    for sample in inventory.itertuples(index=False):
        directory = args.results / sample.sample_id
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if metadata.get("model") != "context_aware_trees_v1" or (
            metadata.get("full_weight"),
            metadata.get("local_weight"),
        ) != (0.45, 0.55):
            raise ValueError("Every run must use the selected 45/55 model.")
        if (
            metadata["sites"] != sample.sites
            or pq.ParquetFile(directory / "sites.parquet").metadata.num_rows
            != sample.sites
        ):
            raise ValueError("A run is incomplete.")
        completed.append((sample, metadata))
    args.output.mkdir(parents=True, exist_ok=True)
    counts = export_reference(joblib.load(args.reference), args.output / "reference")
    rows = []
    for sample, metadata in completed:
        relative = Path("runs") / sample.sample_id
        directory = args.output / relative
        directory.mkdir(parents=True, exist_ok=True)
        source = args.results / sample.sample_id
        shutil.copyfile(source / "sites.parquet", directory / "sites.parquet")
        export_csv(source / "sites.parquet", directory / "predictions.csv.gz")
        row = {
            key: getattr(sample, key)
            for key in [
                "sample_id",
                "cell_line",
                "replicate",
                "run",
                "replicate_id",
                "protocol",
                "kit",
                "platform",
                "spike_in",
                "sites",
                "minimum_reads",
                "maximum_reads",
                "non_ENST_sites",
            ]
        }
        row.update(
            overlaps_data0=metadata["overlaps_data0"],
            total_read_observations=metadata["total_reads"],
            reference_unmatched_sites=metadata["reference_unmatched_sites"],
            predictions_path=(relative / "predictions.csv.gz").as_posix(),
            sites_path=(relative / "sites.parquet").as_posix(),
            signals_url=sample.url,
            index_url=sample.url.rsplit("/", 1)[0] + "/data.info",
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output / "samples.csv", index=False)
    first = completed[0][1]
    manifest = dict(
        format_version=1,
        name="SG-NEx pre-analysis predictions",
        model_name="Context-aware trees",
        model_id="context_aware_trees_v1",
        model_version="1",
        full_weight=0.45,
        local_weight=0.55,
        training_datasets=[0, 1, 2],
        observation_unit="One reported transcript position in one sequencing run.",
        primary_key=["sample_id", "transcript_id", "transcript_position"],
        counts=dict(
            runs=len(rows),
            cell_lines=int(inventory.cell_line.nunique()),
            biological_replicates=int(inventory.replicate_id.nunique()),
            run_site_observations=int(inventory.sites.sum()),
            reference_transcripts=counts,
        ),
        preservation=dict(
            all_reported_sites=True,
            original_row_order=True,
            original_identifiers=True,
            original_scores=True,
            coverage_filter=None,
            score_cutoff=None,
            human_only_filter=False,
            replicate_aggregation=None,
            identifier_conversion_applied=False,
        ),
        prediction_provenance=dict(
            device=first["device"],
            threads=first["threads"],
            batch_size=first["batch_size"],
            dependencies={
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "torch",
                    "xgboost",
                    "catboost",
                    "joblib",
                    "pyarrow",
                ]
            },
        ),
        reference=dict(
            assembly="GRCh38",
            ensembl_release=91,
            schema="transcript_geometry_v2",
            sources=[
                dict(
                    url="https://sg-nex-data.s3.ap-southeast-1.amazonaws.com/data/annotations/"
                    + path
                )
                for path in [
                    "gtf_file/Homo_sapiens.GRCh38.91.gtf",
                    "gtf_file/hg38_sequins_SIRV_ERCCs_longSIRVs_corrected.gtf",
                    "transcriptome_fasta/Homo_sapiens.GRCh38.cdna.ncrna.fa",
                    "transcriptome_fasta/hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa",
                ]
            ],
            extension_policy="Original human entries take priority; only absent synthetic transcript IDs are added. Exact input seven-mer required.",
            export="Exact sequence, gene and geometry values; null denotes an absent reference entry.",
            coordinate_convention="Zero-based positions and half-open transcript intervals.",
        ),
        interpretation="Prediction scores are binary-label model scores, not measured modification fractions. Hct116 replicate 3/run 1 overlaps Dataset 0 training measurements.",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        """# SG-NEx predictions — Context-aware trees v1

Complete unfiltered predictions from the selected model: **45% full CatBoost (41 inputs) + 55% local XGBoost (20 inputs)**, sharing one eight-output neural encoder. All fitted parameters use Datasets 0–2.

`samples.csv` lists 23 runs, 17 biological replicates and seven cell lines. Each `runs/<sample>/` folder contains the original three-column prediction CSV and its aligned Parquet table with sequence, coverage and reference metadata. All 28,670,080 reported run-site observations are retained, including low-coverage and synthetic-control entries.

Use the Parquet tables for analysis. The original transcript and position strings and row order are preserved. `reference/` supplies the fixed transcript sequences, geometry, gene names and verified alias map. Applying the alias map or a coverage filter is an analysis choice; neither changes these source files.

Scores are model predictions, not molecular modification fractions. Missing observations are not zeros. Technical runs belong to biological replicates; average eligible technical runs within a biological replicate before giving biological replicates equal cell weight. Hct116 replicate 3/run 1 overlaps Dataset 0 and is marked in the inventory.

Rebuild with `task 2/package_predictions.py` after completing `task 2/run.py`. No raw nanopore signal or training is needed to read this bundle.
""",
        encoding="utf-8",
    )
    (args.output / "requirements.txt").write_text(
        "pandas==2.1.4\npyarrow==21.0.0\n", encoding="utf-8"
    )
    print(
        f"Packaged {len(rows)} runs and {inventory.sites.sum():,} unfiltered observations.",
        flush=True,
    )


if __name__ == "__main__":
    main()
