"""Compare cell lines on shared positions and reserve replicates for confirmation."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
ALIASES = (
    pd.read_csv(HERE / "transcript_aliases.csv", dtype=str)
    .set_index("input_transcript_id")
    .transcript_id
)
if not ALIASES.index.is_unique or not ALIASES.is_unique:
    raise ValueError("The verified transcript alias map must be one-to-one.")
KEY = ["transcript_id", "transcript_position"]
COLUMNS = KEY + ["kmer", "n_reads", "gene_id", "reference_matched", "score", "model"]
CANDIDATE_COLUMNS = [
    "gene_id",
    "transcript_id",
    "matched_supported_sites",
    "mean_abs_delta",
    "mean_delta_b_minus_a",
    "discovery_delta",
    "confirmation_delta",
    "positions_agreeing",
    "maximum_depth_ratio",
    "cell_a",
    "cell_b",
    "discovery_replicate_a",
    "discovery_replicate_b",
]


def load_sites(path, min_reads=20, transcripts=None, human_only=True):
    condition = ds.field("n_reads") >= min_reads
    if transcripts is not None:
        wanted = set(transcripts)
        wanted.update(ALIASES.index[ALIASES.isin(wanted)])
        condition = condition & ds.field("transcript_id").isin(sorted(wanted))
    frame = (
        ds.dataset(path, format="parquet")
        .to_table(columns=COLUMNS, filter=condition)
        .to_pandas()
    )
    frame["input_transcript_id"] = frame.transcript_id
    frame["transcript_id"] = frame.transcript_id.map(ALIASES).fillna(
        frame.transcript_id
    )
    frame["input_transcript_position"] = frame.transcript_position.astype(str)
    frame["transcript_position"] = frame.transcript_position.astype("int64").astype(str)
    if human_only:
        frame = frame.loc[frame.transcript_id.str.startswith("ENST")].copy()
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate analysis coordinates in {Path(path).parent.name}.")
    return frame


def aggregate_replicate(frames, sample_rows):
    """Each available eligible technical run has equal weight at each coordinate."""
    combined = pd.concat(
        [
            frame.assign(sample_id=sample.sample_id)
            for frame, sample in zip(frames, sample_rows.itertuples())
        ],
        ignore_index=True,
    )
    grouped = combined.groupby(KEY, sort=False, observed=True)
    if grouped.kmer.nunique().gt(1).any():
        raise ValueError(
            "Technical runs disagree on the seven-mer of a matched coordinate."
        )
    result = grouped.agg(
        gene_id=("gene_id", "first"),
        kmer=("kmer", "first"),
        score=("score", "mean"),
        n_reads=("n_reads", "mean"),
        n_runs=("sample_id", "nunique"),
        run_score_min=("score", "min"),
        run_score_max=("score", "max"),
        reference_matched_fraction=("reference_matched", "mean"),
    ).reset_index()
    result["cell_line"] = sample_rows.cell_line.iloc[0]
    result["replicate_id"] = sample_rows.replicate_id.iloc[0]
    result["available_runs"] = len(sample_rows)
    return result


def aggregate_cell(replicates):
    """Each observed biological replicate has equal weight, regardless of runs."""
    grouped = replicates.groupby(KEY, sort=False, observed=True)
    if grouped.kmer.nunique().gt(1).any():
        raise ValueError("Biological replicates disagree on a matched seven-mer.")
    result = grouped.agg(
        gene_id=("gene_id", "first"),
        kmer=("kmer", "first"),
        score=("score", "mean"),
        n_reads=("n_reads", "mean"),
        n_replicates=("replicate_id", "nunique"),
        n_runs=("n_runs", "sum"),
        replicate_score_sd=("score", "std"),
        reference_matched_fraction=("reference_matched_fraction", "mean"),
    ).reset_index()
    result["cell_line"] = replicates.cell_line.iloc[0]
    result["available_replicates"] = replicates.replicate_id.nunique()
    return result


def matched_sites(left, right):
    joined = left.merge(
        right, on=KEY, how="inner", suffixes=("_a", "_b"), validate="one_to_one"
    )
    mismatch = int(joined.kmer_a.ne(joined.kmer_b).sum())
    return joined.loc[joined.kmer_a.eq(joined.kmer_b)].copy(), mismatch


def replicate_details(replicates, order, width):
    """Keep biological replicates distinct; the first inventory replicate is discovery."""
    scores = replicates.pivot(
        index=KEY, columns="replicate_id", values="score"
    ).reindex(columns=order)
    depth = replicates.pivot(
        index=KEY, columns="replicate_id", values="n_reads"
    ).reindex(columns=order)
    result = pd.DataFrame(index=scores.index)
    total, count = np.zeros(len(scores)), np.zeros(len(scores))
    for a, b in combinations(order, 2):
        difference = (scores[a] - scores[b]).abs()
        total += difference.fillna(0).to_numpy()
        count += difference.notna().to_numpy()
    result["within_mae"] = np.divide(
        total, count, out=np.full(len(scores), np.nan), where=count > 0
    )
    result["replicate_min"] = scores.min(axis=1)
    result["replicate_max"] = scores.max(axis=1)
    result["depth_min"] = depth.min(axis=1)
    result["depth_max"] = depth.max(axis=1)
    result["discovery_score"] = scores.iloc[:, 0]
    result["confirmation_score"] = scores.iloc[:, 1:].mean(axis=1)
    for i in range(width):
        result[f"replicate_score_{i}"] = scores.iloc[:, i] if i < len(order) else np.nan
    return result.reset_index()


def supported_comparison(joined):
    """Use two biological replicates per cell and comparable depth across all of them."""
    ratio = joined[["depth_max_a", "depth_max_b"]].max(axis=1) / joined[
        ["depth_min_a", "depth_min_b"]
    ].min(axis=1)
    use = joined.n_replicates_a.ge(2) & joined.n_replicates_b.ge(2) & ratio.le(2)
    result = joined.loc[use].copy()
    result["replicate_depth_ratio"] = ratio.loc[use]
    total, count = np.zeros(len(result)), np.zeros(len(result))
    a_columns = [
        c
        for c in result
        if c.startswith("replicate_score_") and c.endswith("_a") and c[16:-2].isdigit()
    ]
    b_columns = [
        c
        for c in result
        if c.startswith("replicate_score_") and c.endswith("_b") and c[16:-2].isdigit()
    ]
    for a in a_columns:
        for b in b_columns:
            difference = (result[a] - result[b]).abs()
            total += difference.fillna(0).to_numpy()
            count += difference.notna().to_numpy()
    result["between_replicate_mae"] = np.divide(
        total, count, out=np.full(len(result), np.nan), where=count > 0
    )
    result["within_replicate_mae"] = (result.within_mae_a + result.within_mae_b) / 2
    result["delta_b_minus_a"] = result.score_b - result.score_a
    result["replicate_ranges_separated"] = (
        result.replicate_min_b > result.replicate_max_a
    ) | (result.replicate_min_a > result.replicate_max_b)
    result["discovery_delta"] = result.discovery_score_b - result.discovery_score_a
    result["confirmation_delta"] = (
        result.confirmation_score_b - result.confirmation_score_a
    )
    return result


def variation_summary(compared):
    """Average within and between differences over exactly the same coordinates."""
    if compared is None:
        return dict.fromkeys(
            [
                "absolute_cell_delta_median",
                "absolute_cell_delta_p90",
                "absolute_cell_delta_p99",
                "within_replicate_mae",
                "between_replicate_mae",
                "between_minus_within",
                "fraction_replicate_ranges_separated",
            ],
            np.nan,
        )
    within = compared.within_replicate_mae.mean()
    between = compared.between_replicate_mae.mean()
    absolute = compared.delta_b_minus_a.abs()
    return dict(
        absolute_cell_delta_median=absolute.median(),
        absolute_cell_delta_p90=absolute.quantile(0.9),
        absolute_cell_delta_p99=absolute.quantile(0.99),
        within_replicate_mae=within,
        between_replicate_mae=between,
        between_minus_within=between - within,
        fraction_replicate_ranges_separated=compared.replicate_ranges_separated.mean(),
    )


def discovery_sites(compared):
    use = compared[["discovery_delta", "confirmation_delta"]].notna().all(
        axis=1
    ) & compared.gene_id_a.str.startswith("ENSG")
    result = compared.loc[use].copy()
    result["abs_delta"] = result.delta_b_minus_a.abs()
    result["direction_agrees"] = result.discovery_delta * result.confirmation_delta > 0
    return result


def transcript_candidates(compared, a, b, discovery_a, discovery_b):
    eligible = discovery_sites(compared)
    result = (
        eligible.groupby(["gene_id_a", "transcript_id"])
        .agg(
            matched_supported_sites=("discovery_delta", "size"),
            mean_abs_delta=("abs_delta", "mean"),
            mean_delta_b_minus_a=("delta_b_minus_a", "mean"),
            discovery_delta=("discovery_delta", "mean"),
            confirmation_delta=("confirmation_delta", "mean"),
            positions_agreeing=("direction_agrees", "sum"),
            maximum_depth_ratio=("replicate_depth_ratio", "max"),
        )
        .reset_index()
        .rename(columns={"gene_id_a": "gene_id"})
    )
    return result.loc[result.matched_supported_sites.ge(3)].assign(
        cell_a=a,
        cell_b=b,
        discovery_replicate_a=discovery_a,
        discovery_replicate_b=discovery_b,
    )


def rank_candidates(tables):
    result = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    result["discovery_magnitude"] = result.discovery_delta.abs()
    return result.sort_values(
        [
            "discovery_magnitude",
            "matched_supported_sites",
            "gene_id",
            "transcript_id",
            "cell_a",
            "cell_b",
        ],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)


def shared_positions(cells, total_replicates, expected_cells=None):
    """Find positions supported by repeated samples in every analyzed cell line."""
    expected_cells = set(cells) if expected_cells is None else set(expected_cells)
    if not cells:
        return pd.DataFrame(
            columns=KEY
            + [
                "gene_id",
                "kmer",
                "observed_biological_replicates",
                "minimum_replicate_score",
                "maximum_replicate_score",
                "minimum_mean_depth",
                "maximum_mean_depth",
                "minimum_cell_score",
                "maximum_cell_score",
                "cell_score_range",
                "cell_mean_depth_ratio",
                "high_in_every_replicate",
                "all_replicates_observed",
            ]
        )
    parts = []
    for cell, frame in cells.items():
        use = frame.n_replicates.ge(2)
        parts.append(frame.loc[use].assign(cell_line=cell))
    combined = pd.concat(parts, ignore_index=True)
    grouped = combined.groupby(KEY)
    eligible = grouped.cell_line.nunique().eq(
        len(expected_cells)
    ) & grouped.kmer.nunique().eq(1)
    means = combined.pivot(index=KEY, columns="cell_line", values="score").reindex(
        eligible.index[eligible]
    )
    result = grouped.agg(
        gene_id=("gene_id", "first"),
        kmer=("kmer", "first"),
        observed_biological_replicates=("n_replicates", "sum"),
        minimum_replicate_score=("replicate_min", "min"),
        maximum_replicate_score=("replicate_max", "max"),
        minimum_mean_depth=("n_reads", "min"),
        maximum_mean_depth=("n_reads", "max"),
    ).reindex(means.index)
    result["minimum_cell_score"] = means.min(axis=1)
    result["maximum_cell_score"] = means.max(axis=1)
    result["cell_score_range"] = result.maximum_cell_score - result.minimum_cell_score
    result["cell_mean_depth_ratio"] = (
        result.maximum_mean_depth / result.minimum_mean_depth
    )
    result["high_in_every_replicate"] = result.minimum_replicate_score.ge(0.5)
    result["all_replicates_observed"] = result.observed_biological_replicates.eq(
        total_replicates
    )
    for cell in means:
        result["score_" + cell] = means[cell]
    return result.reset_index()
