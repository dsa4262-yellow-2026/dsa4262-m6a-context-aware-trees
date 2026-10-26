"""Exploratory analysis of SG-NEx predictions and their signal features.

Run analyze.py first, then this script. Signal EDA uses indexed JSON reads from
the supplied data. --plots-only reuses tables.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from comparisons import (
    KEY,
    load_sites,
    aggregate_replicate,
    aggregate_cell,
    replicate_details,
    matched_sites,
    supported_comparison,
)

HERE = Path(__file__).resolve().parent
KEY3 = KEY + ["kmer"]
CELLS = ["A549", "H9", "HEYA8", "Hct116", "HepG2", "K562", "MCF7"]
MOTIFS = ["GGACT", "GAACT", "AGACT"]
FEATURES = [
    f"{channel}_{offset}"
    for offset in ["minus1", "center", "plus1"]
    for channel in ["log_dwell", "signal_sd", "current"]
]
SEED = 4262


def csv(frame, root, name):
    frame.to_csv(root / (name + ".csv"), index=False)


def short(name):
    return (
        name.replace("SGNex_", "")
        .replace("_directRNA_replicate", " r")
        .replace("_run", " / run ")
    )


def states(scores, sample_info, threshold=0.5):
    """High/low require ALL declared samples; missing is never a low call."""
    high, low, missing = {}, {}, {}
    for cell in CELLS:
        sub = scores.reindex(
            columns=sample_info.loc[sample_info.cell_line.eq(cell), "replicate_id"]
        )
        observed = sub.notna().all(axis=1) & (sub.shape[1] > 0)
        high[cell] = observed & sub.ge(threshold).all(axis=1)
        low[cell] = observed & sub.lt(threshold).all(axis=1)
        missing[cell] = ~observed
    high, low, missing = [
        pd.DataFrame(x, index=scores.index) for x in (high, low, missing)
    ]
    mixed = ~(high | low | missing)
    return high, low, mixed, missing


def choose_representatives(positions):
    """Choose one transcript per gene by observed-position count, never score."""
    counts = (
        positions.groupby(["gene_id", "transcript_id"], observed=True)
        .size()
        .rename("eligible_positions")
        .reset_index()
    )
    return counts.sort_values(
        ["gene_id", "eligible_positions", "transcript_id"],
        ascending=[True, False, True],
    ).drop_duplicates("gene_id")


def metagene_position(position, geometry):
    if geometry is None or not len(geometry.get("CDS", [])):
        return np.nan
    start = min(a for a, b in geometry["CDS"])
    end = max(
        b for a, b in list(geometry["CDS"]) + list(geometry.get("stop_codon", []))
    )
    length = geometry["boundaries"][-1]
    if not (0 < start < end < length and 0 <= position < length):
        return np.nan
    return (
        position / start
        if position < start
        else (
            1 + (position - start) / (end - start)
            if position < end
            else 2 + (position - end) / (length - end)
        )
    )


def summarize_signal(raw):
    """Same dwell transform as the frozen encoder, before model standardization."""
    values = np.asarray(raw, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 9
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite nonempty N x 9 read features.")
    if (values[:, [0, 1, 3, 4, 6, 7]] < 0).any():
        raise ValueError("Negative dwell time or within-read signal SD.")
    values[:, [0, 3, 6]] = np.log1p(1000 * values[:, [0, 3, 6]])
    return values.mean(axis=0, dtype=np.float64)


def overlap_cell_contrasts(bio, replacement, inventory, table):
    """Audit all six affected comparisons, keeping coordinate intersections explicit."""
    from evidence import retained_contrast

    cells = {}
    for cell, frame in bio.groupby("cell_line"):
        order = (
            inventory.loc[inventory.cell_line.eq(cell)]
            .sort_values(["replicate", "run"])
            .replicate_id.drop_duplicates()
            .tolist()
        )
        cells[cell] = aggregate_cell(frame).merge(
            replicate_details(frame, order, 3), on=KEY, validate="one_to_one"
        )
    sample = "SGNex_Hct116_directRNA_replicate3_run4"
    rep = "SGNex_Hct116_directRNA_replicate3"
    replaced = aggregate_replicate(
        [replacement], inventory.loc[inventory.sample_id.eq(sample)]
    )
    changed = pd.concat(
        [bio.loc[bio.cell_line.eq("Hct116") & bio.replicate_id.ne(rep)], replaced],
        ignore_index=True,
    )
    order = (
        inventory.loc[inventory.cell_line.eq("Hct116")]
        .sort_values(["replicate", "run"])
        .replicate_id.drop_duplicates()
        .tolist()
    )
    changed = aggregate_cell(changed).merge(
        replicate_details(changed, order, 3), on=KEY, validate="one_to_one"
    )
    rows = []
    for cell in CELLS:
        if cell == "Hct116":
            continue
        before = supported_comparison(matched_sites(cells["Hct116"], cells[cell])[0])
        after = supported_comparison(matched_sites(changed, cells[cell])[0])
        result = retained_contrast(before, after)
        result.update(
            cell_a="Hct116",
            cell_b=cell,
            current_matched_positions=len(after),
            baseline_excess=(
                before.between_replicate_mae - before.within_replicate_mae
            ).mean(),
            current_excess=(
                after.between_replicate_mae - after.within_replicate_mae
            ).mean(),
        )
        rows.append(result)
    csv(pd.DataFrame(rows), table, "training_overlap_cell_contrasts")
    return rows


def match_features(bio, cap=200, seed=SEED):
    """Within sample/motif, sample high sites then match unused low sites.

    Match the complete seven-mer exactly and mean read depth within twofold.
    Greedy nearest log-depth matching, deterministic shuffled high-site order.
    This is a descriptive contrast conditional on model prediction, not labels.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (rep, motif), sub in bio.groupby(["replicate_id", "motif"], sort=True):
        if motif not in MOTIFS:
            continue
        sub = sub.sort_values(KEY).reset_index(drop=True)
        high = sub.loc[sub.score.ge(0.5)]
        order = rng.permutation(high.index)
        pools = {
            k: g.index.to_numpy().tolist()
            for k, g in sub.loc[sub.score.lt(0.5)].groupby("kmer")
        }
        count = 0
        for idx in order:
            row = sub.loc[idx]
            candidates = pools.get(row.kmer, [])
            if not candidates:
                continue
            depths = sub.loc[candidates, "n_reads"].to_numpy()
            dist = np.abs(np.log(depths / row.n_reads))
            j = int(np.argmin(dist))
            if dist[j] > np.log(2) + 1e-12:
                continue
            other = pools[row.kmer].pop(j)
            pair = f"{rep}|{motif}|{count}"
            for i, group in [(idx, "high"), (other, "low")]:
                item = sub.loc[i]
                rows.append(
                    dict(
                        pair_id=pair,
                        group=group,
                        **{
                            c: item[c]
                            for c in KEY3
                            + [
                                "replicate_id",
                                "cell_line",
                                "motif",
                                "score",
                                "n_reads",
                                "n_runs",
                            ]
                        },
                    )
                )
            count += 1
            if count >= cap:
                break
    return pd.DataFrame(rows)


def extract_signal(matches, inventory, results, data_root, table):
    output, input_manifest = [], []
    for sample in inventory.itertuples():
        wanted = matches.loc[matches.replicate_id.eq(sample.replicate_id), KEY3]
        if wanted.empty:
            continue
        predictions = load_sites(results / sample.sample_id / "sites.parquet")
        selected = predictions.merge(wanted, on=KEY3, validate="one_to_one")
        rawpath, info = data_root / sample.signals, data_root / sample.info
        if not rawpath.is_file() or not info.is_file():
            raise FileNotFoundError(
                f"Signal EDA requires {rawpath} and its data.info index."
            )
        index = pd.read_csv(
            info, dtype={"transcript_id": str, "transcript_position": str}
        )
        index = index.rename(columns={c: "input_" + c for c in KEY})
        expected = len(selected)
        selected = selected.merge(
            index,
            on=["input_transcript_id", "input_transcript_position"],
            validate="one_to_one",
            suffixes=("", "_index"),
        )
        if len(selected) != expected:
            raise ValueError(
                f"Requested signal offsets missing for {sample.sample_id}."
            )
        digest = hashlib.sha256()
        with rawpath.open("rb") as handle:
            for r in selected.sort_values("start").itertuples():
                if r.n_reads != r.n_reads_index:
                    raise ValueError("Prediction and indexed read counts disagree.")
                handle.seek(r.start)
                payload = handle.read(r.end - r.start)
                digest.update(payload)
                record = json.loads(payload)
                kmers = record[r.input_transcript_id][r.input_transcript_position]
                if list(kmers) != [r.kmer] or len(kmers[r.kmer]) != r.n_reads:
                    raise ValueError(
                        "Signal record does not match its prediction key, motif or depth."
                    )
                features = summarize_signal(kmers[r.kmer])
                output.append(
                    dict(
                        sample_id=sample.sample_id,
                        replicate_id=sample.replicate_id,
                        transcript_id=r.transcript_id,
                        transcript_position=r.transcript_position,
                        kmer=r.kmer,
                        n_reads=r.n_reads,
                        **dict(zip(FEATURES, features)),
                    )
                )
        print(
            f"Signal summaries: {short(sample.sample_id)} ({len(selected):,} indexed sites)",
            flush=True,
        )
        input_manifest.append(
            dict(
                sample_id=sample.sample_id,
                json_bytes=rawpath.stat().st_size,
                index_sha256=hashlib.sha256(info.read_bytes()).hexdigest(),
                selected_run_sites=len(selected),
                selected_records_sha256=digest.hexdigest(),
            )
        )
    run = pd.DataFrame(output)
    run.to_parquet(table / "signal_run_summaries.parquet", index=False)
    # Equal eligible-run means, not pooling reads; matches prediction aggregation.
    aggregate = (
        run.groupby(KEY3 + ["replicate_id"], observed=True)[FEATURES]
        .mean()
        .reset_index()
    )
    counts = (
        run.groupby(KEY3 + ["replicate_id"], observed=True)
        .size()
        .rename("signal_runs")
        .reset_index()
    )
    aggregate = aggregate.merge(counts, on=KEY3 + ["replicate_id"])
    data = matches.merge(
        aggregate, on=KEY3 + ["replicate_id"], how="left", validate="one_to_one"
    )
    if data[FEATURES].isna().any().any():
        raise ValueError(
            "Selected signal sites are missing, rather than silently dropped."
        )
    if not data.signal_runs.eq(data.n_runs).all():
        raise ValueError(
            "Signal summaries must use every eligible run contributing to the biological score."
        )
    csv(pd.DataFrame(input_manifest), table, "signal_input_provenance")
    data.to_parquet(table / "signal_matched_sites.parquet", index=False)
    pairs = data.loc[data.group.eq("high")].merge(
        data.loc[data.group.eq("low")],
        on="pair_id",
        suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    differences = pairs[
        ["pair_id", "replicate_id_high", "cell_line_high", "motif_high"]
    ].rename(columns=lambda c: c.removesuffix("_high"))
    for feature in FEATURES:
        differences[feature] = pairs[feature + "_high"] - pairs[feature + "_low"]
    differences.to_parquet(table / "signal_pair_differences.parquet", index=False)
    # Tighter depth control without resampling sites or looking at signal values.
    ratio = np.maximum(pairs.n_reads_high, pairs.n_reads_low) / np.minimum(
        pairs.n_reads_high, pairs.n_reads_low
    )
    checks = []
    for cutoff in [2.0, 1.25]:
        kept = differences.loc[ratio.le(cutoff)]
        med = (
            kept.groupby(["replicate_id", "cell_line", "motif"])[FEATURES]
            .median()
            .reset_index()
        )
        med = med.merge(
            kept.groupby(["replicate_id", "motif"]).size().rename("pairs"),
            on=["replicate_id", "motif"],
        )
        checks.append(med.assign(maximum_depth_ratio=cutoff))
    csv(pd.concat(checks, ignore_index=True), table, "signal_depth_match_sensitivity")
    summaries = (
        differences.groupby(["replicate_id", "cell_line", "motif"])[FEATURES]
        .median()
        .reset_index()
    )
    summaries = summaries.merge(
        differences.groupby(["replicate_id", "motif"]).size().rename("pairs"),
        on=["replicate_id", "motif"],
    )
    csv(summaries, table, "signal_sample_contrasts")
    completeness = (
        data.groupby(["replicate_id", "motif", "group"])
        .agg(
            sites=("score", "size"),
            score_median=("score", "median"),
            depth_median=("n_reads", "median"),
            technical_run_sites=("signal_runs", "sum"),
        )
        .reset_index()
    )
    csv(completeness, table, "signal_matching_support")
    return dict(
        matched_pairs=len(pairs),
        run_sites=len(run),
        sample_motif_groups=len(summaries),
        max_pairs_per_sample_motif=200,
        motifs=MOTIFS,
        seed=SEED,
    )


def build(args):
    a, table = args.analysis, args.analysis / "tables"
    table.mkdir(exist_ok=True)
    metadata = json.loads((a / "metadata.json").read_text())
    if not metadata["complete"] or metadata["min_reads"] != 20:
        raise ValueError(
            "This analysis requires the complete primary 20-read analysis."
        )
    inventory = pd.read_csv(args.samples)
    info = inventory.sort_values(["cell_line", "replicate", "run"]).drop_duplicates(
        "replicate_id"
    )[["replicate_id", "cell_line", "platform"]]
    info["label"] = info.replicate_id.map(short)
    csv(inventory, table, "sample_manifest")
    bio = pd.read_parquet(a / "biological_sites.parquet")
    shared = pd.read_parquet(a / "shared_sites.parquet")
    complete = shared.loc[shared.all_replicates_observed].copy()
    common = bio.merge(complete[KEY3], on=KEY3, validate="many_to_one")
    matrix = common.pivot(index=KEY3, columns="replicate_id", values="score").reindex(
        columns=info.replicate_id
    )
    depth = common.pivot(
        index=KEY3, columns="replicate_id", values="n_reads"
    ).reindex_like(matrix)
    if matrix.shape[1] != 17 or matrix.isna().any().any() or depth.isna().any().any():
        raise ValueError(
            "Common universe must contain every declared biological sample."
        )
    matrix.to_parquet(table / "common_scores.parquet")
    depth.to_parquet(table / "common_depth.parquet")
    csv(info, table, "biological_sample_info")
    corr = matrix.corr(method="spearman")
    corr.to_csv(table / "sample_correlations.csv")
    order = leaves_list(
        linkage(
            squareform(np.clip(1 - corr.values, 0, 2), checks=False), method="average"
        )
    )
    csv(
        info.set_index("replicate_id").loc[corr.index[order]].reset_index(),
        table,
        "cluster_order",
    )
    neighbor = []
    for rep in matrix.columns:
        top = corr[rep].drop(rep).idxmax()
        neighbor.append(
            dict(
                replicate_id=rep,
                nearest_sample=top,
                same_cell=info.set_index("replicate_id").loc[rep, "cell_line"]
                == info.set_index("replicate_id").loc[top, "cell_line"],
                spearman=corr.loc[rep, top],
            )
        )
    csv(pd.DataFrame(neighbor), table, "nearest_samples")
    cellmean = (
        common.groupby(KEY3 + ["cell_line"])
        .score.mean()
        .unstack("cell_line")
        .reindex(columns=CELLS)
    )
    cellmean.corr(method="spearman").to_csv(table / "cell_correlations.csv")
    flow = (
        bio.groupby(["cell_line", "replicate_id"])
        .agg(
            eligible_positions=("score", "size"),
            median_depth=("n_reads", "median"),
            high_fraction=("score", lambda s: s.ge(0.5).mean()),
        )
        .reset_index()
    )
    cflow = (
        common.groupby("replicate_id")
        .agg(
            common_high_fraction=("score", lambda s: s.ge(0.5).mean()),
            common_median_depth=("n_reads", "median"),
        )
        .reset_index()
    )
    csv(
        flow.merge(cflow, on="replicate_id").merge(
            info[["replicate_id", "platform"]], on="replicate_id"
        ),
        table,
        "sample_support",
    )
    bins = [20, 30, 50, 100, 200, 500, np.inf]
    depthdata = common.assign(depth_bin=pd.cut(common.n_reads, bins=bins, right=False))
    depthstats = (
        depthdata.groupby(["cell_line", "replicate_id", "depth_bin"], observed=True)
        .agg(
            positions=("score", "size"),
            median_score=("score", "median"),
            high_fraction=("score", lambda x: x.ge(0.5).mean()),
        )
        .reset_index()
    )
    depthstats["depth_bin"] = depthstats.depth_bin.astype(str)
    csv(depthstats, table, "depth_score_strata")
    # Avoid shared sites dominating a run: compare coverage selection explicitly.
    rows = []
    for rep, frame in bio.groupby("replicate_id"):
        vals = common.loc[common.replicate_id.eq(rep)]
        rows.append(
            dict(
                replicate_id=rep,
                spearman_score_logdepth=frame.score.corr(
                    np.log1p(frame.n_reads), method="spearman"
                ),
                common_spearman_score_logdepth=vals.score.corr(
                    np.log1p(vals.n_reads), method="spearman"
                ),
            )
        )
    csv(pd.DataFrame(rows), table, "depth_score_association")
    # Cell states and intersections on a single observation-complete universe.
    sensitivity = []
    for threshold in [0.5, 0.7, 0.9]:
        high, low, mixed, missing = states(matrix, info, threshold)
        n = high.sum(axis=1)
        strict = n.eq(1) & low.sum(axis=1).eq(6)
        sensitivity.append(
            dict(
                score_threshold=threshold,
                all_cells_high=int(n.eq(7).sum()),
                any_cell_high=int(n.gt(0).sum()),
                exactly_one_high_cell=int(n.eq(1).sum()),
                strictly_one_high_other_six_low=int(strict.sum()),
                any_discordant=int(mixed.any(axis=1).sum()),
            )
        )
        if threshold == 0.5:
            primary_high = high
            pos = complete.set_index(KEY3).reindex(matrix.index).copy()
            for cell in CELLS:
                pos["state_" + cell] = np.where(
                    high[cell], "high", np.where(low[cell], "low", "discordant")
                )
            pos["high_cells"] = n
            pos["strictly_restricted"] = strict
            pos.reset_index().to_parquet(table / "site_states.parquet", index=False)
            csv(pos.loc[strict].reset_index(), table, "restricted_candidates")
            csv(
                n.value_counts()
                .reindex(range(8), fill_value=0)
                .rename_axis("high_cells")
                .rename("positions")
                .reset_index(),
                table,
                "sharing_counts",
            )
            csv(
                high.loc[n.gt(0)]
                .astype(int)
                .value_counts()
                .rename("positions")
                .reset_index(),
                table,
                "support_intersections",
            )
    csv(pd.DataFrame(sensitivity), table, "score_threshold_sensitivity")
    # Motif composition: all 18 possible central DRACH motifs, fixed common sites.
    common["motif"] = common.kmer.str.slice(1, 6)
    motif = (
        common.groupby(["cell_line", "replicate_id", "motif"])
        .agg(
            positions=("score", "size"),
            high_fraction=("score", lambda x: x.ge(0.5).mean()),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    csv(motif, table, "motif_summary")
    print("Common-universe, coverage, motif and sharing tables complete.", flush=True)
    # Replicate evidence and batch/platform descriptors (no confounding adjustment claim).
    pairs = pd.read_csv(a / "replicate_pairs.csv")
    platforms = (
        inventory.set_index("sample_id").platform.to_dict()
        | info.set_index("replicate_id").platform.to_dict()
    )
    pairs["platform_left"] = pairs.left.map(platforms)
    pairs["platform_right"] = pairs.right.map(platforms)
    pairs["platform_match"] = pairs.platform_left.eq(pairs.platform_right)
    csv(pairs, table, "replicate_pairs_platform")
    # Gene burden: one representative transcript, selected by support, and denominators.
    representatives = choose_representatives(complete)
    csv(representatives, table, "gene_representative_transcripts")
    selected = complete.merge(
        representatives[["gene_id", "transcript_id"]],
        on=["gene_id", "transcript_id"],
        validate="many_to_one",
    )
    scoregene = common.merge(selected[KEY3], on=KEY3, validate="many_to_one")
    names = complete.drop_duplicates("gene_id").set_index("gene_id").gene_name
    genes = (
        scoregene.groupby(["gene_id", "transcript_id", "cell_line", "replicate_id"])
        .agg(
            eligible_positions=("score", "size"),
            high_positions=("score", lambda s: s.ge(0.5).sum()),
            high_fraction=("score", lambda s: s.ge(0.5).mean()),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    genes["gene_name"] = genes.gene_id.map(names)
    genes.to_parquet(table / "gene_sample_burden.parquet", index=False)
    ranking = (
        genes.groupby(["gene_id", "transcript_id", "cell_line"])
        .agg(
            high_fraction=("high_fraction", "mean"),
            eligible_positions=("eligible_positions", "first"),
            high_positions=("high_positions", "mean"),
        )
        .reset_index()
    )
    ranking = (
        ranking.groupby(["gene_id", "transcript_id"])
        .agg(
            eligible_positions=("eligible_positions", "first"),
            mean_high_fraction=("high_fraction", "mean"),
            mean_high_positions=("high_positions", "mean"),
        )
        .reset_index()
    )
    ranking["gene_name"] = ranking.gene_id.map(names)
    ranking = ranking.sort_values(
        ["mean_high_fraction", "mean_high_positions", "gene_id"],
        ascending=[False, False, True],
    )
    csv(ranking, table, "gene_burden_ranking_all")
    csv(
        ranking.loc[ranking.eligible_positions.ge(10)],
        table,
        "gene_burden_ranking_min10",
    )
    threshold_ranks = []
    for cutoff in [5, 10, 20]:
        for rank, r in enumerate(
            ranking.loc[ranking.eligible_positions.ge(cutoff)].head(20).itertuples(), 1
        ):
            threshold_ranks.append(
                dict(
                    minimum_positions=cutoff,
                    rank=rank,
                    gene_id=r.gene_id,
                    gene_name=r.gene_name,
                    mean_high_fraction=r.mean_high_fraction,
                    eligible_positions=r.eligible_positions,
                )
            )
    csv(pd.DataFrame(threshold_ranks), table, "gene_burden_rank_sensitivity")
    # Coding topology, denominator-controlled bin profiles and last-exon context.
    reference = joblib.load(args.reference)
    gmap = reference["geometry"]
    annotation = pd.read_parquet(a / "site_annotations.parquet")
    locations = complete.merge(
        annotation[KEY + ["region"]], on=KEY, validate="one_to_one"
    )
    coord, lastdist, stopdist, gap, last = [], [], [], [], []
    for r in locations.itertuples():
        g = gmap.get(r.transcript_id)
        p = int(r.transcript_position)
        coord.append(metagene_position(p, g))
        valid = g is not None and len(g.get("CDS", [])) and len(g.get("stop_codon", []))
        if valid:
            stop = min(a for a, b in g["stop_codon"])
            start_last = g["boundaries"][-2]
            stopdist.append(p - stop)
            lastdist.append(p - start_last)
            gap.append(stop - start_last)
            last.append(p >= start_last)
        else:
            stopdist.append(np.nan)
            lastdist.append(np.nan)
            gap.append(np.nan)
            last.append(False)
    locations["metagene"] = coord
    locations["distance_stop_start"] = stopdist
    locations["distance_last_exon_start"] = lastdist
    locations["stop_to_last_exon_start"] = gap
    locations["last_exon"] = last
    locations["any_cell_high"] = (
        primary_high.any(axis=1)
        .reindex(pd.MultiIndex.from_frame(locations[KEY3]))
        .to_numpy()
    )
    locations.to_parquet(table / "position_topology.parquet", index=False)
    valid = locations.loc[locations.metagene.notna()].copy()
    edges = np.linspace(0, 3, 31)
    bg = np.histogram(valid.metagene, edges)[0]
    selectedhigh = valid.loc[valid.high_in_every_replicate]
    count = np.histogram(selectedhigh.metagene, edges)[0]
    # Weight eligible positions to reproduce each candidate gene's contribution.
    weights = (
        selectedhigh.groupby("gene_id")
        .size()
        .div(valid.groupby("gene_id").size())
        .fillna(0)
    )
    expected = np.histogram(valid.metagene, edges, weights=valid.gene_id.map(weights))[
        0
    ]
    meta = pd.DataFrame(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "eligible": bg,
            "shared": count,
            "gene_matched_expected": expected,
        }
    )
    meta["eligible_percent"] = 100 * bg / bg.sum()
    meta["shared_percent"] = 100 * count / count.sum()
    meta["gene_matched_percent"] = 100 * expected / expected.sum()
    csv(meta, table, "metagene_bins")
    profile = []
    bins = np.arange(-500, 501, 50)
    for anchor in ["distance_stop_start", "distance_last_exon_start"]:
        for cohort, mask in [
            ("all_coding", locations.distance_stop_start.notna()),
            (
                "stop_at_least_150nt_after_last_exon_start",
                locations.stop_to_last_exon_start.ge(150),
            ),
        ]:
            frame = locations.loc[mask]
            den = np.histogram(frame[anchor], bins)[0]
            num = np.histogram(frame.loc[frame.high_in_every_replicate, anchor], bins)[
                0
            ]
            for i in range(len(den)):
                profile.append(
                    dict(
                        anchor=anchor,
                        cohort=cohort,
                        left=bins[i],
                        right=bins[i + 1],
                        eligible=int(den[i]),
                        shared=int(num[i]),
                        percent_shared=100 * num[i] / den[i] if den[i] else np.nan,
                    )
                )
    csv(pd.DataFrame(profile), table, "anchor_profiles")
    last_exon = (
        locations.loc[locations.distance_stop_start.notna()]
        .groupby("last_exon")
        .agg(
            eligible=("high_in_every_replicate", "size"),
            shared=("high_in_every_replicate", "sum"),
        )
        .reset_index()
    )
    last_exon["shared_fraction"] = last_exon.shared / last_exon.eligible
    csv(last_exon, table, "last_exon_support")
    # Preserve all ranked comparisons, with one discovery-selected row per gene.
    disc = pd.read_csv(a / "discovery_candidates.csv").drop_duplicates("gene_id").copy()
    disc["retained_direction"] = disc.discovery_delta * disc.confirmation_delta > 0
    disc["aligned_confirmation"] = (
        np.sign(disc.discovery_delta) * disc.confirmation_delta
    )
    csv(disc, table, "gene_discovery_confirmation")
    # Save structured transcript models for locus drawings without loading reference in plots-only mode.
    structures = {}
    for tx in pd.read_csv(a / "examples.csv").transcript_id:
        g = gmap[tx]
        structures[tx] = {
            key: np.asarray(g[key]).tolist()
            for key in ["boundaries", "CDS", "UTR", "stop_codon"]
        }
    (table / "locus_structures.json").write_text(
        json.dumps(structures, indent=2), encoding="utf8"
    )
    del reference
    # Remove the run that overlaps Dataset 0, holding other samples fixed.
    replacement = load_sites(
        args.results / "SGNex_Hct116_directRNA_replicate3_run4" / "sites.parquet"
    )
    column = "SGNex_Hct116_directRNA_replicate3"
    replace = replacement.set_index(KEY3).score.reindex(matrix.index)
    keep = replace.notna()
    without = matrix.loc[keep].copy()
    without[column] = replace.loc[keep]
    base = matrix.loc[keep]
    overlap = []
    for threshold in [0.5, 0.7, 0.9]:
        b = base.ge(threshold).all(axis=1)
        c = without.ge(threshold).all(axis=1)
        original = matrix.ge(threshold).all(axis=1)
        overlap.append(
            dict(
                score_threshold=threshold,
                original_common=len(matrix),
                retained_common=len(base),
                original_shared=int(original.sum()),
                original_shared_on_retained=int(b.sum()),
                without_overlap_shared=int(c.sum()),
                lost_observation=int(original.loc[~keep].sum()),
                lost_score=int((b & ~c).sum()),
                gained_score=int((c & ~b).sum()),
                hct116_rep3_mean_absolute_change=float(
                    (base[column] - without[column]).abs().mean()
                ),
            )
        )
    csv(pd.DataFrame(overlap), table, "training_overlap_run_sensitivity")
    overlap_cell_contrasts(bio, replacement, inventory, table)
    print(
        "Gene burden, topology, confirmation and overlap-run sensitivity complete.",
        flush=True,
    )
    # Raw-signal EDA is a separate, reproducible matched subset of the complete universe.
    matches = match_features(common)
    csv(matches, table, "signal_selected_pairs")
    signal = extract_signal(matches, inventory, args.results, args.data_root, table)
    summary = dict(
        analysis_version="1.0",
        common_positions=len(matrix),
        eligible_biological_observations=len(bio),
        biological_samples=len(info),
        cells=len(CELLS),
        runs=len(inventory),
        same_cell_nearest_neighbors=int(sum(n["same_cell"] for n in neighbor)),
        score_sensitivity=sensitivity,
        gene_burden_genes_min10=int(ranking.eligible_positions.ge(10).sum()),
        representative_transcript_genes=len(representatives),
        metagene_eligible=len(valid),
        metagene_shared=len(selectedhigh),
        metagene_excluded=len(complete) - len(valid),
        signal=signal,
        overlap_sensitivity=overlap,
        discovery_genes=len(disc),
        discovery_retained=int(disc.retained_direction.sum()),
        top100_retained=int(disc.head(100).retained_direction.sum()),
        interpretation="Descriptive comparisons of predicted m6A site scores and signal features; biological replicates are the experimental units.",
    )
    (a / "exploration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    inputs = [
        args.samples,
        args.reference,
        a / "metadata.json",
        a / "biological_sites.parquet",
        a / "shared_sites.parquet",
    ]
    manifest = []
    for path in inputs:
        manifest.append(
            dict(
                name=path.name,
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    path = HERE / "explore.py"
    manifest.append(
        dict(
            name=path.name,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
    (a / "exploration_provenance.json").write_text(
        json.dumps(
            dict(
                inputs=manifest,
                seed=SEED,
                python=platform.python_version(),
                pandas=pd.__version__,
                numpy=np.__version__,
                signal_selection="At most 200 high/low pairs per biological sample and prespecified motif; exact seven-mer and <=2-fold biological mean depth; all reads in every eligible contributing technical run.",
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, default=HERE / "analysis_context_trees")
    p.add_argument("--results", type=Path, default=HERE / "results_context_trees")
    p.add_argument("--samples", type=Path, default=HERE / "samples.csv")
    p.add_argument(
        "--reference",
        type=Path,
        default=HERE.parent / "task 1 context aware trees/model/reference.joblib",
    )
    p.add_argument("--data-root", type=Path, default=HERE / "raw_data")
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Render cached analysis tables without reading raw signals.",
    )
    args = p.parse_args()
    if not args.plots_only:
        print(json.dumps(build(args), indent=2), flush=True)
    from scientific_figures import render_all

    render_all(args.analysis)
    from questions import write_findings

    write_findings(args.analysis)
    from report_panels import render

    render(args.analysis)
    record_path = args.analysis / "exploration_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf8"))
    record["rendering_sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ["scientific_figures.py", "report_panels.py", "questions.py"]
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
