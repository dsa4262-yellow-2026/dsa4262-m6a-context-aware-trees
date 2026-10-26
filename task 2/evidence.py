"""Connect repeat-supported predictions to transcript biology and threshold stability."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from comparisons import (
    load_sites,
    aggregate_replicate,
    aggregate_cell,
    matched_sites,
    replicate_details,
    supported_comparison,
    variation_summary,
    discovery_sites,
    shared_positions,
)

KEY = ["transcript_id", "transcript_position"]
VARIATION_COLUMNS = KEY + ["within_replicate_mae", "between_replicate_mae"]
HERE = Path(__file__).resolve().parent
REGIONS = ["5UTR", "CDS", "stop_codon", "3UTR", "noncoding_or_unassigned", "unresolved"]


def region_at(position, geometry):
    """Use zero-based, half-open transcript intervals, including split stop codons."""
    cds, stop, utr = (geometry.get(name, []) for name in ["CDS", "stop_codon", "UTR"])
    contains = lambda spans: any(a <= position < b for a, b in spans)
    if contains(stop):
        return "stop_codon"
    if contains(cds):
        return "CDS"
    if cds and contains(utr):
        if position < min(a for a, b in cds):
            return "5UTR"
        if position >= max(b for a, b in cds + stop):
            return "3UTR"
    return "noncoding_or_unassigned"


def annotate_sites(sites, reference, feature_module):
    sites = sites.copy().reset_index(drop=True)
    original_positions = sites.transcript_position.astype(str)
    sites["transcript_position"] = sites.transcript_position.astype("int64")
    geometry, flanks, distances, matched = feature_module.reference_features(
        sites, reference
    )
    names = (
        feature_module.GEOMETRY_NAMES
        + feature_module.COMPACT_FLANK_NAMES
        + feature_module.A_FLANK_NAMES
        + feature_module.MOTIF_DISTANCE_NAMES
    )
    values = np.column_stack([geometry, flanks, distances])
    result = sites[KEY + ["kmer"]].copy()
    result["transcript_position"] = original_positions
    result["reference_matched"] = matched
    result["static_context_present"] = np.isfinite(values).sum(axis=1)
    result["missing_static_features"] = pd.Series(
        [";".join(np.asarray(names)[~np.isfinite(row)]) for row in values],
        dtype="object",
    )
    result["stop_distance_nt"] = geometry[:, 5]
    result["transcript_length_nt"] = geometry[:, 0]
    result["motif"] = sites.kmer.str.slice(1, 6)
    result["region"] = "unresolved"
    result["exon_number"] = np.nan
    result["exon_count"] = np.nan
    for transcript, rows in sites.groupby("transcript_id", sort=False).indices.items():
        g = reference["geometry"].get(transcript.split(".")[0])
        if g is None:
            continue
        for i in rows:
            if not matched[i] or g["boundaries"][-1] != geometry[i, 0]:
                continue
            position = int(sites.transcript_position.iloc[i])
            result.loc[i, "region"] = region_at(position, g)
            result.loc[i, "exon_number"] = np.searchsorted(
                g["boundaries"], position, side="right"
            )
            result.loc[i, "exon_count"] = len(g["boundaries"]) - 1
    return result


def confirmation_summary(candidates):
    """A single discovery-selected transcript and cell comparison per gene."""
    unique = candidates.drop_duplicates("gene_id")
    rows = []
    for limit in [3, 10, 50, 100, None]:
        selected = unique if limit is None else unique.head(limit)
        agrees = selected.discovery_delta * selected.confirmation_delta > 0
        aligned = np.sign(selected.discovery_delta) * selected.confirmation_delta
        rows.append(
            dict(
                discovery_selection="all" if limit is None else f"top_{limit}",
                genes=len(selected),
                direction_retained=int(agrees.sum()),
                direction_retained_fraction=agrees.mean(),
                median_discovery_magnitude=selected.discovery_delta.abs().median(),
                median_confirmation_magnitude=selected.confirmation_delta.abs().median(),
                median_confirmation_in_discovery_direction=aligned.median(),
            )
        )
    return pd.DataFrame(rows)


def rebuild_cells(results, inventory, threshold):
    cells, observations = {}, 0
    width = inventory.groupby("cell_line").replicate_id.nunique().max()
    for cell, cell_rows in inventory.groupby("cell_line", sort=True):
        replicates = []
        order = (
            cell_rows.sort_values(["replicate", "run"])
            .replicate_id.drop_duplicates()
            .tolist()
        )
        for _, rows in cell_rows.groupby("replicate_id", sort=True):
            frames = [
                load_sites(results / s.sample_id / "sites.parquet", threshold)
                for s in rows.itertuples()
            ]
            observations += sum(map(len, frames))
            if sum(map(len, frames)):
                replicates.append(aggregate_replicate(frames, rows))
        if replicates:
            frame = pd.concat(replicates, ignore_index=True)
            cells[cell] = aggregate_cell(frame).merge(
                replicate_details(frame, order, width), on=KEY, validate="one_to_one"
            )
    return cells, observations


def catalogue_overlap(baseline, current, observed):
    """Compare candidate identities at the same score cutoff and different depths."""
    before = pd.MultiIndex.from_frame(baseline[KEY])
    after = pd.MultiIndex.from_frame(current[KEY])
    eligible = pd.MultiIndex.from_frame(observed[KEY])
    retained = int(after.isin(before).sum())
    lost_coverage = int((~before.isin(eligible)).sum())
    lost_score = int((before.isin(eligible) & ~before.isin(after)).sum())
    return dict(
        baseline_candidates=len(before),
        baseline_candidates_retained=retained,
        baseline_candidates_lost=len(before) - retained,
        new_candidates=len(after) - retained,
        lost_coverage=lost_coverage,
        lost_score=lost_score,
    )


def region_composition(common, candidates, annotations):
    """Use positions that could enter the catalogue before applying its score cutoff."""
    eligible = common.loc[common.all_replicates_observed]
    groups = {
        "eligible_background": eligible,
        "eligible_same_genes": eligible.loc[eligible.gene_id.isin(candidates.gene_id)],
        "shared_candidates": candidates,
    }
    rows = []
    for name, frame in groups.items():
        annotated = frame[KEY].merge(annotations, on=KEY, validate="one_to_one")
        for region in REGIONS:
            n = int(annotated.region.eq(region).sum())
            rows.append(
                dict(
                    cohort=name,
                    region=region,
                    positions=n,
                    denominator=len(annotated),
                    fraction=n / len(annotated) if len(annotated) else np.nan,
                )
            )
    return pd.DataFrame(rows)


def region_by_gene(common, candidates, annotations):
    """Hold each gene's candidate count fixed when comparing 3-prime UTR location."""
    eligible = common.loc[common.all_replicates_observed, KEY + ["gene_id"]]
    eligible = eligible.merge(
        annotations[KEY + ["region"]], on=KEY, validate="one_to_one"
    )
    candidate = candidates[KEY + ["gene_id"]].merge(
        annotations[KEY + ["region"]], on=KEY, validate="one_to_one"
    )
    background = (
        eligible.assign(utr=eligible.region.eq("3UTR"))
        .groupby("gene_id")
        .agg(eligible_positions=("utr", "size"), eligible_3utr=("utr", "sum"))
    )
    selected = (
        candidate.assign(utr=candidate.region.eq("3UTR"))
        .groupby("gene_id")
        .agg(candidate_positions=("utr", "size"), candidate_3utr=("utr", "sum"))
    )
    result = selected.join(background, how="left")
    result["background_3utr_fraction"] = (
        result.eligible_3utr / result.eligible_positions
    )
    result["expected_candidate_3utr"] = (
        result.candidate_positions * result.background_3utr_fraction
    )
    return result.reset_index()


def example_confirmation_pairs(supported, examples, inventory):
    """Show each remaining biological-sample pair on the selected coordinates."""
    rows = []
    for example in examples.itertuples():
        sites = supported.loc[
            supported.transcript_id.eq(example.transcript_id)
            & supported.cell_a.eq(example.cell_a)
            & supported.cell_b.eq(example.cell_b)
        ]
        orders = []
        for cell in [example.cell_a, example.cell_b]:
            values = inventory.loc[inventory.cell_line.eq(cell)].sort_values(
                ["replicate", "run"]
            )
            orders.append(values.replicate_id.drop_duplicates().tolist())
        for i, replicate_a in enumerate(orders[0][1:], start=1):
            for j, replicate_b in enumerate(orders[1][1:], start=1):
                difference = (
                    sites[f"replicate_score_{j}_b"] - sites[f"replicate_score_{i}_a"]
                )
                observed = difference.notna()
                delta = difference.mean()
                rows.append(
                    dict(
                        gene_name=example.gene_name,
                        transcript_id=example.transcript_id,
                        cell_a=example.cell_a,
                        cell_b=example.cell_b,
                        replicate_a=replicate_a,
                        replicate_b=replicate_b,
                        selected_positions=len(sites),
                        paired_positions=int(observed.sum()),
                        all_selected_positions=bool(observed.all()),
                        transcript_positions=";".join(
                            str(p)
                            for p in sorted(
                                sites.loc[observed, "transcript_position"].astype(int)
                            )
                        ),
                        delta_b_minus_a=delta,
                        aligned_delta=np.sign(example.discovery_delta) * delta,
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "gene_name",
            "transcript_id",
            "cell_a",
            "cell_b",
            "replicate_a",
            "replicate_b",
            "selected_positions",
            "paired_positions",
            "all_selected_positions",
            "transcript_positions",
            "delta_b_minus_a",
            "aligned_delta",
        ],
    )


def retained_contrast(before, after):
    """Compare coverage settings on identical positions as well as their own sets."""
    before = (
        pd.DataFrame(columns=VARIATION_COLUMNS)
        if before is None
        else before[VARIATION_COLUMNS]
    )
    after = (
        pd.DataFrame(columns=VARIATION_COLUMNS)
        if after is None
        else after[VARIATION_COLUMNS]
    )
    paired = before.merge(
        after, on=KEY, suffixes=("_base", "_current"), validate="one_to_one"
    )
    baseline = (
        paired.between_replicate_mae_base - paired.within_replicate_mae_base
    ).mean()
    current = (
        paired.between_replicate_mae_current - paired.within_replicate_mae_current
    ).mean()
    return dict(
        baseline_matched_positions=len(before),
        baseline_positions_retained=len(paired),
        baseline_positions_lost=len(before) - len(paired),
        newly_qualifying_positions=len(after) - len(paired),
        baseline_excess_on_retained=baseline,
        current_excess_on_retained=current,
        excess_change_on_retained=current - baseline,
    )


def threshold_tables(
    results, inventory, cells, examples, supported, output, min_reads, observations
):
    baselines = {}
    baseline_pairs = {}
    catalogues, contrasts, fixed_examples, common_sets = [], [], [], []
    for threshold in sorted(
        {min_reads, int(np.ceil(min_reads * 1.5)), int(np.ceil(min_reads * 2.5))}
    ):
        current, n = (
            (cells, observations)
            if threshold == min_reads
            else rebuild_cells(results, inventory, threshold)
        )
        common = shared_positions(
            current, inventory.replicate_id.nunique(), inventory.cell_line.unique()
        )
        common_sets.append((threshold, common))
        for score in [0.5, 0.7, 0.9]:
            selected = common.loc[
                common.all_replicates_observed
                & common.minimum_replicate_score.ge(score)
            ]
            if threshold == min_reads:
                baselines[score] = selected[KEY]
            catalogues.append(
                dict(
                    min_reads=threshold,
                    score_threshold=score,
                    eligible_human_observations=n,
                    common_supported_positions=len(common),
                    all_replicates_observed_positions=int(
                        common.all_replicates_observed.sum()
                    ),
                    candidate_positions=len(selected),
                    candidate_genes=selected.gene_id.nunique(),
                    **catalogue_overlap(
                        baselines[score],
                        selected,
                        common.loc[common.all_replicates_observed],
                    ),
                )
            )
        for a, b in combinations(sorted(inventory.cell_line.unique()), 2):
            compared, unmatched = None, 0
            if a in current and b in current:
                joined, _ = matched_sites(current[a], current[b])
                compared = supported_comparison(joined)
                unmatched = int(
                    (
                        compared.reference_matched_fraction_a.lt(1)
                        | compared.reference_matched_fraction_b.lt(1)
                    ).sum()
                )
            if threshold == min_reads:
                baseline_pairs[a, b] = (
                    None if compared is None else compared[VARIATION_COLUMNS].copy()
                )
            contrasts.append(
                dict(
                    min_reads=threshold,
                    cell_a=a,
                    cell_b=b,
                    matched_positions=0 if compared is None else len(compared),
                    reference_unmatched_positions=unmatched,
                    **variation_summary(compared),
                    **retained_contrast(baseline_pairs[a, b], compared),
                )
            )
            for example in examples.loc[
                examples.cell_a.eq(a) & examples.cell_b.eq(b)
            ].itertuples():
                positions = supported.loc[
                    supported.transcript_id.eq(example.transcript_id)
                    & supported.cell_a.eq(a)
                    & supported.cell_b.eq(b),
                    "transcript_position",
                ]
                fixed = (
                    supported.iloc[:0]
                    if compared is None
                    else discovery_sites(compared)
                )
                fixed = fixed.loc[
                    fixed.transcript_id.eq(example.transcript_id)
                    & fixed.transcript_position.isin(positions)
                ]
                fixed_examples.append(
                    dict(
                        min_reads=threshold,
                        gene_name=example.gene_name,
                        transcript_id=example.transcript_id,
                        cell_a=a,
                        cell_b=b,
                        baseline_positions=len(positions),
                        retained_positions=len(fixed),
                        discovery_delta=fixed.discovery_delta.mean(),
                        confirmation_delta=fixed.confirmation_delta.mean(),
                        positions_retaining_direction=int(fixed.direction_agrees.sum()),
                    )
                )
        print(
            f"Checked analysis at {threshold} reads: {len(common):,} common supported positions.",
            flush=True,
        )
    tables = [pd.DataFrame(rows) for rows in [catalogues, contrasts, fixed_examples]]
    for name, frame in zip(
        ["threshold_stability", "cell_contrast_stability", "example_stability"], tables
    ):
        frame.to_csv(output / (name + ".csv"), index=False)
    selected = pd.concat(
        [
            frame.loc[
                frame.all_replicates_observed & frame.minimum_replicate_score.ge(0.5),
                KEY + ["gene_id", "kmer"],
            ]
            for _, frame in common_sets
        ],
        ignore_index=True,
    ).drop_duplicates(KEY)
    for depth, frame in common_sets:
        values = frame[
            KEY
            + [
                "minimum_replicate_score",
                "observed_biological_replicates",
                "all_replicates_observed",
            ]
        ]
        values = values.rename(
            columns={c: f"{c}_at_{depth}_reads" for c in values if c not in KEY}
        )
        selected = selected.merge(values, on=KEY, how="left", validate="one_to_one")
    selected["score_at_least_09_at_every_depth"] = np.logical_and.reduce(
        [
            selected[f"minimum_replicate_score_at_{depth}_reads"].ge(0.9)
            & selected[f"all_replicates_observed_at_{depth}_reads"].fillna(False)
            for depth, _ in common_sets
        ]
    )
    names = pd.read_csv(HERE / "genes.csv").set_index("gene_id").gene_name
    selected.insert(
        3, "gene_name", selected.gene_id.map(names).fillna(selected.gene_id)
    )
    selected.sort_values(
        [
            "score_at_least_09_at_every_depth",
            f"minimum_replicate_score_at_{min_reads}_reads",
        ],
        ascending=False,
    ).to_csv(output / "shared_candidate_stability.csv", index=False)
    return tables[0]


def complete_analysis(
    results, inventory, output, cells, min_reads, observations, reference_path
):
    shared = pd.read_parquet(output / "shared_sites.parquet")
    high = pd.read_csv(output / "shared_candidates.csv", dtype={k: str for k in KEY})
    examples = pd.read_csv(output / "examples.csv")
    supported = pd.read_csv(
        output / "example_supported_sites.csv", dtype={k: str for k in KEY}
    )
    candidates = pd.read_csv(output / "discovery_candidates.csv")
    confirmation = confirmation_summary(candidates)
    confirmation.to_csv(output / "discovery_confirmation.csv", index=False)
    threshold = threshold_tables(
        results, inventory, cells, examples, supported, output, min_reads, observations
    )
    print(
        "Annotating candidate positions with the bundled transcript reference.",
        flush=True,
    )
    reference = joblib.load(reference_path)
    spec = spec_from_file_location(
        "context_tree_reference_features",
        HERE.parent / "task 1 context aware trees/context_trees/features.py",
    )
    features = module_from_spec(spec)
    spec.loader.exec_module(features)
    targets = [shared[KEY + ["kmer"]]]
    if len(supported):
        targets.append(supported[KEY + ["kmer_a"]].rename(columns={"kmer_a": "kmer"}))
    sites = pd.concat(targets, ignore_index=True).drop_duplicates(KEY)
    annotations = annotate_sites(sites, reference, features)
    annotations.to_parquet(
        output / "site_annotations.parquet", index=False, compression="zstd"
    )
    addition = annotations.drop(columns="kmer")
    stable = pd.read_csv(
        output / "shared_candidate_stability.csv", dtype={k: str for k in KEY}
    )
    for name, frame in [
        ("shared_candidates", high),
        ("example_supported_sites", supported),
        ("shared_candidate_stability", stable),
    ]:
        frame.merge(addition, on=KEY, validate="one_to_one").to_csv(
            output / (name + ".csv"), index=False
        )
    regions = region_composition(shared, high, annotations)
    regions.to_csv(output / "region_summary.csv", index=False)
    gene_regions = region_by_gene(shared, high, annotations)
    gene_regions.to_csv(output / "region_by_gene.csv", index=False)
    example_confirmation_pairs(supported, examples, inventory).to_csv(
        output / "example_confirmation_pairs.csv", index=False
    )
    return dict(
        coverage_thresholds=sorted(threshold.min_reads.unique().tolist()),
        score_thresholds=[0.5, 0.7, 0.9],
        coverage_rule="Reapply depth to each original run, then rebuild biological and cell means; frozen model scores are unchanged.",
        retained_position_comparison="For each cell pair and higher read cutoff, compare primary and current excess on their coordinate intersection; separately count retained, lost and newly qualifying positions. Empty comparisons retain zero counts and unavailable contrasts.",
        example_stability="Keep the primary discovery-selected transcript, cell pair and coordinate list fixed; report retained positions at each depth.",
        confirmation_summary="One highest-discovery-magnitude comparison per gene, taken from the full discovery ranking; zeros and reversals do not retain direction.",
        annotation="Same fixed Ensembl 91 reference as the model; zero-based half-open intervals; stop codons precede CDS in region assignment.",
        region_background="Positions observed in every declared biological replicate before the score cutoff, and the subset in shared-candidate genes. Counts are transcript positions, not unique genomic bases.",
        region_gene_weighting="For each candidate gene, multiply its candidate count by its eligible-position 3-prime UTR fraction, then sum; dividing by total candidate count gives a background with the same gene contributions.",
        candidate_turnover="Loss of observation in any declared biological replicate is a coverage loss; a fully observed baseline candidate below the score cutoff is a score loss. Both use the same score cutoff as the baseline.",
        individual_confirmation="For the three discovery-selected examples, retain each remaining biological-replicate pair, its available original positions and signed mean contrast; sample pairs can share replicates.",
    )
