"""Verify consistency across analysis tables, figures and their source records."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit
import numpy as np
import pandas as pd

from explore import KEY3, FEATURES, choose_representatives



def audit(a):
    t = a / "tables"
    passed = []
    scores = pd.read_parquet(t / "common_scores.parquet")
    depth = pd.read_parquet(t / "common_depth.parquet")
    assert scores.index.equals(depth.index) and scores.columns.equals(depth.columns)
    assert not scores.isna().any().any() and depth.ge(20).all().all()
    assert scores.ge(0).all().all() and scores.le(1).all().all()
    passed.append(
        "Complete score/depth matrices align; no missing or out-of-range scores."
    )
    site = (
        pd.read_parquet(t / "site_states.parquet").set_index(KEY3).reindex(scores.index)
    )
    assert site.high_in_every_replicate.equals(
        scores.ge(0.5).all(axis=1).rename("high_in_every_replicate")
    )
    columns = [c for c in site if c.startswith("state_")]
    assert site.high_cells.eq(site[columns].eq("high").sum(axis=1)).all()
    strict = site[columns].eq("high").sum(axis=1).eq(1) & site[columns].eq("low").sum(
        axis=1
    ).eq(6)
    assert site.strictly_restricted.eq(strict).all()
    assert set(site[columns].to_numpy().ravel()) <= {"high", "low", "discordant"}
    passed.append(
        "High-support, discordance and strict-restriction counts match individual sample scores."
    )
    signal = pd.read_parquet(t / "signal_matched_sites.parquet")
    run = pd.read_parquet(t / "signal_run_summaries.parquet")
    assert not signal.duplicated(KEY3 + ["replicate_id"]).any()
    assert signal.groupby("pair_id").size().eq(2).all()
    high = signal.loc[signal.group.eq("high")]
    low = signal.loc[signal.group.eq("low")]
    pairs = high.merge(low, on="pair_id", suffixes=("_h", "_l"), validate="one_to_one")
    assert (
        pairs.kmer_h.eq(pairs.kmer_l).all()
        and pairs.replicate_id_h.eq(pairs.replicate_id_l).all()
    )
    assert pairs.score_h.ge(0.5).all() and pairs.score_l.lt(0.5).all()
    assert (
        (
            np.maximum(pairs.n_reads_h, pairs.n_reads_l)
            / np.minimum(pairs.n_reads_h, pairs.n_reads_l)
        )
        .le(2 + 1e-12)
        .all()
    )
    assert high.groupby(["replicate_id", "motif"]).size().le(200).all()
    means = run.groupby(KEY3 + ["replicate_id"])[FEATURES].mean()
    got = signal.set_index(KEY3 + ["replicate_id"])[FEATURES].reindex(means.index)
    assert np.allclose(means, got, rtol=0, atol=1e-12)
    assert signal.signal_runs.eq(signal.n_runs).all()
    passed.append(
        "Signal pairs are distinct, sequence-identical, depth-matched and correctly grouped; all contributing runs have equal weight."
    )
    meta = pd.read_csv(t / "metagene_bins.csv")
    loc = pd.read_parquet(t / "position_topology.parquet")
    valid = loc.loc[loc.metagene.notna()]
    assert (
        meta.eligible.sum() == len(valid)
        and meta.shared.sum() == valid.high_in_every_replicate.sum()
    )
    assert np.isclose(meta.gene_matched_expected.sum(), meta.shared.sum())
    assert np.allclose(
        meta[["eligible_percent", "shared_percent", "gene_matched_percent"]].sum(), 100
    )
    passed.append(
        "Metagene denominators and gene-matched weights conserve eligible and candidate totals."
    )
    rep = pd.read_csv(t / "gene_representative_transcripts.csv")
    expected = choose_representatives(site.reset_index())
    pd.testing.assert_frame_equal(
        rep.sort_values("gene_id").reset_index(drop=True),
        expected.sort_values("gene_id").reset_index(drop=True),
        check_dtype=False,
    )
    genes = pd.read_parquet(t / "gene_sample_burden.parquet")
    equalcell = (
        genes.groupby(["gene_id", "cell_line"])
        .high_fraction.mean()
        .groupby("gene_id")
        .mean()
    )
    ranking = (
        pd.read_csv(t / "gene_burden_ranking_all.csv")
        .set_index("gene_id")
        .mean_high_fraction.reindex(equalcell.index)
    )
    assert np.allclose(equalcell, ranking, atol=1e-12)
    passed.append(
        "Gene representatives are selected without scores and burden ranks give cells equal weight."
    )
    overlap = pd.read_csv(t / "training_overlap_cell_contrasts.csv")
    base = pd.read_csv(a / "cell_pairs.csv")
    for r in overlap.itertuples():
        item = base.loc[
            base.scope.eq("matched_replicates")
            & (
                (base.cell_a.eq(r.cell_a) & base.cell_b.eq(r.cell_b))
                | (base.cell_b.eq(r.cell_a) & base.cell_a.eq(r.cell_b))
            )
        ].iloc[0]
        assert np.isclose(item.between_minus_within, r.baseline_excess, atol=1e-12)
        assert item.matched_sites == r.baseline_matched_positions
        assert (
            r.baseline_positions_retained + r.baseline_positions_lost
            == r.baseline_matched_positions
        )
        assert (
            r.baseline_positions_retained + r.newly_qualifying_positions
            == r.current_matched_positions
        )
    passed.append(
        "Overlap-run sensitivity reproduces all six original contrasts and accounts for coordinate turnover."
    )
    if (a / "README.md").is_file():
        findings = (a / "README.md").read_text(encoding="utf8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", findings)
        for link in links:
            url = urlsplit(link)
            if url.scheme or not url.path:
                continue
            assert (a / unquote(url.path)).is_file(), f"Broken findings link: {link}"
        images = re.findall(r"!\[[^\]]*\]\((figures_scientific/[^)]+\.png)\)", findings)
        expected = {p.relative_to(a).as_posix() for p in (a / "figures_scientific").glob("*.png")}
        assert len(images) == len(set(images)) and set(images) == expected
        for folder in ["figures_scientific", "figures_report"]:
            for path in (a / folder).glob("*"):
                if path.suffix in [".png", ".svg", ".pdf"]:
                    assert path.relative_to(a).as_posix() in links, f"Unlinked figure: {path.name}"
        passed.append("Every scientific figure appears once; all figure downloads and evidence links resolve.")
    source = {}
    for file in [
        "analyze.py",
        "comparisons.py",
        "evidence.py",
        "explore.py",
        "scientific_figures.py",
        "report_panels.py",
        "questions.py",
        "REFERENCES.md",
        "exploration_checks.py",
        "audit_analysis.py",
    ]:
        path = Path(__file__).resolve().parent / file
        source[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(
        status="passed",
        checks=passed,
        verified_counts=dict(
            common_positions=len(scores),
            biological_samples=len(scores.columns),
            shared_high=int(site.high_in_every_replicate.sum()),
            strict_restricted=int(strict.sum()),
            matched_signal_pairs=len(high),
            signal_run_sites=len(run),
            scientific_figures=len(list((a / "figures_scientific").glob("*.pdf"))),
            report_figures=len(list((a / "figures_report").glob("*.pdf"))),
        ),
        source_sha256=source,
    )
    (a / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_context_trees",
    )
    audit(p.parse_args().analysis)
