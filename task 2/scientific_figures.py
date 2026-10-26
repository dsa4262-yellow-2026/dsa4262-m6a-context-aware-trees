"""Original scientific figures from cached Task 2 tables (PNG, SVG and PDF).

The layouts follow metagene, UpSet, replicate-point, heatmap and locus-track
conventions cited in REFERENCES.md.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from explore import CELLS, MOTIFS, FEATURES, short

PALETTE = dict(
    zip(
        CELLS,
        ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#009E73", "#D55E00", "#6B5B95"],
    )
)
BLUE, INK, GRAY, ORANGE = "#0072B2", "#243746", "#88969E", "#D55E00"
CAPTIONS = {}


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#84929B",
        }
    )


def save(fig, out, name, caption):
    for ext in ["png", "svg", "pdf"]:
        meta = (
            {"Software": None}
            if ext == "png"
            else (
                {"Date": None, "Creator": None}
                if ext == "svg"
                else {"Creator": None, "CreationDate": None, "ModDate": None}
            )
        )
        fig.savefig(out / f"{name}.{ext}", dpi=240, bbox_inches="tight", metadata=meta)
    CAPTIONS[name] = caption
    plt.close(fig)


def legend_cells(fig, y=0.02, ncol=7):
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=PALETTE[c], label=c) for c in CELLS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=9,
    )


def footer(fig, text):
    fig.text(0.01, 0.015, text, fontsize=9, va="bottom", color="#4C606D")


def support(a, t, out):
    d = pd.read_csv(t / "sample_support.csv")
    info = pd.read_csv(t / "biological_sample_info.csv")
    d = d.set_index("replicate_id").loc[info.replicate_id].reset_index()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 6.0), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    y = np.arange(len(d))
    axes[0].barh(
        y,
        d.eligible_positions / 1000,
        color=[PALETTE[c] for c in d.cell_line],
        alpha=0.9,
    )
    axes[0].axvline(26.024, color=INK, ls="--", lw=1, label="Common to all 17: 26,024")
    axes[0].set_yticks(y, d.replicate_id.map(short))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Eligible transcript positions (thousands)")
    axes[0].set_title("A  Unequal observation depth", loc="left")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    for i, r in enumerate(d.itertuples()):
        axes[1].plot(
            [100 * r.high_fraction, 100 * r.common_high_fraction],
            [i, i],
            color="#B9C2C8",
            lw=1.2,
        )
        axes[1].scatter(
            100 * r.high_fraction,
            i,
            s=26,
            facecolor="white",
            edgecolor=PALETTE[r.cell_line],
            zorder=3,
        )
        axes[1].scatter(
            100 * r.common_high_fraction, i, s=26, color=PALETTE[r.cell_line], zorder=3
        )
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Positions scoring ≥0.5 (%)")
    axes[1].set_title("B  The denominator changes the answer", loc="left")
    axes[1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                lw=0,
                mfc="white",
                mec=INK,
                label="All eligible in this sample",
            ),
            Line2D(
                [],
                [],
                marker="o",
                lw=0,
                color=INK,
                label="Same 26,024 common positions",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    footer(
        fig,
        "Each row is one biological replicate. Contributing run-sites require ≥20 reads.\nTechnical runs are averaged equally; predicted modified sites have scores ≥0.5.",
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.92, wspace=0.15)
    save(
        fig,
        out,
        "01_observation_and_denominators",
        "Site availability and predicted modified-site fractions. A: eligible transcript sites in each biological replicate; the dashed line marks sites common to all 17. B: fractions scoring at least 0.5 among all eligible sites (open symbols) and the common sites (filled symbols). Connected symbols refer to the same replicate.",
    )


def similarity(a, t, out):
    corr = pd.read_csv(t / "sample_correlations.csv", index_col=0)
    order = pd.read_csv(t / "cluster_order.csv").replicate_id
    corr = corr.loc[order, order]
    pairs = pd.read_csv(t / "replicate_pairs_platform.csv")
    fig = plt.figure(figsize=(11.2, 7.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], wspace=0.48, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(corr, cmap="Blues", vmin=0.75, vmax=1)
    ax.set_xticks(
        range(17), [short(x) for x in order], rotation=65, ha="right", fontsize=8
    )
    ax.set_yticks(range(17), [short(x) for x in order], fontsize=8)
    ax.set_title("A  Biological replicates on identical positions", loc="left", pad=15)
    cb = fig.colorbar(im, ax=ax, shrink=0.65, fraction=0.035, pad=0.025)
    cb.set_label("Spearman correlation")
    for k, (metric, label, title) in enumerate(
        [
            ("spearman", "Spearman correlation", "B  Reproducibility"),
            ("mae", "Mean absolute score difference", "C  Absolute disagreement"),
        ]
    ):
        p = fig.add_subplot(gs[k, 1])
        for j, kind in enumerate(["technical", "biological"]):
            sub = pairs.loc[pairs.kind.eq(kind)].sort_values(
                ["cell_line", "left", "right"]
            )
            x = j + np.linspace(-0.16, 0.16, len(sub))
            for xi, row in zip(x, sub.itertuples()):
                p.scatter(
                    xi,
                    getattr(row, metric),
                    color=PALETTE[row.cell_line],
                    s=32,
                    marker="o" if row.platform_match else "D",
                    edgecolor="white",
                    lw=0.5,
                    zorder=3,
                )
            p.plot([j - 0.23, j + 0.23], [sub[metric].median()] * 2, color=INK, lw=1.3)
        p.set_xticks([0, 1], ["Technical\n6 pairs", "Biological\n13 pairs"])
        p.set_ylabel(label)
        p.set_title(title, loc="left")
        p.set_ylim(bottom=0, top=1 if metric == "spearman" else None)
    legend_cells(fig, 0.045)
    footer(
        fig,
        "Heatmap: 26,024 common positions; average linkage of 1 − correlation. Scale 0.75–1.00.\nPair plots use pair-specific matched positions. Lines are medians. Diamonds: different platforms. Pairs can share samples.",
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.92)
    save(
        fig,
        out,
        "02_similarity_and_replicates",
        "Replicate agreement. The clustered correlation heatmap uses sites common to all 17 biological replicates. Technical-run and biological-replicate comparisons use their pair-specific matched sites. Colours identify cell lines, diamonds mark different sequencing platforms and bars mark medians. Pairs can share replicates.",
    )


def contrasts(a, t, out):
    d = pd.read_csv(a / "cell_pairs.csv")
    d = d.loc[d.scope.eq("matched_replicates")].sort_values("between_minus_within")
    fig, ax = plt.subplots(figsize=(10.2, 7))
    y = np.arange(len(d))
    vals = d.between_minus_within
    ax.hlines(y, 0, vals, color=np.where(vals.ge(0), BLUE, ORANGE), lw=1.5)
    ax.scatter(vals, y, c=np.where(vals.ge(0), BLUE, ORANGE), s=32, zorder=3)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, d.cell_a + " / " + d.cell_b)
    ax.set_xlabel("Between-cell-line difference − within-cell-line difference")
    ax.set_title(
        "Cell-line differences are small relative to replicate variation",
        loc="left",
        pad=15,
    )
    ax.set_xlim(-0.0027, 0.0051)
    ax.set_ylim(-1, len(d))
    ax.grid(axis="x", alpha=0.15)
    for i, r in enumerate(d.itertuples()):
        ax.text(0.005, i, f"{r.matched_sites:,}", ha="right", va="center", fontsize=9)
    ax.text(
        0.005, len(d) - 0.2, "Matched positions", ha="right", va="bottom", fontsize=9
    )
    footer(
        fig,
        "≥2 biological replicates per cell line; identical sites for both quantities.\nRead-coverage ratios ≤2 across contributing replicates; points show descriptive score differences.",
    )
    fig.subplots_adjust(left=0.19, bottom=0.15, right=0.98, top=0.90)
    save(
        fig,
        out,
        "03_between_within_effects",
        "All 21 cell-line contrasts. At each matched site, subtract the equal-cell-line mean of within-cell-line absolute score differences from the mean of cross-cell-line differences. Positive values indicate greater variation between cell lines. Each cell line has at least two biological replicates; read-coverage ratios across contributing replicates are at most two. Site counts are shown for each contrast.",
    )


def sharing(a, t, out):
    counts = pd.read_csv(t / "sharing_counts.csv").set_index("high_cells").positions
    top = pd.read_csv(t / "support_intersections.csv").head(12)
    fig = plt.figure(figsize=(10.8, 5.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1.6], height_ratios=[1.4, 1], wspace=0.3, hspace=0.04
    )
    ax = fig.add_subplot(gs[:, 0])
    bars = ax.bar(range(1, 8), counts.loc[1:7], color=BLUE)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Cell lines consistently predicted modified")
    ax.set_ylabel("Transcript positions")
    ax.set_title("A  Sharing spectrum", loc="left")
    ax.margins(y=0.2)
    b = fig.add_subplot(gs[0, 1])
    b.bar(range(len(top)), top.positions, color=BLUE)
    b.set_xticks([])
    b.set_title("B  Most frequent support combinations", loc="left")
    b.set_ylabel("Positions")
    b.margins(y=0.2)
    b.bar_label(b.containers[0], padding=2, fontsize=8)
    c = fig.add_subplot(gs[1, 1])
    for x, row in enumerate(top.itertuples()):
        yy = np.flatnonzero([getattr(row, cell) for cell in CELLS])
        c.scatter([x] * 7, range(7), s=16, color="#D9E0E5")
        c.plot([x] * len(yy), yy, "o-", color=BLUE, ms=5, lw=1.1)
    c.set_yticks(range(7), CELLS)
    c.set_ylim(6.6, -0.6)
    c.set_xlim(b.get_xlim())
    c.set_xticks([])
    c.set_xlabel("Connected dots identify consistently modified cell lines")
    footer(
        fig,
        "Consistently modified: every declared replicate scores ≥0.5. Common sites: 26,024.\n25,345 sites have no consistently modified cell line. Pale dots indicate unmodified or discordant predictions.\n18 of 156 sites modified in exactly one cell line are consistently unmodified in the other six.",
    )
    fig.subplots_adjust(bottom=0.25, top=0.9, left=0.07, right=0.99)
    save(
        fig,
        out,
        "04_shared_and_restricted_sites",
        "Consistent modified predictions across cell lines. Left: number of cell lines in which every biological replicate scores at least 0.5. Right: the 12 most frequent exact combinations. Pale dots include consistently unmodified and discordant predictions. Of 156 sites consistently predicted modified in exactly one cell line, 18 are consistently predicted unmodified in all six others.",
    )


def motifs(a, t, out):
    d = pd.read_csv(t / "motif_summary.csv")
    matrix = (
        d.groupby(["motif", "cell_line"])
        .high_fraction.mean()
        .unstack()
        .reindex(columns=CELLS)
    )
    n = d.groupby("motif").positions.first().reindex(matrix.index)
    fig, ax = plt.subplots(figsize=(8.8, 7.1))
    im = ax.imshow(
        matrix * 100, cmap="YlGnBu", vmin=0, vmax=max(1, matrix.max().max() * 100)
    )
    ax.set_yticks(
        range(len(matrix)), [f"{m}   (n={int(n[m]):,})" for m in matrix.index]
    )
    ax.set_xticks(range(7), CELLS, rotation=35, ha="right")
    for i in range(len(matrix)):
        for j in range(7):
            v = matrix.iloc[i, j] * 100
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.6 * matrix.max().max() * 100 else INK,
            )
    fig.colorbar(
        im, ax=ax, fraction=0.04, pad=0.03, label="Mean fraction scoring ≥0.5 (%)"
    )
    ax.set_title("All observed central DRACH motifs", loc="left", pad=14)
    footer(
        fig,
        "One set of common sites; n is the number of positions of that motif in each biological replicate.\nBiological replicates receive equal weight within each cell. Sequence is a model input, so this is descriptive.",
    )
    fig.subplots_adjust(left=0.28, right=0.88, bottom=0.15, top=0.9)
    save(
        fig,
        out,
        "05_motif_composition",
        "Predicted modified-site fractions for all observed central DRACH motifs. Counts beside rows give each motif’s eligible-site denominator among common sites. Cell-line values average biological replicates equally. Sequence is an input to the model.",
    )


def signals(a, t, out):
    d = pd.read_csv(t / "signal_sample_contrasts.csv")
    features = ["current_center", "signal_sd_center", "log_dwell_center"]
    labels = [
        "Mean current difference\n(input current units)",
        "Signal SD difference\n(input current units)",
        "Mean log(1 + 1000 × dwell)\ndifference",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 8.2))
    for i, motif in enumerate(MOTIFS):
        for j, feature in enumerate(features):
            ax = axes[i, j]
            sub = d.loc[d.motif.eq(motif)]
            for k, cell in enumerate(CELLS):
                group = sub.loc[sub.cell_line.eq(cell)]
                x = (
                    k + np.linspace(-0.17, 0.17, len(group))
                    if len(group) > 1
                    else np.array([k])
                )
                ax.scatter(
                    x,
                    group[feature],
                    color=PALETTE[cell],
                    s=26,
                    edgecolor="white",
                    lw=0.4,
                    zorder=3,
                )
            ax.axhline(0, color=GRAY, lw=0.9)
            ax.set_xticks(range(7), CELLS, rotation=60, ha="right", fontsize=8)
            ax.set_title(
                f"{chr(65+i*3+j)}  {motif} · {int(sub.pairs.sum()):,} pairs",
                loc="left",
                fontsize=10,
            )
            ax.set_ylabel(labels[j], fontsize=9)
            ax.set_xlim(-0.6, 6.6)
            ax.grid(axis="y", alpha=0.13)
    footer(
        fig,
        "Each dot: one biological replicate’s median paired difference (predicted modified minus unmodified).\nModified ≥0.5; unmodified <0.5. Pairs have identical seven-mers and read-coverage ratios ≤2.\nAll reads in eligible runs are summarized. These features are model inputs.",
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.95, bottom=0.18, hspace=0.8, wspace=0.38
    )
    save(
        fig,
        out,
        "06_matched_signal_features",
        "Signal comparisons matched by seven-mer and read coverage. Within each biological replicate and motif, match at most 200 predicted modified sites to unused predicted unmodified sites with identical seven-mers and the closest log read coverage within a twofold ratio. Summarize all reads in each site, average eligible technical runs equally and calculate paired differences. Each point is one biological replicate’s median difference. These panels show the central position; all nine signal summaries are retained in the tables.",
    )
    # Full spatial context: equal-cell summaries of sample medians; preserve all offsets.
    rows = d.groupby(["motif", "cell_line"])[FEATURES].median()
    fig, axes = plt.subplots(1, 3, figsize=(11, 7), sharey=True)
    for j, (channel, label) in enumerate(
        [
            ("current", "Mean current"),
            ("signal_sd", "Signal SD"),
            ("log_dwell", "Mean transformed dwell"),
        ]
    ):
        values = rows[
            [channel + "_" + offset for offset in ["minus1", "center", "plus1"]]
        ]
        bound = np.nanmax(np.abs(values.to_numpy()))
        im = axes[j].imshow(
            values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto"
        )
        axes[j].set_xticks([0, 1, 2], ["−1", "Centre", "+1"])
        axes[j].set_title(label, loc="left")
        axes[j].set_yticks(
            range(len(rows)), [f"{m} / {c}" for m, c in rows.index], fontsize=8
        )
        fig.colorbar(
            im,
            ax=axes[j],
            orientation="horizontal",
            fraction=0.04,
            pad=0.09,
            label="Modified − unmodified",
        )
    footer(
        fig,
        "Each heatmap row summarizes biological-replicate medians within one cell/motif. Each channel has its own scale.\nAll three positions are retained. Full sample values and matching counts accompany this figure.",
    )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.93, bottom=0.16, wspace=0.3)
    save(
        fig,
        out,
        "07_signal_features_all_offsets",
        "Nine signal summaries at three local positions. Each row shows the median of replicate-level paired differences within one cell line and motif. Mean current, signal standard deviation and transformed dwell time have separate colour scales centred on zero. Individual-replicate values and matching counts accompany the figure.",
    )


def topology(a, t, out):
    d = pd.read_csv(t / "metagene_bins.csv")
    edges = np.r_[d.left, d.right.iloc[-1]]
    region = pd.read_csv(a / "region_summary.csv")
    by = pd.read_csv(a / "region_by_gene.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    ax = axes[0]
    for col, color, label in [
        ("eligible_percent", GRAY, "Eligible background (24,356)"),
        ("gene_matched_percent", ORANGE, "Background matched by gene"),
        ("shared_percent", BLUE, "High in every sample (170)"),
    ]:
        ax.stairs(d[col], edges, color=color, label=label, lw=1.6)
    for x in [1, 2]:
        ax.axvline(x, color=GRAY, ls=":", lw=0.8)
    ax.set_xticks([0.5, 1.5, 2.5], ["5′ UTR", "CDS + stop", "3′ UTR"])
    ax.set_xlim(0, 3)
    ax.set_ylabel("Positions in cohort per bin (%)")
    ax.set_title("A  Position along coding transcripts", loc="left")
    ax.legend(frameon=False, fontsize=8)
    vals = [
        region.loc[region.cohort.eq(c) & region.region.eq("3UTR"), "fraction"].iloc[0]
        for c in ["eligible_background", "eligible_same_genes", "shared_candidates"]
    ]
    weighted = by.expected_candidate_3utr.sum() / by.candidate_positions.sum()
    labels = [
        "All eligible positions",
        "Eligible in candidate genes",
        "Gene-weighted background",
        "Shared candidates",
    ]
    vals = [vals[0], vals[1], weighted, vals[2]]
    axes[1].scatter(np.array(vals) * 100, range(4), c=[GRAY, GRAY, ORANGE, BLUE], s=55)
    for i, v in enumerate(vals):
        axes[1].text(v * 100 + 2, i, f"{v:.1%}", va="center", fontsize=9)
    axes[1].set_yticks(range(4), labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Positions in 3′ UTR (%)")
    axes[1].set_title("B  Control the background", loc="left")
    footer(
        fig,
        "Metagene: 10 bins per region; excludes incomplete coding/UTR boundaries. Region panel uses all 26,024 positions.\nGene weighting preserves each candidate gene’s contribution. Location is a model input, so enrichment is descriptive.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.89, wspace=0.7)
    save(
        fig,
        out,
        "08_transcript_topology",
        "Relative transcript location with eligible-site backgrounds. A: equal numbers of bins per transcript region, comparing 170 shared predicted modified sites with 24,356 eligible sites on fully annotated coding transcripts. The gene-weighted background preserves candidate contributions per gene. B: 3′ UTR fractions among all common sites, sites in candidate genes, the gene-weighted background and 181 shared candidates. Location is an input to the model.",
    )
    d = pd.read_csv(t / "anchor_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
    for ax, anchor, title in zip(
        axes,
        ["distance_stop_start", "distance_last_exon_start"],
        ["Stop-codon start", "Last-exon start"],
    ):
        for cohort, color, label in [
            ("all_coding", BLUE, "All annotated coding positions"),
            (
                "stop_at_least_150nt_after_last_exon_start",
                ORANGE,
                "Stop ≥150 nt after last-exon start",
            ),
        ]:
            frame = d.loc[d.anchor.eq(anchor) & d.cohort.eq(cohort)]
            mids = (frame.left + frame.right) / 2
            supported = frame.eligible.ge(20)
            ax.plot(
                mids,
                frame.percent_shared.where(supported),
                "-o",
                color=color,
                ms=3,
                lw=1.2,
                label=label,
            )
        ax.axvline(0, color=GRAY, ls=":", lw=0.9)
        ax.set_xlabel(f"Distance from {title.lower()} (nt)")
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-500, 500)
    axes[0].set_ylabel("Eligible positions high in all samples (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    footer(
        fig,
        "50-nt bins; candidate count divided by observed eligible positions within each bin. Bins with <20 positions are omitted.\nThe restricted cohort separates the two anchors by ≥150 nt; counts are in anchor_profiles.csv. This is descriptive.",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.15)
    save(
        fig,
        out,
        "09_stop_and_last_exon",
        "Stop-codon and last-exon alignments. Each 50-nt bin shows shared predicted modified sites divided by eligible sites. A second curve uses transcripts whose stop codon begins at least 150 nt after the last-exon start. Displayed bins contain at least 20 eligible sites; all numerator and denominator counts are saved.",
    )


def burden(a, t, out):
    ranking = pd.read_csv(t / "gene_burden_ranking_min10.csv").head(12)
    genes = pd.read_parquet(t / "gene_sample_burden.parquet")
    info = pd.read_csv(t / "biological_sample_info.csv")
    matrix = genes.pivot(
        index="gene_id", columns="replicate_id", values="high_fraction"
    ).reindex(index=ranking.gene_id, columns=info.replicate_id)
    fig, ax = plt.subplots(figsize=(10.1, 5.6))
    im = ax.imshow(
        matrix * 100,
        cmap="YlGnBu",
        vmin=0,
        vmax=max(10, matrix.max().max() * 100),
        aspect="auto",
    )
    ax.set_yticks(
        range(len(ranking)),
        [f"{r.gene_name}  (n={r.eligible_positions})" for r in ranking.itertuples()],
    )
    ax.set_xticks(range(17), info.label, rotation=60, ha="right", fontsize=8)
    for edge in (
        np.flatnonzero(info.cell_line.to_numpy()[1:] != info.cell_line.to_numpy()[:-1])
        + 0.5
    ):
        ax.axvline(edge, color="white", lw=2)
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label="Eligible positions scoring ≥0.5 (%)",
    )
    ax.set_title(
        "Genes with the highest predicted modified-site fractions", loc="left", pad=15
    )
    footer(
        fig,
        "One transcript per gene, selected by most commonly observed positions before looking at scores; minimum 10 positions.\nRank: mean predicted modified-site fraction, with equal biological-replicate weight within cells and equal weight across cells.\nEach point represents one biological replicate; n is the eligible-site denominator.",
    )
    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.29, top=0.89)
    save(
        fig,
        out,
        "10_gene_burden",
        "Fraction of predicted modified sites per gene. Select the transcript with the most common observed sites, breaking ties by transcript ID. Require at least ten sites and rank fractions with equal replicate weight within cell lines and equal cell-line weight. Points show all 17 biological replicates for the 12 leading genes; counts beside gene names give the site denominator. Rankings with minima of five and twenty sites are also provided.",
    )


def discovery(a, t, out):
    d = pd.read_csv(t / "gene_discovery_confirmation.csv")
    lim = max(
        0.33,
        np.abs(d[["discovery_delta", "confirmation_delta"]].to_numpy()).max() * 1.04,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.3))
    hb = axes[0].hexbin(
        d.discovery_delta,
        d.confirmation_delta,
        gridsize=40,
        mincnt=1,
        bins="log",
        extent=(-lim, lim, -lim, lim),
        cmap="Blues",
    )
    fig.colorbar(
        hb, ax=axes[0], fraction=0.04, pad=0.02, label="Genes per hexagon (log scale)"
    )
    top = d.head(100)
    axes[1].scatter(
        top.discovery_delta,
        top.confirmation_delta,
        s=20,
        c=np.where(top.retained_direction, BLUE, ORANGE),
        edgecolor="none",
        alpha=0.8,
    )
    for ax in axes:
        ax.plot([-lim, lim], [-lim, lim], "--", color=GRAY, lw=0.9)
        ax.axhline(0, color=GRAY, lw=0.7)
        ax.axvline(0, color=GRAY, lw=0.7)
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            xlabel="Discovery score difference (B − A)",
            ylabel="Confirmation score difference (B − A)",
        )
        ax.set_aspect("equal")
    axes[0].set_title(f"A  All {len(d):,} selected genes", loc="left")
    axes[1].set_title("B  Top 100 discovery-ranked genes", loc="left")
    footer(
        fig,
        "One strongest discovery-ranked transcript/cell comparison per gene. Ranking uses initial-replicate scores.\nDirection retained: 2,571/4,517 overall; 73/100 among top 100. Orange points reverse or lose direction.\nThe dashed line denotes equal effects; near-zero confirmation values show shrinkage.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.89, wspace=0.4)
    save(
        fig,
        out,
        "11_discovery_and_confirmation",
        "Initial and additional biological-replicate comparisons. Select one transcript/cell-line contrast per gene using initial-replicate scores. Plot the signed mean score difference on the selected sites in initial and additional replicates. The diagonal marks equal effects. Coverage eligibility uses both subsets; only initial-replicate scores determine ranking. Direction retention and effect-size reduction are reported separately.",
    )


def gene_comparison_figure(a):
    """Compare the fixed examples; sample counts describe support at each site."""
    examples = pd.read_csv(a / "examples.csv")
    sites = pd.read_csv(a / "example_supported_sites.csv")
    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    labels, support = [], []
    for i, e in enumerate(examples.itertuples()):
        selected = sites.loc[sites.transcript_id.eq(e.transcript_id)]
        labels.append(
            f"{e.gene_name} ({len(selected)} sites)\n{e.cell_b} − {e.cell_a}"
        )
        counts = []
        for suffix in ["b", "a"]:
            columns = [
                c for c in selected
                if c.startswith("replicate_score_") and c.endswith(f"_{suffix}")
                and c[16:-2].isdigit() and int(c[16:-2]) > 0
            ]
            n = selected[columns].notna().sum(axis=1)
            counts.append(str(n.min()) if n.min() == n.max() else f"{n.min()}–{n.max()}")
        support.append(" / ".join(counts))
        ax.plot(
            [e.discovery_delta, e.confirmation_delta], [i, i],
            color=GRAY, lw=1.5, zorder=2,
        )
        ax.scatter(
            e.discovery_delta, i, s=48, facecolors="white", edgecolors=ORANGE,
            linewidths=1.5, zorder=3, label="Initial samples" if i == 0 else None,
        )
        ax.scatter(
            e.confirmation_delta, i, s=44, color=BLUE, zorder=3,
            label="Additional samples" if i == 0 else None,
        )
        for value, dy in [(e.discovery_delta, 11), (e.confirmation_delta, -15)]:
            ax.annotate(
                f"{value:+.3f}", (value, i), xytext=(0, dy),
                textcoords="offset points", ha="center", fontsize=9, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
    limit = max(0.35, examples[["discovery_delta", "confirmation_delta"]].abs().max().max() * 1.18)
    ax.set(xlim=(-limit, limit), ylim=(len(examples) - 0.5, -0.5))
    ax.set_yticks(range(len(examples)), labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Mean prediction-score difference", fontsize=9.5, labelpad=7)
    ax.tick_params(axis="x", labelsize=9)
    for i, label in enumerate(support):
        ax.text(1.10, i, label, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=9.5)
    fig.text(0.865, 0.80, "Additional samples\nper site", ha="center", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.50, 0.94), frameon=False, ncol=2, fontsize=9,
               handletextpad=0.4, columnspacing=1.8)
    fig.text(0.02, 0.98, "Selected gene differences shrink in additional samples",
             ha="left", va="top", weight="bold", fontsize=11)
    fig.subplots_adjust(left=0.25, right=0.81, top=0.73, bottom=0.18)
    caption = (
        "Mean prediction-score differences for the three highest-ranked distinct genes "
        "selected using initial samples. Each row keeps the same cell-line contrast and "
        "three transcript sites in initial and additional samples. At each site, available "
        "biological replicates are averaged within each cell line; site differences then "
        "receive equal weight. Hollow orange and filled blue points denote initial and "
        "additional samples; connectors link the two estimates, and the dashed line is zero. "
        "One initial sample contributes per cell line. Counts on the right give additional "
        "samples per site in the listed cell-line order; ranges indicate unequal observation "
        "across sites. Individual-site scores and read coverage are shown in the gene-specific figures."
    )
    return fig, caption


def gene_comparison(a, t, out):
    fig, caption = gene_comparison_figure(a)
    save(fig, out, "17_gene_effect_comparison", caption)


def loci(a, t, out):
    examples = pd.read_csv(a / "examples.csv")
    raw = pd.read_parquet(a / "example_sites.parquet")
    sup = pd.read_csv(a / "example_supported_sites.csv")
    structures = json.loads((t / "locus_structures.json").read_text())
    for idx, e in enumerate(examples.itertuples()):
        positions = sorted(
            sup.loc[sup.transcript_id.eq(e.transcript_id), "transcript_position"]
            .astype(int)
            .unique()
        )
        sub = raw.loc[
            raw.transcript_id.eq(e.transcript_id)
            & raw.cell_line.isin([e.cell_a, e.cell_b])
            & raw.eligible
        ].copy()
        sub["transcript_position"] = sub.transcript_position.astype(int)
        sub = (
            sub.loc[sub.transcript_position.isin(positions)]
            .groupby(["cell_line", "replicate_id", "transcript_position"])
            .agg(score=("score", "mean"), n_reads=("n_reads", "mean"))
            .reset_index()
        )
        g = structures[e.transcript_id]
        L = g["boundaries"][-1]
        fig = plt.figure(figsize=(9.2, 6.4))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.5, 1.15], hspace=0.6)
        ax = fig.add_subplot(gs[0])
        ax.hlines(0.5, 0, L, color=GRAY, lw=1)
        for start, end in g["UTR"]:
            ax.add_patch(Rectangle((start, 0.37), end - start, 0.26, color="#B7C8D2"))
        for start, end in g["CDS"]:
            ax.add_patch(Rectangle((start, 0.25), end - start, 0.5, color=BLUE))
        for boundary in g["boundaries"][1:-1]:
            ax.axvline(boundary, color="white", lw=1)
        ax.axvspan(min(positions) - 8, max(positions) + 8, color=ORANGE, alpha=0.18)
        ax.set(
            xlim=(0, L),
            ylim=(0, 1),
            yticks=[],
            xlabel="Full transcript coordinate (nt, zero-based)",
        )
        ax.set_title(f"{e.gene_name} · {e.transcript_id}", loc="left")
        axes = [fig.add_subplot(gs[i]) for i in [1, 2]]
        reps = (
            sub[["cell_line", "replicate_id"]]
            .drop_duplicates()
            .sort_values(["cell_line", "replicate_id"])
        )
        for k, r in enumerate(reps.itertuples()):
            rr = sub.loc[sub.replicate_id.eq(r.replicate_id)]
            disc = r.replicate_id in [e.discovery_replicate_a, e.discovery_replicate_b]
            offset = (k - (len(reps) - 1) / 2) * 1.1
            for target, metric in zip(axes, ["score", "n_reads"]):
                target.scatter(
                    rr.transcript_position + offset,
                    rr[metric],
                    color=PALETTE[r.cell_line],
                    facecolor="white" if disc else PALETTE[r.cell_line],
                    edgecolor=PALETTE[r.cell_line],
                    marker="o" if r.cell_line == e.cell_a else "s",
                    s=35,
                    label=short(r.replicate_id)
                    + (" (discovery)" if disc else " (confirmation)"),
                    zorder=3,
                )
        axes[0].set(ylabel="Prediction score", ylim=(0, 1))
        axes[0].axhline(0.5, color=GRAY, ls=":", lw=0.8)
        axes[1].set_ylabel("Mean eligible-run read count")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Selected transcript position (nt, zero-based)")
        for ax in axes:
            ax.set_xticks(positions)
            ax.set_xlim(min(positions) - 12, max(positions) + 12)
            ax.grid(axis="y", alpha=0.15)
        axes[0].legend(
            frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01, 1)
        )
        footer(
            fig,
            f"Discovery difference ({e.cell_b} − {e.cell_a}): {e.discovery_delta:+.3f}; confirmation: {e.confirmation_delta:+.3f}.\nEvery dot is a biological replicate. Horizontal jitter only separates symbols; hollow symbols are discovery samples.\nRead-count points are observed eligible site depths, not a continuous sequencing-coverage track.",
        )
        fig.subplots_adjust(left=0.1, right=0.73, bottom=0.16, top=0.92)
        save(
            fig,
            out,
            f"{12+idx:02d}_locus_{e.gene_name}",
            f"{e.gene_name}: the discovery-selected local example, including all available biological-replicate scores and read counts at the original selected positions. The upper track shows annotated UTRs (thin), CDS (thick), exon boundaries and the highlighted window on full transcript coordinates. Lower panels zoom to selected sites; jitter is only graphical. Hollow points are discovery and filled points confirmation. Missing values are not plotted as zero. Depths are sampled site counts rather than base-resolution coverage.",
        )


def robustness(a, t, out):
    d = pd.read_csv(a / "threshold_stability.csv")
    p = d.loc[d.score_threshold.eq(0.5)]
    overlap = pd.read_csv(t / "training_overlap_run_sensitivity.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.6))
    bottom = np.zeros(len(p))
    for field, label, color in [
        ("baseline_candidates_retained", "Original candidates retained", BLUE),
        ("lost_coverage", "Original candidates: observation lost", GRAY),
        ("lost_score", "Original candidates: score lost", ORANGE),
    ]:
        vals = p[field].to_numpy()
        axes[0].bar(range(len(p)), vals, bottom=bottom, color=color, label=label)
        for i, v in enumerate(vals):
            if v > 5:
                axes[0].text(
                    i,
                    bottom[i] + v / 2,
                    str(v),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        bottom += vals
    axes[0].set_xticks(range(len(p)), p.min_reads.astype(str))
    axes[0].set_xlabel("Minimum reads in each run")
    axes[0].set_ylabel("Original 181 shared candidates")
    axes[0].set_title("A  Most attrition is lost observation", loc="left")
    axes[0].legend(
        frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14)
    )
    q = d.loc[d.min_reads.eq(20)]
    bars = axes[1].bar(q.score_threshold.astype(str), q.candidate_positions, color=BLUE)
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xlabel("Minimum score in every biological replicate")
    axes[1].set_ylabel("Shared candidate positions")
    axes[1].set_title("B  The catalogue depends on the cutoff", loc="left")
    axes[1].margins(y=0.15)
    footer(
        fig,
        f"New candidates after rebuilding run means: 3 at 30 reads; 2 at 50 reads (not in the left stacks).\nRemoving the training-overlap run retains {int(overlap.retained_common):,} common positions and gives {int(overlap.without_overlap_shared)} shared candidates\non that retained universe; {int(overlap.lost_observation)} original candidates lose observation and {int(overlap.lost_score)} lose score support.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.35, top=0.88, wspace=0.3)
    save(
        fig,
        out,
        "15_threshold_robustness",
        "Read-coverage and score-cutoff sensitivity. Reapply coverage cutoffs to each original run, rebuild biological-replicate means and require all declared replicates. Stacks separate loss of observation from loss of score support among the original 181 candidates; newly eligible candidates are reported separately. Score-cutoff comparisons use the fixed 20-read analysis. The run-exclusion analysis removes Hct116 replicate 3 run 1, which overlaps Dataset 0, while retaining run 4.",
    )


def overlap_contrasts(a, t, out):
    d = pd.read_csv(t / "training_overlap_cell_contrasts.csv")
    fig, ax = plt.subplots(figsize=(9.3, 4.9))
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        ax.plot(
            [r.baseline_excess_on_retained, r.current_excess_on_retained],
            [i, i],
            color=GRAY,
            lw=1.5,
        )
        ax.scatter(
            r.baseline_excess_on_retained,
            i,
            facecolor="white",
            edgecolor=BLUE,
            s=50,
            zorder=3,
        )
        ax.scatter(r.current_excess_on_retained, i, color=BLUE, s=50, zorder=3)
        ax.text(
            0.0075,
            i,
            f"{r.baseline_positions_retained:,}",
            ha="right",
            va="center",
            fontsize=10,
        )
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set_yticks(y, "Hct116 / " + d.cell_b)
    ax.set_xlabel("Between − within effect on the same retained positions")
    ax.set_xlim(-0.002, 0.0077)
    ax.set_title(
        "Small cell contrasts are sensitive to the overlapping run", loc="left", pad=18
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                mfc="white",
                mec=BLUE,
                lw=0,
                label="Original run mean",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color=BLUE,
                lw=0,
                label="After removing overlapping run",
            ),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.17),
    )
    ax.text(0.0075, 5.55, "Retained positions", ha="right", va="center", fontsize=9)
    footer(
        fig,
        "Hct116 replicate 3 run 1 is removed; run 4 and every other biological replicate are retained.\nEach connected pair uses identical positions and the original twofold depth rule. Two contrasts change sign.\nRun 4 from the same biological replicate remains included.",
    )
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.31, top=0.85)
    save(
        fig,
        out,
        "16_overlap_run_cell_contrasts",
        "The six Hct116 contrasts before and after excluding the run overlapping Dataset 0. Connected points use identical sites meeting the read-coverage and replicate-support criteria in both analyses. All six effects increase and two change sign. Site turnover and values on each full eligible set are saved in training_overlap_cell_contrasts.csv.",
    )



def render_all(a):
    style()
    t = a / "tables"
    out = a / "figures_scientific"
    out.mkdir(exist_ok=True)
    for function in [
        support,
        similarity,
        contrasts,
        sharing,
        motifs,
        signals,
        topology,
        burden,
        discovery,
        loci,
        robustness,
        overlap_contrasts,
        gene_comparison,
    ]:
        function(a, t, out)
        print(f"Figures: {function.__name__}", flush=True)
    (a / "figure_captions.json").write_text(
        json.dumps(CAPTIONS, indent=2, ensure_ascii=False), encoding="utf8"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_context_trees",
    )
    render_all(p.parse_args().analysis)
