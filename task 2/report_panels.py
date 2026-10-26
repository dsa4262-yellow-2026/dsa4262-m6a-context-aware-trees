"""Ten compact figure layouts, drawn at publication width.

Read existing evidence tables without rerunning prediction or analysis.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scientific_figures import style, PALETTE, BLUE, ORANGE, INK, GRAY, gene_comparison_figure
from explore import CELLS, short

CAPTIONS = {}



def save(fig, out, name, caption):
    for ext in ["png", "svg", "pdf"]:
        metadata = (
            {"Software": None}
            if ext == "png"
            else (
                {"Date": None, "Creator": None}
                if ext == "svg"
                else {"Creator": None, "CreationDate": None, "ModDate": None}
            )
        )
        fig.savefig(out / f"{name}.{ext}", dpi=260, metadata=metadata)
    CAPTIONS[name] = caption
    plt.close(fig)


def cell_legend(fig, y=0.01):
    fig.legend(
        handles=[
            Line2D([], [], marker="o", lw=0, color=PALETTE[c], label=c) for c in CELLS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=7,
        frameon=False,
        fontsize=9,
        handlelength=0.7,
        columnspacing=0.8,
    )


def locus(ax, a, example, panel, depth=False):
    raw = pd.read_parquet(a / "example_sites.parquet")
    supported = pd.read_csv(a / "example_supported_sites.csv")
    positions = sorted(
        supported.loc[
            supported.transcript_id.eq(example.transcript_id), "transcript_position"
        ]
        .astype(int)
        .unique()
    )
    rows = raw.loc[
        raw.transcript_id.eq(example.transcript_id)
        & raw.eligible
        & raw.cell_line.isin([example.cell_a, example.cell_b])
    ].copy()
    rows.transcript_position = rows.transcript_position.astype(int)
    rows = (
        rows.loc[rows.transcript_position.isin(positions)]
        .groupby(["cell_line", "replicate_id", "transcript_position"])[
            ["score", "n_reads"]
        ]
        .mean()
        .reset_index()
    )
    reps = (
        rows[["cell_line", "replicate_id"]]
        .drop_duplicates()
        .sort_values(["cell_line", "replicate_id"])
    )
    for i, row in enumerate(reps.itertuples()):
        rr = rows.loc[rows.replicate_id.eq(row.replicate_id)]
        initial = row.replicate_id in [
            example.discovery_replicate_a,
            example.discovery_replicate_b,
        ]
        xx = (
            np.array([positions.index(p) for p in rr.transcript_position])
            + (i - (len(reps) - 1) / 2) * 0.07
        )
        ax.scatter(
            xx,
            rr.n_reads if depth else rr.score,
            s=28,
            facecolor="white" if initial else PALETTE[row.cell_line],
            edgecolor=PALETTE[row.cell_line],
            marker="o" if row.cell_line == example.cell_a else "s",
            zorder=3,
        )
    ax.set_xticks(range(len(positions)), positions)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_xlabel("Transcript position (nt)")
    ax.set_title(f"{panel}  {example.gene_name}", loc="left")
    ax.set_ylabel("Read coverage" if depth else "Prediction score")
    if depth:
        ax.set_yscale("log")
        ax.axhline(20, ls=":", c=GRAY, lw=0.8)
    else:
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls=":", c=GRAY, lw=0.8)
    ax.text(
        0.02,
        0.97,
        f"{example.cell_a}: circles\n{example.cell_b}: squares",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )


def render(a):
    CAPTIONS.clear()
    style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    t = a / "tables"
    out = a / "figures_report"
    out.mkdir(exist_ok=True)
    d = pd.read_csv(t / "sample_support.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.65))
    for j, cell in enumerate(CELLS):
        g = d.loc[d.cell_line.eq(cell)]
        x = j + np.linspace(-0.16, 0.16, len(g))
        axs[0].scatter(x, g.eligible_positions / 1000, c=PALETTE[cell], s=24)
        axs[1].scatter(x, g.common_high_fraction * 100, c=PALETTE[cell], s=24)
    axs[0].axhline(26.024, color=GRAY, ls="--", lw=1)
    axs[0].text(0.1, 31, "26,024 common sites", fontsize=9)
    axs[0].set_ylabel("Sites with ≥20 reads (×1,000)")
    axs[1].set_ylabel("Predicted modified sites (%)")
    for ax in axs:
        ax.set_xticks(range(7), CELLS, rotation=50, ha="right")
        ax.set_ylim(bottom=0)
    axs[0].set_title("A  Sites available per replicate", loc="left")
    axs[1].set_title("B  The same sites in all replicates", loc="left")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.29, top=0.87, wspace=0.45)
    save(
        fig,
        out,
        "report_01",
        "Each point is one biological replicate. A: available human transcript sites after requiring ≥20 reads in each contributing sequencing run. B: percentage predicted modified (score ≥0.5) among the same 26,024 sites observed in all 17 replicates.",
    )

    d = pd.read_csv(t / "signal_sample_contrasts.csv")
    motifs = ["GGACT", "GAACT", "AGACT"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.85))
    columns = [
        ("current_center", "A  Mean current", "Current difference\n(input units)"),
        ("signal_sd_center", "B  Signal SD", "Signal SD difference\n(input units)"),
        ("log_dwell_center", "C  Dwell time", "Mean log(1 + 1000 × dwell)\ndifference"),
    ]
    for ax, (field, heading, label) in zip(axs, columns):
        for i, m in enumerate(motifs):
            g = d.loc[d.motif.eq(m)].sort_values("replicate_id")
            ax.scatter(
                i + np.linspace(-0.20, 0.20, len(g)),
                g[field],
                c=[PALETTE[c] for c in g.cell_line],
                s=18,
                alpha=0.85,
            )
            ax.plot([i - 0.23, i + 0.23], [g[field].median()] * 2, c=INK, lw=1.2)
        ax.axhline(0, c=GRAY, lw=0.8, ls="--")
        ax.set_xticks(range(3), motifs, rotation=40, ha="right")
        ax.set_ylabel(label)
        ax.set_title(heading, loc="left")
        ax.margins(y=0.17)
    cell_legend(fig, 0)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.32, top=0.86, wspace=0.65)
    save(
        fig,
        out,
        "report_02",
        "Central-position signal features at predicted modified minus predicted unmodified sites, matched within each biological replicate by the full 7-mer and read coverage (ratio ≤2). Each dot is one replicate’s median paired difference; horizontal bars are medians across 17 replicates. Colours identify cell lines. The three motifs contain 5,316 matched pairs in total.",
    )

    cp = pd.read_csv(a / "cell_pairs.csv")
    cp = cp.loc[cp.scope.eq("matched_replicates")]
    counts = pd.read_csv(t / "sharing_counts.csv").set_index("high_cells").positions
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.75))
    axs[0].scatter(
        cp.within_replicate_mae,
        cp.between_replicate_mae,
        s=25,
        c=np.where(cp.between_minus_within.gt(0), BLUE, ORANGE),
    )
    lim = max(cp.within_replicate_mae.max(), cp.between_replicate_mae.max()) * 1.12
    axs[0].plot([0, lim], [0, lim], ls="--", color=GRAY, lw=1)
    axs[0].set(
        xlim=(0, lim),
        ylim=(0, lim),
        xlabel="Within-cell-line score difference",
        ylabel="Between-cell-line score difference",
    )
    axs[0].text(
        0.03,
        0.96,
        "17/21 above the diagonal",
        transform=axs[0].transAxes,
        va="top",
        fontsize=9,
    )
    bars = axs[1].bar(range(1, 8), counts.loc[1:7], color=BLUE)
    axs[1].bar_label(bars, fontsize=9, padding=2)
    axs[1].margins(y=0.2)
    axs[1].set_xticks(range(1, 8))
    axs[1].set_xlabel("Cell lines with consistent predictions")
    axs[1].set_ylabel("Number of transcript sites")
    axs[0].set_title("A  Cell-line and replicate variation", loc="left")
    axs[1].set_title("B  Shared predicted m6A sites", loc="left")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.23, top=0.86, wspace=0.48)
    save(
        fig,
        out,
        "report_03",
        "A: all 21 cell-line pairs. Mean absolute score differences between and within cell lines use identical sites with ≥2 biological replicates per cell line and at most twofold differences in read coverage. B: number of cell lines in which every replicate predicts a site as modified; 25,345 common sites with no such cell line are omitted. Other cell lines may have mixed predictions.",
    )

    meta = pd.read_csv(t / "metagene_bins.csv")
    edges = np.r_[meta.left, meta.right.iloc[-1]]
    rank = pd.read_csv(t / "gene_burden_ranking_min10.csv").head(6)
    g = pd.read_parquet(t / "gene_sample_burden.parquet")
    gm = (
        g.groupby(["gene_id", "cell_line"])
        .high_fraction.mean()
        .unstack()
        .reindex(index=rank.gene_id, columns=CELLS)
        * 100
    )
    fig, axs = plt.subplots(
        1, 2, figsize=(6.9, 3.05), gridspec_kw={"width_ratios": [1.12, 1]}
    )
    for col, c, label in [
        ("eligible_percent", GRAY, "All sites"),
        ("gene_matched_percent", ORANGE, "Gene-matched"),
        ("shared_percent", BLUE, "Shared predicted"),
    ]:
        axs[0].stairs(meta[col], edges, color=c, lw=1.2, label=label)
    for x in [1, 2]:
        axs[0].axvline(x, color=GRAY, ls=":", lw=0.7)
    axs[0].set_xticks([0.5, 1.5, 2.5], ["5′ UTR", "CDS + stop", "3′ UTR"])
    axs[0].set_xlim(0, 3)
    axs[0].set_ylabel("Sites in each bin (%)")
    axs[0].legend(frameon=False, fontsize=9, loc="upper left", handlelength=1.3)
    im = axs[1].imshow(gm, cmap="YlGnBu", vmin=0, vmax=50, aspect="auto")
    axs[1].set_yticks(
        range(6), [f"{r.gene_name} ({r.eligible_positions})" for r in rank.itertuples()]
    )
    axs[1].set_xticks(range(7), CELLS, rotation=55, ha="right")
    cb = fig.colorbar(im, ax=axs[1], orientation="horizontal", fraction=0.05, pad=0.33)
    cb.set_label("Predicted modified sites (%)", fontsize=9)
    axs[0].set_title("A  Relative location along mRNAs", loc="left")
    axs[1].set_title("B  Highest-ranked genes", loc="left")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.26, top=0.87, wspace=0.68)
    save(
        fig,
        out,
        "report_04",
        "A: metagene profiles with 10 bins per region for 170 shared predicted m6A sites and 24,356 analysed sites on fully annotated coding transcripts. Each curve sums to 100%; the orange background matches the shared sites’ contribution from each gene. B: the six highest-ranked genes among 870 with ≥10 common sites on one representative transcript. Colour shows the mean replicate fraction for each cell line; parentheses give analysed-site counts.",
    )

    disc = pd.read_csv(t / "gene_discovery_confirmation.csv").head(100)
    example = pd.read_csv(a / "examples.csv").iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.85))
    lim = 0.33
    axs[0].scatter(
        disc.discovery_delta,
        disc.confirmation_delta,
        c=np.where(disc.retained_direction, BLUE, ORANGE),
        s=17,
        alpha=0.8,
    )
    axs[0].plot([-lim, lim], [-lim, lim], "--", c=GRAY, lw=0.8)
    axs[0].axhline(0, c=GRAY, lw=0.6)
    axs[0].axvline(0, c=GRAY, lw=0.6)
    axs[0].set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="Initial-replicate difference (B − A)",
        ylabel="Additional-replicate difference (B − A)",
    )
    axs[0].set_xticks([-0.3, 0, 0.3])
    axs[0].set_yticks([-0.3, 0, 0.3])
    axs[0].text(
        0.03,
        0.97,
        "73/100 retain direction",
        transform=axs[0].transAxes,
        va="top",
        fontsize=9,
    )
    axs[0].set_title("A  The 100 largest initial differences", loc="left")
    locus(axs[1], a, example, "B")
    fig.text(
        0.70,
        0.02,
        "Hollow: initial; filled: additional replicates",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.27, top=0.86, wspace=0.46)
    save(
        fig,
        out,
        "report_05",
        "A: one comparison per gene, ranked using only the initial biological replicate from each cell line (lowest replicate number). Remaining replicates assess the same sites. Orange points reverse direction; the dashed line is equality. B: RNASEK scores at the three selected transcript sites in every observed biological replicate. Positions are equally spaced for readability, with small horizontal offsets to separate replicates. The HEYA8 − H9 mean difference falls from +0.303 to +0.079. The SAMD4B and C20orf24 compact layout shows their selected-site scores and read coverage.",
    )

    corr = pd.read_csv(t / "sample_correlations.csv", index_col=0)
    order = pd.read_csv(t / "cluster_order.csv").replicate_id
    corr = corr.loc[order, order]
    pairs = pd.read_csv(t / "replicate_pairs_platform.csv")
    fig, axs = plt.subplots(
        1, 2, figsize=(6.9, 4.4), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    im = axs[0].imshow(corr, cmap="Blues", vmin=0.75, vmax=1)
    axs[0].set_xticks(range(17), [short(x) for x in order], rotation=90, fontsize=9)
    axs[0].set_yticks(range(17), [short(x) for x in order], fontsize=9)
    cb = fig.colorbar(im, ax=axs[0], orientation="horizontal", fraction=0.045, pad=0.34)
    cb.set_label("Spearman correlation", fontsize=9)
    for i, kind in enumerate(["technical", "biological"]):
        g = pairs.loc[pairs.kind.eq(kind)].sort_values(["cell_line", "left", "right"])
        axs[1].scatter(
            i + np.linspace(-0.16, 0.16, len(g)),
            g.mae,
            c=[PALETTE[c] for c in g.cell_line],
            s=24,
        )
        axs[1].plot([i - 0.23, i + 0.23], [g.mae.median()] * 2, color=INK, lw=1.3)
    axs[1].set_xticks([0, 1], ["Technical\n6 pairs", "Biological\n13 pairs"])
    axs[1].set_ylabel("Mean absolute score difference")
    axs[1].set_ylim(bottom=0)
    axs[0].set_title("A  All biological replicates", loc="left")
    axs[1].set_title("B  Replicate pairs", loc="left")
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.89, wspace=0.60)
    cell_legend(fig, -0.002)
    save(
        fig,
        out,
        "appendix_01",
        "A: the 17-replicate Spearman matrix on the same 26,024 sites, ordered by average-linkage clustering of 1 − correlation. r denotes biological replicate number. The scale starts at 0.75. B: every eligible technical- and biological-replicate pair, using its shared sites. Bars are medians. Pairwise sites differ from the complete common-site set in A; pairs can share replicates.",
    )

    motifs = pd.read_csv(t / "motif_summary.csv")
    matrix = (
        motifs.groupby(["motif", "cell_line"])
        .high_fraction.mean()
        .unstack()
        .reindex(columns=CELLS)
        * 100
    )
    den = motifs.groupby("motif").positions.first().reindex(matrix.index)
    fig, ax = plt.subplots(figsize=(6.9, 3.75))
    im = ax.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=np.ceil(matrix.to_numpy().max() / 5) * 5,
        aspect="auto",
    )
    ax.set_yticks(
        range(len(matrix)), [f"{m}  (n={int(den[m]):,})" for m in matrix.index]
    )
    ax.set_xticks(range(7), CELLS)
    ax.set_xlabel("Cell line")
    ax.set_ylabel("Central DRACH 5-mer")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("Predicted modified sites (%)")
    fig.subplots_adjust(left=0.29, right=0.89, bottom=0.13, top=0.98)
    save(
        fig,
        out,
        "appendix_02",
        "All 18 DRACH motifs among the 26,024 common sites. Each entry is the fraction scoring ≥0.5, averaged equally over biological replicates in that cell line. The motif-specific number of common sites is shown beside each row and is identical across cell lines. T is retained from the reference sequence notation (U in RNA). Matched signal comparisons additionally match the two outer bases of the 7-mer.",
    )

    fig, caption = gene_comparison_figure(a)
    save(fig, out, "appendix_03", caption)

    examples = pd.read_csv(a / "examples.csv").iloc[1:3]
    fig, axs = plt.subplots(
        2, 2, figsize=(6.9, 4.7), gridspec_kw={"height_ratios": [1.15, 1]}
    )
    for j, (_, example) in enumerate(examples.iterrows()):
        locus(axs[0, j], a, example, "A" if j == 0 else "B")
        locus(axs[1, j], a, example, "C" if j == 0 else "D", depth=True)
        axs[1, j].set_title("")
        axs[1, j].texts[0].remove()
    fig.text(
        0.5,
        0.015,
        "Hollow: initial replicate; filled: additional replicates. Dotted lines: score 0.5 / 20 reads.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.15, top=0.93, wspace=0.44, hspace=0.70
    )
    save(
        fig,
        out,
        "appendix_04",
        "The second and third genes selected by the same initial-replicate ranking as RNASEK. A–B: scores at the fixed three sites per gene. C–D: mean eligible-run read counts at those sites (log scale). SAMD4B nearly loses its mean difference in additional replicates; C20orf24 retains direction at only one of three sites.",
    )

    q = pd.read_csv(a / "threshold_stability.csv")
    q = q.loc[q.score_threshold.eq(0.5)]
    overlap = pd.read_csv(t / "training_overlap_cell_contrasts.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.05))
    base = np.zeros(len(q))
    for field, label, color in [
        ("baseline_candidates_retained", "Retained", BLUE),
        ("lost_coverage", "Insufficient coverage", GRAY),
        ("lost_score", "Below score cutoff", ORANGE),
    ]:
        v = q[field].to_numpy()
        axs[0].bar(range(len(q)), v, bottom=base, color=color, label=label)
        for i, n in enumerate(v):
            if n > 5:
                axs[0].text(
                    i,
                    base[i] + n / 2,
                    str(n),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        base += v
    axs[0].set_xticks(range(len(q)), q.min_reads.astype(str))
    axs[0].set_xlabel("Minimum reads per run")
    axs[0].set_ylabel("Original 181 shared predicted sites")
    axs[0].legend(
        frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(-0.04, -0.25)
    )
    for i, r in enumerate(overlap.itertuples()):
        axs[1].plot(
            [i, i],
            [r.baseline_excess_on_retained, r.current_excess_on_retained],
            c=GRAY,
            lw=1,
        )
    axs[1].scatter(
        range(6),
        overlap.baseline_excess_on_retained,
        s=29,
        facecolors="white",
        edgecolors=INK,
        label="With overlapping run",
    )
    axs[1].scatter(
        range(6),
        overlap.current_excess_on_retained,
        s=29,
        c=BLUE,
        label="Without overlapping run",
    )
    axs[1].axhline(0, c=GRAY, ls="--", lw=0.8)
    axs[1].set_xticks(range(6), overlap.cell_b, rotation=45, ha="right")
    axs[1].set_ylabel("Between − within score difference")
    axs[1].set_xlabel("Cell line compared with Hct116")
    axs[1].legend(
        frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(-0.12, -0.44)
    )
    axs[0].set_title("A  Read-coverage sensitivity", loc="left")
    axs[1].set_title("B  Removing a training-overlap run", loc="left")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.38, top=0.87, wspace=0.53)
    save(
        fig,
        out,
        "appendix_05",
        "A: fate of the original 181 shared predictions after rebuilding replicate means at each read-coverage cutoff. New predictions (3 at 30 reads; 2 at 50) are excluded from these stacks. B: all six comparisons involving Hct116, before and after excluding replicate 3 run 1, which overlaps Dataset 0. Each pair uses identical retained coordinates before and after exclusion. In two comparisons, between-cell-line variation exceeds within-cell-line variation only after exclusion.",
    )
    (a / "report_figure_captions.json").write_text(
        json.dumps(CAPTIONS, indent=2, ensure_ascii=False), encoding="utf8"
    )
    print("Ten compact figure layouts rendered.", flush=True)


if __name__ == "__main__":
    render(Path(__file__).resolve().parent / "analysis_context_trees")
