"""Create the scientific findings page and question-to-evidence index."""

from pathlib import Path
import hashlib
import json
import pandas as pd


SUPPORTING_FIGURES = {
    "Signal features": (
        "Signal differences at all three local positions",
        [
            (
                "07_signal_features_all_offsets",
                "Signal differences at three neighbouring positions",
            )
        ],
    ),
    "Transcript location": (
        "Stop-codon and last-exon patterns",
        [("09_stop_and_last_exon", "Predicted sites near stop codons and last exons")],
    ),
    "Additional replicates": (
        "Compare three gene effects and inspect the additional examples",
        [
            ("17_gene_effect_comparison", "Three selected genes: initial and additional-sample differences"),
            ("13_locus_SAMD4B", "SAMD4B: site scores and read coverage"),
            ("14_locus_C20orf24", "C20orf24: site scores and read coverage"),
        ],
    ),
    "Sensitivity": (
        "All six cell-line comparisons after excluding the overlapping run",
        [
            (
                "16_overlap_run_cell_contrasts",
                "Cell-line differences after excluding the overlapping run",
            )
        ],
    ),
}


def figure_markdown(name, title, captions, *, heading=True):
    base = f"figures_scientific/{name}"
    prefix = f"**{title}**\n\n" if heading else ""
    return (
        f"{prefix}![{title}]({base}.png)\n\n{captions[name]}\n\n"
        f"[PDF]({base}.pdf) · [SVG]({base}.svg) · [PNG]({base}.png)"
    )



def write_findings(a):
    t = a / "tables"
    s = json.loads((a / "exploration_summary.json").read_text())
    support = pd.read_csv(t / "sample_support.csv")
    depth = pd.read_csv(t / "depth_score_association.csv")
    pairs = pd.read_csv(a / "replicate_pairs.csv")
    cell = pd.read_csv(a / "cell_pairs.csv")
    cell = cell.loc[cell.scope.eq("matched_replicates")]
    genes = pd.read_csv(t / "gene_burden_ranking_min10.csv")
    sig = pd.read_csv(t / "signal_sample_contrasts.csv")
    ds = pd.read_csv(t / "signal_depth_match_sensitivity.csv")
    tight = ds.loc[ds.maximum_depth_ratio.eq(1.25)]
    discovery = pd.read_csv(a / "discovery_confirmation.csv").set_index(
        "discovery_selection"
    )
    fields = {
        "sample_eligible_min": int(support.eligible_positions.min()),
        "sample_eligible_max": int(support.eligible_positions.max()),
        "common_high_fraction_min": float(support.common_high_fraction.min()),
        "common_high_fraction_max": float(support.common_high_fraction.max()),
        "depth_rho_min": float(depth.common_spearman_score_logdepth.min()),
        "depth_rho_max": float(depth.common_spearman_score_logdepth.max()),
        "technical_rho_median": float(
            pairs.loc[pairs.kind.eq("technical"), "spearman"].median()
        ),
        "biological_rho_median": float(
            pairs.loc[pairs.kind.eq("biological"), "spearman"].median()
        ),
        "technical_mae_median": float(
            pairs.loc[pairs.kind.eq("technical"), "mae"].median()
        ),
        "biological_mae_median": float(
            pairs.loc[pairs.kind.eq("biological"), "mae"].median()
        ),
        "positive_cell_pairs": int(cell.between_minus_within.gt(0).sum()),
        "cell_pairs": len(cell),
        "cell_excess_min": float(cell.between_minus_within.min()),
        "cell_excess_max": float(cell.between_minus_within.max()),
        "signal_current_negative_groups": int(sig.current_center.lt(0).sum()),
        "signal_dwell_negative_groups": int(sig.log_dwell_center.lt(0).sum()),
        "tight_signal_pairs": int(tight.pairs.sum()),
        "tight_signal_groups": len(tight),
        "tight_current_negative_groups": int(tight.current_center.lt(0).sum()),
        "tight_dwell_negative_groups": int(tight.log_dwell_center.lt(0).sum()),
        "top_gene": genes.iloc[0].gene_name,
        "top_gene_fraction": float(genes.iloc[0].mean_high_fraction),
        "top_gene_positions": int(genes.iloc[0].eligible_positions),
        "top100_discovery_median": float(
            discovery.loc["top_100", "median_discovery_magnitude"]
        ),
        "top100_aligned_confirmation_median": float(
            discovery.loc["top_100", "median_confirmation_in_discovery_direction"]
        ),
    }
    s.update(fields)
    rows = [
        (
            "Site availability",
            "Which sites are observed in every biological replicate?",
            f"Each replicate has {fields['sample_eligible_min']:,}–{fields['sample_eligible_max']:,} eligible sites; {s['common_positions']:,} are observed in all 17.",
            "Use the common sites for sharing and gene-level fractions, and sites matched by read coverage for cell-line contrasts.",
            "01_observation_and_denominators",
            ["tables/sample_support.csv", "tables/depth_score_association.csv"],
        ),
        (
            "Replicate agreement",
            "How reproducible are predictions within cell lines?",
            f"Median technical-run and biological-replicate correlations are {fields['technical_rho_median']:.3f} and {fields['biological_rho_median']:.3f}. The nearest neighbour of {s['same_cell_nearest_neighbors']}/17 replicates belongs to the same cell line.",
            "Rank agreement is substantial. Cell lines overlap in the correlation-based clustering.",
            "02_similarity_and_replicates",
            [
                "tables/sample_correlations.csv",
                "tables/nearest_samples.csv",
                "tables/replicate_pairs_platform.csv",
            ],
        ),
        (
            "Cell-line differences",
            "Do differences between cell lines exceed variation within cell lines?",
            f"{fields['positive_cell_pairs']}/{fields['cell_pairs']} contrasts have greater between-cell-line disagreement. The excess ranges from {fields['cell_excess_min']:+.4f} to {fields['cell_excess_max']:+.4f} score units.",
            "The additional variation between cell lines is small. Each contrast uses the same sites for both quantities and treats biological replicates as the experimental units.",
            "03_between_within_effects",
            ["cell_pairs.csv", "cell_contrast_stability.csv"],
        ),
        (
            "Shared and restricted sites",
            "Which predicted modified sites recur across cell lines?",
            "181 sites in 105 genes score ≥0.5 in all 17 replicates. Of 156 sites consistently predicted modified in exactly one cell line, 18 are consistently predicted unmodified in all other cell lines.",
            "A restricted prediction requires agreement in both the selected cell line and the other six. Counts refer to transcript coordinates.",
            "04_shared_and_restricted_sites",
            [
                "tables/site_states.parquet",
                "tables/restricted_candidates.csv",
                "tables/score_threshold_sensitivity.csv",
            ],
        ),
        (
            "Signal features",
            "How do signal features differ after sequence and read-coverage matching?",
            f"Across {s['signal']['matched_pairs']:,} pairs in 51 replicate/motif groups, all group medians show lower central mean current and transformed dwell time at predicted modified sites. Signal standard deviation is higher for GAACT and lower for GGACT and AGACT.",
            f"Pairs have identical seven-mers and read-coverage ratios ≤2. Tightening the ratio to 1.25 retains {fields['tight_signal_pairs']:,} pairs and {fields['tight_current_negative_groups']}/{fields['tight_signal_groups']} negative current medians. These input-feature associations explain the model predictions.",
            "06_matched_signal_features",
            [
                "tables/signal_sample_contrasts.csv",
                "tables/signal_depth_match_sensitivity.csv",
                "tables/signal_matching_support.csv",
            ],
        ),
        (
            "Sequence",
            "How do predictions vary by central DRACH motif?",
            "Predicted modified-site fractions differ across the 18 observed central motifs and seven cell lines.",
            "Each motif has an explicit eligible-site denominator. Sequence is a model input; signal comparisons therefore match the full seven-mer.",
            "05_motif_composition",
            ["tables/motif_summary.csv"],
        ),
        (
            "Transcript location",
            "Where are shared predictions located within transcripts?",
            "142/181 shared sites (78.5%) are in 3′ UTRs, compared with 38.0% of eligible sites and 46.6% of the background with matching gene contributions. Of 170 shared sites with coding and stop annotations, 168 lie in the last exon.",
            "Relative-position and landmark plots use observed eligible sites as denominators. Location features contribute to these model-predicted patterns.",
            "08_transcript_topology",
            [
                "tables/metagene_bins.csv",
                "region_by_gene.csv",
                "tables/anchor_profiles.csv",
                "tables/last_exon_support.csv",
            ],
        ),
        (
            "Gene-level fractions",
            "Which genes have the highest fraction of predicted modified sites?",
            f"Among {s['gene_burden_genes_min10']} genes with ≥10 common sites, {fields['top_gene']} ranks first at {fields['top_gene_fraction']:.1%} across {fields['top_gene_positions']} eligible sites, followed by C19orf48 and MRPL14.",
            "Select one transcript per gene by site availability, then average fractions with equal weight for replicates within each cell line and for cell lines. Rankings change with the minimum eligible-site count.",
            "10_gene_burden",
            [
                "tables/gene_burden_ranking_min10.csv",
                "tables/gene_burden_rank_sensitivity.csv",
                "tables/gene_sample_burden.parquet",
            ],
        ),
        (
            "Additional replicates",
            "Which local differences recur in additional biological replicates?",
            f"Direction is retained for {s['discovery_retained']:,}/{s['discovery_genes']:,} genes ({s['discovery_retained'] / s['discovery_genes']:.1%}) and 73/100 leading genes. For those 100 genes, the median initial magnitude is {fields['top100_discovery_median']:.3f}, compared with {fields['top100_aligned_confirmation_median']:.3f} in the initial direction in additional replicates.",
            "Ranking uses initial-replicate scores. Coverage eligibility uses both subsets. Repeated direction often accompanies a smaller effect size.",
            "11_discovery_and_confirmation",
            [
                "tables/gene_discovery_confirmation.csv",
                "discovery_confirmation.csv",
                "example_confirmation_pairs.csv",
            ],
        ),
        (
            "Sensitivity",
            "Which findings persist after changing cutoffs or excluding the overlapping run?",
            "There are 181, 18 and 0 shared sites at score cutoffs 0.5, 0.7 and 0.9. Higher read-coverage cutoffs mainly remove observations. Excluding Hct116 replicate 3 run 1 retains all 181 shared sites among 25,962 common sites, while two of six affected cell-line contrasts change sign on identical retained sites.",
            "The shared-site result is stable to this run exclusion; the small cell-line contrasts are more sensitive. Run 4 from the same biological replicate remains included.",
            "15_threshold_robustness",
            [
                "threshold_stability.csv",
                "tables/training_overlap_run_sensitivity.csv",
                "tables/training_overlap_cell_contrasts.csv",
            ],
        ),
    ]
    data = [
        dict(
            topic=x[0],
            question=x[1],
            answer=x[2],
            interpretation=x[3],
            figure=x[4],
            tables=x[5],
        )
        for x in rows
    ]
    (a / "question_evidence.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8"
    )
    (a / "exploration_summary.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False), encoding="utf8"
    )
    pd.DataFrame([{**d, "tables": ";".join(d["tables"])} for d in data]).to_csv(
        t / "question_evidence.csv", index=False
    )
    reading_notes = (
        "A site is a transcript position. Biological replicates are separate samples; "
        "technical runs sequence the same sample more than once. Unless stated otherwise, "
        "sites need at least 20 reads per run, technical-run scores are averaged within "
        "each biological replicate, and scores ≥0.5 are classified as predicted modified."
    )
    terms = (
        "A seven-mer is a sequence of seven bases. Its central DRACH motif has "
        "D=A/G/T, R=A/G and H=A/C/T (T denotes RNA U). CDS means coding sequence; "
        "UTR means untranslated region. Read coverage is the number of reads at a site."
    )
    captions = json.loads((a / "figure_captions.json").read_text(encoding="utf8"))
    rnasek = (
        "12_locus_RNASEK",
        "RNASEK: follow the same sites across biological replicates",
    )
    locus_note = (
        "In these gene figures, discovery denotes the initial biological replicates "
        "and confirmation denotes the additional replicates. The lower panels show "
        "read coverage at the same selected sites."
    )
    markdown = [
        "# SG-NEx findings and figures",
        "23 SG-NEx sequencing runs, 17 biological replicates and seven human cell lines. "
        "These analyses describe predicted m6A sites, their signal features and reproducibility.",
        "[Analysis instructions](../README.md) · "
        "[Scientific references](../REFERENCES.md)",
        reading_notes,
        terms,
        "Each figure includes PNG, SVG and PDF downloads. "
        "[Compact layouts](#compact-layouts) combine selected panels.",
        "## On this page",
        "\n".join(
            f"- [{d['topic']}](#{d['topic'].lower().replace(' ', '-')})"
            for d in data
        ),
    ]
    for d in data:
        links = " · ".join(f"[{Path(path).name}]({path})" for path in d["tables"])
        markdown.extend(
            [
                f"## {d['topic']}",
                f"**{d['question']}**",
                d["answer"],
                d["interpretation"],
                figure_markdown(d["figure"], d["topic"], captions, heading=False),
                f"Numerical evidence: {links}.",
            ]
        )
        if d["topic"] == "Additional replicates":
            markdown.extend([locus_note, figure_markdown(*rnasek, captions)])
        if d["topic"] in SUPPORTING_FIGURES:
            heading, figures = SUPPORTING_FIGURES[d["topic"]]
            contents = "\n\n".join(
                figure_markdown(*figure, captions) for figure in figures
            )
            markdown.append(
                f"<details>\n<summary>{heading}</summary>\n\n{contents}\n\n</details>"
            )
    compact = [
        ("report_01", "Site availability and predicted modified-site fractions"),
        ("appendix_01", "Technical-run and biological-replicate agreement"),
        ("report_03", "Cell-line differences and shared predictions"),
        ("report_02", "Signal differences at matched sites"),
        ("appendix_02", "Predictions across the 18 DRACH motifs"),
        ("report_04", "Transcript location and gene-level fractions"),
        ("report_05", "Additional replicates and RNASEK"),
        ("appendix_03", "Three selected gene effects"),
        ("appendix_04", "SAMD4B and C20orf24 scores and read coverage"),
        ("appendix_05", "Read-coverage sensitivity and training-overlap exclusion"),
    ]
    downloads = ["| Content | Downloads |", "|---|---|"]
    for name, title in compact:
        base = f"figures_report/{name}"
        downloads.append(
            f"| {title} | [PNG]({base}.png) · [SVG]({base}.svg) · [PDF]({base}.pdf) |"
        )
    markdown.extend([
        "## Compact layouts",
        "Download alternative layouts of the results above. "
        "[Captions (JSON)](report_figure_captions.json) accompany these files.",
        "\n".join(downloads),
    ])
    (a / "README.md").write_text("\n\n".join(markdown) + "\n", encoding="utf8")
    provenance = a / "exploration_provenance.json"
    if provenance.is_file():
        record = json.loads(provenance.read_text(encoding="utf8"))
        record.setdefault("rendering_sources", {})["questions.py"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        provenance.write_text(json.dumps(record, indent=2), encoding="utf8")
    return s


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_context_trees",
    )
    write_findings(p.parse_args().analysis)
