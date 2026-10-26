"""Summarise saved SG-NEx predictions without loading or fitting a model."""
import argparse
from collections import Counter
from importlib.metadata import version
from itertools import combinations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from comparisons import (ALIASES, KEY, COLUMNS, load_sites, aggregate_replicate, aggregate_cell,
                         matched_sites, replicate_details, supported_comparison, variation_summary,
                         discovery_sites, transcript_candidates, rank_candidates, shared_positions)

HERE = Path(__file__).resolve().parent
OVERLAP = 'SGNex_Hct116_directRNA_replicate3_run1'
DEPTH_EDGES = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf])
DEPTH_LABELS = ['1–4', '5–9', '10–19', '20–49', '50–99', '100–199',
                '200–499', '500–999', '1000+']


def site_kind(ids):
    """The supplied human reference uses ENST; named controls remain separate."""
    human = ids.str.startswith('ENST')
    synthetic = ids.str.match(r'^(SIRV|ERCC|R[12]_)')
    return np.where(human, 'human', np.where(synthetic, 'synthetic', 'other_nonhuman'))


def frequency_quantiles(frequency, probabilities=(.1, .5, .9, .99)):
    """Exact linear-interpolated quantiles from integer-depth frequencies."""
    if not frequency:
        return [np.nan]*len(probabilities)
    values = np.array(sorted(frequency))
    cumulative = np.cumsum([frequency[v] for v in values])
    ranks = (cumulative[-1]-1)*np.asarray(probabilities)
    low, high = np.floor(ranks).astype(int), np.ceil(ranks).astype(int)
    a, b = values[np.searchsorted(cumulative, low, side='right')], values[np.searchsorted(cumulative, high, side='right')]
    return (a+(b-a)*(ranks-low)).tolist()


def sample_summary(path, sample, eligible, min_reads, run_metadata):
    counts = Counter()
    depths = Counter()
    hist = {kind: np.zeros(len(DEPTH_LABELS), dtype=np.int64)
            for kind in ['human', 'synthetic', 'other_nonhuman']}
    scanner = ds.dataset(path, format='parquet').scanner(
        columns=['transcript_id', 'n_reads', 'reference_matched'], batch_size=131072)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        kind = site_kind(frame.transcript_id)
        depth = frame.n_reads.to_numpy()
        above = depth >= min_reads
        counts['observed_sites'] += len(frame)
        counts['eligible_all'] += int(above.sum())
        counts['excluded_low_coverage'] += int((~above).sum())
        counts['summed_site_coverage'] += int(depth.sum())
        counts['reference_unmatched_all'] += int((~frame.reference_matched).sum())
        for name in hist:
            use = kind == name
            counts[name+'_observed'] += int(use.sum())
            counts['eligible_'+name] += int((use & above).sum())
            hist[name] += np.histogram(depth[use], DEPTH_EDGES)[0]
        values, n = np.unique(depth, return_counts=True)
        depths.update(dict(zip(values.tolist(), n.tolist())))
    q = frequency_quantiles(depths)
    summary = dict(sample_id=sample.sample_id, cell_line=sample.cell_line,
                   replicate_id=sample.replicate_id, replicate=int(sample.replicate), run=int(sample.run),
                   overlaps_data0=sample.sample_id == OVERLAP, min_reads=min_reads, **counts,
                   depth_q10=q[0], depth_median=q[1], depth_q90=q[2], depth_q99=q[3],
                   eligible_human_reference_unmatched=int((~eligible.reference_matched).sum()),
                   mean_score=float(eligible.score.mean()), median_score=float(eligible.score.median()),
                   score_q10=float(eligible.score.quantile(.1)), score_q90=float(eligible.score.quantile(.9)),
                   fraction_score_at_least_05=float(eligible.score.ge(.5).mean()),
                   elapsed_seconds=run_metadata.get('elapsed_seconds'),
                   peak_process_rss_bytes=run_metadata.get('peak_process_rss_bytes'))
    depth_rows = [dict(sample_id=sample.sample_id, cell_line=sample.cell_line, kind=kind,
                       depth_bin=label, count=int(count))
                  for kind, values in hist.items() for label, count in zip(DEPTH_LABELS, values)]
    score_n, score_edges = np.histogram(eligible.score, np.linspace(0, 1, 21))
    score_rows = [dict(sample_id=sample.sample_id, cell_line=sample.cell_line,
                       lower=float(a), upper=float(b), count=int(n))
                  for a, b, n in zip(score_edges[:-1], score_edges[1:], score_n)]
    return summary, depth_rows, score_rows


def pair_summary(joined):
    a, b = joined.score_a.to_numpy(), joined.score_b.to_numpy()
    if not len(a):
        return dict(matched_sites=0, pearson=np.nan, spearman=np.nan, mae=np.nan,
                    mean_delta_b_minus_a=np.nan)
    variable = len(a) > 1 and np.std(a) > 0 and np.std(b) > 0
    ratio = np.log2(joined.n_reads_b.to_numpy()/joined.n_reads_a.to_numpy())
    return dict(matched_sites=len(a), pearson=float(np.corrcoef(a, b)[0, 1]) if variable else np.nan,
                spearman=float(spearmanr(a, b).statistic) if variable else np.nan,
                mae=float(np.mean(np.abs(b-a))), mean_delta_b_minus_a=float(np.mean(b-a)),
                score_mean_a=float(a.mean()), score_mean_b=float(b.mean()),
                score_median_a=float(np.median(a)), score_median_b=float(np.median(b)),
                score_q10_a=float(np.quantile(a, .1)), score_q90_a=float(np.quantile(a, .9)),
                score_q10_b=float(np.quantile(b, .1)), score_q90_b=float(np.quantile(b, .9)),
                depth_median_a=float(joined.n_reads_a.median()), depth_median_b=float(joined.n_reads_b.median()),
                log2_depth_ratio_q10=float(np.quantile(ratio, .1)),
                log2_depth_ratio_median=float(np.median(ratio)),
                log2_depth_ratio_q90=float(np.quantile(ratio, .9)))


def write_group(writer, path, frame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression='zstd')
    writer.write_table(table)
    return writer


def figures(output, summary, replicate_pairs, cell_pairs, cells, examples, min_reads, font=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    family = 'Arial'
    if font is not None:
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        try:
            font_manager.findfont('Arial', fallback_to_default=False)
        except ValueError:
            family = 'DejaVu Sans'
            print('Arial is unavailable. Supply --font /path/to/arial.ttf for submission figures.', flush=True)
    plt.rcParams.update({'font.family': family, 'font.size': 10, 'pdf.fonttype': 42})
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    if examples.empty:
        for suffix in ['png', 'pdf']:
            (folder/f'example_transcripts.{suffix}').unlink(missing_ok=True)

    def save(fig, name):
        fig.savefig(folder/(name+'.png'), dpi=300, bbox_inches='tight', metadata={'Software': None})
        fig.savefig(folder/(name+'.pdf'), bbox_inches='tight',
                    metadata={'Creator': None, 'Producer': None, 'CreationDate': None, 'ModDate': None})
        plt.close(fig)

    labels = summary.apply(lambda r: f'{r.cell_line} rep{r.replicate} run{r.run}', axis=1)
    fig, ax = plt.subplots(figsize=(7.2, max(3, .23*len(summary))))
    left = np.zeros(len(summary))
    for column, label, color in [('eligible_human', f'Human, ≥{min_reads} reads', '#247c89'),
                                ('eligible_synthetic', 'Synthetic, eligible', '#bf8057'),
                                ('eligible_other_nonhuman', 'Other nonhuman, eligible', '#9067a1'),
                                ('excluded_low_coverage', f'Below {min_reads} reads', '#d6dadd')]:
        values = summary[column].to_numpy()/1e6
        ax.barh(labels, values, left=left, color=color, label=label)
        left += values
    ax.invert_yaxis(); ax.set_xlabel('Observed sites (millions)')
    ax.legend(fontsize=10, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    ax.set_title('Observed sites and coverage eligibility', pad=42)
    save(fig, 'coverage')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = dict(zip(sorted(summary.cell_line.unique()), plt.get_cmap('tab10').colors))
    markers = {}
    shapes = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for _, rows in summary.groupby('cell_line'):
        for i, replicate in enumerate(sorted(rows.replicate_id.unique())):
            markers[replicate] = shapes[i % len(shapes)]
    kinds = [kind for kind in ['technical', 'biological'] if kind in set(replicate_pairs.kind)]
    for i, kind in enumerate(kinds):
        group = replicate_pairs.loc[replicate_pairs.kind.eq(kind)]
        jitter = np.linspace(-.15, .15, len(group))
        for ax, metric in zip(axes, ['spearman', 'mae']):
            ax.scatter(i+jitter, group[metric], c=[colors[x] for x in group.cell_line], s=25)
            ax.set_ylabel('Spearman correlation' if metric == 'spearman' else 'Mean absolute score difference')
            ax.set_xticks(range(len(kinds)), [name.title() for name in kinds])
    if replicate_pairs.empty:
        for ax in axes:
            ax.text(.5, .5, 'No completed replicate pair yet', ha='center', transform=ax.transAxes)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle='', marker='o', color=color, label=cell)
               for cell, color in colors.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Replicate agreement at shared eligible human coordinates')
    fig.tight_layout(rect=(0, .15, 1, .95))
    save(fig, 'replicate_agreement')
    from matplotlib.lines import Line2D
    matched = cell_pairs.loc[cell_pairs.scope.eq('matched_replicates')].dropna(subset=['mae']).copy()
    matched = matched.sort_values('between_minus_within', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(3, .27*len(matched)+1.5)),
                             sharey=True, gridspec_kw={'width_ratios': [1, 1]})
    if len(matched):
        y = np.arange(len(matched))
        axes[0].hlines(y, matched.within_replicate_mae, matched.between_replicate_mae, color='#b6c4c9')
        axes[0].scatter(matched.within_replicate_mae, y, color='#76848e', s=25, label='Within cell (W)')
        axes[0].scatter(matched.between_replicate_mae, y, color='#146e76', s=25, label='Between cells (B)')
        labels = [f'{r.cell_a} / {r.cell_b} (n={r.matched_sites:,})' for r in matched.itertuples()]
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        axes[1].hlines(y, matched.absolute_cell_delta_median, matched.absolute_cell_delta_p99, color='#b6c4c9')
        for name, label, color, marker in [('median','Median','#146e76','o'),
                ('p90','90th percentile','#a87926','s'), ('p99','99th percentile','#a24e36','^')]:
            axes[1].scatter(matched['absolute_cell_delta_'+name], y, color=color, marker=marker, s=25, label=label)
        for ax in axes:
            ax.set_xlim(left=0)
            ax.legend(loc='upper left', bbox_to_anchor=(0,-.10), frameon=False, fontsize=9)
    else:
        for ax in axes:
            ax.text(.5, .5, 'No jointly supported positions', ha='center', transform=ax.transAxes)
    axes[0].set_title('Replicate variation', pad=12)
    axes[0].set_xlabel('Mean absolute replicate-score difference')
    axes[1].set_title('Spread of local differences', pad=12)
    axes[1].set_xlabel('Absolute cell-mean difference (D)')
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
    fig.tight_layout(w_pad=2.5)
    save(fig, 'cell_comparison')

    shared = pd.read_parquet(output/'shared_sites.parquet')
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), gridspec_kw={'width_ratios': [1.1, 1]})
    names = sorted(cells)
    if len(shared) and len(names) > 1:
        correlation = shared[['score_'+c for c in names]].corr(method='spearman').to_numpy()
        display = np.ma.masked_where(np.triu(np.ones_like(correlation), 1).astype(bool), correlation)
        image = axes[0].imshow(display, cmap='Blues', vmin=.8, vmax=1)
        for i in range(len(names)):
            for j in range(i+1):
                axes[0].text(j, i, f'{correlation[i,j]:.2f}', ha='center', va='center', fontsize=10,
                             color='white' if correlation[i,j] > .94 else 'black')
        axes[0].set_xticks(range(len(names)), names, rotation=45, ha='right')
        axes[0].set_yticks(range(len(names)), names)
        axes[0].set_title(f'Rank agreement on the same {len(shared):,} sites', fontsize=10)
        fig.colorbar(image, ax=axes[0], fraction=.05, pad=.03, label='Spearman correlation')
        counts = [int(shared.minimum_cell_score.ge(.5).sum()), int(shared.high_in_every_replicate.sum()),
                  int((shared.high_in_every_replicate & shared.all_replicates_observed).sum())]
        axes[1].barh(range(3), counts, color=['#b9d6d8', '#6ba2a6', '#146e76'])
        axes[1].set_yticks(range(3), ['Every cell mean', 'Every observed replicate', 'All declared replicates'])
        axes[1].invert_yaxis()
        for i, n in enumerate(counts): axes[1].text(n+max(counts)*.02, i, f'{n:,}', va='center')
        axes[1].set_xlim(0, max(counts)*1.2 if max(counts) else 1)
        axes[1].set_xlabel('Positions with every stated score >= 0.5')
        axes[1].set_title('A shared set of high-scoring candidates', fontsize=10)
    else:
        for ax in axes: ax.text(.5, .5, 'No common supported set across cells', ha='center', transform=ax.transAxes)
    fig.tight_layout(w_pad=2.5)
    save(fig, 'shared_sites')

    if len(examples):
        observations = pd.read_parquet(output/'example_sites.parquet')
        supported_sites = pd.read_csv(output/'example_supported_sites.csv', dtype={k: str for k in KEY})
        fig, axes = plt.subplots(len(examples), 2, figsize=(8.2, 2.9*len(examples)), squeeze=False)
        for pair_axes, example in zip(axes, examples.itertuples()):
            supported = supported_sites.loc[supported_sites.transcript_id.eq(example.transcript_id)
                & supported_sites.cell_a.eq(example.cell_a) & supported_sites.cell_b.eq(example.cell_b)]
            positions = sorted(supported.transcript_position.astype(int).tolist())
            observed = observations.loc[observations.transcript_id.eq(example.transcript_id)
                & observations.cell_line.isin([example.cell_a, example.cell_b]) & observations.eligible
                & observations.transcript_position.astype(int).isin(positions)]
            replicas = observed.groupby(['cell_line', 'replicate_id', 'transcript_position']).agg(
                score=('score', 'mean'), n_reads=('n_reads', 'mean')).reset_index()
            identities = sorted(replicas.replicate_id.unique())
            for j, (replicate, shown) in enumerate(replicas.groupby('replicate_id', sort=True)):
                cell = shown.cell_line.iloc[0]
                discovery = replicate in [example.discovery_replicate_a, example.discovery_replicate_b]
                x = shown.transcript_position.astype(int).map({p:i for i,p in enumerate(positions)}).to_numpy()
                offset = (j-(len(identities)-1)/2)*.055
                label = f'{cell} r{replicate.rsplit("replicate",1)[-1]}' + (' (D)' if discovery else '')
                for ax, metric in zip(pair_axes, ['score', 'n_reads']):
                    ax.scatter(x+offset, shown[metric], s=28, marker=markers[replicate],
                        facecolors='white' if discovery else colors[cell], edgecolors=colors[cell], label=label)
            pair_axes[0].set_title(f'{example.gene_name} / {example.transcript_id}', fontsize=10)
            pair_axes[0].set_ylim(0,1); pair_axes[0].set_ylabel('Prediction score')
            pair_axes[1].set_ylim(bottom=0); pair_axes[1].set_ylabel('Mean reads per run')
            pair_axes[1].legend(fontsize=9, ncol=3, frameon=False, loc='lower left',
                bbox_to_anchor=(0,1.01), borderaxespad=0, columnspacing=.65, handletextpad=.2)
            for ax in pair_axes:
                ax.set_xticks(range(len(positions)), [str(p) for p in positions], rotation=45, ha='right')
                ax.set_xlabel('Qualifying transcript positions (equally spaced)')
                ax.grid(axis='y', alpha=.15)
        fig.suptitle('Discovery replicates: hollow symbols (D) | Confirmation: filled symbols', fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96), h_pad=2.1)
        save(fig, 'example_transcripts')
    return family


def analyze(results, samples, output, min_reads=20, allow_partial=False, font=None, reference=None):
    started = time.perf_counter()
    inventory = pd.read_csv(samples).fillna('')
    if inventory.empty or inventory.sample_id.duplicated().any():
        raise ValueError('The inventory must be nonempty with unique sample IDs.')
    declared_runs = inventory.sample_id.tolist()
    completed = inventory.sample_id.map(lambda s: (results/s/'sites.parquet').exists() and (results/s/'run.json').exists())
    missing = inventory.loc[~completed, 'sample_id'].tolist()
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} runs lack completed predictions. Finish run.py or explicitly use --allow-partial.')
    inventory = inventory.loc[completed].copy()
    if inventory.empty:
        raise ValueError('No completed prediction tables are available yet.')
    run_info = {sample: json.loads((results/sample/'run.json').read_text()) for sample in inventory.sample_id}
    for sample in inventory.itertuples():
        saved = run_info[sample.sample_id]
        if saved.get('model') != 'context_aware_trees_v1':
            raise ValueError('Select context-aware trees results.')
        if any(saved[k] != getattr(sample, k) for k in
                ['sample_id', 'cell_line', 'replicate_id', 'replicate', 'run']):
            raise ValueError(f'{sample.sample_id}: sample details differ from inventory.')
        rows = pq.read_metadata(results/sample.sample_id/'sites.parquet').num_rows
        if rows != saved['sites'] or rows != sample.sites:
            raise ValueError(f'{sample.sample_id}: prediction counts differ from inventory.')
    reference = reference or HERE.parent/'task 1 context aware trees/model/reference.joblib'
    overlap_replicates = set(inventory.loc[inventory.sample_id.eq(OVERLAP), 'replicate_id'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'metadata.json').unlink(missing_ok=True)
    (output/'input_inventory.csv').write_bytes(Path(samples).read_bytes())
    summaries, depth_rows, score_rows, pair_rows = [], [], [], []
    writer = None
    for replicate_id, rows in inventory.groupby('replicate_id', sort=True):
        frames = []
        for sample in rows.itertuples():
            path = results/sample.sample_id/'sites.parquet'
            saved = run_info[sample.sample_id]
            frame = load_sites(path, min_reads)
            summary, depth, score = sample_summary(path, sample, frame, min_reads, saved)
            summaries.append(summary); depth_rows.extend(depth); score_rows.extend(score); frames.append(frame)
            print(f'Analysed {sample.sample_id}: {len(frame):,} eligible human sites.', flush=True)
        for i, j in combinations(range(len(frames)), 2):
            joined, mismatch = matched_sites(frames[i], frames[j])
            pair_rows.append(dict(kind='technical', cell_line=rows.cell_line.iloc[0],
                left=rows.sample_id.iloc[i], right=rows.sample_id.iloc[j],
                overlaps_data0=OVERLAP in [rows.sample_id.iloc[i], rows.sample_id.iloc[j]],
                kmer_mismatches=mismatch, **pair_summary(joined)))
        if sum(map(len, frames)):
            replica = aggregate_replicate(frames, rows)
            writer = write_group(writer, output/'biological_sites.parquet', replica)
    if writer is None:
        raise ValueError('No human sites meet the requested coverage threshold.')
    writer.close()
    cells, cell_rows, candidate_tables = {}, [], []
    replicate_order = {cell: rows.sort_values(["replicate", "run"]).replicate_id.drop_duplicates().tolist()
                       for cell, rows in inventory.groupby("cell_line")}
    replicate_width = max(map(len, replicate_order.values()))
    writer = None
    for cell in sorted(inventory.cell_line.unique()):
        replicates = ds.dataset(output/'biological_sites.parquet').to_table(filter=ds.field('cell_line') == cell).to_pandas()
        if replicates.empty:
            continue
        groups = list(replicates.groupby('replicate_id', sort=True))
        for (a, left), (b, right) in combinations(groups, 2):
            joined, mismatch = matched_sites(left, right)
            pair_rows.append(dict(kind='biological', cell_line=cell, left=a, right=b,
                                  overlaps_data0=bool({a, b} & overlap_replicates),
                                  kmer_mismatches=mismatch, **pair_summary(joined)))
        base = aggregate_cell(replicates)
        writer = write_group(writer, output/'cell_sites.parquet', base)
        cells[cell] = base.merge(replicate_details(replicates, replicate_order[cell], replicate_width),
                                 on=KEY, validate='one_to_one')
    writer.close()
    for a, b in combinations(sorted(cells), 2):
        joined, mismatch = matched_sites(cells[a], cells[b])
        compared = supported_comparison(joined)
        cell_rows.append(dict(cell_a=a, cell_b=b, scope='matched_replicates',
            eligible_sites_a=len(cells[a]), eligible_sites_b=len(cells[b]), kmer_mismatches=mismatch,
            overlaps_data0=bool(overlap_replicates) and 'Hct116' in (a, b),
            **pair_summary(compared), **variation_summary(compared)))
        candidate_tables.append(transcript_candidates(compared, a, b,
            replicate_order[a][0], replicate_order[b][0]))
    summary = pd.DataFrame(summaries).sort_values(['cell_line', 'replicate', 'run'])
    cell_summary = []
    for cell, values in cells.items():
        runs = summary.loc[summary.cell_line.eq(cell)]
        cell_summary.append(dict(cell_line=cell, runs=len(runs), biological_replicates=runs.replicate_id.nunique(),
            observed_site_observations=int(runs.observed_sites.sum()),
            eligible_human_site_observations=int(runs.eligible_human.sum()),
            unique_eligible_human_sites=len(values), sites_with_two_replicates=int(values.n_replicates.ge(2).sum()),
            mean_score=float(values.score.mean()), median_score=float(values.score.median()),
            score_q10=float(values.score.quantile(.1)), score_q90=float(values.score.quantile(.9))))
    pd.DataFrame(cell_summary).to_csv(output/'cell_summary.csv', index=False)
    pairs = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(
        columns=['kind', 'cell_line', 'left', 'right', 'spearman', 'mae'])
    cell_pairs = pd.DataFrame(cell_rows) if cell_rows else pd.DataFrame(
        columns=['cell_a', 'cell_b', 'scope', 'matched_sites', 'mean_delta_b_minus_a', 'mae', 'spearman',
                 'within_replicate_mae', 'between_replicate_mae', 'between_minus_within',
                 'fraction_replicate_ranges_separated'])
    for name, table in [('run_summary', summary), ('depth_histogram', pd.DataFrame(depth_rows)),
                        ('score_histogram', pd.DataFrame(score_rows)), ('replicate_pairs', pairs), ('cell_pairs', cell_pairs)]:
        table.to_csv(output/(name+'.csv'), index=False)
    candidates = rank_candidates(candidate_tables)
    gene_path = HERE/'genes.csv'
    genes = pd.read_csv(gene_path).fillna('') if gene_path.exists() else pd.DataFrame(columns=['gene_id', 'gene_name'])
    names = genes.set_index('gene_id').gene_name
    candidates['gene_name'] = candidates.gene_id.map(names).fillna(candidates.gene_id)
    candidates.to_csv(output/'discovery_candidates.csv', index=False)
    examples = candidates.drop_duplicates('gene_id').head(3).copy()
    shared = shared_positions(cells, inventory.replicate_id.nunique(), inventory.cell_line.unique())
    shared['gene_name'] = shared.gene_id.map(names).fillna(shared.gene_id)
    shared.to_parquet(output/'shared_sites.parquet', index=False, compression='zstd')
    shared_candidates = shared.loc[shared.high_in_every_replicate & shared.all_replicates_observed]
    shared_candidates.sort_values(['minimum_replicate_score', *KEY], ascending=[False, True, True]).to_csv(
        output/'shared_candidates.csv', index=False)
    examples.to_csv(output/'examples.csv', index=False)
    supported_examples = []
    for example in examples.itertuples():
        joined, _ = matched_sites(cells[example.cell_a], cells[example.cell_b])
        eligible = discovery_sites(supported_comparison(joined))
        supported_examples.append(eligible.loc[eligible.transcript_id.eq(example.transcript_id)].assign(
            cell_a=example.cell_a, cell_b=example.cell_b))
    supported_table = pd.concat(supported_examples, ignore_index=True) if supported_examples else pd.DataFrame(
        columns=KEY+['cell_a', 'cell_b', 'score_a', 'score_b', 'delta_b_minus_a'])
    supported_table.to_csv(output/'example_supported_sites.csv', index=False)
    example_rows = []
    if len(examples):
        for sample in inventory.itertuples():
            frame = load_sites(results/sample.sample_id/'sites.parquet', min_reads=1,
                               transcripts=examples.transcript_id, human_only=True)
            frame = frame.assign(sample_id=sample.sample_id, cell_line=sample.cell_line,
                                  replicate_id=sample.replicate_id, eligible=lambda x: x.n_reads.ge(min_reads),
                                  overlaps_data0=sample.sample_id == OVERLAP)
            example_rows.append(frame)
    example_table = pd.concat(example_rows, ignore_index=True) if example_rows else pd.DataFrame(
        columns=COLUMNS+['input_transcript_id', 'input_transcript_position', 'sample_id', 'cell_line', 'replicate_id', 'eligible', 'overlaps_data0'])
    example_table.to_parquet(output/'example_sites.parquet', index=False, compression='zstd')
    index = pd.concat([frame[['gene_id', 'transcript_id']] for frame in cells.values()]).drop_duplicates()
    index['gene_name'] = index.gene_id.map(names).fillna(index.gene_id)
    index.sort_values(['gene_id', 'transcript_id']).to_parquet(
        output/'transcript_index.parquet', index=False, compression='zstd')
    font_used = figures(output, summary, pairs, cell_pairs, cells, examples, min_reads, font)
    evidence = {}
    if len(cells) > 1:
        from evidence import complete_analysis
        evidence = complete_analysis(results, inventory, output, cells, min_reads,
                                     int(summary.eligible_human.sum()), reference)
    metadata = dict(completed_runs=len(inventory), completed_cell_lines=inventory.cell_line.nunique(),
        eligible_human_cell_lines=len(cells),
        missing_runs=missing, complete=not missing, min_reads=min_reads,
        model='context_aware_trees_v1', full_weight=.45, local_weight=.55,
        dependencies={name: version(name) for name in ['numpy', 'pandas', 'pyarrow', 'scipy', 'matplotlib', 'joblib']},
        evidence=evidence, observed_sites=int(summary.observed_sites.sum()),
        eligible_site_observations=int(summary.eligible_all.sum()),
        declared_runs=declared_runs,
        eligible_human_site_observations=int(summary.eligible_human.sum()),
        scope=(f'Context-aware trees: {len(inventory)} of {len(declared_runs)} declared processed runs, '
               f'{inventory.cell_line.nunique()} cell lines, {inventory.replicate_id.nunique()} biological replicates.'),
        coordinates='Original input keys are retained in per-run prediction tables. Analysis maps only explicitly verified transcript version aliases to the fixed reference IDs, preserving input_transcript_id and input_transcript_position; analysis positions use canonical integer strings and matched seven-mers must agree.',
        transcript_aliases=dict(file='transcript_aliases.csv', verified_ids=len(ALIASES),
            audit='Exact version headers and full sequences match the fixed Ensembl 91 reference. Unlisted IDs remain unchanged.'),
        human_rule='ENST transcript IDs. SIRV, ERCC and R1_/R2_ sequin IDs are synthetic; other nonhuman IDs are separately counted.',
        aggregation='Mean available eligible technical runs at each coordinate, then mean available biological replicates. No coverage weighting or missing-to-zero imputation. Support counts accompany every aggregate.',
        depth_comparison='Primary matched_replicates comparisons require at least two biological replicates per cell and a maximum/minimum biological-replicate mean read-depth ratio no greater than 2 across both cells.',
        matched_variation='At each matched position, average absolute differences over all cross-cell biological-replicate pairs. Within each cell, average all distinct biological-replicate pairs, then give the two cells equal weight. Average both quantities over the same positions.',
        discovery_replicates={cell: order[0] for cell, order in replicate_order.items()},
        confirmation_replicates={cell: order[1:] for cell, order in replicate_order.items()},
        shared_positions='Same transcript coordinate and seven-mer, with at least two observed biological replicates in every cell. Shared candidates additionally require all declared biological replicates and every replicate score at least 0.5.',
        examples='Use matched_replicates positions with a score in the lowest-numbered biological replicate of each cell (discovery) and at least one remaining replicate (confirmation). Require at least 3 qualifying positions per transcript. Rank the absolute signed mean discovery contrast, then site count; select 3 distinct genes without using confirmation scores in ranking. Confirmation is the equal mean of remaining replicates at each same position. All candidates and qualifying positions are saved.',
        interpretation=('Scores predict binary m6A site labels; they are not molecular modification fractions. Technical runs are grouped within biological replicates.' + (' Hct116 replicate 3/run 1 overlaps Dataset 0 training measurements.' if overlap_replicates else '')),
        font_family=font_used, figure_formats='PNG300dpi and embedded TrueType PDF')
    from findings import write_findings
    write_findings(output, metadata)
    metadata["analysis_elapsed_seconds"] = round(time.perf_counter()-started, 3)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results', type=Path, default=HERE/'results_context_trees', help='Folder containing one result directory per run.')
    parser.add_argument('--samples', type=Path, default=HERE/'samples.csv', help='Run and biological-replicate inventory.')
    parser.add_argument('--output', type=Path, default=HERE/'analysis_context_trees', help='Destination for findings, tables and figures.')
    parser.add_argument('--min-reads', type=int, default=20, help='Minimum coverage per run-site for comparisons.')
    parser.add_argument('--allow-partial', action='store_true', help='Permit an explicitly incomplete preview.')
    parser.add_argument('--font', type=Path, help='Optional Arial TTF path on systems without Arial installed.')
    parser.add_argument('--reference', type=Path, help='Transcript reference used by the saved model; defaults to the bundled reference.')
    args = parser.parse_args()
    if args.min_reads < 1:
        parser.error('--min-reads must be positive')
    result = analyze(args.results, args.samples, args.output, args.min_reads, args.allow_partial, args.font, args.reference)
    status = 'Complete analysis' if result['complete'] else 'Incomplete preview'
    print(f"{status}: {result['completed_runs']} runs, {result['completed_cell_lines']} cell lines, "
          f"{result['eligible_human_site_observations']:,} eligible human observations.")
    print(f"Read the findings: {(args.output/'findings.html').resolve()}")


if __name__ == '__main__':
    main()
