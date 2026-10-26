"""Build the SG-NEx findings page directly from the saved analysis tables."""
from html import escape

import pandas as pd


def table(frame):
    return '<div class="scroll">' + frame.to_html(index=False, border=0, escape=True, na_rep='Unavailable') + '</div>'


def number(value, signed=False):
    return 'Unavailable' if pd.isna(value) else format(value, '+.3f' if signed else '.3f')


def figure(name, caption):
    return (f'<figure><img src="figures/{name}.png" alt="{escape(caption)}">'
            f'<figcaption>{caption} <a href="figures/{name}.pdf">PDF</a></figcaption></figure>')


BIOLOGY = {
    'RNASEK': ('RNASEK encodes a membrane protein associated with the V-ATPase proton pump. '
              'Experiments in human cells connect it to endocytosis, the uptake of material into cells.',
              'Perreira et al. (2015), Cell Reports', 'https://pubmed.ncbi.nlm.nih.gov/26212330/'),
    'ANAPC11': ('ANAPC11 encodes the catalytic RING subunit of the anaphase-promoting complex. '
               'This complex tags cell-cycle proteins, including cyclin B and securin, for degradation during cell division.',
               'Gmachl et al. (2000), PNAS', 'https://pubmed.ncbi.nlm.nih.gov/10922056/'),
    'CHTF8': ('CHTF8 encodes part of the human CTF18-RFC complex. Biochemical experiments show that this complex '
             'loads the PCNA clamp onto DNA, linking DNA replication with the machinery that keeps sister chromatids together.',
             'Bermudez et al. (2003), PNAS', 'https://pubmed.ncbi.nlm.nih.gov/12930902/')
}


def shared_findings(output, metadata):
    stability = pd.read_csv(output/'threshold_stability.csv')
    regions = pd.read_csv(output/'region_summary.csv')
    genes = pd.read_csv(output/'region_by_gene.csv')
    candidates = pd.read_csv(output/'shared_candidates.csv')
    identities = pd.read_csv(output/'shared_candidate_stability.csv')
    depth = metadata['min_reads']
    primary = stability.loc[stability.min_reads.eq(depth) & stability.score_threshold.eq(.5)].iloc[0]
    counts = stability.pivot(index='min_reads', columns='score_threshold', values='candidate_positions').reset_index()
    counts.columns = ['Reads per run', 'Every score ≥0.5', 'Every score ≥0.7', 'Every score ≥0.9']
    if candidates.empty:
        return ('<section id="shared"><h2>2. Which candidates are shared across cells, and where are they?</h2>'
            '<p>A shared candidate must be observed in every declared biological replicate and '
            'score at least 0.5 in each. No positions meet this rule at the primary read cutoff.</p>'
            '<p>The coverage and score settings give the following counts:</p>'+table(counts)+'</section>')
    stable = identities.loc[identities.score_at_least_09_at_every_depth].copy()
    stable = stable[['gene_name','transcript_id','transcript_position',f'minimum_replicate_score_at_{depth}_reads','region']]
    stable.iloc[:, 3] = stable.iloc[:, 3].map(lambda x: f'{x:.3f}')
    stable.columns = ['Gene','Transcript','Position',f'Minimum score at {depth} reads','Region']
    region = regions.loc[regions.region.eq('3UTR')].set_index('cohort')
    r, b, g = [region.loc[name] for name in ['shared_candidates','eligible_background','eligible_same_genes']]
    expected = genes.expected_candidate_3utr.sum()
    count = genes.candidate_positions.sum()
    expected_fraction = expected / count if count else float('nan')
    return f"""<section id="shared"><h2>2. Which candidates are shared across cells, and where are they?</h2>
<p class="answer"><strong>{len(candidates):,} transcript positions in {candidates.gene_id.nunique():,} genes score at least 0.5 in every biological replicate.</strong>
Among them, {len(stable)} positions maintain scores of at least 0.9 at every tested read cutoff. These are specific, repeatedly supported targets across the sampled cell lines.</p>
<h3>Select a shared set from positions that every sample can observe</h3>
<p>A common position has the same coordinate and seven-base sequence in at least two biological replicates of every cell line.
This gives <strong>{int(primary.common_supported_positions):,} common positions</strong>; {int(primary.all_replicates_observed_positions):,}
are observed in every declared biological replicate. The shared catalogue selects from this latter set by requiring every replicate score ≥0.5.
The observation rule defines the population to which this shared-site conclusion applies.
The pairwise twofold-depth rule serves cell contrasts; this catalogue instead requires observation and high scores throughout the collection.</p>
{figure('shared_sites', 'Left: cell-mean rank agreement on the common positions. Right: progressively stronger requirements for a score of at least 0.5.')}
<h3>A compact set with persistent model support</h3>
<p>The following {len(stable)} positions in {stable.Gene.nunique()} genes meet the stronger 0.9 score rule in every replicate at all tested read cutoffs.
The table identifies transcript coordinates; isoforms can share a genomic base. <a href="#support">Question 4</a> explains how coverage and score thresholds change the wider catalogue.</p>
{table(stable)}
<h3>High scores concentrate in the 3′ untranslated region</h3>
<p>A 3′ UTR is transcribed sequence after the protein-coding region. It contains {int(r.positions)}/{int(r.denominator)}
({100*r.fraction:.1f}%) shared candidates, compared with {int(b.positions):,}/{int(b.denominator):,}
({100*b.fraction:.1f}%) of all positions eligible for the catalogue and {int(g.positions):,}/{int(g.denominator):,}
({100*g.fraction:.1f}%) of eligible positions in the same genes. Each background requires observation in every biological replicate,
so its positions could have entered the catalogue before the score cutoff.</p>
<p>Genes contribute unequal numbers of candidates. To hold these contributions constant, multiply each gene's candidate count by the
3′ UTR fraction of its eligible positions, then sum. This gives {expected:.1f} expected 3′ UTR candidates out of {int(r.denominator)}
({100*expected_fraction:.1f}%), compared with {int(r.positions)} observed ({100*r.fraction:.1f}%). The difference is
{100*(r.fraction-expected_fraction):.1f} percentage points with the candidate gene contributions fixed.</p>
{figure('transcript_regions', '3′ UTR proportions on catalogue-eligible positions. The gene-matched expectation gives each gene exactly its observed share of candidates. Full region counts remain in region_summary.csv.')}
<p>The concentration persists when gene contributions are held constant.

Experimental m6A maps motivate examining 3′ UTRs and stop-codon regions
(<a href="https://doi.org/10.1016/j.cell.2012.05.003">Meyer et al., 2012</a>).
Transcript location is also a model input: this analysis describes where high predictions occur.</p>
<p class="sources">Tables: <a href="shared_candidates.csv">shared candidates</a> ·
<a href="shared_candidate_stability.csv">individual-site stability</a> ·
<a href="threshold_stability.csv">threshold counts and identity changes</a> ·
<a href="region_summary.csv">region counts and denominators</a> · <a href="region_by_gene.csv">gene-matched calculation</a></p></section>"""


def local_findings(output, examples):
    if examples.empty:
        return ('<section id="local"><h2>3. Which local differences recur in additional samples?</h2>'
                '<p>No transcript/cell pair has at least three positions with the required coverage, '
                'replication and discovery/confirmation measurements. There are no local examples '
                'to rank for this collection.</p></section>')
    confirmation = pd.read_csv(output/'discovery_confirmation.csv')
    individual = pd.read_csv(output/'example_confirmation_pairs.csv')
    sample_view = individual[['gene_name','replicate_a','replicate_b','paired_positions','selected_positions','delta_b_minus_a']].copy()
    sample_view['Replicate A'] = sample_view.replicate_a.str.extract(r'(replicate\d+)$', expand=False)
    sample_view['Replicate B'] = sample_view.replicate_b.str.extract(r'(replicate\d+)$', expand=False)
    sample_view['Positions'] = sample_view.paired_positions.astype(str)+'/'+sample_view.selected_positions.astype(str)
    sample_view['B − A'] = sample_view.delta_b_minus_a.map(lambda x: number(x, True))
    sample_view = sample_view.rename(columns={'gene_name':'Gene'})[['Gene','Replicate A','Replicate B','Positions','B − A']]
    top = confirmation.set_index('discovery_selection').loc['top_100']
    all_genes = confirmation.set_index('discovery_selection').loc['all']
    stronger = ('The leading list retains direction more often than the full list.'
                if top.direction_retained_fraction > all_genes.direction_retained_fraction else
                'The leading list does not retain direction more often than the full list.')
    selected_sites = pd.read_csv(output/'example_supported_sites.csv')
    utr_count = int(selected_sites.region.eq('3UTR').sum())
    chosen = examples[['gene_name','cell_a','cell_b','matched_supported_sites','discovery_delta','confirmation_delta','positions_agreeing']].copy()
    for name in ['discovery_delta', 'confirmation_delta']:
        chosen[name] = chosen[name].map(lambda x: number(x, True))
    chosen.columns = ['Gene','Cell A','Cell B','Positions','Discovery B − A','Confirmation B − A','Positions retaining direction']
    view = confirmation[['discovery_selection','genes','direction_retained','direction_retained_fraction',
                         'median_discovery_magnitude','median_confirmation_in_discovery_direction']].copy()
    view.direction_retained_fraction = view.direction_retained_fraction.map(lambda x: f'{100*x:.1f}%')
    for name in ['median_discovery_magnitude','median_confirmation_in_discovery_direction']:
        view[name] = view[name].map(lambda x: number(x, True))
    view.columns = ['Discovery list','Genes','Direction retained','Percentage','Median discovery magnitude','Median aligned confirmation']
    biology = []
    for gene in examples.gene_name:
        if gene in BIOLOGY:
            explanation, title, url = BIOLOGY[gene]
            biology.append(f'<li><strong>{escape(gene)}.</strong> {explanation} <a href="{url}">{title}</a>.</li>')
    return f"""<section id="local"><h2>3. Which local differences recur in additional samples?</h2>
<p class="answer"><strong>{int(top.direction_retained)}/{int(top.genes)} leading genes retain their discovery direction in confirmation</strong>,
compared with {100*all_genes.direction_retained_fraction:.1f}% across all {int(all_genes.genes):,} eligible genes.
The leading list has a median discovery magnitude of {top.median_discovery_magnitude:.3f} and a median confirmation contrast of
{top.median_confirmation_in_discovery_direction:+.3f} when aligned with that direction. The recurring differences are smaller in confirmation.</p>
<h3>Select on discovery samples, then inspect the same positions in confirmation</h3>
<p>The lowest-numbered biological replicate in each cell is the discovery sample. The remaining replicates supply confirmation means.
Numbering fixes the sample split without using scores.
For a transcript and cell pair, require at least three matched positions with discovery and confirmation values and the same support/depth rule as the broad comparisons.
Rank the absolute <em>signed mean</em> discovery difference, B − A; keep each gene's highest-ranked comparison.
Using a signed mean favors differences that point in a consistent direction across positions.
The separation of selection and score checking follows the principle of avoiding circular analysis
(<a href="https://doi.org/10.1038/nn.2303">Kriegeskorte et al., 2009</a>).</p>
<p>{stronger} At each selected position, average the remaining replicate scores, then average the signed differences over positions.
Align confirmation by multiplying its signed contrast by the discovery sign: positive means the original direction, negative means reversal.
For example, discovery −0.20 and confirmation −0.08 give aligned confirmation +0.08.</p>
{figure('evidence_stability', 'Left: direction retention in nested discovery-ranked gene lists. Right: median discovery magnitude and median confirmation aligned with the discovery direction; negative values indicate reversal.')}
<details><summary>Direction and magnitude across the complete candidate ranking</summary>
<p>Aligned confirmation is positive for the discovery direction, negative for a reversal and zero for no difference.
Confirmation scores do not determine the ranking; coverage and availability determine eligibility in both sets.</p>{table(view)}</details>
<h3>Locate the differences and show the sample support behind each mean</h3>
{table(chosen)}
<p>Positive B − A means cell B has the higher mean prediction. Each discovery and confirmation contrast uses the same positions.
The per-position measurements show whether a transcript mean is supported consistently across its sites.</p>
{figure('example_transcripts', 'Each point is the mean for one biological replicate at one of three selected transcript positions; technical runs are averaged. Color identifies cell line, while hollow and filled circles distinguish discovery and confirmation replicates. Sites are equally spaced for display. The row annotation gives the mean score difference (cell B − cell A) across the selected positions in discovery and confirmation samples.')}
<h3>Check what the confirmation mean contains</h3>
<p>For each remaining biological-sample pair, average its score differences over the original selected positions observed in both samples.
Keep the paired-position count beside the contrast: a missing position cannot supply a direction. Pairs sharing a sample are overlapping comparisons.</p>
{table(sample_view)}
<p>These values show how consistently a discovery-selected contrast is represented by the individual samples, and whether all of its original positions are available.</p>
{figure('example_locations', 'Selected positions mapped onto their spliced transcripts, with coding sequence, untranslated regions and stop codons.')}
<p>{utr_count}/{len(selected_sites)} selected positions lie in the 3′ UTR. Their exact coordinates and named cell comparisons give researchers
defined RNA targets. <a href="#support">Question 4</a> distinguishes examples that retain support at higher coverage from those that become unavailable.</p>
<details><summary>Biological context for the named RNA targets</summary><ul>{''.join(biology)}</ul>
<p>Experiments show that the m6A reader YTHDF2 can promote RNA decay
(<a href="https://doi.org/10.1038/nature12730">Wang et al., 2014</a>).
For candidates in that region, joint measurements of site-specific m6A and transcript lifetime in the named cell pairs test a focused hypothesis about RNA regulation.
The gene studies above explain the functions of these RNA targets; our measured contrasts concern their prediction scores.</p></details>
<p class="sources">Tables: <a href="discovery_candidates.csv">all transcript comparisons</a> ·
<a href="discovery_confirmation.csv">gene-level confirmation</a> · <a href="example_supported_sites.csv">exact example positions</a> ·
<a href="example_stability.csv">fixed-example depth checks</a> · <a href="example_confirmation_pairs.csv">individual confirmation samples</a></p></section>"""


def support_findings(output, metadata):
    """Connect changes in catalogue membership and contrasts to observed support."""
    depth = metadata['min_reads']
    thresholds = pd.read_csv(output/'threshold_stability.csv')
    strict = thresholds.loc[thresholds.min_reads.eq(thresholds.min_reads.max()) &
                            thresholds.score_threshold.eq(.5)].iloc[0]
    counts = thresholds.pivot(index='min_reads', columns='score_threshold', values='candidate_positions').reset_index()
    counts.columns = ['Reads per run', 'Every score ≥0.5', 'Every score ≥0.7', 'Every score ≥0.9']
    retained = thresholds[['min_reads','score_threshold','baseline_candidates','baseline_candidates_retained',
                           'lost_coverage','lost_score','new_candidates']].copy()
    retained.columns = ['Reads per run','Score cutoff',f'Candidates at {depth} reads','Retained','Lost coverage','Lost score','New']
    settings = pd.read_csv(output/'cell_contrast_stability.csv')
    broad_rows = []
    for cutoff, group in settings.groupby('min_reads'):
        fractions = group.baseline_positions_retained / group.baseline_matched_positions.replace(0, float('nan'))
        retained_range = 'Unavailable' if fractions.isna().all() else f'{100*fractions.min():.1f}–{100*fractions.max():.1f}%'
        change = group.excess_change_on_retained.abs().max()
        broad_rows.append({'Reads per run':cutoff,
            'Cell pairs with observations':int(group.matched_positions.gt(0).sum()),
            'Pairs with B > W':int(group.between_minus_within.gt(0).sum()),
            'Original positions retained':retained_range,
            'Largest |change in E| on retained positions':'Unavailable' if pd.isna(change) else f'{change:.6f}'})
    fixed = pd.read_csv(output/'example_stability.csv')
    coverage = fixed[['gene_name','min_reads','retained_positions','baseline_positions','confirmation_delta']].copy()
    coverage['Positions retained'] = coverage.retained_positions.astype(str) + '/' + coverage.baseline_positions.astype(str)
    coverage['Confirmation B − A'] = coverage.confirmation_delta.map(lambda x: number(x, True))
    coverage = coverage.rename(columns={'gene_name':'Gene','min_reads':'Reads per run'})[
        ['Gene','Reads per run','Positions retained','Confirmation B − A']]
    losses = int(strict.baseline_candidates_lost)
    if losses:
        answer = (f'At {int(strict.min_reads)} reads, {int(strict.lost_coverage)} of {losses} losses from the '
                  f'primary shared catalogue lose all-replicate observation; {int(strict.lost_score)} remain '
                  'fully observed but fall below the score cutoff.')
    else:
        answer = f'At {int(strict.min_reads)} reads, no candidates are lost from the primary shared catalogue.'
    return f"""<section id="support"><h2>4. How does read support change the conclusions?</h2>
<p class="answer"><strong>{answer}</strong> Read support determines which statements can still be evaluated.
Compare candidate identities and keep the original coordinates when following a local contrast.</p>
<h3>Separate observation support from model support</h3>
<p>A higher read cutoff asks for more measurements per contributing run. A higher score cutoff asks for stronger model support in every replicate.
The primary score cutoff is 0.5; 0.7 and 0.9 produce more selective catalogues. At each read cutoff, filter the original run observations,
then rebuild replicate and cell means. The fitted model, individual run-site scores and full-run context remain fixed.</p>
{table(counts)}
<p>At {int(strict.min_reads)} reads and score ≥0.5, {int(strict.baseline_candidates_retained)} original candidates remain and
{int(strict.new_candidates)} newly qualify. A coverage loss means at least one biological replicate lacks an eligible observation.
A score loss means all replicates remain observed but at least one score falls below the same cutoff.
Removing a technical run can change a replicate mean, so a stricter read cutoff can introduce a candidate.</p>
<details><summary>Retained, lost and new identities at every setting</summary>
<p>Each row compares with the {depth}-read catalogue at the same score cutoff. A position lost through coverage supplies no conclusion about its modification status.</p>
{table(retained)}</details>
<h3>Compare broad patterns on the same retained positions</h3>
<p>For each cell pair, compute E = B − W using the primary-depth means and again using the higher-depth means on their coordinate intersection.
This separates changes in contributing samples from changes in the positions being compared.</p>
{table(pd.DataFrame(broad_rows))}
<p>The saved comparison table also records newly qualifying positions. A position can enter when excluding a low-depth run brings the replicate-depth ratio within twofold.</p>
<h3>Follow the original local examples without replacing lost sites</h3>
<p>Keep the transcript, cell pair and selected coordinates fixed. The count beside each contrast shows how much of the original example remains observable.</p>
{table(coverage)}
<p>The shared candidates and local contrasts provide two complementary uses of the predictions: inspect sites repeatedly supported across cells,
or investigate a specific cell comparison with its replicate and coverage evidence attached.</p>
<p class="sources">Tables: <a href="threshold_stability.csv">catalogue counts and identity changes</a> ·
<a href="cell_contrast_stability.csv">same-position cell comparisons</a> · <a href="example_stability.csv">original local examples</a></p></section>"""


def write_findings(output, metadata):
    summary = pd.read_csv(output/'run_summary.csv')
    cells = pd.read_csv(output/'cell_summary.csv')
    pairs = pd.read_csv(output/'replicate_pairs.csv')
    contrasts = pd.read_csv(output/'cell_pairs.csv')
    examples = pd.read_csv(output/'examples.csv')
    depth = metadata['min_reads']
    scope = ('Complete declared collection' if metadata['complete'] else 'Partial collection')
    total, eligible = summary.observed_sites.sum(), summary.eligible_human.sum()
    counts = cells[['cell_line','runs','biological_replicates','eligible_human_site_observations']].copy()
    counts.columns = ['Cell line','Sequencing runs','Biological replicates',f'Human run-sites ≥{depth} reads']
    counts.iloc[:, 3] = counts.iloc[:, 3].map(lambda x: f'{x:,}')
    repeat = []
    for kind, rows in pairs.groupby('kind', sort=False):
        repeat.append({'Comparison':kind.replace('_',' '), 'Pairs':len(rows),
                       'Spearman range':f'{rows.spearman.min():.3f}–{rows.spearman.max():.3f}',
                       'Absolute difference range':f'{rows.mae.min():.3f}–{rows.mae.max():.3f}'})
    broad = '<p>No cell pair has jointly supported positions at this read cutoff.</p>'
    broad_answer = ''
    if contrasts.matched_sites.gt(0).any():
        values = contrasts.loc[contrasts.matched_sites.gt(0)]
        broad_answer = f'''<p class="answer"><strong>The median absolute difference between cell means is
{values.absolute_cell_delta_median.min():.4f}–{values.absolute_cell_delta_median.max():.4f} across the {len(values)} cell pairs.</strong>
Between-cell variation exceeds within-cell variation in {int(values.between_minus_within.gt(0).sum())} pairs,
with excesses from {values.between_minus_within.min():+.4f} to {values.between_minus_within.max():+.4f}.
The analysis below compares broad variation first; Question 3 then follows specific local contrasts in additional samples.</p>'''
        view = values[['cell_a','cell_b','matched_sites','within_replicate_mae','between_replicate_mae',
                       'between_minus_within','absolute_cell_delta_median','absolute_cell_delta_p90','absolute_cell_delta_p99']].copy()
        for name in view.columns[3:]:
            view[name] = view[name].map(lambda x: f'{x:.4f}')
        view.columns = ['Cell A','Cell B','Positions','Within W','Between B','Excess B − W','Median D','90th percentile D','99th percentile D']
        broad = f"""<p>Each comparison uses {values.matched_sites.min():,}–{values.matched_sites.max():,} matched positions.
E = B − W compares the size of cross-cell variation with the repeat variation shown alongside it.</p>
<p>To see differences hidden by averages, also calculate D = |mean score in B − mean score in A| at each position.
The median D ranges from {values.absolute_cell_delta_median.min():.4f} to {values.absolute_cell_delta_median.max():.4f} across pairs;
the 99th percentile ranges from {values.absolute_cell_delta_p99.min():.3f} to {values.absolute_cell_delta_p99.max():.3f}.
Thus a small global excess can coexist with larger local contrasts. D compares cell means, whereas W and B compare individual replicate scores.</p>
{figure('cell_comparison', 'Left: within-cell and between-cell variation on identical positions. Right: the distribution of absolute cell-mean differences D on those positions.')}
<details><summary>All cell pairs and the spread of their site-level differences</summary>{table(view)}</details>"""
    shared = local = context = support_section = overview = ''
    if metadata.get('evidence'):
        shared = shared_findings(output, metadata)
        local = local_findings(output, examples)
        support_section = support_findings(output, metadata)
        thresholds = pd.read_csv(output/'threshold_stability.csv')
        primary = thresholds.loc[thresholds.min_reads.eq(depth) & thresholds.score_threshold.eq(.5)].iloc[0]
        confirmation = pd.read_csv(output/'discovery_confirmation.csv').set_index('discovery_selection')
        top = confirmation.loc['top_100']
        overview = f'''<div class="questions">
<a href="#variation"><strong>1 · How different are the cells overall?</strong><span>Compare cell differences with the variation between repeated samples.</span></a>
<a href="#shared"><strong>2 · Which candidates are shared, and where?</strong><span>{int(primary.candidate_positions):,} positions in {int(primary.candidate_genes):,} genes meet the all-replicate rule.</span></a>
<a href="#local"><strong>3 · Which local differences recur?</strong><span>{int(top.direction_retained)}/{int(top.genes)} leading genes retain direction in confirmation samples.</span></a>
<a href="#support"><strong>4 · What changes with stronger read support?</strong><span>Separate lost observations from lower scores, and follow the same selected sites.</span></a>
</div>'''
        support = pd.read_csv(output/'context_support.csv')
        support_view = support[['cohort','positions','reference_unmatched_positions','complete_static_context_positions',
                                'eligible_run_observations','background_incomplete_observations','all_context_complete_observations']].copy()
        support_view.columns = ['Finding set','Positions','Reference unmatched','All 16 reference inputs',
                                'Run-site observations','Background incomplete','All 21 context inputs']
        labels = {'common':'Common positions', 'shared_candidates':'Shared candidates', 'examples':'Local examples'}
        support_view['Finding set'] = support_view['Finding set'].replace(labels)
        for column in support_view.columns[1:]:
            support_view[column] = support_view[column].map(lambda x: f'{x:,}')
        missing_view = support[['cohort','missing_stop_distance','missing_previous_motif','missing_next_motif',
                                'missing_exon_features','missing_flank_features']].copy()
        missing_view.columns = ['Finding set','Stop distance','Previous motif','Next motif','Exon inputs','Flank inputs']
        missing_view['Finding set'] = missing_view['Finding set'].replace(labels)
        context = f"""<section id="context"><h2>Reference: model-input availability</h2>
<p>The findings above use the fixed Task 1 model. This reference records which transcript descriptors and sample-background inputs were available at the reported candidates.</p>
<details><summary>Context availability for common positions, shared candidates and local examples</summary>
<p>There are 16 reference descriptors and five summaries of other reference-matched sites on the same transcript in the complete run.
An absent stop codon or neighboring motif removes its corresponding descriptor; no other reported site makes the four background means missing and its count zero.
The fitted trees process missing values directly.</p>{table(support_view)}
<p>Reference availability is counted by transcript position. Full context is counted by run-site observation, since background depends on the sample.
For local examples, the observation counts use only the two cells defining each contrast.</p>
<h3>Which reference inputs are absent?</h3>{table(missing_view)}
<p>A position can lack more than one input, so these columns can overlap.</p>
<p>These counts describe the evidence entering the model. Matched experimental labels would be needed to quantify how missing inputs affect accuracy.</p>
<p class="sources">Tables: <a href="context_support.csv">context support</a> · <a href="site_annotations.parquet">position annotations</a></p></details></section>"""
    overlap = (' Hct116 replicate 3/run 1 also contributed measurements to Dataset 0 training; this application describes predictions, while Task 1 evaluates accuracy.'
               if summary.overlaps_data0.any() else '')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Context-aware trees · SG-NEx findings</title>
<style>
:root{{--ink:#183a4a;--accent:#146c70;--pale:#eef5f5;--muted:#52636d;--line:#ccdadd}}
*{{box-sizing:border-box}}body{{margin:0;background:#f7f9fa;color:var(--ink);font:17px/1.6 Arial,sans-serif}}
main{{max-width:1110px;margin:auto;background:white;padding:44px 48px 60px}}h1{{font-size:34px;line-height:1.2;margin:8px 0 18px}}
h2{{font-size:26px;line-height:1.3;margin:0 0 18px}}h3{{font-size:20px;margin:25px 0 10px}}p{{margin:12px 0}}
.eyebrow,.sources,figcaption{{font-size:14px;color:var(--muted)}}a{{color:var(--accent);text-underline-offset:3px}}nav{{display:flex;flex-wrap:wrap;gap:9px 22px;margin:25px 0}}
section{{border-top:1px solid var(--line);padding-top:30px;margin-top:34px;scroll-margin-top:18px}}.lead,.answer{{background:var(--pale);padding:15px 20px;border-left:4px solid var(--accent)}}
.questions{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:24px 0}}.questions a{{border:1px solid var(--line);border-radius:6px;padding:17px;text-decoration:none}}.questions a:hover{{background:var(--pale)}}.questions strong,.questions span{{display:block}}.questions span{{font-size:15px;color:var(--muted);margin-top:8px}}
.scroll{{overflow-x:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;line-height:1.45}}th{{background:var(--pale);text-align:left}}td,th{{border-bottom:1px solid var(--line);padding:9px 10px;vertical-align:top}}
figure{{margin:25px 0}}img{{display:block;width:100%;height:auto}}figcaption{{margin-top:9px}}details{{border:1px solid var(--line);padding:12px 16px;margin:18px 0}}summary{{cursor:pointer;font-weight:bold}}
li{{margin:10px 0}}code{{font-size:.9em}}footer{{border-top:1px solid var(--line);margin-top:35px;padding-top:18px;font-size:14px}}
@media(max-width:700px){{main{{padding:25px 18px}}h1{{font-size:28px}}h2{{font-size:23px}}body{{font-size:16px}}.questions{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}details{{display:block}}section,figure{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Context-aware trees · SG-NEx direct RNA sequencing · {scope}</div><h1>What do the predicted m6A patterns show across cells?</h1>
<p>We connect the fixed Task 1 model to four questions about the RNA samples: how predictions differ across cells,
which candidates are shared, which local differences recur, and how observation support affects each conclusion.</p>
<p class="lead"><strong>{total:,} run-site predictions</strong> across {len(summary)} sequencing runs,
{summary.replicate_id.nunique()} biological replicates and {summary.cell_line.nunique()} cell lines.
The comparisons below use {eligible:,} human observations with at least {depth} reads per run.</p>
{overview}<nav><a href="#data">How the samples are compared</a><a href="#context">Reference: input availability</a></nav></header>
<section id="data"><h2>Read the results at the correct sample level</h2>
<p>A run-site observation is a transcript coordinate reported in one sequencing run. A biological replicate is a separate sample preparation;
technical runs sequence the same replicate. The declared collection contains processed direct-RNA files from
<a href="https://doi.org/10.1038/s41592-025-02623-4">SG-NEx</a>. Every reported site and supplied read enters prediction.
The Task 1 fit and transformations remain fixed. A score expresses support for the binary m6A site label, not the fraction of molecules carrying m6A.</p>
<p>After prediction, retain human ENST observations with at least {depth} reads per run: {100*eligible/total:.1f}% of all run-site observations.
Match positions by sequence-verified transcript ID, zero-based coordinate and agreeing seven-base sequence. A coordinate missing from a run contributes no score.</p>
<p><strong>Average technical runs within each biological replicate; then average biological replicates within a cell.</strong>
This gives each biological sample equal weight, following the distinction between technical measurements and biological replication
(<a href="https://doi.org/10.1083/jcb.202001064">Lord et al., 2020</a>).
Read counts are averaged in the same two stages, and support counts accompany each mean.</p>
<details><summary>Sample inventory, coverage and an averaging example</summary>{table(counts)}
<p>Technical scores 0.2 and 0.4 give one replicate score of 0.3; with another replicate scoring 0.9, the cell mean is 0.6.
The biological sample, rather than its number of sequencing runs, determines its weight.</p>
{figure('coverage', 'All reported observations were predicted. This plot shows how many enter the primary human comparison and how many belong to other support or transcript categories.')}
</details><p>{escape(overlap.strip())}</p><p class="sources">Tables: <a href="input_inventory.csv">input inventory</a> ·
<a href="run_summary.csv">run summaries</a> · <a href="cell_summary.csv">cell summaries</a> ·
<a href="biological_sites.parquet">biological-replicate means</a> · <a href="cell_sites.parquet">cell means</a></p></section>
<section id="variation"><h2>1. How different are the cells overall?</h2>{broad_answer}
<h3>Establish repeatability at shared positions</h3>
<p>For technical runs within a sample and biological replicates within a cell, compare their shared eligible positions.
Spearman correlation measures agreement in score ranking; mean absolute difference measures the size of score disagreement.</p>
{table(pd.DataFrame(repeat))}<details><summary>View each technical and biological replicate comparison</summary>
{figure('replicate_agreement', 'Each point is a pair of repeated measurements compared on its shared eligible human positions.')}</details>
<h3>Use the same positions for within-cell and between-cell differences</h3>
<p>A position needs at least two biological replicates per cell. The largest biological-replicate mean read depth across both cells
must be at most twice the smallest. This limits unequal read support. At each qualifying position, average all absolute cross-cell replicate differences (B);
average within-cell differences separately and give the two cells equal weight (W). Average both quantities over exactly the same positions.</p>
<p>Positive B − W means the between-cell gap exceeds the within-cell gap. It describes score variation in this collection;
cell identity and sequencing conditions both vary.</p>{broad}
<p>These averages describe the broad pattern. Shared high-scoring positions and discovery-selected local contrasts answer the next two questions;
<a href="#support">Question 4</a> follows the comparisons as read support increases.</p>
<p class="sources">Tables: <a href="replicate_pairs.csv">repeat comparisons</a> · <a href="cell_pairs.csv">cell comparisons</a></p></section>
{shared}{local}{support_section}{context}
<footer>Reproduce these tables, figures and findings with <code>python analyze.py</code> using the saved predictions and declared sample inventory.
<a href="metadata.json">Analysis settings</a> · <a href="../README.md">Run instructions and output reference</a>.</footer>
</main></body></html>"""
    (output/'findings.html').write_text(html, encoding='utf-8')
