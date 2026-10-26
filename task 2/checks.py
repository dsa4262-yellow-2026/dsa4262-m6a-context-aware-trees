"""Focused checks for final-model prediction alignment and replicate averaging."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze import aggregate_cell, aggregate_replicate, analyze, load_sites
from run import prediction_settings
from comparisons import replicate_details, supported_comparison, variation_summary, rank_candidates, shared_positions
from analyze import matched_sites
from evidence import (region_at, confirmation_summary, catalogue_overlap, region_composition,
                      region_by_gene, example_confirmation_pairs, retained_contrast, threshold_tables)

HERE = Path(__file__).resolve().parent
PREDICTOR = HERE.parent/'task 1 context aware trees'
KEY = ['transcript_id', 'transcript_position']


def fixture(directory):
    records = []
    for line in (PREDICTOR/'example/input.json').read_text(encoding='utf-8-sig').splitlines():
        for transcript, positions in json.loads(line).items():
            records.extend((transcript, position, context) for position, context in positions.items())
    versioned = records[0][0]+'.999'
    records = [(versioned if t == records[0][0] else t, p, c) for t, p, c in records]
    records.append(('UNMATCHED_FIXTURE', '00010', records[0][2]))
    rows = []
    with (directory/'data.json').open('wb') as stream:
        for transcript, position, context in records:
            start = stream.tell()
            stream.write((json.dumps({transcript: {position: context}}, separators=(',', ':'))+'\n').encode())
            rows.append((transcript, position, start, stream.tell(), len(next(iter(context.values())))))
    index = pd.DataFrame(rows, columns=KEY+['start', 'end', 'n_reads'])
    index.to_csv(directory/'data.info', index=False)
    inventory = pd.DataFrame([dict(sample_id='fixture', cell_line='fixture', replicate=1, run=1,
        replicate_id='fixture_r1', signals='data.json', info='data.info', sites=len(index),
        sites_at_least_20_reads=int(index.n_reads.ge(20).sum()))])
    inventory.to_csv(directory/'samples.csv', index=False)
    return versioned, inventory


def check_averaging():
    def row(score, depth, matched):
        return pd.DataFrame([dict(transcript_id='ENSTtest', transcript_position='10', kmer='AAACTAA',
            n_reads=depth, gene_id='ENSGtest', reference_matched=matched, score=score,
            model='context_aware_trees_v1')])
    samples = pd.DataFrame([dict(sample_id=s, cell_line='cell', replicate_id=r)
                           for s, r in [('a1', 'a'), ('a2', 'a'), ('b1', 'b')]])
    a = aggregate_replicate([row(.2, 20, True), row(.4, 40, False)], samples.iloc[:2])
    b = aggregate_replicate([row(.9, 90, True)], samples.iloc[2:])
    combined = aggregate_cell(pd.concat([a, b]))
    np.testing.assert_allclose(combined[['score', 'n_reads', 'reference_matched_fraction']], [[.6, 60, .75]])
    assert combined.n_replicates.iloc[0] == 2
    missing = aggregate_cell(a)
    np.testing.assert_allclose(missing.score, [.3])
    assert missing.n_replicates.iloc[0] == 1



def check_matched_comparisons():
    cells = {}
    for cell, scores, depths in [('A', [.1,.3], [20,40]), ('B', [.7,.9,.5], [20,30,40])]:
        rows = []
        for i, (score, depth) in enumerate(zip(scores, depths)):
            for position in ['10','20','30']:
                if position == '30' and cell == 'B' and i > 0:
                    continue
                rows.append(dict(transcript_id='ENSTtest', transcript_position=position,
                    gene_id='ENSGtest', kmer='AAACTAA', score=score, n_reads=80 if position=='20' and cell=='B' and i==0 else depth,
                    n_runs=1, reference_matched_fraction=1., cell_line=cell, replicate_id=f'{cell}{i}', available_runs=1))
        frame = pd.DataFrame(rows)
        order = [f'{cell}{i}' for i in range(len(scores))]
        cells[cell] = aggregate_cell(frame).merge(replicate_details(frame, order, 3), on=KEY, validate='one_to_one')
    joined, _ = matched_sites(cells['A'], cells['B'])
    result = supported_comparison(joined)
    assert result.transcript_position.tolist() == ['10']
    # Six cross-cell pairs: their gaps average 0.5; within cells receive equal weight.
    np.testing.assert_allclose(result.between_replicate_mae, [.5])
    np.testing.assert_allclose(result.within_replicate_mae, [(.2 + (.2+.2+.4)/3)/2])
    np.testing.assert_allclose(result.discovery_delta, [.6])
    np.testing.assert_allclose(result.confirmation_delta, [.4])
    np.testing.assert_allclose(variation_summary(result)['between_minus_within'], .5-7/30)
    shared = shared_positions(cells, 5)
    assert set(shared.transcript_position) == {'10','20'}
    assert shared.all_replicates_observed.all() and not shared.high_in_every_replicate.any()
    assert shared_positions(cells, 7, ['A', 'B', 'C']).empty
    assert shared_positions({}, 5, ['A', 'B']).empty
    for name in ['absolute_cell_delta_median', 'absolute_cell_delta_p90', 'absolute_cell_delta_p99']:
        np.testing.assert_allclose(variation_summary(result)[name], .5)
    candidates = pd.DataFrame([dict(gene_id='g1', transcript_id='t1', matched_supported_sites=3,
        discovery_delta=.2, confirmation_delta=.9, cell_a='A', cell_b='B'),
        dict(gene_id='g2', transcript_id='t2', matched_supported_sites=3,
        discovery_delta=-.3, confirmation_delta=0., cell_a='A', cell_b='B')])
    before = rank_candidates([candidates]).gene_id.tolist()
    candidates['confirmation_delta'] = [-100., 100.]
    assert rank_candidates([candidates]).gene_id.tolist() == before == ['g2','g1']


def check_catalogue_background():
    common = pd.DataFrame(dict(transcript_id=['t']*4, transcript_position=['1','2','3','4'],
        gene_id=['a','a','b','b'], all_replicates_observed=[True, False, True, True],
        minimum_replicate_score=[.95,.99,.8,.2]))
    annotations = common[KEY].assign(region=['3UTR','3UTR','CDS','CDS'])
    candidate = common.loc[common.all_replicates_observed & common.minimum_replicate_score.ge(.9)]
    regions = region_composition(common, candidate, annotations)
    background = regions.loc[regions.cohort.eq('eligible_background')].set_index('region')
    assert background.loc['3UTR', 'positions'] == 1
    assert background.loc['3UTR', 'denominator'] == 3
    same_genes = regions.loc[regions.cohort.eq('eligible_same_genes')]
    assert same_genes.denominator.eq(1).all()
    # A stricter depth can change a run average and introduce another candidate.
    current = common.iloc[[0, 2]]
    overlap = catalogue_overlap(candidate, current, current)
    assert overlap == dict(baseline_candidates=1, baseline_candidates_retained=1,
                           baseline_candidates_lost=0, new_candidates=1, lost_coverage=0, lost_score=0)


def check_evidence_extensions():
    # Four baseline candidates: one retained, two lose coverage, one loses score.
    sites = pd.DataFrame({'transcript_id':['t']*5, 'transcript_position':list('12345')})
    result = catalogue_overlap(sites.iloc[:4], sites.iloc[[0,4]], sites.iloc[[0,3,4]])
    assert result['lost_coverage'] == 2 and result['lost_score'] == 1
    assert result['baseline_candidates_retained'] == 1 and result['new_candidates'] == 1
    # Fixing candidate counts gives (2*1/2 + 1*0/2)/3 = 1/3 expected UTR share.
    common = sites.iloc[:4].assign(gene_id=['a','a','b','b'], all_replicates_observed=True)
    ann = common[KEY].assign(region=['3UTR','CDS','CDS','CDS'])
    candidate = common.iloc[[0,1,2]]
    gene = region_by_gene(common, candidate, ann)
    np.testing.assert_allclose(gene.expected_candidate_3utr.sum()/gene.candidate_positions.sum(), 1/3)
    # The second confirmation pair covers only one of two selected positions.
    supported = sites.iloc[:2].assign(cell_a='A',cell_b='B',
        replicate_score_1_a=[.1,.3], replicate_score_2_a=[.9,np.nan], replicate_score_1_b=[.5,.7])
    examples = pd.DataFrame([dict(gene_name='g',transcript_id='t',cell_a='A',cell_b='B',discovery_delta=.2)])
    inventory = pd.DataFrame([dict(cell_line=c,replicate=i,run=1,replicate_id=f'{c}{i}')
        for c, n in [('A',3),('B',2)] for i in range(n)])
    pairs = example_confirmation_pairs(supported, examples, inventory)
    assert pairs.paired_positions.tolist() == [2,1]
    assert pairs.all_selected_positions.tolist() == [True,False]
    np.testing.assert_allclose(pairs.aligned_delta, [.4,-.4])
    empty = example_confirmation_pairs(supported, examples.iloc[:0], inventory)
    assert empty.empty and 'paired_positions' in empty


def check_coverage_comparisons():
    before = pd.DataFrame(dict(transcript_id=['t','t'], transcript_position=['1','2'],
        within_replicate_mae=[.1,.2], between_replicate_mae=[.4,.3]))
    after = pd.DataFrame(dict(transcript_id=['t','t'], transcript_position=['2','3'],
        within_replicate_mae=[.1,.2], between_replicate_mae=[.5,.9]))
    result = retained_contrast(before, after)
    assert result['baseline_positions_retained'] == 1
    assert result['baseline_positions_lost'] == result['newly_qualifying_positions'] == 1
    np.testing.assert_allclose([result['baseline_excess_on_retained'],
        result['current_excess_on_retained'], result['excess_change_on_retained']], [.1,.4,.3])
    absent = retained_contrast(before, None)
    assert absent['baseline_positions_lost'] == 2 and np.isnan(absent['current_excess_on_retained'])
    # If a cell has no remaining observations, keep its comparison and example explicit.
    inventory = pd.DataFrame(dict(cell_line=['A','B'], replicate_id=['a','b']))
    examples = pd.DataFrame([dict(gene_name='g', transcript_id='t', cell_a='A', cell_b='B')])
    supported = before[KEY].assign(cell_a='A',cell_b='B',discovery_delta=.2,
                                  confirmation_delta=.1,direction_agrees=True)
    with tempfile.TemporaryDirectory() as temporary, patch('evidence.rebuild_cells', return_value=({},0)):
        folder = Path(temporary)
        threshold_tables(folder, inventory, {}, examples, supported, folder, 20, 0)
        contrasts = pd.read_csv(folder/'cell_contrast_stability.csv')
        fixed = pd.read_csv(folder/'example_stability.csv')
        assert len(contrasts)==3 and contrasts.matched_positions.eq(0).all()
        assert len(fixed)==3 and fixed.retained_positions.eq(0).all()
        assert fixed.confirmation_delta.isna().all()


def main():
    check_averaging()
    check_matched_comparisons()
    check_catalogue_background()
    check_evidence_extensions()
    check_coverage_comparisons()
    # Interval boundaries must preserve the stop codon and adjacent untranslated bases.
    geometry = {'CDS':[(10,20)], 'stop_codon':[(20,22),(22,23)], 'UTR':[(0,10),(23,30)]}
    assert [region_at(p, geometry) for p in [9,10,19,20,22,23]] == ['5UTR','CDS','CDS','stop_codon','stop_codon','3UTR']
    assert region_at(5, {'UTR':[(0,10)]}) == 'noncoding_or_unassigned'
    chosen = pd.DataFrame(dict(gene_id=['a','a','b','c'], discovery_delta=[.8,.7,-.6,.5],
                              confirmation_delta=[.2,-.9,.1,0.]))
    measured = confirmation_summary(chosen).iloc[0]
    assert measured.genes == 3 and measured.direction_retained == 1
    with tempfile.TemporaryDirectory(prefix='context_trees_task2_check_') as temporary:
        folder = Path(temporary)
        versioned, inventory = fixture(folder)
        command = [sys.executable, '-B', str(HERE/'run.py'), '--data-root', str(folder),
                   '--samples', str(folder/'samples.csv'), '--output', str(folder/'results')]
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, '-B', str(PREDICTOR/'predict.py'), '--data', str(folder/'data.json'),
                        '--output', str(folder/'native.csv')], check=True)
        read = lambda path: pd.read_csv(path, dtype={k: str for k in KEY}, float_precision='round_trip')
        native = read(folder/'native.csv')
        stored = pd.read_parquet(folder/'results/fixture/sites.parquet')
        pd.testing.assert_frame_equal(native, read(folder/'results/fixture/predictions.csv.gz'), check_exact=True)
        pd.testing.assert_frame_equal(native, stored[KEY+['score']], check_exact=True)
        assert stored.loc[stored.transcript_id.eq(versioned), 'reference_matched'].all()
        assert not stored.iloc[-1].reference_matched and stored.iloc[-1].transcript_position == '00010'
        compared = load_sites(folder/'results/fixture/sites.parquet', min_reads=1, human_only=False)
        assert compared.iloc[-1].transcript_position == '10'
        assert compared.iloc[-1].input_transcript_position == '00010'
        marker = folder/'results/fixture/run.json'
        before = marker.read_bytes()
        existing = subprocess.run(command, capture_output=True, text=True)
        assert existing.returncode and 'results already exist' in existing.stderr
        subprocess.run(command + ['--resume'], check=True)
        assert marker.read_bytes() == before
        more = pd.concat([inventory, inventory.assign(sample_id='missing', replicate_id='fixture_r2', replicate=2)])
        more.to_csv(folder/'incomplete.csv', index=False)
        try:
            analyze(folder/'results', folder/'incomplete.csv', folder/'analysis')
        except ValueError as error:
            assert 'lack completed predictions' in str(error)
        else:
            raise AssertionError('Incomplete collection accepted without --allow-partial.')
        with patch('analyze.figures', return_value='not rendered in fixture'):
            metadata = analyze(folder/'results', folder/'incomplete.csv', folder/'analysis', allow_partial=True)
        assert not metadata['complete'] and metadata['missing_runs'] == ['missing']
        different = inventory.assign(cell_line='different_sample')
        different.to_csv(folder/'different.csv', index=False)
        try:
            analyze(folder/'results', folder/'different.csv', folder/'wrong_sample')
        except ValueError as error:
            assert 'sample details differ' in str(error)
        else:
            raise AssertionError('Mismatched sample annotation accepted.')
        with (folder/'data.json').open('ab') as stream:
            stream.write(b'\n')
        changed = subprocess.run(command + ['--resume'], capture_output=True, text=True)
        assert changed.returncode and 'Input index does not cover' in changed.stderr
        invalid = folder/'invalid_model'
        invalid.mkdir()
        (invalid/'manifest.json').write_bytes((PREDICTOR/'model/manifest.json').read_bytes())
        (invalid/'encoder.pt').write_bytes(b'changed component')
        try:
            prediction_settings(invalid, 'cpu', 3)
        except ValueError as error:
            assert 'saved context-aware trees model directory' in str(error)
        else:
            raise AssertionError('Incomplete fitted model accepted.')
    print('Passed: coverage-versus-score turnover, gene-weighted UTR background, individual confirmation support, prediction alignment, original keys, reference flags, explicit resume, sample alignment, incomplete collection/model, replicate averaging, matched-pair arithmetic, discovery-only ranking, declared-cell support, score-specific retention and catalogue-eligible region backgrounds.')


if __name__ == '__main__':
    main()
