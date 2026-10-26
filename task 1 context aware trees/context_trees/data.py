"""Read original signals, match labels by key, and keep evaluation groups intact."""
from pathlib import Path

import numpy as np
import pandas as pd

from .features import KEY, iter_sites, reference_features, summarize_reads, transformed_reads


def attach_labels(sites, path, reference):
    labels = pd.read_csv(path, dtype={'transcript_id': str, 'gene_id': str})
    position = pd.to_numeric(labels.transcript_position, errors='raise')
    if position.isna().any() or not position.eq(np.floor(position)).all():
        raise ValueError('Label positions must be integers.')
    labels['transcript_position'] = position.astype(np.int64)
    if labels.duplicated(KEY).any() or not labels.label.isin([0, 1]).all():
        raise ValueError('Binary labels must be unique by site and equal to 0 or 1.')
    mapped = labels.transcript_id.str.split('.').str[0].map(reference['transcript_gene'])
    if 'gene_id' in labels:
        known = mapped.notna()
        if not labels.loc[known, 'gene_id'].eq(mapped[known]).all():
            raise ValueError('Label genes disagree with the reference.')
    else:
        labels['gene_id'] = mapped
    columns = KEY + ['gene_id', 'label']
    if 'n_reads' in labels:
        columns.append('n_reads')
        labels = labels.rename(columns={'n_reads': 'label_read_count'})
        columns[-1] = 'label_read_count'
    joined = sites.merge(labels[columns], on=KEY, how='left', sort=False, validate='one_to_one')
    if len(joined) != len(labels) or joined[['gene_id', 'label']].isna().any().any():
        raise ValueError('Every signal site must match exactly one label and a known gene.')
    if 'label_read_count' in joined:
        if not joined.n_reads.eq(joined.label_read_count).all():
            raise ValueError('Declared read counts disagree with the original signals.')
        joined = joined.drop(columns='label_read_count')
    joined['label'] = joined.label.astype(np.int8)
    return joined


def prepare_sites(datasets, directory, reference):
    """Prepare fixed summaries and on-disk reads, preserving original site order.

    Each dataset is (source_number, signal_path, label_path_or_None).
    Only dwell time is transformed: log(1 + 1000 * dwell), after a float32 cast.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames, summaries, offset = [], [], 0
    with (directory / 'reads.bin').open('wb') as stream:
        for source, signal_path, label_path in datasets:
            rows = []
            for transcript, position, kmer, raw in iter_sites(signal_path):
                reads = transformed_reads(raw)
                reads.tofile(stream)
                summaries.append(summarize_reads(reads))
                rows.append((source, transcript, int(position), position, kmer,
                             len(reads), offset, offset + len(reads)))
                offset += len(reads)
                if len(rows) % 40000 == 0:
                    print(f'Source {source}: prepared {len(rows):,} sites.', flush=True)
            sites = pd.DataFrame(rows, columns=['source', *KEY, 'position_key', 'kmer',
                                                'n_reads', 'read_start', 'read_stop'])
            if sites.empty:
                raise ValueError('An input file has no sites.')
            if label_path is not None:
                sites = attach_labels(sites, label_path, reference)
            frames.append(sites)
    sites = pd.concat(frames, ignore_index=True)
    if 'gene_id' in sites and not sites.groupby('transcript_id').gene_id.nunique().eq(1).all():
        raise ValueError('Gene assignments conflict across datasets.')
    geometry, flanks, distances, reference_matched = reference_features(sites, reference)
    return dict(sites=sites,
                reads=np.memmap(directory / 'reads.bin', mode='r', dtype=np.float32, shape=(offset, 9)),
                summary=np.stack(summaries),
                geometry=geometry, flanks=flanks, motif_distances=distances, reference_matched=reference_matched)



def close_reads(*arrays):
    for array in arrays:
        array._mmap.close()


def attach_order(sites, path, order_column, group_columns, source):
    """Join declared order and group metadata by site identity, never row order."""
    labels = pd.read_csv(path, dtype={'transcript_id': str})
    position = pd.to_numeric(labels.transcript_position, errors='raise')
    if position.isna().any() or not position.eq(np.floor(position)).all():
        raise ValueError('Label positions must be integers.')
    labels['transcript_position'] = position.astype(np.int64)
    if labels.duplicated(KEY).any():
        raise ValueError('Ordered labels must be unique by site.')
    levels = pd.to_numeric(labels[order_column], errors='raise')
    if not np.isfinite(levels).all() or labels[group_columns].isna().any().any():
        raise ValueError('Ordered labels need finite levels and nonmissing groups.')
    metadata = labels[KEY].copy()
    metadata['order_level'] = levels
    metadata['order_key'] = [(source, *row) for row in labels[group_columns].itertuples(index=False, name=None)]
    if 'n_reads' in labels:
        metadata['declared_reads'] = labels.n_reads
    joined = sites.merge(metadata, on=KEY, how='left', sort=False, validate='one_to_one')
    if len(joined) != len(labels) or joined.order_level.isna().any():
        raise ValueError('Every signal site must match exactly one ordered label.')
    if 'declared_reads' in joined:
        if not joined.n_reads.eq(joined.declared_reads).all():
            raise ValueError('Ordered read counts disagree with signals.')
        joined = joined.drop(columns='declared_reads')
    return joined


def prepare_training(config, base, directory, reference):
    """Load arbitrary named binary datasets and optional within-group order labels."""
    entries = config['datasets']
    names = [entry['name'] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError('Each dataset name must be unique.')
    if any(entry['kind'] not in ['binary', 'ordered'] for entry in entries):
        raise ValueError('Dataset kind must be binary or ordered.')
    base, directory = Path(base), Path(directory)
    binary = [entry for entry in entries if entry['kind'] == 'binary']
    if not binary:
        raise ValueError('At least one binary labelled dataset is required.')
    prepared = prepare_sites([(e['name'], base / e['signals'], base / e['labels']) for e in binary],
                             directory / 'binary', reference)
    ordered = None
    try:
        entries = [entry for entry in entries if entry['kind'] == 'ordered']
        if entries:
            ordered = prepare_sites([(e['name'], base / e['signals'], None) for e in entries],
                                    directory / 'ordered', reference)
            frames = []
            for entry in entries:
                sites = ordered['sites'].loc[ordered['sites'].source.eq(entry['name'])]
                frames.append(attach_order(sites, base / entry['labels'], entry.get('order_column', 'label'),
                                           entry.get('group_columns', KEY),
                                           entry.get('group_namespace', entry['name'])))
            sites = pd.concat(frames, ignore_index=True)
            sites['order_group'] = pd.factorize(sites.order_key, sort=False)[0]
            ordered['sites'] = sites.drop(columns='order_key')
        return prepared, ordered
    except Exception:
        close_reads(prepared['reads'])
        if ordered is not None:
            close_reads(ordered['reads'])
        raise
