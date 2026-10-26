"""Measured signal, local sequence and optional transcript context."""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["transcript_id", "transcript_position"]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def iter_sites(path):
    """Yield original identifiers, normalized context and finite N-by-9 signals."""
    with Path(path).open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    seen = set()
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            for transcript, positions in record.items():
                if not transcript or not isinstance(positions, dict):
                    raise ValueError(f"Line {line_number}: invalid transcript record")
                for position, contexts in positions.items():
                    if not re.fullmatch(r"[0-9]+", position):
                        raise ValueError(
                            f"Expected a nonnegative integer position: {position}"
                        )
                    key = (transcript, int(position))
                    if (
                        key in seen
                        or not isinstance(contexts, dict)
                        or len(contexts) != 1
                    ):
                        raise ValueError(f"Duplicate site or ambiguous sequence: {key}")
                    seen.add(key)
                    context, values = next(iter(contexts.items()))
                    context = context.upper().replace("U", "T")
                    if not re.fullmatch("[ACGT][AGT][AG]AC[ACT][ACGT]", context):
                        raise ValueError(
                            f"Expected a seven-base DRACH context at {key}"
                        )
                    reads = np.asarray(values, dtype=np.float64)
                    if (
                        reads.ndim != 2
                        or reads.shape[1] != 9
                        or len(reads) == 0
                        or not np.isfinite(reads).all()
                    ):
                        raise ValueError(
                            f"Expected finite, nonempty N-by-9 signals at {key}"
                        )
                    if (reads[:, [0, 1, 3, 4, 6, 7]] < 0).any():
                        raise ValueError(f"Negative dwell time or signal SD at {key}")
                    yield transcript, position, context, reads


def transformed_reads(raw):
    """Transform dwell times: cast first, then log1p dwell in float32."""
    values = np.array(raw, dtype=np.float32, copy=True)
    values[:, [0, 3, 6]] = np.log1p(values[:, [0, 3, 6]] * 1000.0)
    return values


def sequence_encoding(kmers):
    lookup = np.full(256, -1, dtype=np.int16)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    letters = np.frombuffer("".join(kmers).encode("ascii"), dtype=np.uint8).reshape(
        -1, 7
    )
    codes = lookup[letters]
    if np.any(codes < 0):
        raise ValueError("Seven-mers must contain A, C, G and T only.")
    return np.eye(4, dtype=np.float32)[codes].reshape(-1, 28)


def physical_contexts(sequence):
    """Three base-4 five-mer codes from the 28-column seven-mer one-hot encoding."""
    codes = sequence.reshape(-1, 7, 4).argmax(2)
    weights = np.array([256, 64, 16, 4, 1])
    return np.column_stack([codes[:, w : w + 5] @ weights for w in range(3)])


MOTIF_COLUMNS = np.array([4, 6, 8, 20, 21])
MOTIF_NAMES = [f"sequence_p{i // 4}_{'ACGT'[i % 4]}" for i in MOTIF_COLUMNS]
TRANSCRIPT_CONTEXT_NAMES = [
    f"other_mean_{channel}" for channel in ["log_dwell", "signal_sd", "current_mean"]
] + ["other_mean_log_read_count", "other_site_count"]


def transcript_signal_context(sites, summary, scalers, reference_matched):
    """Other-site background within reference-verified transcripts; exclude synthetic groups."""
    contexts = physical_contexts(sequence_encoding(sites.kmer.to_numpy()))
    mean = scalers["read_mean"][contexts].reshape(-1, 9)
    scale = scalers["read_scale"][contexts].reshape(-1, 9)
    center = (summary[:, :9].astype(np.float64) - mean) / scale
    sizes = sites.n_reads.to_numpy(dtype=np.float64)
    groups, identities = pd.factorize(
        pd.MultiIndex.from_frame(sites[["source", "transcript_id"]]), sort=False
    )
    valid = np.asarray(reference_matched, dtype=bool)
    other_count = np.bincount(groups, weights=valid)[groups] - valid
    values = np.column_stack((center, np.log1p(sizes))) * valid[:, None]
    sums = np.zeros((len(identities), values.shape[1]))
    np.add.at(sums, groups, values)
    other = np.divide(
        sums[groups] - values,
        other_count[:, None],
        out=np.zeros_like(values),
        where=other_count[:, None] > 0,
    )
    result = np.column_stack((other, other_count)).astype(np.float32)
    # Equal averages across the left, center and right normalized windows.
    pooled = result[:, :9].reshape(-1, 3, 3).mean(axis=1)
    output = np.column_stack((pooled, result[:, 9:]))
    output[other_count == 0, :4] = np.nan
    output[~valid] = np.nan
    return output


GEOMETRY_NAMES = [
    "transcript_length_nt",
    "transcript_position_fraction",
    "exon_length_nt",
    "exon_rank_fraction",
    "exon_position_fraction",
    "signed_distance_stop_codon_nt",
]
MOTIF_DISTANCE_NAMES = ["distance_previous_drach_nt", "distance_next_drach_nt"]
REGIONS = [
    ("upstream_far", -30, -16),
    ("upstream_near", -15, -4),
    ("downstream_near", 4, 15),
    ("downstream_far", 16, 30),
]
FLANK_OFFSETS = np.r_[np.arange(-30, -3), np.arange(4, 31)]
COMPACT_FLANK_NAMES = [f"{name}_gc_frequency" for name, _, _ in REGIONS]
A_FLANK_NAMES = [f"{name}_a_frequency" for name, _, _ in REGIONS]


def compact_flanks(flanks):
    """GC and A proportions among observed bases; an unobserved region is missing."""
    bases = np.asarray(flanks, dtype=np.float32).reshape(-1, len(FLANK_OFFSETS), 4)
    gc, adenine = [], []
    for _, start, stop in REGIONS:
        region = bases[:, (FLANK_OFFSETS >= start) & (FLANK_OFFSETS <= stop)]
        count = region.sum(axis=(1, 2))
        for output, numerator in [
            (gc, region[:, :, 1:3].sum(axis=(1, 2))),
            (adenine, region[:, :, 0].sum(axis=1)),
        ]:
            output.append(
                np.divide(
                    numerator,
                    count,
                    out=np.full(len(count), np.nan, dtype=np.float32),
                    where=count > 0,
                )
            )
    return np.column_stack(gc + adenine)


def reference_features(sites, reference):
    """Attach six location features, eight flank proportions and two motif distances."""
    geometry = np.full((len(sites), 6), np.nan, dtype=np.float32)
    flanks = np.full((len(sites), 8), np.nan, dtype=np.float32)
    distances = np.full((len(sites), 2), np.nan, dtype=np.float32)
    available = np.zeros(len(sites), dtype=bool)
    positions, kmers = sites.transcript_position.to_numpy(), sites.kmer.to_numpy()
    lookup = np.full(256, 4, dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    onehot = np.eye(5, 4, dtype=np.uint8)
    for tid, rows in sites.groupby("transcript_id", sort=False).indices.items():
        key = tid.split(".")[0]
        sequence = reference["sequences"].get(key)
        if sequence is None:
            continue
        length = len(sequence)
        matches = np.array(
            [
                3 <= positions[i] < length - 3
                and sequence[positions[i] - 3 : positions[i] + 4] == kmers[i]
                for i in rows
            ]
        )
        rows = rows[matches]
        if not len(rows):
            continue
        available[rows] = True
        p = positions[rows]
        geometry[rows, 0] = length
        geometry[rows, 1] = p / (length - 1)
        letters = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        index = p[:, None] + FLANK_OFFSETS
        in_bounds = (index >= 0) & (index < length)
        codes = lookup[letters[np.clip(index, 0, length - 1)]]
        codes[~in_bounds] = 4
        flanks[rows] = compact_flanks(onehot[codes].reshape(len(rows), -1))
        centers = np.array(
            [m.start() + 2 for m in re.finditer(r"(?=[AGT][AG]AC[ACT])", sequence)]
        )
        rank = np.searchsorted(centers, p)
        if not np.array_equal(centers[rank], p):
            raise ValueError("A reference-matched input is not a candidate motif.")
        before, after = rank > 0, rank < len(centers) - 1
        distances[rows[before], 0] = p[before] - centers[rank[before] - 1]
        distances[rows[after], 1] = centers[rank[after] + 1] - p[after]
        g = reference["geometry"].get(key)
        if g is None or g["boundaries"][-1] != length:
            continue
        boundaries = g["boundaries"]
        exon = np.searchsorted(boundaries, p, side="right") - 1
        exon_length = np.diff(boundaries)[exon]
        geometry[rows, 2] = exon_length
        geometry[rows, 3] = (exon + 0.5) / (len(boundaries) - 1)
        geometry[rows, 4] = (p - boundaries[exon] + 0.5) / exon_length
        stop = g["stop_codon"]
        if sum(b - a for a, b in stop) == 3:
            geometry[rows, 5] = p - stop[0][0]
    return geometry, flanks, distances, available


SUMMARY_NAMES = [
    f"mean_{window}_current_mean" for window in ("left", "center", "right")
]


def summarize_reads(reads):
    """Nine means for normalization/background; three current SDs for the final trees."""
    return np.concatenate((reads.mean(0), reads.std(0)[[2, 5, 8]])).astype(np.float32)


CURRENT_SPREAD_NAMES = [
    f"across_read_sd_{window}_current_mean" for window in ("left", "center", "right")
]
