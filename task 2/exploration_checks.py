"""Focused checks for the inference risks introduced by the extended EDA."""
import numpy as np
import pandas as pd
from explore import (
    states,
    summarize_signal,
    metagene_position,
    choose_representatives,
    match_features,
    KEY3,
)


def main():
    # A missing sample, an absent cell and a discordant cell must not become low.
    m = pd.DataFrame({"a1": [0.8, 0.8, np.nan, 0.2], "a2": [0.9, 0.2, 0.8, 0.3]})
    info = pd.DataFrame({"replicate_id": ["a1", "a2"], "cell_line": ["A549", "A549"]})
    high, low, mixed, missing = states(m, info)
    assert high.A549.tolist() == [True, False, False, False]
    assert low.A549.tolist() == [False, False, False, True]
    assert mixed.A549.tolist() == [False, True, False, False]
    assert missing.A549.tolist() == [False, False, True, False]
    assert missing.H9.all() and not high.H9.any() and not low.H9.any()
    # Apply nonlinear dwell transform before the per-site mean, not afterwards.
    raw = [
        [0.001, 2, 100, 0.001, 3, 120, 0.001, 4, 130],
        [0.003, 4, 110, 0.003, 5, 140, 0.003, 6, 150],
    ]
    result = summarize_signal(raw)
    assert np.allclose(result[[0, 3, 6]], (np.log(2) + np.log(4)) / 2)
    assert np.allclose(result[[1, 2, 4, 5, 7, 8]], [3, 105, 4, 130, 5, 140])
    assert not np.isclose(result[0], np.log1p(1000 * 0.002))
    for bad in [[], [[1, 2]], [[float("nan")] * 9], [[-0.1] * 9]]:
        try:
            summarize_signal(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid read matrix accepted")
    # Half-open boundaries: CDS start and stop-end enter their next region.
    g = {"CDS": [(100, 397)], "stop_codon": [(397, 400)], "boundaries": [0, 250, 600]}
    assert metagene_position(100, g) == 1 and metagene_position(400, g) == 2
    assert metagene_position(250, g) == 1.5 and np.isnan(metagene_position(600, g))
    # Gene representatives depend only on support; score changes do not select them.
    p = pd.DataFrame(
        {
            "gene_id": ["g"] * 5,
            "transcript_id": ["b", "b", "a", "a", "c"],
            "score": [0, 0, 1, 1, 0.9],
        }
    )
    assert choose_representatives(p).transcript_id.iloc[0] == "a"
    p.score = 1 - p.score
    assert choose_representatives(p).transcript_id.iloc[0] == "a"
    # Matching cannot silently pair different 7-mers or depth ratios >2, or reuse lows.
    p = pd.DataFrame(
        {
            "transcript_id": ["t"] * 5,
            "transcript_position": ["1", "2", "3", "4", "5"],
            "kmer": ["AGGACTA", "AGGACTA", "TGGACTT", "AGGACTA", "AGGACTA"],
            "score": [0.8, 0.1, 0.2, 0.9, 0.05],
            "n_reads": [30, 32, 30, 100, 300],
            "n_runs": [1] * 5,
            "replicate_id": ["r"] * 5,
            "cell_line": ["A549"] * 5,
            "motif": ["GGACT"] * 5,
        }
    )
    matches = match_features(p)
    assert len(matches) == 2 and set(matches.transcript_position) == {"1", "2"}
    assert not matches.duplicated(KEY3 + ["replicate_id"]).any()
    print(
        "Passed: missing versus low states; per-read transform; annotation boundaries; score-independent transcript selection; exact-sequence/depth matching without reuse."
    )


if __name__ == "__main__":
    main()
