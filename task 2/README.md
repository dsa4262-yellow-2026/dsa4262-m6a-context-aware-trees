# SG-NEx analysis

This package analyses Context Aware Trees predictions for **23 SG-NEx sequencing runs from 17 biological replicates across seven human cell lines**. See the [model guide](<../task 1 context aware trees/README.md>) for prediction methods and training data.

The analysis describes predicted m6A sites, their signal features and reproducibility. SG-NEx has no matched site labels for calculating accuracy here. A site score represents predicted modification status, not the fraction of modified RNA molecules.

A **site** is a transcript position. A **biological replicate** is a separate biological sample; **technical runs** sequence the same sample more than once. **Read coverage** counts the RNA reads observed at a site. Unless stated otherwise, the analyses require at least 20 reads per run and classify scores ≥0.5 as predicted modified.

## Results

- [Results by question (CSV)](analysis_context_trees/tables/question_evidence.csv)
- [Computed summary (JSON)](analysis_context_trees/exploration_summary.json)

## Install

Run these commands from this folder with Python 3.11:

```text
python -m pip install -r requirements.txt
```

New predictions and the prediction-alignment checks also require the sibling model package:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "../task 1 context aware trees/requirements.txt"
python -m pip check
```

The transcript reference is bundled in `../task 1 context aware trees/model/`. Figures use Arial when available, with DejaVu Sans as the portable fallback.

## Reproduce the analysis

The saved predictions are in `results_context_trees/`, with one folder per run. Each `sites.parquet` retains every original identifier, position string, score and row, together with sequence and coverage metadata; `run.json` records the prediction settings. Build the analysis in a new output folder:

```text
python analyze.py --results results_context_trees --output reproduced_analysis
python explore.py --analysis reproduced_analysis --results results_context_trees --data-root YOUR_DATA_FOLDER
```

`analyze.py` calculates replicate and cell-line aggregates, matched comparisons, candidate rankings, transcript annotations and cutoff sensitivity. `explore.py` calculates the remaining analyses, extracts selected raw signals, and creates the findings page and both figure sets.

Signal analysis requires uncompressed JSON and matching byte indexes at the relative paths in [samples.csv](samples.csv), for example:

```text
YOUR_DATA_FOLDER/
  sgnex/m6anet/SGNex_A549_directRNA_replicate5_run1/data.json
  sgnex/m6anet/SGNex_A549_directRNA_replicate5_run1/data.info
  ...
```

The signal comparison reads **13,482 selected run-sites for 5,316 matched biological-site pairs**, using every read in each contributing run-site. The extractor checks site identity, sequence and read counts against the prediction records.

### Generate figures from supplied tables

All numerical tables are included, so figures can be generated without raw signal files:

```text
python explore.py --analysis analysis_context_trees --plots-only
```

This generates 17 scientific figures, ten compact layouts and a local findings page. For a narrower edit, `python report_panels.py` redraws only the compact layouts; `python questions.py` refreshes the findings and download links. Changes to calculations require rerunning both full analysis stages.

### Export the prediction data

Create a portable bundle with three-column CSV exports, the Parquet tables and reference annotations:

```text
python package_predictions.py --output exported_predictions
```

CSV exports preserve the identifiers, position strings, row order and scores in the supplied Parquet tables.

## Reproduce predictions from the original inputs

Processed SG-NEx inputs occupy about 31.1 GiB. Allow 32 GB RAM and at least 80 GB working storage for full prediction. A GPU is optional. The downloader requires `curl` and resumes partial downloads:

```text
python download.py --data-root YOUR_DATA_FOLDER
python run.py --data-root YOUR_DATA_FOLDER --output reproduced_results
python analyze.py --results reproduced_results --output reproduced_analysis
python explore.py --analysis reproduced_analysis --results reproduced_results --data-root YOUR_DATA_FOLDER
```

For one sample, add `--sample SGNex_A549_directRNA_replicate5_run1` to the download and prediction commands. Each prediction input must contain the complete sample because transcript context uses other reported sites. Reads are batched internally.

`run.py --resume` checks completed runs before skipping them. `--work-dir` selects scratch storage, `--threads` controls CPU parallelism and `--device cuda` selects a GPU. Complete analysis requires all declared runs; `analyze.py --allow-partial` supports inspection of an incomplete collection.

## Analysis definitions

- **Eligibility and weighting:** human ENST sites with at least 20 reads per contributing run, verified transcript aliases and matching seven-mers. Average technical runs equally within each biological replicate, then biological replicates equally within each cell line. Missing scores remain missing.
- **Shared and restricted predictions:** a cell line is consistently predicted modified when every declared replicate scores at least the cutoff, consistently unmodified when every replicate scores below it, and discordant otherwise. Restricted sites are consistently modified in one cell line and unmodified in the other six. Counts use transcript coordinates.
- **Between- versus within-cell-line variation:** require at least two replicates per cell line and a maximum/minimum mean read-coverage ratio of at most two. At each site, compare the mean absolute score difference across cell lines with the equal-cell-line mean of within-cell-line differences.
- **Signal features:** match sites within each replicate and the GGACT, GAACT or AGACT motif, using identical seven-mers (seven consecutive bases) and read-coverage ratios at most two. Use seed 4262 and at most 200 pairs per replicate/motif. Transform dwell time per read as `log(1 + 1000 × dwell time)` with time in seconds, then average transformed dwell time, signal standard deviation and mean current at the three local positions. Report median paired differences for each replicate. These features are model inputs, so their associations explain the predictions.
- **Gene-level fractions:** select the transcript with the most common observed sites for each gene, independently of scores. Require at least ten sites; average modified-site fractions equally within cell lines, then across cell lines. Repeat with minima of five and twenty sites.
- **Initial and additional replicates:** rank transcript contrasts using the lowest-numbered replicate per cell line and at least three supported sites. Assess the same sites using the remaining replicates. Coverage eligibility uses both subsets; only initial-replicate scores determine ranking.
- **Training overlap:** Hct116 replicate 3 run 1 overlaps Dataset 0. The sensitivity analysis removes that run, retains run 4 and compares the six affected cell-line contrasts on identical retained sites.

## Code and outputs

| Files | Purpose |
|---|---|
| `download.py`, `run.py`, `package_predictions.py` | Download inputs, predict complete samples and export prediction bundles. |
| `analyze.py`, `comparisons.py`, `evidence.py` | Aggregation, matched comparisons, selection, annotations and cutoff sensitivity. |
| `explore.py` | Signal matching, replicate EDA, gene-level fractions, transcript location and overlap sensitivity. |
| `scientific_figures.py`, `report_panels.py` | Scientific figures and compact layouts. |
| `questions.py` | Findings and links to numerical evidence. |
| `REFERENCES.md` | Sources for data, analysis methods and figure design. |
| `analysis_context_trees/` | Aggregates, comparisons and metadata. |
| `analysis_context_trees/tables/` | Detailed results, selected signal pairs and input provenance. |

## Verify

```text
python checks.py
python exploration_checks.py
python audit_analysis.py --analysis analysis_context_trees
```

The checks cover prediction identity, sample/model compatibility, averaging, matched comparisons, ranking, missing-data states, transcript boundaries and signal matching. The audit verifies the supplied results against one another and checks the findings page and its links when generated. It writes `verification.json` with counts and source hashes. Input hashes, dependency versions and extraction settings are recorded in `exploration_provenance.json` and `tables/signal_input_provenance.csv`.

Scientific methods and figure-design references are in [REFERENCES.md](REFERENCES.md).
