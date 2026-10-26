# Context-aware trees: SG-NEx application

This folder applies the saved final model to complete SG-NEx samples, compares the observed site scores across biological replicates and cell lines, and produces a readable findings page.

The model uses a neural read encoder, XGBoost and CatBoost. The fixed score is 45% full CatBoost (41 inputs) plus 55% local XGBoost (20 inputs); both share eight learned read summaries. It was trained with datasets 0, 1 and 2 and is kept fixed during this analysis. A score between 0 and 1 estimates the binary m6A site-label target; it is not a measured fraction of modified molecules.

## Read the completed results

The saved predictions are in `results_context_trees/`; analysis tables are in `analysis_context_trees/`.

## Four questions connect the results

| Question | What the analysis answers | Where to start |
|---|---|---|
| 1. How different are the cells overall? | Compare cell differences with repeat variation at the same supported positions. | [Cell comparisons](analysis_context_trees/cell_pairs.csv) |
| 2. Which candidates are shared, and where? | Identify sites scoring highly in every biological replicate and locate them within transcripts. | [Shared candidates](analysis_context_trees/shared_candidates.csv) |
| 3. Which local differences recur? | Select using discovery samples, then show direction, magnitude and site support in confirmation samples. | [Local contrasts](analysis_context_trees/examples.csv) |
| 4. How does read support change the conclusions? | Distinguish lost observations from lower scores and follow the original positions at higher coverage. | [Observation support](analysis_context_trees/threshold_stability.csv) |

The linked tables provide the numerical results for each question. Sample preparation and context availability support the interpretation; the full prediction tables retain every reported input position.

## Reproduce in three steps

Open a terminal in this folder. Use Python 3.11 and retain the sibling `../task 1 context aware trees/` folder, including its saved `model/`.

### 1. Install and choose the data folder

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "../task 1 context aware trees/requirements.txt" -r requirements.txt
python -m pip check
```

The chosen data folder must contain the relative `signals` and `info` paths listed in `samples.csv`, for example:

```text
your_data_folder/
  sgnex/m6anet/
    SGNex_A549_directRNA_replicate5_run1/
      data.json
      data.info
    ...
```

`data.json` contains the read measurements. `data.info` indexes the same sites, with byte offsets and read counts; it contains no methylation labels. Each inventory row supplies the original JSON URL. The matching index is `data.info` in that URL's directory. Prediction checks that the site identifiers, read counts and byte index match the signal file.

The declared collection contains 23 processed direct-RNA runs, seven cell lines and 17 biological replicates. Input JSON files occupy about 31.1 GiB; prediction also needs space for scratch and outputs. Inputs may be downloaded and processed one run at a time. Keep the original uncompressed JSON with its matching byte index.

With `curl` installed, download the complete declared collection directly from this folder:

```text
python download.py --data-root "YOUR_DATA_FOLDER"
```

The downloader resumes partial transfers and checks file lengths against the source server. Existing complete files are kept. Add `--sample SGNex_A549_directRNA_replicate5_run1` to download only the initial pilot. On Ubuntu, `sudo apt install curl` installs the download tool if needed. For the complete prediction and analysis run, use a CPU machine with 32 GB RAM and at least 80 GB of free working storage; a GPU is optional.

### 2. Predict complete samples

Replace `YOUR_DATA_FOLDER` with the folder chosen above:

```text
python run.py --data-root "YOUR_DATA_FOLDER" --output reproduced_results
```

This command selects the bundled context-aware trees fit automatically. It retains every reported site and writes `reproduced_results/`, keeping this run separate from the supplied results. CPU execution with three threads is the default. Labels and retraining are unnecessary.

The model uses other reported sites on the same transcript as sample context. Supply each complete sample; the predictor batches reads internally. To continue a stopped run with the same inputs and fitted model, add `--resume`; completed samples are checked against their site index and skipped. Use a new output directory after changing inputs, the fitted model or execution settings.

For the initial complete-sample pilot:

```text
python run.py --data-root "YOUR_DATA_FOLDER" --output reproduced_results --sample SGNex_A549_directRNA_replicate5_run1
```

`--work-dir` chooses scratch storage. `--threads` changes CPU parallelism; `--device cuda` selects an available CUDA environment. `--output` chooses another result folder. `--samples` selects a different inventory with the same documented columns. `--model` selects another saved fit of the same context-aware trees implementation.

### 3. Generate comparisons and read the findings

To analyze the **supplied predictions** immediately, no raw-data download is needed:

```text
python analyze.py --results results_context_trees --output reproduced_analysis
```

To analyze predictions you generated in Step 2:

```text
python analyze.py --results reproduced_results --output reproduced_analysis
```

Open **`reproduced_analysis/findings.html`** in a browser. Follow the four questions above: overall cell differences, shared candidates and their locations, recurring local contrasts, and the effect of read support. It also shows coverage/score-threshold stability, transcript-region annotations and confirmation across the candidate ranking. Every numerical finding links to its source table. Figures are available as PNG and PDF.

The command requires every run in the inventory. `--allow-partial` creates an explicitly incomplete development preview. For different folders:

```text
python analyze.py --results YOUR_RESULTS --samples YOUR_INVENTORY.csv --output YOUR_ANALYSIS
```

Figures use Arial when installed; `--font` accepts an Arial TTF file. If unavailable, the actual substitute font is recorded in `metadata.json`. Supply Arial when preparing course-report figures.

Annotations use the reference bundled with the final model. If analyzing another saved fit with a different reference, pass that same file using `--reference YOUR_REFERENCE.joblib`.

## How a comparison is computed

1. Score all observed sites in each complete sample.
2. For the primary biological analysis, retain human transcript sites with at least 20 reads in that run. Synthetic/control counts remain available separately.
3. Match only compatible transcript coordinates and seven-base sequences. Positions use the input's zero-based convention: 0 is the first base. Verified transcript aliases are applied during analysis; original IDs and position spellings remain in the prediction files and the `input_transcript_id` / `input_transcript_position` fields. Analysis compares positions by their integer value.
4. Average available technical-run scores within a biological replicate. Then average biological-replicate means equally within each cell line. Missing observations remain missing.
5. For primary cell comparisons, require at least two biological replicates in each cell and a maximum/minimum replicate mean read-depth ratio no greater than 2 across both cells. Compare all cross-cell replicate-pair score gaps with the average within-cell gaps on exactly the same positions. Give each cell equal weight for the within-cell reference. Also report the median, 90th and 99th percentiles of the absolute difference between cell means at those same positions. These distinguish typical differences from the upper tail.

For example, technical scores 0.2 and 0.4 give one replicate mean of 0.3. A second replicate with score 0.9 gives a cell mean of **0.6**. If the second replicate is absent, report 0.3 with one contributing replicate.

Transcript examples use the lowest-numbered biological replicate of each cell for discovery and the remaining replicates for confirmation. At least three positions must meet the joint support/depth rule and have both discovery and confirmation measurements. Rank transcripts by the absolute signed mean discovery contrast, then site count; select three distinct genes without using confirmation scores in the ranking. The same positions are used for confirmation, and individual-position direction agreement is reported. All candidate results, including reversals, remain in `discovery_candidates.csv`. For the three selected examples, `example_confirmation_pairs.csv` shows every remaining biological-sample pair, its original selected coordinates observed in both samples, its mean signed difference and its paired-position count. Missing positions do not supply a direction; pairs sharing a replicate are overlapping comparisons.

The shared-position analysis requires the same coordinate and seven-mer with at least two biological replicates in every cell. `shared_candidates.csv` additionally requires all declared biological replicates to be observed, each with score at least 0.5. Counts refer to transcript coordinates; isoforms can share a genomic base.

At a matched coordinate, the reference sequence and annotated transcript location are fixed across cells. Read signals, coverage and the other observed sites can still differ; the replicate and depth comparisons describe these sample-dependent predictions. The all-replicate rule prioritizes recurring support across this collection. It selects well-observed transcripts and does not estimate transcriptome-wide modification prevalence.

The [held-out score-cutoff evaluation](<../task 1 context aware trees/README.md#interpret-a-score-cutoff>) shows how 0.5, 0.7 and 0.9 trade precision against recall on the two labelled samples. SG-NEx catalogues apply these cutoffs and record repeated model support across the observed biological samples.

The primary 20-read setting provides repeated read measurements at each contributing run-site. Stability analysis repeats the aggregation at 30 and 50 reads and compares shared-candidate score cutoffs of 0.5, 0.7 and 0.9. The read cutoff governs observation support; score cutoffs govern how much model support a candidate requires. For a different `--min-reads`, the coverage settings are the supplied value and its rounded-up 1.5-fold and 2.5-fold multiples. Every setting filters the original run observations before averaging; filtering an already averaged cell depth would answer a different question. Model scores remain fixed. At each score cutoff, compare identities with the primary-depth catalogue at that same cutoff and count retained, lost and new candidates. Split losses into loss of observation in any declared biological replicate (`lost_coverage`) and fully observed positions below the score cutoff (`lost_score`). Removing a technical run can change a replicate mean, so a stricter depth can introduce a candidate. The original local examples and coordinate lists are held fixed, with lost positions counted explicitly.

For each cell pair, `cell_contrast_stability.csv` also compares the two coverage settings on their position intersection. `baseline_excess_on_retained` and `current_excess_on_retained` calculate excess with primary and higher-depth means on the same positions; their difference is `excess_change_on_retained`. This separates changes in the contributing runs/replicates from changes in position membership. Retained, lost and newly qualifying counts accompany each row. Removing a run can improve depth balance and let a previously excluded position qualify. An empty comparison has zero positions and unavailable contrasts, including when an entire cell loses observations.

For the wider discovery/confirmation summary, keep the first occurrence of each gene in the discovery-only ranking. Report direction agreement and signed confirmation for the leading 3, 10, 50 and 100 genes and the complete list. A zero or reversed confirmation difference is not counted as direction agreement. Aligned confirmation multiplies the signed confirmation contrast by the discovery sign: discovery -0.20 and confirmation -0.08 give aligned confirmation +0.08. Average remaining replicates at each position first, then average positions equally. Pair-specific summaries can differ from this mean when pairs cover different subsets of sites.

Transcript annotations use the model's Ensembl 91 sequence and exon/CDS/UTR intervals. Intervals are zero-based and half-open: `[start, end)` includes `start` and excludes `end`. The stop codon is assigned before the coding sequence; 5′ and 3′ UTRs require an annotated UTR on the appropriate side of the coding sequence. `noncoding_or_unassigned` covers other positions, including transcripts without an annotated CDS. Gene biotype alone is not used to infer a transcript's coding status. Compare shared-candidate region proportions with all positions observed in every declared biological replicate before applying the score cutoff, then with the subset in the same candidate genes. Both backgrounds have the same observation requirement as the catalogue. `region_by_gene.csv` additionally holds gene contributions fixed: for each candidate gene, multiply its candidate count by the 3′ UTR fraction of its eligible positions, then sum. Divide by total candidate count for a gene-matched expected fraction. The observed 142/181 (78.5%) compares with 84.4/181 (46.6%) expected under these fixed gene contributions.

The 16 static context inputs are recomputed using the final model's feature functions. For each original run, background availability counts reference-matched reported positions on the same original transcript before applying the coverage filter. The focal position is excluded. All five background inputs are available when the focal reference matches and at least one other reference-matched position is reported. This measures input availability; it does not measure prediction error caused by missingness.

Cell-pair comparisons can use different shared-site sets. Cell identity and sequencing conditions both vary across the collection. Hct116 replicate 3/run 1 contains measurements used in Dataset 0; its overlap flag is retained in summaries.

## Files you can reuse

Each result folder contains one `SAMPLE_ID/` directory with:

| File | Meaning |
|---|---|
| `predictions.csv.gz` | Original `transcript_id,transcript_position,score`, in input order. |
| `sites.parquet` | Exactly the same keys and scores, with seven-mer, read count, gene ID, reference-match flag and model name. |
| `run.json` | Sample and replicate identifiers, counts and model execution settings. |

`reference_matched` means the input seven-mer exactly matches the bundled transcript sequence at that coordinate. Missing reference context is handled by the final model; every valid site still receives a score.

The analysis folder contains:

| File | Question answered |
|---|---|
| `run_summary.csv`, `cell_summary.csv` | How much data supports each run and cell? |
| `depth_histogram.csv`, `score_histogram.csv` | What are the observed coverage and score distributions? |
| `replicate_pairs.csv` | Do repeated measurements agree at shared positions? |
| `cell_pairs.csv` | Within/between replicate variation and median/90th/99th-percentile absolute cell-mean differences on the same jointly supported positions |
| `biological_sites.parquet`, `cell_sites.parquet` | What score and replicate support belong to each position? |
| `discovery_candidates.csv` | Complete discovery ranking and confirmation results for all eligible transcript/cell-pair candidates |
| `examples.csv`, `example_supported_sites.csv`, `example_sites.parquet` | Selected discovery examples, exact qualifying positions, confirmation and underlying measurements |
| `example_confirmation_pairs.csv` | Individual confirmation-sample contrasts, paired original positions and full sample IDs for each selected example |
| `shared_sites.parquet`, `shared_candidates.csv` | Positions shared across every cell and candidates with scores >=0.5 in every declared biological replicate |
| `threshold_stability.csv` | Shared-candidate counts at each coverage/score setting, including baseline, retained, lost-by-coverage, lost-by-score and new identities at the same score cutoff |
| `shared_candidate_stability.csv` | Candidate-specific minimum replicate scores and support at each coverage setting; a flag identifies scores >=0.9 in every declared replicate at all settings |
| `cell_contrast_stability.csv`, `example_stability.csv` | Broad contrasts at each depth, retained/lost/new positions and excess at both depths on their intersection; original local examples with retained counts and unavailable contrasts when no sites remain |
| `discovery_confirmation.csv` | Confirmation direction and effect size across discovery-ranked lists with one comparison per gene |
| `site_annotations.parquet`, `region_summary.csv` | Transcript regions, exon/stop-codon locations, absent static inputs and region composition with explicit observed backgrounds |
| `region_by_gene.csv` | Candidate and eligible 3′ UTR counts per gene, and the expected candidate count at fixed gene contributions |
| `context_support.csv` | Reference and context availability for common positions, shared candidates and the two cells contributing to each local example |
| `transcript_index.parquet` | Which human transcripts and gene names occur in the eligible analysis? |
| `input_inventory.csv`, `metadata.json` | Which samples, coverage rules and analysis settings produced the outputs? |

`transcript_aliases.csv` contains 46,297 sequence-verified version aliases. `genes.csv` contains gene names and biotypes from the model's Ensembl 91 human annotation. Sample details and analysis settings are recorded in each `run.json` and the analysis `metadata.json`.

## Check the implementation

```text
python checks.py
```

The focused checks compare the wrapper with native model predictions, preserve original keys and scores, check reference flags, check site-index alignment, model components and complete collections, and verify replicate averaging, matched-pair arithmetic, joint coverage/support filtering, and ranking that is independent of confirmation scores. Additional hand-calculated cases check catalogue-eligible annotation backgrounds, score-specific candidate retention, coverage-versus-score losses, gene-matched region expectations, individual confirmation-pair support and missing declared cells. They use temporary data derived from the bundled small prediction example.

The supplied analysis includes 2,973,240 eligible human observations, six technical pairs, 13 biological pairs and 21 cell pairs. Its 181 shared candidates occur in 105 genes. The three leading discovery examples are RNASEK, SAMD4B and C20orf24. All three retain their mean direction, but only two, two and one of their respective three positions retain direction individually. Consult the complete site and sample-pair tables before interpreting recurrence.

## Export the complete pre-analysis collection

```text
python package_predictions.py --results reproduced_results --output outputs/preanalysis
```

Use `--results results_context_trees` to package the supplied predictions. This writes an unfiltered portable collection with all 28,670,080 original run-site observations, the sample inventory and fixed reference. It does not apply the 20-read cutoff or change source scores. The separate three-column course-prediction CSVs are not SG-NEx analysis inputs.

For a future original-format sample without a byte index, use the standalone Task 1 `predict.py` directly. To include it in this multi-sample analysis, supply its matching index and an inventory row with the correct sample and biological-replicate identities.

`comparisons.py` loads compatible coordinates, averages samples and implements the matched comparisons and discovery/confirmation split. `evidence.py` adds coverage stability, transcript annotation and context support. `findings.py` turns the generated tables into a connected explanation with primary biological references.
