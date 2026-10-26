# Task 1: Context-aware trees

Context-aware trees v1 combines **45% full CatBoost** and **55% local XGBoost**.
One shared neural encoder produces eight mean-pooled read summaries. The local
predictor uses 20 features; the full predictor adds 21 transcript-context
features. There is no origin classifier or origin-score input.

## Install and predict

Use Python 3.11 on Windows or Ubuntu. CPU prediction needs no GPU.

```sh
conda create -n context-aware-trees-v1 -c conda-forge python=3.11 pip -y
conda activate context-aware-trees-v1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python predict.py --data example/input.json --output outputs/example_predictions.csv
python -m unittest checks.ScientificChecks.test_prediction_identity_and_cpu_batching
```

The example contains 60 sites. Expected scores are in
`example/expected_predictions.csv`. Floating-point tolerance allows ordinary
platform differences. Run `python checks.py` for the scientific checks.

```sh
python predict.py --data inputs/new_sample.json.gz --output outputs/predictions.csv
```

The input is the supplied JSON-lines format, plain or gzip-compressed. Each
site has a transcript identifier, zero-based transcript position, seven-mer
and all contributing reads. Each read contains nine measurements: dwell time,
current standard deviation and current mean in three successive windows.
Prediction does not require labels. Supply one complete reported-site sample
so that transcript background uses the other sites actually observed in it.

Output is exactly `transcript_id,transcript_position,score`. Original input
identifiers, coordinate spelling and order are retained. Scores rank m6A
evidence in [0,1]; they are not measured modification fractions.

`--model` selects another saved fit. `--reference` supplies a compatible
annotation for another reference. Unknown or mismatching transcripts receive
missing context descriptors. `--threads`, `--batch-size` and `--work-dir`
control resources. Use `--device cuda` with a compatible PyTorch installation
for optional GPU inference; CatBoost scoring remains on CPU.

## Explain an individual prediction

From this folder, generate a self-contained HTML explanation for a site in the
included example:

```sh
python explain.py --data example/input.json --transcript ENST00000000412 --position 496 --output outputs/site_explanation.html
```

Open the output in a browser. For another sample, supply the same complete
JSON or JSON.gz file used by `predict.py`, and copy the transcript identifier
and position exactly from that input. Preparing the complete sample preserves
the background from other reported sites; only the selected site needs neural
encoding. The command uses the saved fit on CPU and does not train a model.
`--model` and `--reference` must match the prediction being explained.
`--threads` controls CPU use; `--work-dir` selects temporary storage.

The explanation includes:

- **The final score:** 55% local XGBoost plus 45% full CatBoost, matching
  `predict.py` for the same input and fitted model.
- **Feature contributions:** native TreeSHAP values for the local model's
  20 inputs and the full model's 41 inputs, grouped by feature family and
  available individually. Context inputs are marked **Not used** for XGBoost.
- **Score reconstruction:** add each tree's baseline and raw-margin
  contributions, add CatBoost's separate prevalence intercept once, apply
  sigmoid to each component, then combine the probabilities. Margin
  contributions cannot be averaged into probability contributions.
- **Read-removal comparisons:** remove each read independently and recompute
  the eight learned means, log read count, three current means and three
  across-read current SDs. Sequence and reference features stay fixed; the
  focal site's other-site background already excludes that site. The table
  ranks the 12 largest absolute changes and includes all reads in an expandable
  table, with their normalized measurements and eight encoder responses.

A positive read-removal change means including that read raises the final
score. Each removal starts from the complete original site. These changes
are not additive or labels for individual molecules. For a one-read site,
feature contributions remain available but removal would leave an empty site,
so its removal score is undefined. Feature contributions explain this fitted
predictor rather than causal modification mechanisms.

Run `python -m unittest checks.ExplanationChecks` for score reconstruction,
read removal checked against rebuilt input files, and gzip/HTML checks.

## Train from the course data

Download the original six course files with the standard-library downloader:

```sh
python download_data.py --output inputs/training
```

The [original training-data release](https://github.com/dsa4262-yellow-2026/dsa4262-m6a-context-aware-trees/releases/tag/training-data-v1) contains 328 MB of compressed inputs. Allow about 700 MB for download and extraction, plus training scratch space. This source archive is shared with the earlier model; its measurements and labels are unchanged.

For an offline archive, use `python download_data.py --archive /path/to/training-data-v1.zip --output inputs/training`. Existing files are retained. The resulting layout is:

```text
inputs/training/data0/dataset0.json.gz
inputs/training/data0/data.info.labelled
inputs/training/data1/dataset1.json.gz
inputs/training/data1/data.info
inputs/training/data2/dataset2.json.gz
inputs/training/data2/data.info
```

```sh
python train.py --data-dir inputs/training --output outputs/retrained_model
python predict.py --data example/input.json --model outputs/retrained_model --output outputs/retrained_example.csv
```

Use a new output directory for each fit. Default seed is 4262; the encoder
trains for six epochs. The saved encoder excludes the temporary training head.
The transcript reference and normalization parameters accompany the two trees.

Datasets 0 and 1 provide sample-specific binary site labels. Dataset 2 supplies
seven ordered observations at each of 189 positions. Binary classification and
within-group ordering train the shared encoder and both final predictors.
Order values convey order only: their numerical distances are not fitted as
methylation percentages. Dataset 2 endpoint chemistry is unconfirmed.

The encoder receives nine normalized measurements and 28 one-hot sequence
entries for each read. Its 37-128-128-8 network uses SiLU between hidden layers.
Mean pooling retains every observed read. The temporary site head learns
class-balanced binary log loss and logistic ordering loss. XGBoost learns
binary log loss plus ordering loss; CatBoost uses PairLogit comparisons and
a training-fitted intercept that matches binary-label prevalence.

Both trees use depth 7, learning rate 0.03 and L2 regularization 25. XGBoost has
800 rounds; CatBoost has 960. The final combination is fixed at 45/55. See
`model/manifest.json` for exact feature names and settings.

## Retrain on other datasets

Pass `--config training.json` instead of `--data-dir`. Paths in the manifest
are relative to that manifest. Any number of binary datasets may be listed;
ordered datasets are optional. Dataset names are bookkeeping, not features.

```json
{
  "datasets": [
    {"name": "sample_a", "kind": "binary", "signals": "sample_a.json.gz", "labels": "sample_a_labels.csv"},
    {"name": "series", "kind": "ordered", "signals": "series.json.gz", "labels": "series_labels.csv", "order_column": "level", "group_columns": ["site_group"]}
  ]
}
```

Every label table joins by `transcript_id,transcript_position`. Binary labels
must be 0 or 1. Ordered rows additionally need a finite level and a group that
identifies comparable observations at the same site/context. Higher levels
must have a justified direction of stronger positive-class evidence. Keep
complete genes and ordered groups together when evaluating unseen groups.

## Evaluate

Recalculate the supplied grouped results:

```text
python evaluate.py
```

To refit the entire pipeline within each of the five folds:

```text
python reproduce_evaluation.py --data-dir /path/to/data --output /path/to/new_folds --mode genes
python evaluate.py --results /path/to/new_folds --baselines results/baselines.npz
```

Genes remain together across Datasets 0/1; all settings of a Dataset 2 position remain together. Each fold fits its own normalizers, encoder and trees. The folds measure held-out genes and positions within the supplied releases.

Cell-line transfer is evaluated separately. Each run holds out one entire binary sample and fits every stage on the other sample plus Dataset 2. Shared human genes and coordinates are allowed; labels stay with their sample. To reproduce both transfer runs:

```text
python reproduce_evaluation.py --data-dir /path/to/data --output /path/to/transfer_results --mode samples
python evaluate.py --results /path/to/transfer_results --transfer
```

The supplied results are in `results/transfer/`. To check coverage-dependent reporting, each run retains sites with read count at least that sample's median and recomputes their observed transcript context. A second check retains a uniformly selected half of the sites. Both compare the same retained sites before and after context recomputation, with their reads and labels fixed. These evaluation fits are separate from the final model trained on all three datasets.

| Held-out quantity | Final 45/55 | Local XGBoost | Full CatBoost |
|---|---:|---:|---:|
| Binary ROC AUC | 0.908609 | 0.882255 | 0.919256 |
| Binary PR AUC | 0.526907 | 0.453146 | 0.549385 |
| Within-position pair agreement | 94.36% | 91.26% | 97.23% |

PR AUC uses trapezoidal area; average precision is reported separately. `results/metrics.csv` includes dataset-specific results and aligned logistic/m6Anet baselines. `results/baseline_protocol.json` records their settings. `results/reliability.csv` and Brier scores describe probability agreement. The retrained m6Anet baseline uses the binary-data folds; the final model additionally uses Dataset 2 and reference context. External development comparisons motivated the selected trade-off; the full tree is stronger on these internal folds.

### Interpret a score cutoff

The same evaluation command writes [operating_points.csv](results/operating_points.csv), using each sample's held-out predictions and its own binary labels. A site is selected when `score >= cutoff`. Precision is true positives divided by selected sites; recall is true positives divided by all positive labels. The table includes the counts needed to recompute both, and the mean selected score for comparison with observed precision. An empty selection has undefined precision.

At cutoff 0.5, Dataset 0 precision/recall is 73.0%/38.7% and Dataset 1 is 64.0%/25.7%. At 0.7 these become 90.0%/10.5% and 76.2%/6.2%. Only five and six sites respectively reach 0.9, so the precision estimates there are unstable. These fixed thresholds describe a selectivity/recall trade-off. Task 2 adds an all-biological-sample support rule; its actual accuracy cannot be inferred without SG-NEx labels.

### Consolidated comparison

In the complete repository, export all three evaluation sections together:

```text
python evaluate.py --comparison "../model evaluation/comparison.csv"
```

The sections are binary site classification on held-out genes (Datasets 0/1), source separation and mixture ordering on held-out positions (Dataset 2), and the final release. Dataset 2 retains its own target and metrics. The final-release row is pending until evaluation results are available. For this model folder on its own, use `--comparison outputs/comparison.csv` instead.

### Evaluate a future labelled release

Keep the final model fixed and predict each complete new sample using the normal prediction command. Once its own binary labels are available:

```text
python predict.py --data inputs/new_sample.json.gz --output outputs/new_sample.csv
python evaluate.py --predictions outputs/new_sample.csv --labels inputs/new_sample_labels.csv --dataset final_release --output results/final_release
python evaluate.py --comparison "../model evaluation/comparison.csv" --external results/final_release/metrics.csv
```

Labels require `transcript_id,transcript_position,label`; predictions require `transcript_id,transcript_position,score`. Additional columns are allowed. The evaluator matches integer positions and exact transcript IDs, checks unique keys and identical site coverage, and requires both binary classes. One invocation represents one sample. To evaluate a table containing multiple samples, include `sample_id` in both files so the same coordinate in different samples stays distinct. Labels must belong to the predicted sample.

`metrics.csv` contains ROC AUC, trapezoidal PR AUC and AP. `protocol.json` records the sample name, site keys, counts and metric definitions. `evaluated_sites.csv.gz` retains the labelled coordinates used to compare models on exactly the same sites. For another fitted model, supply its `--model-manifest`; `--model-name` changes the comparison label. Use a distinct dataset name/output folder for each further release, and pass all desired metric CSVs after `--external` whenever rebuilding the consolidated comparison. Local comparisons on the same dataset require identical labelled-site identities.

If labels are withheld, submit the predictions and use the organiser's returned results. The repository's [evaluation instructions](<../model evaluation/README.md#when-labels-remain-private>) explain how to import those results while retaining their metric definition and source. Existing Dataset 0/1/2 results continue to use the held-out fold predictions, separately from the final all-data fit.

## Run the official m6Anet benchmark

The saved results include released and retrained m6Anet comparisons on all binary sites. `results/native_m6anet/` additionally contains native m6Anet 2.1.0 predictions from Python 3.8/Conda on all 44,144 sites of held-out fold 0. On these same sites, ROC/PR AUC is 0.9096/0.5366 for context-aware trees and 0.8752/0.4057 for native m6Anet. `python evaluate.py` verifies the matched site keys and recalculates this separate comparison.

To rerun the native released model, use the separate Python 3.8 Conda environment required by the course:

```text
conda env create -f benchmark/environment.yml
conda activate task1-m6anet
python benchmark/run_m6anet.py --dataset data0 /path/to/data0/dataset0.json.gz --dataset data1 /path/to/data1/dataset1.json.gz --fold 0 --output /path/to/benchmark_results --work-dir /path/to/benchmark_scratch
```

Replace both input paths and choose new, empty output and scratch folders. The runner reads the bundled held-out fold, creates native indexed inputs with the original nine measurements, and runs m6Anet 2.1.0 with its `HCT116_RNA002` checkpoint and official normalization. It uses 1,000 iterations of 20-read sampling with replacement. It writes native predictions, the run settings and a matched comparison with context-aware trees. Fold 0 contains 44,144 sites; `--fold 1` through `--fold 4` select the other folds. The released model's original training overlap is unknown. These native sampled scores differ from the all-site expected-inference scores in `results/baselines.npz`.

Return to the Python 3.11 environment for the final model. The m6Anet environment is only for this benchmark.

## Reproduce the all-site baselines

The three columns in `results/baselines.npz` are the logistic baseline, released m6Anet with expected inference, and the m6Anet architecture retrained on the same five gene folds. Their complete numerical implementations are in `benchmark/`. Use the main Python 3.11 environment and choose new output and scratch directories:

```text
python benchmark/reproduce_baselines.py --data-dir /path/to/data --output outputs/baselines --work-dir /path/to/baseline_scratch
python evaluate.py --baselines outputs/baselines/baselines.npz
```

The command verifies the original sample-specific labels and raw site identities against the held-out records. It fits logistic regression on the nine raw channel means, using training-fold constant removal and standardization. The released m6Anet weights and normalization are bundled under `benchmark/m6anet_reference/`, with the upstream MIT license.

Both m6Anet comparisons calculate `1 - (1 - mean_read_probability)**20`, the expected noisy-OR score for 20 reads sampled with replacement. The retrained comparison fits raw five-mer normalization on training genes, selects the epoch count using an inner 25% gene holdout, then refits on all outer training genes. It uses at most 60 epochs, balanced site sampling, 20 reads per training bag, batch size 512, and Adam AMSGrad with learning rate 0.0004. This differs from the native sampled-inference benchmark above, which runs the official installed package under Python 3.8.

The full command retrains five baselines. To run it in stages, add `--stage prepare`, then `--stage released`, `--stage retrain`, and finally `--stage collect`, retaining the same paths. The retraining stage accepts `--fold 0` through `--fold 4`; collection requires all five. `--device cuda` is optional. Allow 10 GB of scratch space for the original read arrays. The output includes per-fold scores, fitted checkpoints, the combined `baselines.npz` and its protocol.

## External development panels

```sh
python evaluate_external_panels.py
```

This recalculates the common-site comparisons from the included per-site scores. [The panel protocol](results/external/README.md) distinguishes three-seed means from the released seed-4262 fit, mouse assays from human native/IVT comparisons, and development selection from an untouched test. Six panel results are not six independent biological datasets.

## Files and portability

| File or folder | Purpose |
|---|---|
| `train.py`, `predict.py` | Train a model and predict a complete new sample |
| `explain.py` | Explain one prediction with per-view TreeSHAP and independent read removals |
| `download_data.py` | Obtain the original signal and label files |
| `context_trees/` | Data handling, features, encoder and boosting objectives |
| `model/` | Saved encoder, normalizers, trees, reference and manifest |
| `example/` | 60 original-format sites and expected predictions |
| `reproduce_evaluation.py`, `evaluate.py`, `results/` | Refit grouped or sample-held-out experiments; recalculate metrics |
| `benchmark/` | Logistic and m6Anet comparisons, including a separate Python 3.8 Conda environment |
| `checks.py` | Label identity, training-only normalization, group integrity, feature views, prediction alignment and explanation reconstruction |

The model folder is self-contained. Input and output paths are arguments; bundled assets resolve relative to the scripts. CPU is the default. For CUDA, install the appropriate PyTorch build and pass `--device cuda`; CatBoost PairLogit remains on CPU. Windows CPU checks are recorded for this release; Ubuntu/Colab execution has not been verified. Teaching notebooks and report authoring sources are separate deliverables.
