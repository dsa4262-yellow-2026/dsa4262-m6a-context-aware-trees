# Model and prediction

Context Aware Trees combines **55% local XGBoost (20 features)** and **45% full CatBoost (41 features)**. Both use the same eight learned signal summaries. The bundled model was fitted on Datasets 0–2, provided through DSA4262: Data Science in Genomics.

## Install and predict

Follow the [installation guide](../docs/installation.md) for RONIN, Ubuntu or Windows setup. The [quick start](../README.md#quick-start) shows the example commands and expected output.

After installation, run commands from `task 1 context aware trees/`. To predict a new sample:

```sh
python predict.py --data inputs/new_sample.json.gz --output outputs/predictions.csv
python validate_predictions.py --data inputs/new_sample.json.gz --predictions outputs/predictions.csv
```

### Input format

Input uses JSON Lines, plain or gzip-compressed: transcript identifier, zero-based transcript position, seven-mer (seven consecutive bases) and all reads at that site. Each read has nine values: dwell time (signal duration), current standard deviation (SD) and mean current in each of three successive windows. Supply **all observed sites from one sample**, because background features use other sites on its transcripts.

### Output

The CSV has three columns: `transcript_id,transcript_position,score`. Original identifiers, position strings and row order are retained. Scores lie in [0,1] and indicate predicted m6A site status; they do not measure modification fractions.

### Prediction options

`--model` selects a trained model; `--reference` supplies a compatible reference; `--threads`, `--batch-size` and `--work-dir` control resources. `--device cuda` enables neural inference with a compatible GPU installation. Unknown or sequence-mismatching transcripts receive missing context features.

## Model and features

Transform dwell time in seconds to `log(1 + 1000 × dwell time)`, then standardize each signal channel using the mean and SD for its five-mer sequence in the binary training reads. These normalization statistics are reused for new samples.

Each read supplies nine normalized signal measurements and 28 one-hot entries for its full seven-mer: four base indicators at each of seven positions. A **37 → 128 → 128 → 8** neural encoder uses the sigmoid linear unit (SiLU), a nonlinear activation, between hidden layers. Averaging its eight outputs over **all reads at the site** produces summaries H₁–H₈.

During training, a temporary site prediction network (the **head**) receives H₁–H₈ and five motif indicators through **13 → 128 → 64 → 1** layers. It outputs one site score used to calculate the binary and ordering losses. Backpropagation sends these errors through the head and averaging step to update the encoder. After training, fix the encoder's weights and discard the head. The two tree ensembles then learn from its eight summaries and the features below.

| Feature positions | Features | Local XGBoost | Full CatBoost |
|---|---|:---:|:---:|
| 1–8 | Eight mean-pooled neural summaries | ✓ | ✓ |
| 9–13 | Five indicators: D=A, D=G, R=A, H=A, H=C in the central DRACH motif | ✓ | ✓ |
| 14 | Standardized log read count | ✓ | ✓ |
| 15–20 | Three mean currents and three across-read current SDs | ✓ | ✓ |
| 21–26 | Transcript length and relative position; exon length, rank and relative position; distance to stop-codon start | | ✓ |
| 27–34 | GC and A fractions in four flanking windows | | ✓ |
| 35–36 | Distance to the previous and next reference DRACH adenine | | ✓ |
| 37–41 | Other-site transcript background: three normalized signal means, mean log read count and site count | | ✓ |

DRACH is a five-base motif: D=A/G/T, R=A/G and H=A/C/T; T denotes RNA U in the supplied sequences. The five indicators encode the variable bases of this central motif; the encoder also receives the two outer bases of the seven-mer.

Flanking windows are −30…−16, −15…−4, +4…+15 and +16…+30 relative to the candidate adenine. Reference features require an exact seven-mer match. Background excludes the candidate site and gives other observed sites in the same sample and transcript equal weight. Missing context remains missing. Exact feature names and settings are in [model/manifest.json](model/manifest.json).

Both boosted-tree ensembles use depth 7, learning rate 0.03 and L2 regularization 25. XGBoost uses 800 rounds and CatBoost 960. The final site score is `0.55 × local + 0.45 × full`.

## Train on Datasets 0–2

Download `training-data-v1.zip` from the [training-data release](https://github.com/dsa4262-yellow-2026/m6a-context-aware-trees/releases/tag/training-data-v1). Replace `/path/to/training-data-v1.zip` below with the downloaded file's path:

```sh
python download_data.py --archive /path/to/training-data-v1.zip --output inputs/training
python train.py --data-dir inputs/training --output outputs/retrained_model
python predict.py --data example/input.json --model outputs/retrained_model --output outputs/retrained_example.csv
```

The 328 MB archive needs about 700 MB for download and extraction, plus training scratch space. Quote paths containing spaces and use a new model output directory for each fit. Alternatively, `python download_data.py --output inputs/training` downloads and extracts the release directly.

```text
inputs/training/data0/dataset0.json.gz
inputs/training/data0/data.info.labelled
inputs/training/data1/dataset1.json.gz
inputs/training/data1/data.info
inputs/training/data2/dataset2.json.gz
inputs/training/data2/data.info
```

- **Datasets 0/1:** sample-specific binary site labels train class-balanced binary cross-entropy in the neural head, giving the two classes equal total weight. XGBoost uses binary cross-entropy with equal weight per labelled site. CatBoost compares positive-labelled sites with negative-labelled sites.
- **Dataset 2:** seven ordered observations at each of 189 positions train within-position comparisons in the encoder and both trees. The supplied levels specify order; their numerical distances are not fitted as modification percentages.
- **Training:** normalizers are fitted on the binary training reads. The encoder uses six AdamW epochs, learning rate 0.001 and weight decay 0.0001. Its ordering loss averages pairs within groups, then groups equally, and is added to binary loss with weight 1. All reads in a sampled site are used. CatBoost's final intercept is fitted to binary training-label prevalence.

Default seed is 4262. Each saved fit includes the encoder, normalizers, two trees, reference and manifest.

### Retrain on other data

Use `--config training.json` instead of `--data-dir`. Paths are relative to the configuration file. Any number of binary datasets can be supplied; ordered datasets are optional.

```json
{
  "datasets": [
    {"name": "sample_a", "kind": "binary", "signals": "sample_a.json.gz", "labels": "sample_a_labels.csv"},
    {"name": "series", "kind": "ordered", "signals": "series.json.gz", "labels": "series_labels.csv", "order_column": "level", "group_columns": ["site_group"]}
  ]
}
```

Labels join by `transcript_id,transcript_position`. Binary labels must be 0 or 1. Ordered data additionally require a finite level and a group identifying comparable observations at the same site/context. Higher levels must indicate stronger positive-class evidence. Keep complete genes and ordered groups together during validation.

## Reproduce the evaluation

Recalculate the supplied held-out metrics:

```sh
python evaluate.py
python evaluate.py --results results/transfer --transfer
```

Five folds keep each gene together across Datasets 0/1 and all seven settings of each Dataset 2 position together. Every fold refits normalization, encoder and trees. These are held-out predictions, separate from the bundled all-data model.

| Method | Dataset 0 ROC / PR AUC | Dataset 1 ROC / PR AUC |
|---|---:|---:|
| Context Aware Trees | 0.9489 / 0.6131 | 0.8614 / 0.4582 |
| Logistic regression | 0.7150 / 0.0992 | 0.6578 / 0.1203 |
| Released m6Anet, expected inference | 0.9322 / 0.5032 | 0.8194 / 0.3556 |
| Retrained m6Anet, expected inference | 0.9237 / 0.4884 | 0.8277 / 0.3702 |

ROC AUC is the area under the receiver operating characteristic curve; PR AUC is the trapezoidal area under the precision–recall curve. Average precision is reported separately. Dataset 2 pair agreement is 94.36%, with ROC/PR AUC 0.9614/0.9636 for its endpoints (mixture levels 0 and 1). These endpoints distinguish the two read sources.

To refit all folds or the two held-out-sample experiments:

```sh
python reproduce_evaluation.py --data-dir inputs/training --output outputs/folds --mode genes
python evaluate.py --results outputs/folds --baselines results/baselines.npz
python reproduce_evaluation.py --data-dir inputs/training --output outputs/transfer --mode samples
python evaluate.py --results outputs/transfer --transfer
```

Sample transfer fits on one binary sample plus Dataset 2 and evaluates the other sample. It also tests transcript-background sensitivity by retaining a random half of sites or sites above the sample's median read count, then comparing the same retained sites before and after recomputing background.

`results/metrics.csv` contains dataset-specific and pooled results; `reliability.csv` and Brier scores assess score–label agreement. `operating_points.csv` reports counts, precision and recall at cutoffs 0.5, 0.7 and 0.9. The feature set and blend weight were selected using external development comparisons; model fitting uses only Datasets 0–2.

See [benchmark/README.md](benchmark/README.md) for all-site baseline reproduction and the separate **official m6Anet Python 3.8/Conda benchmark**.

### Evaluate a new labelled sample

```sh
python evaluate.py --predictions outputs/predictions.csv --labels inputs/labels.csv --dataset new_sample --output outputs/new_sample_metrics
```

Labels need `transcript_id,transcript_position,label` and must describe exactly the predicted sites in that sample. Multiple-sample tables require `sample_id` in both files. The evaluator writes ROC AUC, trapezoidal PR AUC, average precision, labelled-site keys and the evaluation settings. `--model-name` labels results from another predictor.

## Explain a prediction

```sh
python explain.py --data example/input.json --transcript ENST00000000412 --position 496 --output outputs/site_explanation.html
```

The self-contained HTML shows the final score, TreeSHAP feature contributions for each ensemble and changes after removing individual reads. TreeSHAP values add up to the raw model score (the margin). The sigmoid function maps each margin to [0,1] before the two scores are combined. CatBoost's fitted intercept is included. Read removal recomputes the learned summaries, read count and current summaries; each removal is a separate comparison with the complete site.

Use the same complete sample, model and reference as the prediction being explained. A one-read site has feature contributions but no defined removal score.

## Code and checks

| File or folder | Purpose |
|---|---|
| `train.py` / `predict.py` | Fit a model / predict a complete new sample. |
| `validate_predictions.py` | Check output columns, exact input coordinates and score bounds; optionally compare example scores. Uses only the Python standard library. |
| `context_trees/` | Input preparation, features, shared neural encoder and trees. |
| `evaluate.py` / `reproduce_evaluation.py` | Recalculate metrics / refit grouped and sample-transfer experiments. |
| `benchmark/` | Logistic and m6Anet baselines, including upstream licence. |
| `explain.py` | Individual prediction explanations. |
| `model/` / `example/` / `results/` | Fitted model, runnable example and held-out results. |

```sh
python checks.py
```

Checks cover prediction identity and batching, context construction, ordered-group weighting, model feature sets, score reconstruction and read-removal calculations.
