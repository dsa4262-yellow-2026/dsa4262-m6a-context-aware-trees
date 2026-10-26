<img src="assets/team-icon.png" alt="Yellow logo: a smiling yellow tree" width="144" align="right">

# Context Aware Trees

**By Yellow**

Context Aware Trees predicts N6-methyladenosine (m6A) sites from processed nanopore direct RNA sequencing data. One neural encoder summarizes the reads at each site. A local XGBoost predictor uses 20 features; a full CatBoost predictor adds 21 transcript-context features. The final score is **55% local + 45% full**.

[Model and features](<task 1 context aware trees/README.md#model-and-features>) · [SG-NEx analysis](<task 2/README.md>)

![Model overview: a shared encoder produces eight summaries per site. Together with 12 direct features, these enter both tree ensembles. Local XGBoost uses 20 features and contributes 55% of the final score; full CatBoost adds 21 context features and contributes 45%.](assets/model-overview.png)

## Quick start

Run the bundled **60-site example** to produce a prediction CSV and verify its scores. The fitted model, input data and expected output are included.

Choose an [installation guide](docs/installation.md):

| Platform | Instructions |
|---|---|
| RONIN Ubuntu instance, accessed from Windows, macOS or Linux | [RONIN setup](docs/installation.md#ronin) |
| Ubuntu computer or Ubuntu through WSL | [Ubuntu setup](docs/installation.md#ubuntu) |
| Windows computer with an Intel or AMD processor | [Windows setup](docs/installation.md#windows) |

**Requirements:** Python 3.11 on x86-64 Ubuntu or Windows. **Recommended resources:** 8 CPU cores or vCPUs, 16 GB RAM and 20 GB free disk space for the example. Prediction runs on CPU. Mac users can follow the RONIN instructions.

### Run and verify

After following the installation guide, activate `context-aware-trees` and enter the `task 1 context aware trees` folder. Run:

```sh
python predict.py --data example/input.json --output outputs/example_predictions.csv
python validate_predictions.py --data example/input.json --predictions outputs/example_predictions.csv --expected example/expected_predictions.csv
```

Successful output:

```text
Wrote 60 scores to example_predictions.csv.
PASS: 60 sites; exact columns, input coordinates/order, finite scores in [0,1]; expected scores match.
```

**Output:** `task 1 context aware trees/outputs/example_predictions.csv`, containing **60 rows and exactly three columns**. The first two rows are:

```csv
transcript_id,transcript_position,score
ENST00000000412,496,0.01605253872787089
ENST00000000412,2440,0.5593107649860982
```

`PASS` confirms that the output format and scores match the bundled example within numerical tolerance.

### Predict another sample

From `task 1 context aware trees/`:

```sh
python predict.py --data "/path/to/data.json" --output outputs/new_sample.csv
python validate_predictions.py --data "/path/to/data.json" --predictions outputs/new_sample.csv
```

Replace the input path with your own JSON or JSON.gz file in the [required input format](<task 1 context aware trees/README.md#input-format>). Include all observed sites from one sample so transcript-background features can be calculated. Prediction requires no labels.

To predict an SG-NEx sample, follow the [sample download and prediction instructions](<task 2/README.md#reproduce-predictions-from-the-original-inputs>).

## Documentation

| Topic | Guide |
|---|---|
| Model, features and prediction options | [Model guide](<task 1 context aware trees/README.md>) |
| Training on supplied or new data | [Training guide](<task 1 context aware trees/README.md#train-on-datasets-02>) |
| Evaluation and benchmark comparisons | [Evaluation](<task 1 context aware trees/README.md#reproduce-the-evaluation>) · [Baselines](<task 1 context aware trees/benchmark/README.md>) |
| SG-NEx analysis | [Analysis and reproduction instructions](<task 2/README.md>) |
| Installation help | [Troubleshooting](docs/installation.md#troubleshooting) · [Reopen the environment](docs/installation.md#reopen-the-environment) |

## Data

The bundled model was trained on Datasets 0, 1 and 2, provided through DSA4262: Data Science in Genomics. Download the inputs from the [training-data release](https://github.com/dsa4262-yellow-2026/m6a-context-aware-trees/releases/tag/training-data-v1).

SG-NEx predictions are included for 23 sequencing runs from 17 biological replicates across seven human cell lines. The [sample inventory](<task 2/samples.csv>) lists the inputs; the [analysis guide](<task 2/README.md#generate-figures-from-supplied-tables>) explains how to generate figures from the supplied tables.

## License

Original code and documentation use the [MIT License](LICENSE). Bundled m6Anet assets retain their [upstream MIT notice](<task 1 context aware trees/benchmark/m6anet_reference/LICENSE.txt>). Input datasets retain their source terms.
