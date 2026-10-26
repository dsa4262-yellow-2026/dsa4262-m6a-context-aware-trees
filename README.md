<img src="assets/team-icon.png" alt="Yellow team logo: a smiling yellow tree" width="144" align="right">

# Context-aware trees

DSA4262: Data Science in Genomics. Team: yellow.

Context-aware trees predicts m6A sites from processed nanopore direct-RNA reads. A neural encoder summarizes the reads; XGBoost and CatBoost combine these summaries with measured signal and transcript context. The final score is 45% full CatBoost (41 inputs) plus 55% local XGBoost (20 inputs), sharing an eight-output encoder. The final fit uses datasets 0, 1 and 2.

## Install and run the example

Download the repository and open a terminal in its root folder. Use **Python 3.11** and an x86-64 CPU. The prediction example uses the included fitted model and needs no training or data download.

With Conda installed, create the environment on Ubuntu or Windows:

```sh
conda create -n context-aware-trees-v1 -c conda-forge python=3.11 pip -y
conda activate context-aware-trees-v1
```

For a fresh Ubuntu machine without Conda, first install [Miniforge](https://github.com/conda-forge/miniforge#install):

```sh
curl -L --fail -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Then run the environment commands above. If Python 3.11 is already installed, a virtual environment is also suitable: `python3.11 -m venv .venv`, followed by `source .venv/bin/activate` on Ubuntu or `.venv\Scripts\Activate.ps1` in Windows PowerShell. On Windows, use `py -3.11` in place of `python3.11`.

Install the dependencies and run the example:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "task 1 context aware trees/requirements.txt"
python -m pip check
cd "task 1 context aware trees"
python predict.py --data example/input.json --output outputs/example_predictions.csv
```

The example produces 60 rows with exactly these columns:

```text
transcript_id,transcript_position,score
```

Expected values are in `example/expected_predictions.csv`. Each score is between 0 and 1; identifiers and row order match the input.

For the automated example check, run `python -m unittest checks.ScientificChecks.test_prediction_identity_and_cpu_batching`. Allow small floating-point differences across platforms. A CPU machine with 8 GB RAM is sufficient for this small example; full SG-NEx reproduction is documented separately.

## Predict another sample

From the model folder:

```sh
python predict.py --data /path/to/sample.json.gz --output /path/to/predictions.csv
```

Both plain and gzip-compressed original-format JSON inputs are accepted. Supply the complete set of reported sites for one sample. Labels are not required, and unreported positions need not be added.

The [model instructions](<task 1 context aware trees/README.md>) cover training from original data, CPU/GPU options, evaluation, prediction explanations and the separate Python 3.8 Conda m6Anet benchmark. That model folder can also be copied and used independently.

To [explain an individual prediction](<task 1 context aware trees/README.md#explain-an-individual-prediction>), run from the model folder:

```sh
python explain.py --data example/input.json --transcript ENST00000000412 --position 496 --output outputs/site_explanation.html
```

The HTML shows how each tree's feature contributions reconstruct its score, how the 55/45 combination produces the final prediction, and how that prediction changes when each read is removed separately.

## Train from the original releases

The [training-data release](https://github.com/dsa4262-yellow-2026/dsa4262-m6a-context-aware-trees/releases/tag/training-data-v1) contains the original datasets 0, 1 and 2 with their matching labels (328 MB download). From the model folder, download and train:

```sh
python download_data.py --output inputs/training
python train.py --data-dir inputs/training --output outputs/retrained_model
```

The downloader uses Python's standard library, extracts the six training files and keeps existing inputs. This fits the encoder and both trees using datasets 0, 1 and 2. Pass `--model outputs/retrained_model` to `predict.py` to use that new fit. The [training instructions](<task 1 context aware trees/README.md#train-from-the-course-data>) also cover a different data folder and offline download. Allow about 700 MB for downloading and extracting the inputs, in addition to training scratch space and model outputs.

## Code and results

| Folder | Contents |
|---|---|
| [task 1 context aware trees](<task 1 context aware trees/README.md>) | Final model, training and prediction scripts, example, held-out results and evaluation. |
| [model evaluation](<model evaluation/README.md>) | Final-model, component, logistic-baseline and m6Anet comparisons, with a separate external-development panel index. |
| [task 2](<task 2/README.md>) | All 23 SG-NEx prediction runs, reproducible analysis and tables. |

The [SG-NEx analysis guide](<task 2/README.md>) covers overall cell differences, shared candidates and their locations, recurring local contrasts, and read support.

The final model's pooled gene-grouped evaluation gives ROC AUC 0.908609 and trapezoidal PR AUC 0.526907. All observations of a gene stay together across datasets 0 and 1; all seven mixture settings of a Dataset 2 position stay together. Complete-sample transfer is evaluated separately. Recalculate the saved metrics from the model folder:

```sh
python evaluate.py
python evaluate.py --results results/transfer --transfer
```

Only Datasets 0–2 fit model parameters. External panels informed the feature recipe and 45/55 selection; these comparisons are development evidence.

The repository includes fitted models and saved results. The six processed training inputs are available through the linked release; Task 2 documents downloads of the public SG-NEx inputs. The report and hosted interactive platform are separate deliverables.

## Licence

Original project code and documentation are available under the [MIT License](LICENSE). Bundled m6Anet code and weights retain their [upstream MIT notice](<task 1 context aware trees/benchmark/m6anet_reference/LICENSE.txt>). Input datasets retain their source terms.
