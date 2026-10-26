# Baseline comparisons

Run commands from `task 1 context aware trees/`. The comparison uses the same site labels and held-out gene folds as Context Aware Trees.

## All-site comparisons

The main Python 3.11 environment runs three baselines:

- **Logistic regression:** nine raw channel means, with constant removal and standardization fitted within each training fold.
- **Released m6Anet:** bundled HCT116 RNA002 weights and official normalization.
- **Retrained m6Anet:** the same architecture, fitted only on the binary training folds of Datasets 0/1.

```sh
python benchmark/reproduce_baselines.py --data-dir inputs/training --output outputs/baselines --work-dir benchmark_scratch
python evaluate.py --baselines outputs/baselines/baselines.npz
```

Both m6Anet comparisons use `1 - (1 - mean_read_probability)**20`: expected noisy-OR inference over 20 reads sampled with replacement. Retraining fits five-mer normalization on training genes, selects the epoch count with an inner 25% gene holdout, then refits on all outer training genes. It uses at most 60 epochs, balanced site sampling, 20 reads per training bag, batch size 512, and Adam AMSGrad with learning rate 0.0004.

Allow about 10 GB of scratch space for read arrays. To run in stages, use `--stage prepare`, `--stage released`, `--stage retrain` and `--stage collect` with identical paths. Retraining accepts `--fold 0` through `--fold 4`; collection requires all five. `--device cuda` is optional.

The output includes held-out scores, fitted checkpoints and the protocol. The supplied `results/baselines.npz` and `results/baseline_protocol.json` support immediate metric reproduction.

## Official m6Anet benchmark

Run the official m6Anet implementation in a separate **Python 3.8 Conda environment**:

```sh
conda env create -f benchmark/environment.yml
conda activate task1-m6anet
python benchmark/run_m6anet.py --dataset data0 inputs/training/data0/dataset0.json.gz --dataset data1 inputs/training/data1/dataset1.json.gz --fold 0 --output outputs/native_m6anet --work-dir benchmark_scratch/native
```

Use empty output and scratch folders. The runner converts the original nine measurements into indexed native inputs and runs m6Anet 2.1.0 with its `HCT116_RNA002` checkpoint. Native inference averages 1,000 draws of 20 reads with replacement, so it differs from the expected-inference calculation above.

The supplied native results cover all **44,144 sites in held-out fold 0**. On these same sites, ROC/PR AUC is **0.9096/0.5366** for Context Aware Trees and **0.8752/0.4057** for native m6Anet. `python evaluate.py` verifies the site match and recalculates this comparison. The released checkpoint's original training overlap is unknown.

Return to `conda activate context-aware-trees` for the main model and analyses.

## Source and attribution

[m6Anet source](https://github.com/GoekeLab/m6anet) and [Hendra et al., Nature Methods (2022)](https://doi.org/10.1038/s41592-022-01666-1) describe the model. Bundled reference weights and normalization retain the upstream [MIT licence](m6anet_reference/LICENSE.txt). `expected_m6anet.py` implements expected inference; `retrain_m6anet.py` implements the grouped retraining comparison.
