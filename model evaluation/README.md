# Task 1 model evaluation

`comparison.csv` keeps three evaluation questions together:

| Section | Data and held-out unit | Target and metrics |
|---|---|---|
| A. Site classification | Datasets 0/1; five gene-grouped folds | Binary m6A site labels: ROC AUC, trapezoidal PR AUC and AP. |
| B. Mixture response | Dataset 2; five position-grouped folds | Endpoint-source ROC/PR AUC and agreement across within-position mixture pairs, with half credit for ties. |
| C. Final test | New release, scored with a fixed model | Local metrics when labels are supplied, or results returned by the organiser. Pending until available. |

The Dataset 0/1 comparison includes the final model, its tree components, a logistic baseline and released/retrained m6Anet. Dataset 2 has 378 endpoint observations and 1,323 settings at 189 positions, giving 3,969 within-position pairs. Its endpoints identify source collections. No metric pools these targets with binary site labels.

## Recalculate and export

From `../task 1 context aware trees/`, run:

```text
python evaluate.py --comparison "../model evaluation/comparison.csv"
python evaluate.py --results results/transfer --transfer
```

The first command checks observation identities and grouped folds, recalculates metrics and exports the table. It also refreshes `results/native_m6anet/comparison.csv`: native Python 3.8 m6Anet and the final model on the same 44,144 fold-0 sites. The second command evaluates the separate sample-transfer experiments.

`results/metrics.csv` retains dataset-specific and individual-fold results; `results/summary.json` also contains mixture ordering. The observation archives, baseline predictions and protocols remain in that results folder. [Baseline reproduction instructions](<../task 1 context aware trees/README.md#reproduce-the-all-site-baselines>) describe the original-signal calculations and native benchmark.

## Score cutoffs and candidate selection

The same command produces [operating_points.csv](<../task 1 context aware trees/results/operating_points.csv>): six rows for cutoffs 0.5, 0.7 and 0.9 in Datasets 0 and 1. Each row gives selected sites, true/false positives, false negatives, precision, recall and mean selected score. This connects the ranking evaluation to the score cutoffs used for Task 2 candidate catalogues. The cutoffs are fixed summaries, with no threshold optimization or model refitting. The [model instructions](<../task 1 context aware trees/README.md#interpret-a-score-cutoff>) explain the measured tradeoff and its sample-specific interpretation.

## Add the final release

Use the final model's [new-sample evaluation commands](<../task 1 context aware trees/README.md#evaluate-a-future-labelled-release>). After evaluating the matching predictions and labels, include its metrics:

```text
python evaluate.py --comparison "../model evaluation/comparison.csv" --external results/final_release/metrics.csv
```

Run this command from the model folder. For several final-test samples or models, list their metric CSVs after `--external` in the same command. The exporter checks duplicate dataset/model entries and that locally evaluated models for the same dataset used identical labelled sites. Passing final-test metrics replaces the pending section; omit them to export development results with a pending final section. No training runs are started by evaluation.

## When labels remain private

Keep the returned results as a CSV with these columns:

```text
dataset,model,observations,roc_auc,pr_auc,metric_origin,pr_auc_definition,evidence
```

Use `context_aware_trees_v1` as the model name and `organizer` as `metric_origin`. Fill the ROC and PR numbers exactly as returned, specify the organiser's PR definition, and give the official result's URL or saved filename in `evidence`. Leave `observations` empty if it is not reported. Import this CSV with `--external` as above. If the reported PR metric is average precision, state `average_precision` in `pr_auc_definition`; it is distinct from the development table's trapezoidal PR AUC. Panel C identifies the result's origin and PR definition.

The consolidated CSV retains each section's target, protocol, observation count and metric definition. Report authoring files are kept outside the code repository.

## External development panels

[External results](<../task 1 context aware trees/results/external/README.md>) retain per-site scores, three-seed means and the released fit separately. Recalculate them with `python evaluate_external_panels.py` from the model folder. These panels informed model selection and remain separate from the pending final test.
