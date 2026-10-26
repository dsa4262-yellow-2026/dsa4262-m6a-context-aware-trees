"""Recalculate classification and ordering metrics from grouped held-out predictions."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from context_trees.model import combine_predictions
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
KEYS = [
    "source",
    "transcript_id",
    "transcript_position",
    "gene_id",
    "label",
    "n_reads",
    "kmer",
]


def ranking(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return dict(
        roc_auc=float(roc_auc_score(labels, scores)),
        pr_auc=float(auc(recall, precision)),
        ap=float(average_precision_score(labels, scores)),
    )


def operating_points(
    frame, model_name="context_aware_trees_v1", thresholds=(0.5, 0.7, 0.9)
):
    """Describe fixed score cutoffs using each sample's own held-out labels."""
    rows = []
    for source, sample in frame.groupby("source", sort=True):
        positives = int(sample.label.sum())
        for threshold in thresholds:
            selected = sample.score.ge(threshold)
            count = int(selected.sum())
            true_positive = int(sample.loc[selected, "label"].sum())
            rows.append(
                dict(
                    dataset=source,
                    model=model_name,
                    score_threshold=threshold,
                    observations=len(sample),
                    positive_labels=positives,
                    selected_sites=count,
                    true_positives=true_positive,
                    false_positives=count - true_positive,
                    false_negatives=positives - true_positive,
                    precision=true_positive / count if count else np.nan,
                    recall=true_positive / positives if positives else np.nan,
                    mean_selected_score=float(sample.loc[selected, "score"].mean()),
                )
            )
    return pd.DataFrame(rows)


def load_results(directory):
    binary, mixtures = [], []
    for fold in range(5):
        with np.load(directory / f"fold{fold}.npz", allow_pickle=False) as saved:
            frame = pd.DataFrame({name: saved[name] for name in KEYS})
            frame["index"], frame["fold"] = saved["index"], fold
            frame["score"] = combine_predictions(saved["prediction"])
            frame["xgboost_score"], frame["catboost_score"] = saved["prediction"].T
            binary.append(frame)
            mixture = pd.DataFrame(
                {
                    name: saved["mixture_" + name]
                    for name in ["index", "position", "nominal"]
                }
            )
            mixture["score"], mixture["fold"] = (
                combine_predictions(saved["mixture_prediction"]),
                fold,
            )
            mixture["xgboost_score"], mixture["catboost_score"] = saved[
                "mixture_prediction"
            ].T
            mixtures.append(mixture)
    binary = pd.concat(binary).sort_values("index").reset_index(drop=True)
    mixtures = pd.concat(mixtures).sort_values("index").reset_index(drop=True)
    for frame in (binary, mixtures):
        if not np.array_equal(frame["index"], np.arange(len(frame))):
            raise ValueError(
                "Every input observation must have exactly one held-out prediction."
            )
        scores = frame.filter(regex="(^score$|_score$)").to_numpy()
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid held-out score.")
    if binary.duplicated(["source", "transcript_id", "transcript_position"]).any():
        raise ValueError("Duplicate binary observation.")
    if not binary.groupby("gene_id").fold.nunique().eq(1).all():
        raise ValueError("A gene crosses evaluation folds.")
    if not mixtures.groupby("position").fold.nunique().eq(1).all():
        raise ValueError("A mixture position crosses evaluation folds.")
    if not mixtures.groupby("position").nominal.nunique().eq(7).all():
        raise ValueError(
            "Every mixture position must retain its seven distinct settings."
        )
    mapping = (
        pd.read_csv(directory / "gene_folds.csv")
        .set_index("gene_id", verify_integrity=True)
        .fold
    )
    if not np.array_equal(binary.gene_id.map(mapping), binary.fold):
        raise ValueError("Prediction folds disagree with the supplied gene map.")
    return binary, mixtures


def evaluate(directory, baselines=None):
    binary, mixture = load_results(directory)
    models = {
        "context_aware_trees_v1": binary.score.to_numpy(),
        "local_xgboost": binary.xgboost_score.to_numpy(),
        "full_catboost": binary.catboost_score.to_numpy(),
    }
    if baselines is not None:
        with np.load(baselines, allow_pickle=False) as saved:
            for name in KEYS + ["fold"]:
                if not np.array_equal(binary[name], saved[name]):
                    raise ValueError(f"Baseline observations differ: {name}.")
            for column, name in enumerate(saved["model_names"]):
                models[str(name)] = saved["prediction"][:, column]
    records, reliability = [], []
    for model, scores in models.items():
        for source in ["all", 0, 1]:
            use_source = (
                np.ones(len(binary), dtype=bool)
                if source == "all"
                else binary.source.to_numpy() == source
            )
            if model in ["context_aware_trees_v1", "local_xgboost", "full_catboost"]:
                ordered = np.flatnonzero(use_source)
                ordered = ordered[np.argsort(scores[ordered], kind="stable")]
                for number, rows in enumerate(np.array_split(ordered, 10), 1):
                    reliability.append(
                        dict(
                            dataset=source,
                            model=model,
                            bin=number,
                            observations=len(rows),
                            mean_prediction=float(scores[rows].mean()),
                            observed_fraction=float(
                                binary.label.to_numpy()[rows].mean()
                            ),
                            score_minimum=float(scores[rows].min()),
                            score_maximum=float(scores[rows].max()),
                        )
                    )
            for fold in ["pooled", 0, 1, 2, 3, 4]:
                use = (
                    use_source
                    if fold == "pooled"
                    else use_source & (binary.fold.to_numpy() == fold)
                )
                records.append(
                    dict(
                        dataset=source,
                        fold=fold,
                        model=model,
                        observations=int(use.sum()),
                        **ranking(binary.label.to_numpy()[use], scores[use]),
                        mean_prediction=float(scores[use].mean()),
                        observed_fraction=float(binary.label.to_numpy()[use].mean()),
                        brier=float(
                            brier_score_loss(binary.label.to_numpy()[use], scores[use])
                        ),
                        log_loss=float(
                            log_loss(binary.label.to_numpy()[use], scores[use])
                        ),
                    )
                )
    endpoints = mixture.nominal.isin([0, 1]).to_numpy()
    ordering = []
    for model, column in [
        ("context_aware_trees_v1", "score"),
        ("local_xgboost", "xgboost_score"),
        ("full_catboost", "catboost_score"),
    ]:
        records.append(
            dict(
                dataset="2_endpoints",
                fold="pooled",
                model=model,
                observations=int(endpoints.sum()),
                **ranking(
                    mixture.nominal.to_numpy()[endpoints],
                    mixture[column].to_numpy()[endpoints],
                ),
            )
        )
        correlations, inverted_positions = [], 0
        correct, pairs = 0.0, 0
        for _, group in mixture.groupby("position", sort=True):
            group = group.sort_values("nominal")
            scores = group[column].to_numpy()
            lower, higher = np.triu_indices(len(scores), 1)
            difference = scores[higher] - scores[lower]
            correct += float(((difference > 0) + 0.5 * (difference == 0)).sum())
            pairs += len(difference)
            inverted_positions += int(np.any(difference < 0))
            correlations.append(
                group[["nominal", column]].corr(method="spearman").iloc[0, 1]
            )
        defined = np.asarray(correlations, dtype=float)
        defined = defined[np.isfinite(defined)]
        ordering.append(
            dict(
                model=model,
                positions=len(correlations),
                pairs=pairs,
                positions_with_inversions=inverted_positions,
                pair_agreement=correct / pairs,
                undefined_correlations=len(correlations) - len(defined),
                spearman_minimum=float(defined.min()) if len(defined) else None,
                spearman_10th_percentile=float(np.quantile(defined, 0.1))
                if len(defined)
                else None,
                spearman_median=float(np.median(defined)) if len(defined) else None,
            )
        )
    metrics = pd.DataFrame(records)
    metrics.to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(directory / "reliability.csv", index=False)
    operating_points(binary).to_csv(directory / "operating_points.csv", index=False)
    summary = dict(
        scope="Five-fold evaluation of held-out genes and mixture positions within the supplied releases.",
        binary_sites=len(binary),
        genes=int(binary.gene_id.nunique()),
        mixture_bags=len(mixture),
        mixture_positions=int(mixture.position.nunique()),
        within_position_ordering=ordering,
        definitions={
            "roc_auc": "ROC area",
            "pr_auc": "Trapezoidal precision-recall area",
            "ap": "Average precision; distinct from trapezoidal PR AUC",
            "brier": "Mean squared probability error on binary site labels; lower is better.",
            "log_loss": "Unweighted binary log loss on site labels; lower is better.",
            "reliability": "Ten equal-count bins of held-out scores; compare mean_prediction with observed_fraction.",
            "operating_points": "Fixed score cutoffs 0.5, 0.7 and 0.9 on each binary sample: selected counts, precision and recall; no threshold fitting.",
            "pair_agreement": "Fraction of correctly ordered within-position pairs, with half credit for ties.",
            "dataset2_endpoints": "Discrimination between the two known source collections; not a labelled methylation occupancy test.",
        },
        pooled=[record for record in records if record["fold"] == "pooled"],
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(metrics[metrics.fold == "pooled"].to_string(index=False))
    if (directory / "native_m6anet").exists():
        evaluate_native(directory / "native_m6anet", binary)
    return summary


def evaluate_sample(predictions, labels, dataset, output, model_name):
    """Evaluate one new binary-labelled sample, or samples explicitly keyed by sample_id."""
    dtype = {"transcript_id": str, "transcript_position": str, "sample_id": str}
    predicted = pd.read_csv(predictions, dtype=dtype, keep_default_na=False)
    labelled = pd.read_csv(labels, dtype=dtype, keep_default_na=False)
    keys = ["transcript_id", "transcript_position"]
    if "sample_id" in predicted or "sample_id" in labelled:
        if "sample_id" not in predicted or "sample_id" not in labelled:
            raise ValueError(
                "Multiple-sample evaluation requires sample_id in both files."
            )
        keys = ["sample_id"] + keys
    for frame, value in [(predicted, "score"), (labelled, "label")]:
        if not set(keys + [value]).issubset(frame.columns) or frame.empty:
            raise ValueError(f"Expected a nonempty table with {keys + [value]}.")
        if frame[keys].eq("").any().any():
            raise ValueError("Sample and site identifiers must not be empty.")
        position = pd.to_numeric(frame.transcript_position, errors="raise")
        if (
            not np.isfinite(position).all()
            or not position.eq(np.floor(position)).all()
            or (position < 0).any()
        ):
            raise ValueError("Transcript positions must be nonnegative integers.")
        frame["transcript_position"] = position.astype(np.int64)
        if frame.duplicated(keys).any():
            raise ValueError("Each sample/site key must occur exactly once.")
        frame[value] = pd.to_numeric(frame[value], errors="raise")
    if (
        not np.isfinite(predicted.score).all()
        or not predicted.score.between(0, 1).all()
    ):
        raise ValueError("Prediction scores must be finite and between 0 and 1.")
    if not labelled.label.isin([0, 1]).all() or labelled.label.nunique() != 2:
        raise ValueError(
            "Binary evaluation requires labels 0 and 1, with both classes present."
        )
    joined = predicted[keys + ["score"]].merge(
        labelled[keys + ["label"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError(
            "Predictions and labels must contain exactly the same sample/site keys."
        )
    target = joined[keys + ["label"]].sort_values(keys).copy()
    target["label"] = target.label.astype(int)
    row = dict(
        dataset=dataset,
        model=model_name,
        observations=len(joined),
        positive_fraction=float(joined.label.mean()),
        **ranking(joined.label, joined.score),
    )
    output = Path(output)
    protected = {Path(path).resolve() for path in [predictions, labels]}
    if any(
        (output / name).resolve() in protected
        for name in ["metrics.csv", "protocol.json", "evaluated_sites.csv.gz"]
    ):
        raise ValueError("Evaluation output must not replace an input file.")
    if output.resolve().is_relative_to((HERE / "model").resolve()):
        raise ValueError("Write evaluation results outside the fitted model directory.")
    protocol = dict(
        dataset=dataset,
        model=model_name,
        site_keys=keys,
        labelled_sites="evaluated_sites.csv.gz",
        observations=len(joined),
        positives=int(joined.label.sum()),
        pr_auc="Trapezoidal precision-recall area",
        ap="Average precision",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluated_sites.csv.gz").open("wb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as compressed:
            compressed.write(target.to_csv(index=False, lineterminator="\n").encode())
    pd.DataFrame([row]).to_csv(output / "metrics.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(
        pd.DataFrame([row])[
            ["dataset", "model", "observations", "roc_auc", "pr_auc", "ap"]
        ].to_string(index=False)
    )
    return row


def evaluate_native(directory, binary):
    """Compare native m6Anet and the final model on exactly the same held-out sites."""
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    selected = binary.loc[binary.fold.eq(protocol["fold"])].copy()
    selected["dataset"] = "data" + selected.source.astype(str)
    native = pd.read_csv(directory / "predictions.csv.gz", dtype={"transcript_id": str})
    native = native.rename(columns={"score": "m6anet_native"})
    keys = ["dataset", "transcript_id", "transcript_position"]
    compared = selected.merge(
        native, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not compared._merge.eq("both").all():
        raise ValueError(
            "Native m6Anet and the final model must have identical held-out site keys."
        )
    if (
        not np.isfinite(compared.m6anet_native).all()
        or not compared.m6anet_native.between(0, 1).all()
    ):
        raise ValueError("Invalid native m6Anet score.")
    rows = []
    for source in ["all", "data0", "data1"]:
        values = (
            compared if source == "all" else compared.loc[compared.dataset.eq(source)]
        )
        for name, column in [
            ("context_aware_trees_v1", "score"),
            ("m6anet_native", "m6anet_native"),
        ]:
            rows.append(
                dict(
                    dataset=source,
                    fold=protocol["fold"],
                    model=name,
                    observations=len(values),
                    **ranking(values.label, values[column]),
                )
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "comparison.csv", index=False)
    print("\nNative m6Anet comparison on held-out fold " + str(protocol["fold"]))
    print(metrics.to_string(index=False))
    return metrics


def evaluate_transfer(directory):
    """Score held-out samples and compare the same sites under two context collections."""
    records = []
    for sample in [0, 1]:
        with np.load(directory / f"sample{sample}.npz", allow_pickle=False) as saved:
            index, retained = saved["index"], saved["retained_index"]
            labels = saved["label"]
            score = combine_predictions(saved["prediction"])
            partial = combine_predictions(saved["retained_prediction"])
            rows = np.searchsorted(index, retained)
            covered = saved["covered_index"]
            covered_rows = np.searchsorted(index, covered)
            covered_score = combine_predictions(saved["covered_prediction"])
            if not np.all(saved["source"] == sample) or not np.array_equal(
                index[rows], retained
            ):
                raise ValueError(
                    "Held-out sample identities or retained positions disagree."
                )
            if not np.all(np.diff(index) > 0) or len(np.unique(retained)) != len(
                retained
            ):
                raise ValueError(
                    "Transfer evaluation contains duplicate or unordered observations."
                )
            if not np.array_equal(index[covered_rows], covered) or not np.array_equal(
                index[saved["n_reads"] >= saved["coverage_threshold"]], covered
            ):
                raise ValueError(
                    "Coverage-selected sites disagree with their recorded read counts."
                )
            probabilities = np.r_[score, partial, covered_score]
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0) | (probabilities > 1)
            ):
                raise ValueError("Invalid transfer prediction.")
            conditions = [
                ("all_reported_sites", labels, score, 0.0),
                ("retained_sites_full_context", labels[rows], score[rows], 0.0),
                (
                    "retained_sites_recomputed_context",
                    labels[rows],
                    partial,
                    float(np.abs(partial - score[rows]).mean()),
                ),
                (
                    "covered_sites_full_context",
                    labels[covered_rows],
                    score[covered_rows],
                    0.0,
                ),
                (
                    "covered_sites_recomputed_context",
                    labels[covered_rows],
                    covered_score,
                    float(np.abs(covered_score - score[covered_rows]).mean()),
                ),
            ]
            for condition, y, p, context_change in conditions:
                records.append(
                    dict(
                        held_out_sample=sample,
                        training_binary_sample=1 - sample,
                        condition=condition,
                        observations=len(y),
                        observed_fraction=float(y.mean()),
                        **ranking(y, p),
                        brier=float(brier_score_loss(y, p)),
                        log_loss=float(log_loss(y, p)),
                        mean_absolute_context_change=context_change,
                    )
                )
    metrics = pd.DataFrame(records)
    metrics.to_csv(directory / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument(
        "--baselines",
        type=Path,
        help="Optional baseline prediction archive aligned to the same observations and folds.",
    )
    parser.add_argument(
        "--transfer", action="store_true", help="Evaluate the two held-out sample runs."
    )
    parser.add_argument(
        "--predictions", type=Path, help="Prediction CSV for a new labelled sample."
    )
    parser.add_argument(
        "--labels", type=Path, help="Matching CSV containing binary label values."
    )
    parser.add_argument("--dataset", help="Name of the new sample or release.")
    parser.add_argument(
        "--output", type=Path, help="Directory for new-sample metrics and their record."
    )
    parser.add_argument("--model-name", default="context_aware_trees_v1")
    args = parser.parse_args()
    if args.predictions:
        if not args.labels or not args.dataset or not args.output:
            parser.error("--predictions requires --labels, --dataset and --output.")
        if args.transfer or args.baselines:
            parser.error(
                "Choose either new-sample evaluation or saved held-out results."
            )
        evaluate_sample(
            args.predictions, args.labels, args.dataset, args.output, args.model_name
        )
    elif (
        args.labels
        or args.dataset
        or args.output
        or args.model_name != "context_aware_trees_v1"
    ):
        parser.error("New-sample options require --predictions.")
    elif args.transfer:
        evaluate_transfer(args.results)
    else:
        baselines = args.baselines
        if baselines is None and args.results.resolve() == (HERE / "results").resolve():
            baselines = HERE / "results/baselines.npz"
        evaluate(args.results, baselines)
