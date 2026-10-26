"""Check site identity, supervision boundaries and the deployed feature views."""
from pathlib import Path
from itertools import islice
import copy
import csv
import gzip
import json
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from context_trees.data import attach_labels, close_reads, prepare_sites
from context_trees.features import iter_sites
from context_trees.model import (
    load_model,
    feature_names,
    ordered_groups,
    fit_encoder,
    fit_scalers,
    site_matrix,
    tree_predictions,
    combine_predictions,
)
from explain import explain_site, read_removal_predictions, tree_contributions
from predict import predict_file
from validate_predictions import validate_predictions

HERE = Path(__file__).resolve().parent


class PredictionFormatChecks(unittest.TestCase):
    def test_output_contract_and_expected_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data, output = folder / "input.json.gz", folder / "output.csv"
            with gzip.open(data, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"ENSTtest": {"0007": {}, "9": {}}}) + "\n")
            rows = [["ENSTtest", "0007", "0.25"], ["ENSTtest", "9", "0.75"]]

            def write(values, header=None):
                with output.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(
                        header or ["transcript_id", "transcript_position", "score"]
                    )
                    writer.writerows(values)

            write(rows)
            expected = folder / "expected.csv"
            expected.write_bytes(output.read_bytes())
            self.assertEqual(validate_predictions(data, output, expected), 2)
            for invalid in [
                rows[:1],
                rows + [rows[1]],
                rows[::-1],
                [["ENSTtest", "7", "0.25"], rows[1]],
                [["ENSTtest", "0007", "nan"], rows[1]],
                [["ENSTtest", "0007", "1.1"], rows[1]],
            ]:
                with self.subTest(rows=invalid):
                    write(invalid)
                    with self.assertRaises(ValueError):
                        validate_predictions(data, output)
            write(rows, ["transcript_id", "transcript_position", "probability"])
            with self.assertRaises(ValueError):
                validate_predictions(data, output)
            write([["ENSTtest", "0007", "0.5"], rows[1]])
            with self.assertRaises(ValueError):
                validate_predictions(data, output, expected)


class ScientificChecks(unittest.TestCase):
    def test_label_identity(self):
        sites = pd.DataFrame(
            dict(transcript_id=["t", "t"], transcript_position=[4, 9], n_reads=[3, 5])
        )
        reference = {"transcript_gene": {"t": "g"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            pd.DataFrame(
                dict(
                    transcript_id=["t", "t"],
                    transcript_position=[9, 4],
                    label=[1, 0],
                    n_reads=[5, 3],
                )
            ).to_csv(path, index=False)
            joined = attach_labels(sites, path, reference)
            np.testing.assert_array_equal(joined.label, [0, 1])
            pd.DataFrame(
                dict(transcript_id=["t", "t"], transcript_position=[4, 4], label=[1, 0])
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                attach_labels(sites, path, reference)

    def test_training_only_normalization(self):
        sites = pd.DataFrame(
            dict(
                kmer=["AAGACTA"] * 3,
                n_reads=[2, 2, 2],
                read_start=[0, 2, 4],
                read_stop=[2, 4, 6],
            )
        )
        reads = np.arange(54, dtype=np.float32).reshape(6, 9)
        before = fit_scalers(reads, sites, np.array([0, 1]))
        reads[4:] = 1e8
        after = fit_scalers(reads, sites, np.array([0, 1]))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_ordered_groups_stay_together(self):
        sites = pd.DataFrame(
            dict(
                order_group=[0, 0, 0, 1, 1],
                order_level=[0, 0.5, 1, 0, 1],
                kmer=["AAGACTA"] * 5,
            )
        )
        self.assertEqual(len(ordered_groups(sites, np.array([0, 1, 2]))), 1)
        with self.assertRaises(ValueError):
            ordered_groups(sites, np.array([0, 2]))

    def test_shared_encoder_training(self):
        sites = pd.DataFrame(
            dict(
                kmer=["AAGACTA"] * 4,
                n_reads=[3] * 4,
                read_start=[0, 3, 6, 9],
                read_stop=[3, 6, 9, 12],
                label=[0, 1, 0, 1],
            )
        )
        reads = np.random.default_rng(4262).normal(size=(12, 9)).astype(np.float32)
        binary = dict(sites=sites, reads=reads)
        ordered = dict(
            sites=sites.assign(
                order_group=[0, 0, 1, 1], order_level=[0.0, 1.0, 0.0, 1.0]
            ),
            reads=reads,
        )
        network, scalers, history = fit_encoder(binary, ordered, epochs=1)
        self.assertTrue(np.isfinite(history[0]["binary_loss"]))
        self.assertGreater(history[0]["ordered_loss"], 0.0)
        self.assertTrue(all(np.isfinite(value).all() for value in scalers.values()))
        self.assertTrue(
            all(
                parameter.grad is not None for parameter in network.encoder.parameters()
            )
        )

    def test_model_feature_views(self):
        network, _, tree, cat = load_model(HERE / "model")
        self.assertEqual(tree.num_features(), 20)
        self.assertEqual(len(cat.feature_names_), 41)
        self.assertEqual(len(feature_names()), 41)
        self.assertFalse(any("site_head" in name for name in network.state_dict()))
        values = np.zeros((3, 41), dtype=np.float32)
        first = tree_predictions(tree, cat, values)
        values[:, 20:] = np.nan
        second = tree_predictions(tree, cat, values)
        np.testing.assert_array_equal(first[:, 0], second[:, 0])
        np.testing.assert_allclose(
            combine_predictions(first),
            0.55 * first[:, 0] + 0.45 * first[:, 1],
            rtol=0,
            atol=0,
        )

    def test_prediction_identity_and_cpu_batching(self):
        with tempfile.TemporaryDirectory() as folder:
            first = predict_file(
                HERE / "example/input.json",
                Path(folder) / "one.csv",
                HERE / "model",
                batch_size=1,
            )
            second = predict_file(
                HERE / "example/input.json",
                Path(folder) / "many.csv",
                HERE / "model",
                batch_size=128,
            )
        expected = pd.read_csv(
            HERE / "example/expected_predictions.csv",
            dtype={"transcript_id": str, "transcript_position": str},
        )
        self.assertEqual(list(first), ["transcript_id", "transcript_position", "score"])
        np.testing.assert_array_equal(first.iloc[:, :2], expected.iloc[:, :2])
        np.testing.assert_allclose(first.score, expected.score, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(first.iloc[:, :2], second.iloc[:, :2])
        np.testing.assert_allclose(first.score, second.score, rtol=1e-6, atol=1e-7)


class ExplanationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network, cls.scalers, cls.tree, cls.cat = load_model(HERE / "model")
        cls.reference = joblib.load(HERE / "model/reference.joblib")

    def test_native_contributions_reconstruct_both_views(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared = prepare_sites(
                [(0, HERE / "example/input.json", None)], folder, self.reference
            )
            try:
                values = site_matrix(self.network, prepared, self.scalers)
                values = np.vstack((values, np.r_[values[0, :20], np.full(21, np.nan)]))
                local, full, intercept = tree_contributions(self.tree, self.cat, values)
                self.assertEqual(local.shape, (61, 21))
                self.assertEqual(full.shape, (61, 42))
                self.assertNotEqual(intercept, 0.0)
                local_margin = self.tree.inplace_predict(
                    values[:, :20], predict_type="margin"
                )
                full_margin = self.cat.predict(values, prediction_type="RawFormulaVal")
                np.testing.assert_allclose(
                    local.sum(1), local_margin, rtol=1e-5, atol=2e-5
                )
                np.testing.assert_allclose(
                    full.sum(1), full_margin, rtol=1e-5, atol=2e-5
                )
            finally:
                close_reads(prepared["reads"])

    def test_read_removal_matches_recomputed_input(self):
        first, neighbor = list(islice(iter_sites(HERE / "example/input.json"), 2))
        transcript, position, kmer, raw = first
        self.assertEqual(transcript, neighbor[0])
        for identifier, count in [
            (transcript, 6),
            ("unmapped_transcript", 2),
            ("unmapped_transcript", 1),
        ]:
            with self.subTest(
                transcript=identifier, reads=count
            ), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                record = {
                    identifier: {
                        position: {kmer: raw[:count].tolist()},
                        neighbor[1]: {neighbor[2]: neighbor[3][:4].tolist()},
                    }
                }
                source = folder / "sample.json"
                source.write_text(json.dumps(record) + "\n", encoding="utf-8")
                prepared = prepare_sites(
                    [(0, source, None)], folder / "complete", self.reference
                )
                try:
                    values = site_matrix(self.network, prepared, self.scalers, rows=[0])
                    without, responses, normalized = read_removal_predictions(
                        self.network,
                        prepared,
                        self.scalers,
                        self.tree,
                        self.cat,
                        0,
                        values[0],
                    )
                    np.testing.assert_allclose(
                        responses.mean(0), values[0, :8], rtol=1e-5, atol=1e-6
                    )
                    self.assertEqual(normalized.shape, (count, 9))
                    self.assertEqual(without.shape, (count if count > 1 else 0, 2))
                    fixed_context = values[0, 20:].copy()
                finally:
                    close_reads(prepared["reads"])
                for removed in range(count if count > 1 else 0):
                    changed = copy.deepcopy(record)
                    del changed[identifier][position][kmer][removed]
                    source.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                    smaller = prepare_sites(
                        [(0, source, None)],
                        folder / f"removed-{removed}",
                        self.reference,
                    )
                    try:
                        rebuilt = site_matrix(
                            self.network, smaller, self.scalers, rows=[0]
                        )
                        direct = tree_predictions(self.tree, self.cat, rebuilt)
                        np.testing.assert_allclose(
                            rebuilt[0, 20:], fixed_context, rtol=1e-6, atol=1e-6
                        )
                        if count == 2:
                            np.testing.assert_array_equal(rebuilt[0, 17:20], 0.0)
                        np.testing.assert_allclose(
                            without[removed], direct[0], rtol=1e-6, atol=1e-7
                        )
                    finally:
                        close_reads(smaller["reads"])

    def test_portable_html_matches_example_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output = folder / "sample.json.gz", folder / "explanation.html"
            with gzip.open(source, "wb") as stream:
                stream.write((HERE / "example/input.json").read_bytes())
            expected = pd.read_csv(
                HERE / "example/expected_predictions.csv",
                dtype={"transcript_position": str},
            ).iloc[0]
            scores = explain_site(
                source,
                expected.transcript_id,
                expected.transcript_position,
                output,
                HERE / "model",
                work_dir=folder / "scratch",
            )
            score = combine_predictions(scores[None, :])[0]
            self.assertAlmostEqual(score, expected.score, places=6)
            document = output.read_text(encoding="utf-8")
            self.assertIn(f"<strong>{score:.6f}</strong>", document)
            self.assertIn("0.55 ×", document)
            self.assertIn("0.45 ×", document)
            self.assertIn("All 41 feature values and contributions", document)
            self.assertIn("Not used", document)
            self.assertIn("Response 7", document)
            self.assertNotIn(str(folder), document)


if __name__ == "__main__":
    unittest.main()
