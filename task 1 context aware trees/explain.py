"""Explain a site score through native TreeSHAP and one-read removal comparisons."""
import argparse
import html
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from catboost import Pool
from scipy.special import expit

from context_trees.data import close_reads, prepare_sites
from context_trees.features import MOTIF_COLUMNS
from context_trees.model import (RECIPE, ReadBags, combine_predictions, encode_reads,
                                 feature_names, load_model, site_matrix, tree_predictions)


def tree_contributions(tree, cat, values, threads=3):
    """Return each view's raw-margin contributions, including its own baseline."""
    local = tree.predict(xgb.DMatrix(values[:, :tree.num_features()], nthread=threads),
                         pred_contribs=True)
    full = cat.get_feature_importance(Pool(values), type='ShapValues', thread_count=threads)
    intercept = float(cat.get_metadata().get('site_intercept', 0.))
    reconstructed = np.column_stack((expit(local.sum(axis=1)),
                                     expit(full.sum(axis=1) + intercept)))
    if not np.allclose(reconstructed, tree_predictions(tree, cat, values), rtol=1e-5, atol=2e-6):
        raise ValueError('Feature contributions do not reconstruct the saved model scores.')
    return local, full, intercept


@torch.inference_mode()
def read_removal_predictions(network, prepared, scalers, tree, cat, row, values):
    """Remove each read independently, updating every affected local feature."""
    data = ReadBags(prepared['reads'], prepared['sites'], scalers)
    reads, sequence, counts, _ = data.batch(np.array([row]), 'cpu')
    network.cpu().eval()
    responses = encode_reads(network.encoder, reads, sequence, counts).numpy()
    n, width = responses.shape
    if n == 1:
        return np.empty((0, 2)), responses, reads.numpy()

    remaining = np.repeat(np.asarray(values, dtype=np.float32).reshape(1, -1), n, axis=0)
    response64 = responses.astype(np.float64)
    remaining[:, :width] = (response64.sum(0) - response64) / (n - 1)
    coverage = width + len(MOTIF_COLUMNS)
    remaining[:, coverage] = (np.log1p(n - 1) - scalers['count_mean'][0]) / scalers['count_scale'][0]
    currents = prepared['reads'][data.starts[row]:data.stops[row]][:, [2, 5, 8]].astype(np.float64)
    centered = currents - currents.mean(0)
    remaining[:, coverage + 1:coverage + 4] = (currents.sum(0) - currents) / (n - 1)
    variance = ((centered * centered).sum(0) - n / (n - 1) * centered * centered) / (n - 1)
    remaining[:, coverage + 4:coverage + 7] = 0 if n == 2 else np.sqrt(np.maximum(variance, 0))
    # Sequence and reference stay fixed. Other-site context excludes this focal site.
    return tree_predictions(tree, cat, remaining), responses, reads.numpy()


def render_explanation(sample, transcript, position, site, matched, values, local, full,
                       intercept, scores, without, responses, normalized, center_current):
    """Build a portable HTML explanation without local paths or external assets."""
    width = responses.shape[1]
    local_count, full_count = len(local) - 1, len(full) - 1
    local_weight, full_weight = RECIPE['local_weight'], RECIPE['full_weight']
    final = combine_predictions(scores[None, :])[0]
    groups = [('Learned read responses', 0, width),
              ('Motif and read count', width, width + 6),
              ('Measured current', width + 6, width + 12),
              ('Transcript and exon location', width + 12, width + 18),
              ('Flanking sequence', width + 18, width + 26),
              ('Motif spacing', width + 26, width + 28),
              ('Other sites on this transcript', width + 28, width + 33)]
    family_rows = ''
    for name, start, stop in groups:
        local_value = f'{local[start:stop].sum():+.4f}' if stop <= local_count else 'Not used'
        family_rows += (f'<tr><th>{name}</th><td>{local_value}</td>'
                        f'<td>{full[start:stop].sum():+.4f}</td></tr>')
    feature_rows = ''
    for j, (name, value) in enumerate(zip(feature_names(width), values)):
        display_value = 'missing' if np.isnan(value) else f'{value:.5g}'
        local_value = f'{local[j]:+.4f}' if j < local_count else 'Not used'
        feature_rows += (f'<tr><th>{html.escape(name)}</th><td>{display_value}</td>'
                         f'<td>{local_value}</td><td>{full[j]:+.4f}</td></tr>')

    if len(without):
        removed_score = combine_predictions(without)
        change = final - removed_score
        largest = np.argsort(-np.abs(change), kind='stable')[:12]
        read_rows = ''.join(
            f'<tr><th>{j + 1}</th><td>{center_current[j]:.4g}</td>'
            f'<td>{without[j, 0]:.6f}</td><td>{without[j, 1]:.6f}</td>'
            f'<td>{removed_score[j]:.6f}</td><td>{change[j]:+.6f}</td></tr>'
            for j in largest)
        channel_names = [f'{window}: {channel}' for window in ['Left', 'Centre', 'Right']
                         for channel in ['log dwell', 'signal SD', 'mean current']]
        headers = ''.join(f'<th>{name}</th>' for name in channel_names)
        all_rows = ''.join(
            f'<tr><th>{j + 1}</th>' + ''.join(f'<td>{v:.3f}</td>' for v in normalized[j])
            + f'<td>{removed_score[j]:.6f}</td><td>{change[j]:+.6f}</td></tr>'
            for j in range(len(change)))
        read_section = f'''<h2>Which reads move this prediction?</h2>
<p>Remove one read, recompute the learned summaries, read count and current summaries,
then predict again. Other-site context and reference features stay fixed for this site's score.
The change is the full score minus the score after removing that read. A positive change
means including the read raises the final prediction.</p>
<div class="scroll"><table><thead><tr><th>Read number</th><th>Centre current</th>
<th>Local score without read</th><th>Full score without read</th>
<th>Final score without read</th><th>Score change</th></tr></thead><tbody>{read_rows}</tbody></table></div>
<p>Up to 12 reads with the largest absolute changes are shown. Read numbers start at 1
and follow the original input order. Each comparison removes one read from the complete
original site; removals are not cumulative.</p>
<details><summary>All {len(change)} reads: normalized measurements and score changes</summary>
<p>Zero is the training mean for that five-base sequence; one unit is one training
standard deviation. Dwell time is log-transformed before normalization.</p>
<div class="scroll"><table><thead><tr><th>Read</th>{headers}<th>Final score without read</th>
<th>Score change</th></tr></thead><tbody>{all_rows}</tbody></table></div></details>'''
    else:
        read_section = '''<h2>Read-level explanation</h2><p>This site has one read.
Removing it would leave no measurements, so a read-removal score is undefined.</p>'''
    response_headers = ''.join(f'<th>Response {j}</th>' for j in range(width))
    response_rows = ''.join(
        f'<tr><th>{j + 1}</th>' + ''.join(f'<td>{v:.5f}</td>' for v in response) + '</tr>'
        for j, response in enumerate(responses))
    response_section = f'''<details><summary>The {width} encoder responses for each read</summary>
<p>Each column is averaged across all reads to form one learned site feature. These
responses are not probabilities and have no fixed biological names across retraining.</p>
<div class="scroll"><table><thead><tr><th>Read</th>{response_headers}</tr></thead>
<tbody>{response_rows}</tbody></table></div></details>'''

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Site prediction explanation</title>
<style>body{{font:16px/1.6 Arial,sans-serif;color:#183a4a;max-width:1100px;margin:40px auto;padding:0 24px}}
h1,h2{{line-height:1.2}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 12px;border-bottom:1px solid #d8e2e6;text-align:right}}th:first-child{{text-align:left}}
thead,footer{{background:#edf5f5}}.score{{font-size:24px;color:#146c70}}.scroll{{overflow:auto}}
details{{margin-top:30px}}summary{{cursor:pointer;font-weight:bold}}footer{{padding:15px;margin-top:25px}}</style></head>
<body><h1>How this site received its score</h1>
<p>Sample input: {html.escape(sample)}<br>
{html.escape(transcript)} · position {html.escape(str(position))} · {site.n_reads} reads · {html.escape(site.kmer)}<br>
Reference context: {'matched' if matched else 'missing'}.</p>
<p class="score">Final score = {local_weight:.2f} × {scores[0]:.6f} + {full_weight:.2f} × {scores[1]:.6f}
= <strong>{final:.6f}</strong></p>
<p>Local XGBoost uses {local_count} features and full CatBoost uses {full_count}. Each ensemble
starts from its own baseline and adds feature contributions in internal-score (raw-margin)
units. CatBoost then adds its fitted prevalence intercept. Apply sigmoid to each total,
then combine the two probabilities with the fixed weights above.</p>
<h2>Which feature families move each tree score?</h2>
<div class="scroll"><table><thead><tr><th>Feature family</th><th>Local XGBoost</th><th>Full CatBoost</th></tr></thead>
<tbody><tr><th>Tree baseline</th><td>{local[-1]:.4f}</td><td>{full[-1]:.4f}</td></tr>{family_rows}
<tr><th>Separate prevalence intercept</th><td>None</td><td>{intercept:+.4f}</td></tr>
<tr><th>Total margin before sigmoid</th><td>{local.sum():.4f}</td><td>{full.sum() + intercept:.4f}</td></tr>
<tr><th>Score after sigmoid</th><td>{scores[0]:.6f}</td><td>{scores[1]:.6f}</td></tr></tbody></table></div>
<p>A positive contribution raises that ensemble's margin relative to its baseline.
Context features marked “Not used” are absent from the local predictor. The learned
responses are grouped because they jointly describe the reads. Do not average margin
contributions into probability contributions: sigmoid is nonlinear.</p>
<details><summary>All {full_count} feature values and contributions</summary><div class="scroll"><table>
<thead><tr><th>Feature</th><th>Value</th><th>Local XGBoost</th><th>Full CatBoost</th></tr></thead>
<tbody>{feature_rows}</tbody></table></div></details>
{read_section}{response_section}
<footer>Native TreeSHAP uses each ensemble's stored tree-path statistics as its reference.
Contributions describe the fitted model, not causal effects. Read-removal changes are
separate comparisons and are not additive or molecule-level modification labels.
The final score is a model prediction, not a measured modification fraction.
<a href="https://doi.org/10.1038/s42256-019-0138-9">TreeSHAP</a> and
<a href="https://www.jmlr.org/papers/v22/20-1316.html">removal-based explanations</a>
provide the methodological background.</footer></body></html>'''


def explain_site(data, transcript, position, output, model_directory, threads=3,
                 work_dir=None, reference_path=None):
    data, output, model_directory = Path(data), Path(output), Path(model_directory)
    reference_path = Path(reference_path) if reference_path is not None else model_directory / 'reference.joblib'
    if output.resolve() in {data.resolve(), reference_path.resolve()}:
        raise ValueError('Write the explanation outside the input and reference files.')
    if output.resolve().is_relative_to(model_directory.resolve()):
        raise ValueError('Write the explanation outside the fitted model directory.')
    network, scalers, tree, cat = load_model(model_directory, threads)
    reference = joblib.load(reference_path)
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='context_trees_explain_', dir=work_dir) as scratch:
        prepared = prepare_sites([(0, data, None)], scratch, reference)
        try:
            sites = prepared['sites']
            rows = np.flatnonzero((sites.transcript_id == transcript) & (sites.position_key == str(position)))
            if len(rows) != 1:
                raise ValueError('Choose a transcript and position exactly as they appear in the input file.')
            row = int(rows[0])
            # Prepare the complete sample for background; encode only the requested site.
            values = site_matrix(network, prepared, scalers, rows=[row])
            local, full, intercept = tree_contributions(tree, cat, values, threads)
            scores = tree_predictions(tree, cat, values)[0]
            without, responses, normalized = read_removal_predictions(
                network, prepared, scalers, tree, cat, row, values[0])
            site = sites.iloc[row]
            document = render_explanation(data.name, transcript, position, site,
                bool(prepared['reference_matched'][row]), values[0], local[0], full[0], intercept,
                scores, without, responses, normalized,
                prepared['reads'][site.read_start:site.read_stop, 5])
        finally:
            close_reads(prepared['reads'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Complete sample JSON or JSON.gz, as for predict.py.')
    parser.add_argument('--transcript', required=True)
    parser.add_argument('--position', required=True, help='Exact position spelling from the input.')
    parser.add_argument('--output', type=Path, required=True, help='Destination HTML file.')
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'model')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--work-dir', type=Path, help='Optional directory for temporary prepared reads.')
    parser.add_argument('--reference', type=Path, help='Use the same annotation override as predict.py.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Threads must be positive.')
    scores = explain_site(args.data, args.transcript, args.position, args.output, args.model,
                          args.threads, args.work_dir, args.reference)
    final = combine_predictions(scores[None, :])[0]
    print(f'Wrote {args.output.name}: local={scores[0]:.6f}, full={scores[1]:.6f}, final={final:.6f}.')


if __name__ == '__main__':
    main()
