"""Encode reads and combine local XGBoost with full-feature CatBoost."""
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from catboost import CatBoost
from torch import nn
from scipy.special import expit, logit
from scipy.optimize import brentq

from .features import (COMPACT_FLANK_NAMES, MOTIF_COLUMNS, MOTIF_NAMES, GEOMETRY_NAMES, SUMMARY_NAMES,
                      TRANSCRIPT_CONTEXT_NAMES, MOTIF_DISTANCE_NAMES, physical_contexts, sequence_encoding,
                      transcript_signal_context, A_FLANK_NAMES, CURRENT_SPREAD_NAMES)

WIDTH = 8
HIDDEN = 128
EPOCHS = 6
SEED = 4262
RECIPE = dict(name='Context-aware trees', version='1', full_weight=.45, local_weight=.55,
              depth=7, regularization=25,
              xgb_rounds=800, cat_rounds=960, learning_rate=.03,
              context_feature_weight=.2, column_fraction=.5, order_weight=1.,
              binary_partners=4, query_size=1024)


def column_scaler(values):
    mean = values.mean(0)
    scale = np.sqrt(((values - mean) ** 2).mean(0))
    scale[scale < 1e-8] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_scalers(reads, sites, train):
    """Use training reads only; pool each physical five-mer across the three windows."""
    contexts = physical_contexts(sequence_encoding(sites.kmer.to_numpy()))
    starts, stops = sites.read_start.to_numpy(), sites.read_stop.to_numpy()
    total, square, count = np.zeros(9), np.zeros(9), 0
    context_total, context_square = np.zeros((1024, 3)), np.zeros((1024, 3))
    context_count = np.zeros(1024)
    for start in range(0, len(train), 1024):
        rows = train[start:start + 1024]
        values = np.concatenate([reads[starts[i]:stops[i]] for i in rows]).astype(np.float64)
        total += values.sum(0)
        square += (values * values).sum(0)
        count += len(values)
        sizes = stops[rows] - starts[rows]
        offsets = np.r_[0, np.cumsum(sizes)[:-1]]
        site_total = np.add.reduceat(values, offsets, axis=0)
        site_square = np.add.reduceat(values * values, offsets, axis=0)
        for window in range(3):
            key = contexts[rows, window]
            context_count += np.bincount(key, weights=sizes, minlength=1024)
            for channel in range(3):
                column = 3 * window + channel
                context_total[:, channel] += np.bincount(key, weights=site_total[:, column], minlength=1024)
                context_square[:, channel] += np.bincount(key, weights=site_square[:, column], minlength=1024)
    pooled_mean = total.reshape(3, 3).sum(0) / (3 * count)
    pooled_variance = square.reshape(3, 3).sum(0) / (3 * count) - pooled_mean ** 2
    mean = np.broadcast_to(pooled_mean, (1024, 3)).copy()
    variance = np.broadcast_to(pooled_variance, (1024, 3)).copy()
    seen = context_count > 0
    mean[seen] = context_total[seen] / context_count[seen, None]
    variance[seen] = context_square[seen] / context_count[seen, None] - mean[seen] ** 2
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale < 1e-8] = 1
    count_mean, count_scale = column_scaler(np.log1p(sites.n_reads.to_numpy()[train, None]))
    return dict(read_mean=mean.astype(np.float32), read_scale=scale.astype(np.float32),
                count_mean=count_mean, count_scale=count_scale)


class ReadBags:
    """A compact index into all original reads, with training-fitted normalization."""
    def __init__(self, reads, sites, scalers):
        self.reads, self.scalers = reads, scalers
        self.starts, self.stops = sites.read_start.to_numpy(), sites.read_stop.to_numpy()
        self.counts = self.stops - self.starts
        if not np.array_equal(self.counts, sites.n_reads.to_numpy()) or np.any(self.counts < 1):
            raise ValueError('Every read interval must agree with its nonempty site.')
        self.sequence = sequence_encoding(sites.kmer.to_numpy())
        self.contexts = physical_contexts(self.sequence)
        self.coverage = ((np.log1p(self.counts[:, None]) - scalers['count_mean']) /
                         scalers['count_scale']).astype(np.float32)

    def batch(self, rows, device):
        counts = self.counts[rows]
        values = np.concatenate([self.reads[self.starts[i]:self.stops[i]] for i in rows])
        mean = self.scalers['read_mean'][self.contexts[rows]].reshape(-1, 9)
        scale = self.scalers['read_scale'][self.contexts[rows]].reshape(-1, 9)
        values = (values - np.repeat(mean, counts, axis=0)) / np.repeat(scale, counts, axis=0)
        arrays = values, self.sequence[rows], counts, self.coverage[rows]
        return tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays)


def read_batches(rows, counts, batch_size=128, read_budget=65536):
    """Split computation without discarding reads or changing the optimizer batch."""
    start, maximum = 0, 0
    for stop, row in enumerate(rows):
        candidate = max(maximum, int(counts[row]))
        if stop > start and (stop - start >= batch_size or candidate * (stop - start + 1) > read_budget):
            yield rows[start:stop]
            start, maximum = stop, 0
        maximum = max(maximum, int(counts[row]))
    if start < len(rows):
        yield rows[start:]


def make_encoder(width=WIDTH):
    return nn.Sequential(nn.Linear(37, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN),
                         nn.SiLU(), nn.Linear(HIDDEN, width))


def encode_reads(encoder, reads, sequence, counts):
    return encoder(torch.cat((reads, torch.repeat_interleave(sequence, counts, dim=0)), dim=1))


class LearningModel(nn.Module):
    """One read encoder and one temporary site head for both supervision types."""
    def __init__(self, with_site_head=False, width=WIDTH):
        super().__init__()
        self.encoder = make_encoder(width)
        if with_site_head:
            self.site_head = nn.Sequential(nn.Linear(width + 5, 128), nn.SiLU(),
                                           nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def summaries(self, reads, sequence, counts):
        encoded = encode_reads(self.encoder, reads, sequence, counts)
        return torch.segment_reduce(encoded, reduce='mean', lengths=counts)

    def site(self, reads, sequence, counts, coverage):
        # Coverage stays a tree feature, but cannot explain ordered supervision.
        values = torch.cat((self.summaries(reads, sequence, counts), sequence[:, MOTIF_COLUMNS]), dim=1)
        return self.site_head(values).squeeze(1)


def ordered_groups(sites, rows):
    """Indices of complete groups; numeric levels convey order, not probabilities."""
    selected = sites.iloc[rows]
    if selected[['order_group', 'order_level']].isna().any().any():
        raise ValueError('Ordered observations need a group and a finite order level.')
    if not np.isfinite(selected.order_level.to_numpy(dtype=float)).all():
        raise ValueError('Order levels must be finite.')
    result = []
    for key, group in selected.groupby('order_group', sort=False):
        if len(group) != int(sites.order_group.eq(key).sum()):
            raise ValueError('Keep every observation of an ordered group together.')
        if group.order_level.nunique() < 2 or group.kmer.nunique() != 1:
            raise ValueError('Each ordered group needs at least two levels at one common sequence context.')
        result.append(group.index.to_numpy(dtype=int))
    return result


def ordering_loss(scores, levels):
    """Mean logistic pair loss; ties are not comparisons."""
    lower, higher = torch.where(levels[:, None] < levels[None, :])
    if not lower.numel():
        raise ValueError('An ordered group needs at least two different levels.')
    return nn.functional.softplus(scores[lower] - scores[higher]).mean()


def fit_encoder(prepared, ordered=None, train=None, ordered_train=None,
                device='cpu', threads=3, seed=SEED, epochs=EPOCHS, rank_weight=1., width=WIDTH):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    sites = prepared['sites']
    train = np.arange(len(sites)) if train is None else np.asarray(train, dtype=int)
    y = sites.label.to_numpy(dtype=np.float32)
    if not np.array_equal(np.unique(y[train]), [0, 1]):
        raise ValueError('Encoder training needs both binary classes.')
    if rank_weight < 0 or epochs < 1 or width < 1:
        raise ValueError('Use nonnegative rank weight and positive epochs and width.')
    scalers = fit_scalers(prepared['reads'], sites, train)
    data = ReadBags(prepared['reads'], sites, scalers)
    groups, ordered_data = [], None
    if ordered is not None:
        order_sites = ordered['sites']
        ordered_train = np.arange(len(order_sites)) if ordered_train is None else np.asarray(ordered_train, dtype=int)
        groups = ordered_groups(order_sites, ordered_train)
        ordered_data = ReadBags(ordered['reads'], order_sites, scalers)
    prevalence = y[train].mean()
    model = LearningModel(with_site_head=True, width=width).to(device)
    positive_weight = torch.tensor((1 - prevalence) / prevalence, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng, history = np.random.default_rng(seed), []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(train)
        binary_total, rank_total, steps = 0., 0., 0
        for start in range(0, len(order), 256):
            rows = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            for part in read_batches(rows, data.counts, batch_size=256):
                logits = model.site(*data.batch(part, device))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.as_tensor(y[part], device=device), pos_weight=positive_weight)
                loss = loss / (2 * (1 - prevalence))
                (loss * len(part) / len(rows)).backward()
                binary_total += float(loss.detach()) * len(part)
            if groups and rank_weight:
                chosen = rng.choice(len(groups), size=min(4, len(groups)), replace=False)
                for group_id in chosen:
                    group_rows = groups[group_id]
                    scores = torch.cat([model.site(*ordered_data.batch(part, device))
                                        for part in read_batches(group_rows, ordered_data.counts)])
                    levels = torch.as_tensor(order_sites.order_level.to_numpy()[group_rows], device=device)
                    loss = ordering_loss(scores, levels)
                    (rank_weight * loss / len(chosen)).backward()
                    rank_total += float(loss.detach()) / len(chosen)
            optimizer.step()
            steps += 1
        history.append(dict(epoch=epoch, binary_loss=binary_total / len(train),
                            ordered_loss=rank_total / steps, rank_weight=rank_weight,
                            optimizer_updates=steps, seconds=time.perf_counter() - started))
        print(history[-1], flush=True)
    return model.eval(), scalers, history


@torch.inference_mode()
def site_matrix(network, prepared, scalers, device='cpu', batch_size=128, rows=None):
    data = ReadBags(prepared['reads'], prepared['sites'], scalers)
    context = transcript_signal_context(prepared['sites'], prepared['summary'], scalers,
                                        prepared['reference_matched'])
    rows = np.arange(len(data.counts)) if rows is None else np.asarray(rows, dtype=int)
    learned = np.empty((len(rows), network.encoder[-1].out_features), dtype=np.float32)
    network.to(device).eval()
    for indices in read_batches(np.arange(len(rows)), data.counts[rows], batch_size):
        reads, sequence, counts, _ = data.batch(rows[indices], device)
        learned[indices] = network.summaries(reads, sequence, counts).cpu().numpy()
    groups = [learned, data.sequence[rows][:, MOTIF_COLUMNS], data.coverage[rows],
              prepared['summary'][rows][:, [2, 5, 8]], prepared['summary'][rows, 9:12],
              prepared['geometry'][rows], prepared['flanks'][rows], prepared['motif_distances'][rows], context[rows]]
    return np.column_stack(groups).astype(np.float32)


def order_pairs(sites):
    """Unequal-level comparisons; weights give each observed site equal total degree."""
    if sites is None or len(sites) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)
    sites = sites.reset_index(drop=True)
    lower, higher = [], []
    degree = np.zeros(len(sites))
    levels = sites.order_level.to_numpy()
    for rows in ordered_groups(sites, np.arange(len(sites))):
        level = levels[rows]
        a, b = np.where(level[:, None] < level[None, :])
        lower.extend(rows[a])
        higher.extend(rows[b])
        degree[rows] = (level[:, None] != level[None, :]).sum(1)
    lower, higher = np.asarray(lower, dtype=int), np.asarray(higher, dtype=int)
    order = np.lexsort((higher, lower))
    lower, higher = lower[order], higher[order]
    return lower, higher, (1/degree[lower] + 1/degree[higher]).astype(np.float32)


def ranking_queries(labels, ordered_sites, seed, partners=4, query_size=1024):
    """Valid comparisons of all binary observations and complete ordered groups."""
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 1))
    negative = rng.permutation(np.flatnonzero(labels == 0))
    count = min(len(positive), len(negative), max(1, int(np.ceil(len(labels)/query_size))))
    blocks, winners, losers = [], [], []
    for pos, neg in zip(np.array_split(positive, count), np.array_split(negative, count)):
        blocks.append(np.r_[pos, neg])
        winners.extend(np.repeat(pos, partners))
        losers.extend(rng.choice(neg, partners*len(pos)))
        winners.extend(rng.choice(pos, partners*len(neg)))
        losers.extend(np.repeat(neg, partners))
    winners, losers = np.asarray(winners, dtype=int), np.asarray(losers, dtype=int)
    degree = np.bincount(np.r_[winners, losers], minlength=len(labels))
    weights = (1/degree[winners] + 1/degree[losers]).astype(np.float32)
    low, high, ordered_weight = order_pairs(ordered_sites)
    if ordered_sites is not None and len(ordered_sites):
        sites = ordered_sites.reset_index(drop=True)
        blocks.extend(rows+len(labels) for rows in ordered_groups(sites, np.arange(len(sites))))
    permutation = np.concatenate(blocks)
    inverse = np.empty(len(permutation), dtype=int)
    inverse[permutation] = np.arange(len(permutation))
    pairs = np.column_stack((inverse[np.r_[winners, high+len(labels)]],
                             inverse[np.r_[losers, low+len(labels)]]))
    groups = np.repeat(np.arange(len(blocks)), [len(block) for block in blocks])
    return permutation, groups, pairs, weights, ordered_weight


def fit_trees(values, labels, ordered_values=None, ordered_sites=None,
              device='cpu', threads=3, seed=SEED, recipe=None, input_names=None):
    """Fit local XGBoost and full CatBoost with binary and ordering evidence."""
    settings = dict(RECIPE, **(recipe or {}))
    labels = np.asarray(labels, dtype=np.float32)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('Tree training needs both binary classes.')
    if ordered_values is None:
        ordered_values = np.empty((0, values.shape[1]), dtype=np.float32)
    if len(ordered_values) != (0 if ordered_sites is None else len(ordered_sites)):
        raise ValueError('Ordered feature rows and group labels must align.')
    if settings['order_weight'] < 0 or settings['binary_partners'] < 1 or settings['query_size'] < 2:
        raise ValueError('Use nonnegative ordering weight and positive comparison counts.')
    names = feature_names(values.shape[1] - 33) if input_names is None else list(input_names)
    if len(names) != values.shape[1]:
        raise ValueError('Feature names must match the input columns.')
    lower_priority = ['standardized_log_read_count', *GEOMETRY_NAMES, *TRANSCRIPT_CONTEXT_NAMES[-2:]]
    weights = np.array([settings['context_feature_weight'] if name in lower_priority else 1.
                        for name in names], dtype=np.float32)
    parameters = dict(objective='reg:squarederror', tree_method='hist', max_depth=settings['depth'],
                      eta=settings['learning_rate'], reg_lambda=settings['regularization'],
                      colsample_bynode=settings['column_fraction'], nthread=threads, seed=seed, device=device,
                      base_score=float(logit(labels.mean())), disable_default_eval_metric=True)
    low, high, pair_weight = order_pairs(ordered_sites)
    low, high = low+len(labels), high+len(labels)
    pair_weight *= settings['order_weight']
    def objective(margin, matrix):
        gradient, curvature = np.zeros_like(margin), np.zeros_like(margin)
        probability = expit(margin[:len(labels)])
        gradient[:len(labels)] = probability-labels
        curvature[:len(labels)] = probability*(1-probability)
        reversal = expit(margin[low]-margin[high])
        pair_gradient = pair_weight*reversal
        bound = 2*pair_weight*reversal*(1-reversal)
        np.add.at(gradient, low, pair_gradient)
        np.add.at(gradient, high, -pair_gradient)
        np.add.at(curvature, low, bound)
        np.add.at(curvature, high, bound)
        return gradient, np.maximum(curvature, 1e-6)
    combined = np.vstack((values, ordered_values))
    local_width = values.shape[1] - 21
    if local_width < 13 or names[local_width-1] != 'across_read_sd_right_current_mean':
        raise ValueError('The feature matrix must contain the documented local then context columns.')
    matrix = xgb.DMatrix(combined[:, :local_width], nthread=threads,
                         feature_weights=weights[:local_width])
    tree = xgb.train(parameters, matrix, num_boost_round=settings['xgb_rounds'], obj=objective)
    tree.set_param({'device': 'cpu'})
    tree.set_attr(prediction_link='sigmoid of raw margin',
                  training_objective='binary site log loss plus within-group pair ordering')
    permutation, groups, pairs, binary_weight, ordered_weight = ranking_queries(
        labels, ordered_sites, seed, settings['binary_partners'], settings['query_size'])
    target = np.r_[labels, np.zeros(len(ordered_values))][permutation]
    cat = CatBoost(dict(iterations=settings['cat_rounds'], depth=settings['depth'],
                       learning_rate=settings['learning_rate'], l2_leaf_reg=settings['regularization'],
                       bootstrap_type='Bernoulli', subsample=.8, border_count=254,
                       boosting_type='Plain', grow_policy='SymmetricTree', nan_mode='Min',
                       random_seed=seed, thread_count=threads, allow_writing_files=False,
                       loss_function='PairLogit', task_type='CPU', feature_weights=weights.tolist()))
    cat.fit(combined[permutation], target, group_id=groups, pairs=pairs,
            pairs_weight=np.r_[binary_weight, settings['order_weight']*ordered_weight], verbose=False)
    # Pair ordering has no absolute intercept. Match the binary training prevalence.
    margin = cat.predict(values, prediction_type='RawFormulaVal')
    intercept = brentq(lambda shift: expit(margin+shift).mean()-labels.mean(), -100, 100)
    cat.get_metadata()['site_intercept'] = str(intercept)
    return tree, cat

def tree_predictions(tree, cat, values):
    intercept = float(cat.get_metadata().get('site_intercept', 0.))
    prediction = np.column_stack((expit(tree.inplace_predict(values[:, :tree.num_features()], predict_type='margin')),
                                  expit(cat.predict(values, prediction_type='RawFormulaVal')+intercept)))
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError('The predictors did not return finite scores in [0, 1].')
    return prediction



def combine_predictions(prediction):
    """Columns are local XGBoost and full CatBoost, in that order."""
    return .55 * prediction[:, 0] + .45 * prediction[:, 1]


def feature_names(width=WIDTH):
    return ([f'learned_read_mean_{i}' for i in range(width)] 
            + MOTIF_NAMES + ['standardized_log_read_count'] + SUMMARY_NAMES + CURRENT_SPREAD_NAMES
            + GEOMETRY_NAMES + COMPACT_FLANK_NAMES + A_FLANK_NAMES + MOTIF_DISTANCE_NAMES
            + TRANSCRIPT_CONTEXT_NAMES)


def save_model(directory, network, scalers, tree, cat):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(width=network.encoder[-1].out_features,
                      state={k: v.detach().cpu() for k, v in network.state_dict().items()
                             if not k.startswith('site_head.')},
                      scalers={k: torch.as_tensor(v) for k, v in scalers.items()})
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    with zipfile.ZipFile(buffer) as source, zipfile.ZipFile(directory / 'encoder.pt', 'w') as target:
        for name in source.namelist():
            if not name.endswith('/.data/serialization_id'):
                target.writestr(zipfile.ZipInfo(name), source.read(name))
    tree.save_model(directory / 'xgboost.ubj')
    # Native parameter inspection also reads output_options.
    metadata = cat.get_metadata()
    for name in ('catboost_version_info', 'train_finish_time', 'model_guid', 'training'):
        if name in metadata:
            del metadata[name]
    cat.save_model(str(directory / 'catboost.cbm'))


def load_model(directory, threads=3):
    directory = Path(directory)
    checkpoint = torch.load(directory / 'encoder.pt', map_location='cpu', weights_only=True)
    network = LearningModel(width=checkpoint['width'])
    network.load_state_dict(checkpoint['state'])
    scalers = {k: v.numpy() for k, v in checkpoint['scalers'].items()}
    tree = xgb.Booster(model_file=directory / 'xgboost.ubj')
    tree.set_param({'device': 'cpu', 'nthread': threads})
    cat = CatBoost(dict(thread_count=threads))
    cat.load_model(str(directory / 'catboost.cbm'))
    torch.set_num_threads(threads)
    return network.eval(), scalers, tree, cat
