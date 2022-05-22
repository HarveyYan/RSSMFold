import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import diags
from sklearn.metrics import average_precision_score
from collections import defaultdict

from RSSMFold.lib.utils import matrix_sampling, matrix_sampling_beam_search_v2
from RSSMFold.lib.diff_lp_utils import postprocess_vanilla


def discretize_contact_map(contact_map, batch_x, batch_len, map_threshold=0.9, nonoverlapping=True,
                           constrained_pairing=False, matrix_sampling_mode='greedy', **kwargs):
    contact_map_discretized = []
    cumsum_triu_size = 0

    t_device = kwargs.get('device')
    min_hairpin_span = kwargs.get('min_hairpin_span', 1)
    ret_sampled_pairs = kwargs.get('ret_sampled_pairs', False)
    all_sampled_pairs = []

    for i, seq_len in enumerate(batch_len):
        triu_size = seq_len * (seq_len + 1) // 2
        pred = contact_map[cumsum_triu_size: cumsum_triu_size + triu_size]

        if nonoverlapping:
            pred_mat = torch.zeros(seq_len, seq_len).to(t_device)
            row, col = torch.triu_indices(seq_len, seq_len)
            pred_mat[row, col] = ((pred[:, 0] > map_threshold).float() * pred[:, 0]).to(t_device)
            pred_mat *= torch.as_tensor(
                1. - diags([1] * (2 * min_hairpin_span + 1), list(range(-min_hairpin_span, min_hairpin_span + 1)),
                           shape=(seq_len, seq_len), dtype=np.float32).toarray()).to(t_device)

            seq = batch_x[i][:seq_len]
            idx = torch.where((seq != 0.) & (seq != 1.) & (seq != 2.) & (seq != 3.))[0]
            pred_mat[idx, :] = 0.
            pred_mat[:, idx] = 0.

            if constrained_pairing:
                # constrained pairing
                base_a = (seq == 0.).float()
                base_c = (seq == 1.).float()
                base_g = (seq == 2.).float()
                base_u = (seq == 3.).float()
                au = torch.matmul(base_a[:, None], base_u[None, :])
                au_ua = au + au.t()
                cg = torch.matmul(base_c[:, None], base_g[None, :])
                cg_gc = cg + cg.t()
                ug = torch.matmul(base_u[:, None], base_g[None, :])
                ug_gu = ug + ug.t()
                m = au_ua + cg_gc + ug_gu
                pred_mat *= m.to(t_device)

            if matrix_sampling_mode == 'greedy':
                sampled_pairs = matrix_sampling(pred_mat, np.inf, True, True, 'torch')
            elif matrix_sampling_mode == 'greedy_bm':
                sampled_pairs, _ = matrix_sampling_beam_search_v2(pred_mat, 5000)
            elif matrix_sampling_mode == 'greedy_bm_shuffled':
                max_score = 0
                sampled_pairs = []
                for _ in range(5):
                    pairs, score = matrix_sampling_beam_search_v2(pred_mat, 1000, sort_prob=False)
                    if score > max_score:
                        max_score = score
                        sampled_pairs = pairs
            elif matrix_sampling_mode == 'diff_lp':
                pred_mat_original = torch.zeros(seq_len, seq_len).to(t_device)
                pred_mat_original[row, col] = pred[:, 0].to(t_device)
                pred_mat_original[col, row] = pred[:, 0].to(t_device)
                diff_lp_ret = postprocess_vanilla(
                    torch.logit(pred_mat_original), seq_len, 0.01, 0.1, 50, 1., True, threshold=map_threshold)
                # diff_lp_ret = (diff_lp_ret > 0.5).float()
                row_idx, col_idx = torch.nonzero(diff_lp_ret, as_tuple=True)
                row_idx = row_idx.cpu().numpy()
                col_idx = col_idx.cpu().numpy()
                sampled_pairs = [[from_idx, to_idx] for from_idx, to_idx in zip(row_idx, col_idx) if from_idx < to_idx]
            elif matrix_sampling_mode == 'conflict_sampling':
                sampled_pairs = conflict_sampling(pred_mat)
            elif matrix_sampling_mode == 'no_sampling':
                # no sampling
                sampled_pairs = torch.nonzero(pred_mat).cpu().numpy()
            elif matrix_sampling_mode == 'probknot':
                pred_mat = pred_mat + pred_mat.T
                row_max = (pred_mat == torch.max(pred_mat, dim=0)[0][:, None]) & (pred_mat > 0.)
                col_max = (pred_mat == torch.max(pred_mat, dim=1)[0][None, :]) & (pred_mat > 0.)
                ret = row_max & col_max
                ret[np.tril_indices(seq_len)] = False
                sampled_pairs = torch.nonzero(ret).cpu().numpy()
            else:
                raise ValueError('unknown matrix_sampling_mode', matrix_sampling_mode)

            all_sampled_pairs.append(sampled_pairs)
            idx_in_flattened_triu = [(2 * seq_len - i + 1) * i // 2 + j - i for i, j in sampled_pairs]
            pred = torch.zeros(seq_len * (seq_len + 1) // 2, 1)
            pred[idx_in_flattened_triu] = 1.
        else:
            pred = (pred > map_threshold).float()

        contact_map_discretized.append(pred)
        cumsum_triu_size += triu_size

    if ret_sampled_pairs:
        return all_sampled_pairs
    else:
        return torch.cat(contact_map_discretized, dim=0)


def evaluate_contact_matrix_ap_score(batch_pred, batch_label, batch_len, batch_pos_labels):
    if isinstance(batch_pred, torch.Tensor):
        batch_pred = batch_pred.detach().cpu().numpy()
        batch_label = batch_label.detach().cpu().numpy()

    cumsum_triu_size = 0
    batch_ap_score = []
    for seq_len, nb_pos_labels in zip(batch_len, batch_pos_labels):
        triu_size = seq_len * (seq_len + 1) // 2
        pred = batch_pred[cumsum_triu_size: cumsum_triu_size + triu_size]
        label = batch_label[cumsum_triu_size: cumsum_triu_size + triu_size]

        if nb_pos_labels > 0:
            batch_ap_score.append(average_precision_score(label, pred))

        cumsum_triu_size += triu_size

    return batch_ap_score


def evaluate_contact_matrix_f1(
        batch_pred, batch_label, batch_len, batch_pos_labels, batch_neg_labels,
        allow_shift=False, **kwargs):
    cumsum_triu_size = 0
    batch_recall, batch_precision, batch_f1, batch_mcc = [], [], [], []
    for seq_len, nb_pos_labels, nb_neg_labels in zip(batch_len, batch_pos_labels, batch_neg_labels):
        triu_size = seq_len * (seq_len + 1) // 2
        pred = batch_pred[cumsum_triu_size: cumsum_triu_size + triu_size]
        label = batch_label[cumsum_triu_size: cumsum_triu_size + triu_size]

        nb_pos_preds = pred.sum().item()
        if allow_shift is False:
            tp = torch.sum(pred * label).item()
            fp = nb_pos_preds - tp
            fn = nb_pos_labels - tp
        else:
            g_device = kwargs.get('device')
            pred_mat = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32).to(g_device)
            seq_len = int(seq_len)
            row, col = torch.triu_indices(seq_len, seq_len)
            pred_mat[0, 0, row, col] = pred[:, 0]

            shift_kernel = torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0]]).unsqueeze(0).unsqueeze(0).to(g_device)

            pred_filtered = F.conv2d(pred_mat, shift_kernel, bias=None, padding=1, stride=(1, 1))

            fn = torch.sum(label.squeeze() > pred_filtered[0, 0, row, col]).item()
            tp = nb_pos_labels - fn
            fp = nb_pos_preds - tp

        if nb_pos_labels == 0:
            batch_recall.append(0.)
        else:
            batch_recall.append(tp / (tp + fn))
        if nb_pos_preds == 0:
            batch_precision.append(0.)
        else:
            batch_precision.append(tp / (tp + fp))
        if nb_pos_labels == 0 and nb_pos_preds == 0:
            batch_f1.append(0.)
        else:
            batch_f1.append(2 * tp / (2 * tp + fp + fn))
        if nb_pos_labels == 0 or nb_pos_preds == 0:
            batch_mcc.append(0.)
        else:
            s = (tp + fn) / triu_size
            p = (tp + fp) / triu_size
            batch_mcc.append((tp / triu_size - s * p) / np.sqrt(p * s * (1 - p) * (1 - s)))

        cumsum_triu_size += triu_size
    return batch_recall, batch_precision, batch_f1, batch_mcc


def find_conflicts(pairs):
    idx_to_pairs = defaultdict(list)
    for pair in pairs:
        idx_to_pairs[pair[0]].append(pair)
        idx_to_pairs[pair[1]].append(pair)

    conflict_pairs = []
    for idx, pairs in idx_to_pairs.items():
        if len(pairs) > 1:
            conflict_pairs.append(pairs)

    return conflict_pairs


def conflict_sampling(matrix, return_values=False):
    row, col = torch.nonzero(matrix, as_tuple=True)
    all_values = matrix[row, col].cpu().numpy()
    all_pairs = list(zip(row.cpu().numpy(), col.cpu().numpy()))
    pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}

    conflict_pairs = find_conflicts(all_pairs)

    while len(conflict_pairs) > 0:

        pair_idx_to_remove = []
        for pairs in conflict_pairs:
            pair_values = []
            pair_idx = []
            for pair in pairs:
                idx = pair_to_idx[pair]
                pair_idx.append(idx)
                pair_values.append(all_values[idx])
            pair_idx_to_remove.append(pair_idx[int(np.argmin(pair_values))])

        all_values = np.delete(all_values, pair_idx_to_remove)
        all_pairs = [pair for i, pair in enumerate(all_pairs) if i not in pair_idx_to_remove]
        pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}

        conflict_pairs = find_conflicts(all_pairs)

    if return_values:
        return all_pairs, all_values
    else:
        return all_pairs


if __name__ == "__main__":
    import time

    matrix = torch.triu(torch.rand(1000, 1000))
    matrix = (matrix > 0.99) * matrix
    print(torch.sum(matrix > 0.99))

    # start = time.time()
    # indices = matrix_sampling(matrix.clone(), 10000, greedy=True, mat_type='torch')
    # print(float(torch.sum(matrix[np.array(indices)[:, 0], np.array(indices)[:, 1]])), time.time() - start)
    #
    # start = time.time()
    # indices, max_score_b = matrix_sampling_beam_search_v2(matrix.clone(), 1000)
    # print(float(torch.sum(matrix[np.array(indices)[:, 0], np.array(indices)[:, 1]])), time.time() - start)

    start = time.time()
    indices = conflict_sampling(matrix.clone())
    print(float(torch.sum(matrix[np.array(indices)[:, 0], np.array(indices)[:, 1]])), time.time() - start)
