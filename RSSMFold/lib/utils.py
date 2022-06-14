import math
import torch
import random
import numpy as np
from scipy.sparse import diags
from functools import partial


def get_original_pe(seq_lens, max_len, d_model, return_flattened=True):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    if return_flattened:
        final_pe = []
        for i in range(len(seq_lens)):
            final_pe.append(pe[:seq_lens[i], :])
        return torch.cat(final_pe, dim=0)  # total_len, dim
    else:
        return pe  # max_len, dim


VIABLE_NUC_VOCAB = ['A', 'C', 'G', 'U']


def constraint_matrix(x, sharp_loop_constr=3, distance_contr=None):
    base_a = np.array([VIABLE_NUC_VOCAB[nuc_idx] == 'A' for nuc_idx in x]).astype(np.int32)
    base_u = np.array([VIABLE_NUC_VOCAB[nuc_idx] == 'U' for nuc_idx in x]).astype(np.int32)
    base_c = np.array([VIABLE_NUC_VOCAB[nuc_idx] == 'C' for nuc_idx in x]).astype(np.int32)
    base_g = np.array([VIABLE_NUC_VOCAB[nuc_idx] == 'G' for nuc_idx in x]).astype(np.int32)
    au = np.matmul(base_a[:, None], base_u[None, :])
    au_ua = au + au.T
    cg = np.matmul(base_c[:, None], base_g[None, :])
    cg_gc = cg + cg.T
    ug = np.matmul(base_u[:, None], base_g[None, :])
    ug_gu = ug + ug.T
    m = au_ua + cg_gc + ug_gu

    # for sharp loops
    sharp_loop_offset = list(range(-sharp_loop_constr, sharp_loop_constr + 1))
    all_mat = sharp_loop_mat = diags([1] * len(sharp_loop_offset), sharp_loop_offset, shape=m.shape)

    if distance_contr is not None and isinstance(distance_contr, int):
        dim = m.shape[0]
        distance_offset = list(range(distance_contr + 1, dim))
        distance_offset += [-offset for offset in distance_offset]
        if len(distance_offset) > 0:
            distance_mat = diags([1] * len(distance_offset), distance_offset, shape=m.shape)
            all_mat += distance_mat

    mask = 1 - all_mat.toarray().astype(np.int32)
    m = m * mask
    return m


def get_nonoverlapping_edges(list_edge_idx):
    random.shuffle(list_edge_idx)

    used_indices = []
    final_edge_idx = []
    for edge in list_edge_idx:
        if edge[0] not in used_indices and edge[1] not in used_indices:
            final_edge_idx.append(edge)
            used_indices.append(edge[0])
            used_indices.append(edge[1])

    return final_edge_idx


def matrix_sampling_by_arr(matrix, n_samples):
    sampled_idx = []
    pool_edge_idx = np.array(list(zip(*np.nonzero(matrix))))

    while len(sampled_idx) < n_samples:

        if len(pool_edge_idx) == 0:
            break

        idx = random.sample(range(len(pool_edge_idx)), 1)[0]
        row_idx, col_idx = pool_edge_idx[idx]
        sampled_idx.append([row_idx, col_idx])

        row_idx_to_del = list(np.where(pool_edge_idx == row_idx)[0]) + \
                         list(np.where(pool_edge_idx == col_idx)[0])

        row_idx_to_retrain = list(set(range(len(pool_edge_idx))).difference(set(row_idx_to_del)))

        pool_edge_idx = pool_edge_idx[row_idx_to_retrain]

    return sampled_idx


def matrix_sampling(matrix, n_samples, enforce_nonoverlapping=True, greedy=False, mat_type='numpy',
                    return_values=False):
    assert mat_type in ['numpy', 'torch']

    sampled_idx = []
    sampled_vals = []
    while len(sampled_idx) < n_samples:
        if mat_type == 'numpy':
            row, col = np.nonzero(matrix)
        else:
            row, col = torch.nonzero(matrix, as_tuple=True)

        if len(row) == 0:
            break

        if greedy:
            values = matrix[row, col]
            if mat_type == 'numpy':
                idx = np.argmax(values)
            else:
                idx = torch.argmax(values)
        else:
            idx = random.sample(range(len(row)), 1)[0]

        row_idx, col_idx = row[idx], col[idx]
        sampled_idx.append([int(row_idx), int(col_idx)])
        sampled_vals.append(matrix[row_idx, col_idx])

        if enforce_nonoverlapping:
            matrix[row_idx, :] = 0
            matrix[col_idx, :] = 0
            matrix[:, row_idx] = 0
            matrix[:, col_idx] = 0
        else:
            matrix[row_idx, col_idx] = 0

    if return_values:
        return sampled_idx, sampled_vals
    else:
        return sampled_idx


def matrix_sampling_beam_search(matrix, beam_width):
    all_solved_scores = []
    all_solved_samples = []

    row_indices, col_indices = torch.nonzero(matrix, as_tuple=True)
    all_sorted_values, all_sorted_indices = torch.sort(matrix[row_indices, col_indices], descending=True)
    all_sorted_values = all_sorted_values.cpu().numpy()

    sorted_row_indices, sorted_col_indices = row_indices[all_sorted_indices], col_indices[all_sorted_indices]
    sorted_row_indices = sorted_row_indices.cpu().numpy()
    sorted_col_indices = sorted_col_indices.cpu().numpy()
    sorted_indices = list(zip(sorted_row_indices, sorted_col_indices))

    beam_scores = [0.]
    beam_samples = [[]]
    beam_viable_idx = [sorted_indices]

    # we will visit the values in the order specified by sorted_idx
    for step in range(len(all_sorted_indices)):

        if len(beam_viable_idx) == 0:
            break

        row_idx, col_idx = sorted_row_indices[step], sorted_col_indices[step]
        cur_val = all_sorted_values[step]

        current_scores = []
        current_viable_idx = []
        current_samples = []

        for i, viable_idx in enumerate(beam_viable_idx):

            if len(viable_idx) == 0:
                # all viable indices is none means this expansion is depleted
                all_solved_scores.append(beam_scores[i])
                all_solved_samples.append(beam_samples[i])
                continue

            if (row_idx, col_idx) in viable_idx:
                viable_idx.remove((row_idx, col_idx))
                current_viable_idx.append([
                    (r_idx, c_idx) for r_idx, c_idx in viable_idx if
                    r_idx != row_idx and r_idx != col_idx and
                    c_idx != row_idx and c_idx != col_idx])
                current_scores.append(beam_scores[i] + cur_val)
                current_samples.append(beam_samples[i] + [[row_idx, col_idx]])

            current_viable_idx.append(viable_idx)
            current_scores.append(beam_scores[i])
            current_samples.append(beam_samples[i])

        if len(current_scores) > beam_width:
            _, indices = torch.sort(torch.as_tensor(current_scores).to(matrix.device), descending=True)
            indices = indices[:beam_width].cpu().numpy()
            beam_scores = [current_scores[idx] for idx in indices]
            beam_viable_idx = [current_viable_idx[idx] for idx in indices]
            beam_samples = [current_samples[idx] for idx in indices]
        else:
            beam_scores = current_scores
            beam_viable_idx = current_viable_idx
            beam_samples = current_samples

    all_solved_scores += beam_scores
    all_solved_samples += beam_samples
    max_idx = int(np.argmax(all_solved_scores))
    return all_solved_samples[max_idx], all_solved_scores[max_idx]


def matrix_sampling_beam_search_v2(matrix, beam_width, sort_prob=True):
    row_indices, col_indices = torch.nonzero(matrix, as_tuple=True)
    if sort_prob:
        all_sorted_values, all_sorted_indices = torch.sort(matrix[row_indices, col_indices], descending=True)
        all_sorted_values = all_sorted_values.cpu().numpy()
        sorted_row_indices, sorted_col_indices = row_indices[all_sorted_indices], col_indices[all_sorted_indices]
        sorted_row_indices = sorted_row_indices.cpu().numpy()
        sorted_col_indices = sorted_col_indices.cpu().numpy()
    else:
        # shuffling matrices
        all_sorted_indices = np.random.permutation(len(row_indices))
        sorted_row_indices = row_indices[all_sorted_indices].cpu().numpy()
        sorted_col_indices = col_indices[all_sorted_indices].cpu().numpy()
        all_sorted_values = matrix[sorted_row_indices, sorted_col_indices].cpu().numpy()

    beam_scores = [0.]
    beam_samples = [[]]
    beam_nonviable_idx = [[]]

    # we will visit the values in the order specified by sorted_idx
    for step in range(len(all_sorted_indices)):

        if len(beam_nonviable_idx) == 0:
            break

        row_idx, col_idx = sorted_row_indices[step], sorted_col_indices[step]
        cur_val = all_sorted_values[step]

        current_scores = []
        current_nonviable_idx = []
        current_samples = []

        for i, nonviable_idx in enumerate(beam_nonviable_idx):

            if row_idx not in nonviable_idx and col_idx not in nonviable_idx:

                tmp_nonviable_idx = nonviable_idx.copy()
                tmp_nonviable_idx.append(row_idx)
                if col_idx != row_idx:
                    tmp_nonviable_idx.append(col_idx)
                current_nonviable_idx.append(tmp_nonviable_idx)
                current_scores.append(beam_scores[i] + cur_val)
                current_samples.append(beam_samples[i] + [[row_idx, col_idx]])

            current_nonviable_idx.append(nonviable_idx)
            current_scores.append(beam_scores[i])
            current_samples.append(beam_samples[i])

        if len(current_scores) > beam_width:
            _, indices = torch.sort(torch.as_tensor(current_scores).to(matrix.device), descending=True)
            indices = indices[:beam_width].cpu().numpy()
            beam_scores = [current_scores[idx] for idx in indices]
            beam_nonviable_idx = [current_nonviable_idx[idx] for idx in indices]
            beam_samples = [current_samples[idx] for idx in indices]
        else:
            beam_scores = current_scores
            beam_nonviable_idx = current_nonviable_idx
            beam_samples = current_samples

    max_idx = int(np.argmax(beam_scores))
    return beam_samples[max_idx], beam_scores[max_idx]


def matrix_sampling_beam_search_mp(matrix, beam_width, pool):
    """actually slower than the version not using multiprocessing"""
    all_solved_scores = []
    all_solved_samples = []

    row_indices, col_indices = torch.nonzero(matrix, as_tuple=True)
    all_sorted_values, all_sorted_indices = torch.sort(matrix[row_indices, col_indices], descending=True)
    all_sorted_values = all_sorted_values.cpu().numpy()

    sorted_row_indices, sorted_col_indices = row_indices[all_sorted_indices], col_indices[all_sorted_indices]
    sorted_row_indices = sorted_row_indices.cpu().numpy()
    sorted_col_indices = sorted_col_indices.cpu().numpy()
    sorted_indices = list(zip(sorted_row_indices, sorted_col_indices))

    beam_scores = [0.]
    beam_samples = [[]]
    beam_viable_idx = [sorted_indices]

    # we will visit the values in the order specified by sorted_idx
    for step in range(len(all_sorted_indices)):

        if len(beam_viable_idx) == 0:
            break

        row_idx, col_idx = sorted_row_indices[step], sorted_col_indices[step]
        cur_val = all_sorted_values[step]

        func = partial(beam_search_subroutine, cur_val=cur_val, row_idx=row_idx, col_idx=col_idx)
        ret_ = list(pool.imap(func, zip(beam_viable_idx, beam_scores, beam_samples)))
        # print(ret_)
        ret = []
        for item in ret_:
            if len(item) == 1:
                if item[0][0] is None:
                    all_solved_scores.append(item[0][1])
                    all_solved_samples.append(item[0][2])
                    continue

            ret.extend(item)

        if len(ret) == 0:
            break
        ret = np.array(ret)
        current_viable_idx = ret[:, 0]
        current_scores = ret[:, 1].astype(np.float32)
        current_samples = ret[:, 2]

        if len(current_scores) > beam_width:
            _, indices = torch.sort(torch.as_tensor(current_scores).to(matrix.device), descending=True)
            indices = indices[:beam_width].cpu().numpy()
            beam_scores = [current_scores[idx] for idx in indices]
            beam_viable_idx = [current_viable_idx[idx] for idx in indices]
            beam_samples = [current_samples[idx] for idx in indices]
        else:
            beam_scores = current_scores
            beam_viable_idx = current_viable_idx
            beam_samples = current_samples

    all_solved_scores += list(beam_scores)
    all_solved_samples += list(beam_samples)
    max_idx = int(np.argmax(all_solved_scores))
    return all_solved_samples[max_idx], all_solved_scores[max_idx]


def beam_search_subroutine(args, cur_val, row_idx, col_idx):
    viable_idx, beam_score, beam_sample = args

    if len(viable_idx) == 0:
        return [[None, beam_score, beam_sample]]

    if (row_idx, col_idx) in viable_idx:
        viable_idx.remove((row_idx, col_idx))
        new_viable_idx = [row_col for row_col in viable_idx if
                          row_col[0] not in (row_idx, col_idx) and
                          row_col[1] not in (row_idx, col_idx)]
        current_score = beam_score + cur_val
        current_sample = beam_sample + [(row_idx, col_idx)]

        return [[new_viable_idx, current_score, current_sample], [viable_idx, beam_score, beam_sample]]
    else:
        return [[viable_idx, beam_score, beam_sample]]

