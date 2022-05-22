'''
adaptations from pydca (https://github.com/KIT-MBS/pydca)
trade time for space when handling deep MSAs, while at the same time use numba to accelerate computation
'''

from numba import jit, prange
import numpy as np

VOCAB = ['A', 'C', 'G', 'U', '-']
N_VOCAB = len(VOCAB)


@jit(nopython=True, parallel=True)
def get_msa_eff(msa, eff_cutoff):
    # care not to trigger the race condition — multiple threads trying to write
    # the same slice/element in an array simultaneously
    msa_depth, msa_length = msa.shape[0], msa.shape[1]
    seq_weights = np.zeros(msa_depth)
    for i in prange(msa_depth):
        seq_i = msa[i]
        for j in range(msa_depth):
            seq_j = msa[j]
            if np.sum(seq_i == seq_j) / msa_length >= eff_cutoff:
                seq_weights[i] += 1
    seq_weights = 1 / seq_weights
    return seq_weights


def get_single_site_stats(msa, seq_weights, lamb=None):
    m_eff = np.sum(seq_weights)
    if lamb is None:
        lamb = m_eff

    offset = lamb / (lamb + m_eff) / N_VOCAB
    # trading space for time
    single_site_freqs = (np.eye(N_VOCAB)[msa] * seq_weights[:, None, None]).sum(0) / (lamb + m_eff) + offset
    # single_site_freqs = np.zeros((msa.shape[1], N_VOCAB))
    # for i in range(msa.shape[1]):
    #     for a in range(N_VOCAB):
    #         single_site_freqs[i, a] = np.sum((msa[:, i] == a) * seq_weights) / (lamb + m_eff) + offset

    return single_site_freqs


@jit(nopython=True, parallel=True)
def get_pair_site_stats(msa, seq_weights, lamb=None):
    m_eff = np.sum(seq_weights)
    if lamb is None:
        lamb = m_eff

    msa_depth, msa_length = msa.shape[0], msa.shape[1]
    offset = lamb / (lamb + m_eff) / N_VOCAB ** 2

    pair_site_freqs = np.zeros(((msa_length - 1) * msa_length // 2, N_VOCAB, N_VOCAB))
    for i in prange(msa_length):
        column_i = msa[:, i]
        for j in range(i + 1, msa_length):
            column_j = msa[:, j]
            pair_site_idx = int(
                (msa_length * (msa_length - 1) / 2) - (msa_length - i) * ((msa_length - i) - 1) / 2 + j - i - 1)
            for a in range(N_VOCAB):
                count_ai = column_i == a
                for b in range(N_VOCAB):
                    count_bj = column_j == b
                    count_ai_bj = count_ai * count_bj
                    freq_ia_jb = np.sum(count_ai_bj * seq_weights)
                    pair_site_freqs[pair_site_idx, a, b] += float(freq_ia_jb / (lamb + m_eff) + offset)

    return pair_site_freqs


@jit(nopython=True)
def get_mf_coupling(covariance, msa_length):
    # remove gaps and compute couplings (direct and indirect)
    couplings = - np.linalg.inv(
        covariance[:, :-1, :, :-1].copy().reshape((msa_length * (N_VOCAB - 1), msa_length * (N_VOCAB - 1))))
    couplings = couplings.reshape((msa_length, (N_VOCAB - 1), msa_length, (N_VOCAB - 1)))

    return couplings


@jit(nopython=True, parallel=True)
def compute_direct_information(couplings, single_site_stats, msa_length):
    '''
    requires estimating the local fields in a two-site model
    recommend numba latest version 0.53.1
    somehow if you use an earlier version of numba, say version 0.48, it
    will throw some inexplicable errors
    '''
    direct_info = np.zeros((msa_length * (msa_length - 1) // 2, N_VOCAB - 1, N_VOCAB - 1))
    for i in prange(msa_length):
        freq_i = np.expand_dims(single_site_stats[i], 1)
        for j in range(i + 1, msa_length):
            freq_j = np.expand_dims(single_site_stats[j], 1)
            pair_site_idx = int(
                (msa_length * (msa_length - 1) / 2) - (msa_length - i) * ((msa_length - i) - 1) / 2 + j - i - 1)

            # np.pad not supported by numba
            couplings_ij = np.full((N_VOCAB, N_VOCAB), 1.)
            couplings_ij[:N_VOCAB - 1, :N_VOCAB - 1] = np.exp(couplings[i, :, j, :])

            # computing local fields
            fields_i_new = fields_i_old = np.full((N_VOCAB, 1), 1 / N_VOCAB)
            fields_j_new = fields_j_old = np.full((N_VOCAB, 1), 1 / N_VOCAB)
            max_fields_change = 10.0
            while max_fields_change >= 1e-4:
                x_i = np.dot(couplings_ij, fields_j_old)
                x_j = np.dot(np.transpose(couplings_ij), fields_i_old)
                fields_i_new = freq_i / x_i
                fields_i_new /= np.sum(fields_i_new)
                fields_j_new = freq_j / x_j
                fields_j_new /= np.sum(fields_j_new)
                max_fields_change = max(
                    np.max(np.absolute(fields_i_new - fields_i_old)),
                    np.max(np.absolute(fields_j_new - fields_j_old)))
                fields_i_old = fields_i_new
                fields_j_old = fields_j_new

            hij = np.dot(fields_i_new, np.transpose(fields_j_new))
            pdir_ij = couplings_ij * hij
            pdir_ij /= np.sum(pdir_ij)
            fij = np.dot(freq_i, np.transpose(freq_j))
            # Only take into account residue residue interactions for computing direct information
            fij = fij[:N_VOCAB - 1, :N_VOCAB - 1] + 1e-10
            pdir_ij = pdir_ij[:N_VOCAB - 1, :N_VOCAB - 1] + 1e-10
            direct_info[pair_site_idx] = pdir_ij * np.log(pdir_ij / fij)

    return direct_info
