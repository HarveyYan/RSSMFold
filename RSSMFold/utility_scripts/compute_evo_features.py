import os
import numba
import numpy as np
import pickle
import argparse
import multiprocessing
import h5py
from glob import glob
from tqdm import tqdm
import pathlib
import shutil

import RSSMFold
from RSSMFold.lib.msa_utils import match_to_rfam_id, save_rfam_msa, save_rnacmap_msa, read_msa, get_msa_covariance
from RSSMFold.lib.dca_support import get_mf_coupling, compute_direct_information
from RSSMFold.single_seq_rssm import run as single_seq_rssm_run
from RSSMFold.utility_scripts.service_utils import read_fasta_file

basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()


def compute_one_seq(seq, seq_id, out_dir, save_with_path, struct, args):
    method = args.method
    ncores = args.ncores
    verbose = args.verbose
    specify_blastn_database_path = args.specify_blastn_database_path

    if verbose:
        print('computing evolutionary features for', seq_id, ':', seq, flush=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if method == 'rfamdb':
        rfam_id = match_to_rfam_id(seq, seq_id, out_dir, ncores=ncores)
        if rfam_id is None:
            if verbose:
                print('not matched to any RFAM family')
                print('#' * 100)
            if save_with_path is not None:
                pickle.dump([None], open(save_with_path, 'wb'))
            return None
        elif verbose:
            print('matched to RFAM', rfam_id)
        msa_fp = save_rfam_msa(seq, seq_id, rfam_id, out_dir, cap_cmalign_msa_depth=200000, ncores=ncores)
    elif method == 'rnacmap_rnafold':
        msa_fp, size_msa = save_rnacmap_msa(
            seq, seq_id, out_dir, cap_rnacmap_msa_depth=200000, ncores=ncores, ret_size=True,
            specify_blastn_database_path=specify_blastn_database_path)
        if size_msa == 0:
            if verbose:
                print('MSA is empty')
                print('#' * 100)
            if save_with_path is not None:
                pickle.dump([None], open(save_with_path, 'wb'))
            return None
    else:
        assert struct is not None and len(struct) == len(seq), \
            'the given \'struct\' is not compatible with the sequence'
        msa_fp, size_msa = save_rnacmap_msa(
            seq, seq_id, out_dir, cap_rnacmap_msa_depth=200000, ncores=ncores, dl_struct=struct, ret_size=True,
            specify_blastn_database_path=specify_blastn_database_path)
        if size_msa == 0:
            if verbose:
                print('MSA is empty')
                print('#' * 100)
            if save_with_path is not None:
                pickle.dump([None], open(save_with_path, 'wb'))
            return None

    msa, msa_w, v_idx = read_msa(msa_fp, cap_msa_depth=100000)
    neff = np.sum(msa_w).astype(np.float32)
    if verbose:
        print('MSA contains %d rows, %d columns, Neff %.2f' % (msa.shape[0], msa.shape[1], sum(msa_w)))
    ret = get_msa_covariance(msa, msa_w)
    covariance = ret['covariance_mat']  # l, n_vocab, l, n_vocab
    mf_couplings = get_mf_coupling(covariance, msa.shape[1])  # l, n_vocab - 1, l, n_vocab - 1
    mf_direct_info = compute_direct_information(mf_couplings, ret['single_site_stats'], msa.shape[1])

    ret = {
        'neff': neff.astype(np.float32),
        'cov_triu_k1': covariance.transpose(0, 2, 1, 3).astype(np.float32)[np.triu_indices(msa.shape[1], k=1)],
        'mf_couplings_triu_k1': mf_couplings.transpose(0, 2, 1, 3).astype(np.float32)[
            np.triu_indices(msa.shape[1], k=1)],
        'mf_direct_info_triu_k1': mf_direct_info.astype(np.float32),
    }

    if 'rfam_id' in locals():
        ret['rfam_id'] = rfam_id

    ret.update({
        'full_msa': msa.astype(np.int8),
        'full_msa_w': msa_w.astype(np.float32),
    })

    shutil.rmtree(out_dir)
    if verbose:
        print('#' * 100)

    if save_with_path is not None:
        pickle.dump([ret], open(save_with_path, 'wb'))
        return ret
    else:
        return ret


def worker_func(args):
    rna_seq, rna_id, save_with_path, rna_struct, expr_args = args
    tmp_dir = os.path.join(expr_args.out_dir, rna_id)
    return compute_one_seq(rna_seq, rna_id, tmp_dir, save_with_path, rna_struct, expr_args)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generating evolutionary features')

    basic_group = parser.add_argument_group('Basic input/output options')
    basic_group.add_argument('--input_fasta_path', type=str, metavar='example.fa',
                             help='Path to the input RNA sequences in fasta format')
    basic_group.add_argument('--out_dir', type=str, metavar='./output',
                             help='Output directory where the evolutionary feature library will be stored')
    basic_group.add_argument('--out_filename', type=str,
                             help='Output file name for the evolutionary feature library')
    basic_group.add_argument('--verbose', type=eval, default=True, choices=[True, False],
                             help='Show a progress bar during evolutionary feature computation')

    evo_comp_group = parser.add_argument_group(
        "Options for evolutionary feature computation")
    evo_comp_group.add_argument('--method', type=str, default='rfamdb',
                                choices=['rfamdb', 'rnacmap_rnafold', 'rnacmap_rssm'])
    evo_comp_group.add_argument('--ncores', type=int, default=20,
                                help='Number of CPU cores for parallel computation in '
                                     'numba, blastn and Infernal')
    evo_comp_group.add_argument('--enable_mp', type=eval, default=True, choices=[True, False],
                                help='If enabled, this program will compute evolutionary features '
                                     'for 4 RNA sequences at the same time')
    evo_comp_group.add_argument('--specify_blastn_database_path', type=str, default=None,
                                help='Specify the location of NCBI nucleotide database; ignoring '
                                     'this option will make the program use the default location:'
                                     'RNAcmap/nt_database/nt')

    rssm_group = parser.add_argument_group(
        "Options for single sequence based RSSMs, when \"rnacmap_rssm\" is selected")
    rssm_group.add_argument('--use_gpu_device', nargs='*', type=int, default=[], metavar='0',
                            help='Specify a list of GPU devices to host RSSM; leave this option empty to only use CPU')
    rssm_group.add_argument('--batch_size', type=int, default=8, metavar='8',
                            help='Specify batch size for RSSM inference. '
                                 'Larger batch size may accelerate the computation, '
                                 'but runs the risk of exceeding GPU memory')
    rssm_group.add_argument('--constrained_pairing', type=eval, default=False, choices=[True, False],
                            help='Option to only predict canonical RNA basepairs')
    rssm_group.add_argument('--nonoverlapping', type=eval, default=True, choices=[True, False],
                            help='Option to discard potential multiplets')
    rssm_group.add_argument('--map_threshold', type=float, default=0.2, metavar='0.2',
                            help='Discretization threshold for predicted RNA contact map basepairing probabilities')
    rssm_group.add_argument('--matrix_sampling_mode', type=str, default='greedy',
                            choices=['greedy', 'greedy_bm', 'greedy_bm_shuffled',
                                     'conflict_sampling', 'diff_lp', 'no_sampling', 'probknot'],
                            help='Select a sampling method for discretizing '
                                 'the predicted RNA contact map probabilities')
    rssm_group.add_argument('--use_lp_pred', type=eval, default=False, choices=[True, False],
                            help='Add LinearPartition predictions into RSSM ensembles')
    rssm_group.add_argument('--enable_sliding_window', type=eval, default=False, choices=[True, False],
                            help='Only predicting local structures by enabling the sliding window.'
                                 'This option improves the accuracy of predicted local structures '
                                 'in long RNA sequences (>1000 nts), in addition to fitting long RNAs into GPU memory')
    rssm_group.add_argument('--window_size', type=int, default=1000, metavar='500',
                            help='Size of the sliding window. RSSM will not predict any basepairs '
                                 'spanning longer than this window size')
    rssm_group.add_argument('--window_move_increment', type=int, default=500, metavar='100',
                            help='Step size for moving the sliding window from 5\'UTR to 3\'UTR ')

    return parser


def run():
    parser = get_parser()
    args = parser.parse_args()

    ncores = max(4, args.ncores)
    numba.set_num_threads(ncores)

    if args.enable_mp:
        pool = multiprocessing.get_context('spawn').Pool(4)

    all_ids, all_seqs = read_fasta_file(args.input_fasta_path)

    method = args.method
    if method == 'rnacmap_rssm':
        print('Generating nested RSSM secondary structure predictions')
        args.generate_dot_bracket = True
        single_seq_rssm_run(args)
        all_struct = []
        out_filename = args.input_fasta_path.split(os.path.sep)[-1].split('.')[0]
        with open(os.path.join(args.out_dir, f'{out_filename}.dot_bracket'), 'r') as file:
            for line in file:
                line = line.rstrip()
                if line[0] == '>':
                    continue
                all_struct.append(line)

    all_rna = []
    for i, (rna_id, rna_seq) in enumerate(zip(all_ids, all_seqs)):
        rna_id = rna_id.replace('(', '').replace(')', '')
        file_sig = f'{args.out_filename}_evofeatures_{args.method}_num_{i}.pkl'
        save_path = os.path.join(args.out_dir, file_sig)
        if method == 'rnacmap_rssm':
            struct = all_struct[i]
        else:
            struct = None
        all_rna.append([rna_seq, rna_id, save_path, struct, args])

    if args.enable_mp:
        list(pool.imap(worker_func, all_rna))
    else:
        for rna_seq, rna_id, save_with_path, rna_struct, expr_args in all_rna:
            tmp_dir = os.path.join(expr_args.out_dir, rna_id)
            compute_one_seq(rna_seq, rna_id, tmp_dir, save_with_path, rna_struct, expr_args)

    search_file_sig = f'{args.out_filename}_evofeatures_{args.method}_num_*.pkl'
    save_path = os.path.join(args.out_dir, search_file_sig)
    print(f'Merging results into evolutionary feature library {save_path}')

    all_extracted_features = []
    all_to_merge = glob(save_path)
    all_to_merge.sort(key=lambda fp: int(fp.split('_')[-1].split('.')[0]))
    for fp in all_to_merge:
        print(f'Opening {fp}')
        all_extracted_features.extend(pickle.load(open(fp, 'rb')))

    assert len(all_extracted_features) == len(all_seqs), \
        'size of original dataset %d not equal to number of extracted features %d' % (
            len(all_rna), len(all_extracted_features))

    f = h5py.File(os.path.join(args.out_dir, f'{args.out_filename}_evofeatures_{args.method}.hdf5'), 'w')
    for i, data in tqdm(enumerate(all_extracted_features), total=len(all_extracted_features)):
        seq, seq_id, _, _, _ = all_rna[i]

        grp = f.create_group('%d_%s' % (i, seq_id))
        grp.create_dataset('seq', data=seq)

        if data is not None:
            for k, v in data.items():
                if k == 'rfam_id' or k == 'neff':
                    grp.create_dataset(k, data=v)
                else:
                    if k == 'msa' or k == 'full_msa':
                        dtype = np.int8
                    else:
                        dtype = np.float32
                    grp.create_dataset(k, data=v.astype(dtype), compression="gzip", compression_opts=9)


if __name__ == "__main__":
    run()
