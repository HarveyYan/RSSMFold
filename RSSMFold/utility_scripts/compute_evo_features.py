import os
import numba
import numpy as np
import pickle
import argparse
import multiprocessing
from glob import glob
import h5py
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from RSSMFold.lib.msa_utils import match_to_rfam_id, save_rfam_msa, save_rnacmap_msa, read_msa, get_msa_covariance
from RSSMFold.lib.dca_support import get_mf_coupling, compute_direct_information
from RSSMFold.lib.neural_gremlin_support import get_neural_gremlin_mrf, get_gremlin_mrf

bprna_path = '../../data/bprna_processed/'
rnastralign_path = '../data/rnastralign_processed/'
rfam_dataset_path = '../../data/rfam_ood_processed/'


def compute_one_seq(seq, seq_id, out_dir, method='rfamdb', verbose=False, ncores=20, save_with_path=None,
                    dl_struct=None, include_grem_features=True):
    if verbose:
        print('computing evolutionary features for', seq_id, ':', seq, flush=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if method == 'rfamdb':
        rfam_id = match_to_rfam_id(seq, seq_id, out_dir, ncores=ncores)
        if rfam_id is None:
            if verbose:
                print('not matched to any RFAMS')
                print('#' * 100)
            if save_with_path is not None:
                pickle.dump([None], open(save_with_path, 'wb'))
            return None
        elif verbose:
            print('matched to RFAM', rfam_id)
        msa_fp = save_rfam_msa(seq, seq_id, rfam_id, out_dir, cap_cmalign_msa_depth=200000, ncores=ncores)
    elif method == 'rnacmap_rnafold':
        msa_fp, size_msa = save_rnacmap_msa(
            seq, seq_id, out_dir, cap_rnacmap_msa_depth=200000, ncores=ncores, ret_size=True)
        if size_msa == 0:
            if verbose:
                print('MSA is empty')
                print('#' * 100)
            if save_with_path is not None:
                pickle.dump([None], open(save_with_path, 'wb'))
            return None
    else:
        assert dl_struct is not None and len(dl_struct) == len(seq), \
            'under \'rnacmap_dl\' mode you need to provide a predicted structure for each sequence'
        msa_fp, size_msa = save_rnacmap_msa(
            seq, seq_id, out_dir, cap_rnacmap_msa_depth=200000, ncores=ncores, dl_struct=dl_struct, ret_size=True)
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

    # GPU is often necessary for the computation of gremlin features
    if include_grem_features:
        if neff == 1.:
            # current tensorflow implementation of gremlin cannot handle cases where neff equals to 1
            # (caused by nan during local field initialization)
            # use gremlin_cpp instead
            with open(os.path.join(out_dir, 'tmp_gapsfilt.a2m'), 'w') as file:
                for i, seq in enumerate(msa):
                    file.write('>%d\n%s\n' % (i, ''.join(['ACGU-'[idx] for idx in seq])))
            mrf = get_gremlin_mrf('tmp_gapsfilt', out_dir)
            gremlin_couplings = mrf['pairwise_coupling_triu_k1'].reshape(-1, 5, 5)[:, :-1, :-1]
            gremlin_conserv = mrf['local_conservation'][:, :-1]
        else:
            gremlin_couplings, gremlin_conserv = get_neural_gremlin_mrf(msa, msa_w)
            gremlin_couplings = gremlin_couplings.transpose(0, 2, 1, 3)[np.triu_indices(msa.shape[1], k=1)]

        # we will only keep 2000 sequences in the alignment
        if msa.shape[0] > 2000:
            sampled_idx = np.random.choice(np.arange(1, msa.shape[0]), 1999, False, msa_w[1:] / np.sum(msa_w[1:]))
            sampled_idx = np.sort(sampled_idx)
            msa = np.concatenate([msa[:1, :], msa[sampled_idx]], axis=0)
            msa_w = np.concatenate([msa_w[:1], msa_w[sampled_idx]], axis=0)

        ret.update({  # truncated MSA
            'msa': msa.astype(np.int8),
            'msa_w': msa_w.astype(np.float32),
            'gremlin_couplings_triu_k1': gremlin_couplings.astype(np.float32),
            'gremlin_conserv': gremlin_conserv.astype(np.float32)
        })

    else:
        # we don't limit the size of MSA for now — need them for gremlin later
        ret.update({
            'full_msa': msa.astype(np.int8),
            'full_msa_w': msa_w.astype(np.float32),
        })

    # shutil.rmtree(out_dir, ignore_errors=True)
    if verbose:
        print('#' * 100)

    if save_with_path is not None:
        pickle.dump([ret], open(save_with_path, 'wb'))
        return ret
    else:
        return ret


def worker_func(args):
    rna_seq, rna_id, method, ncores, save_with_path, dl_struct, include_grem_features = args
    return compute_one_seq(rna_seq, rna_id, './' + rna_id, method=method, verbose=False, ncores=ncores,
                           save_with_path=save_with_path, dl_struct=dl_struct,
                           include_grem_features=include_grem_features)


def parse_dl_struct_fasta(path, dataset):
    id_to_struct = {}
    with open(path, 'r') as file:
        for line in file:
            if line[0] == '>':
                id = line[1:].rstrip()
            else:
                seq = line.rstrip()
                id_to_struct[dataset + '-' + id] = seq
    return id_to_struct


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generating evolutionary features')
    parser.add_argument('--dataset', type=str, default='bprna', choices=['bprna', 'rnastralign', 'rfam_external'])
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--chunk', type=int, default=-1, )
    parser.add_argument('--chunk_size', type=int, default=500, )
    parser.add_argument('--method', type=str, default='rfamdb', choices=['rfamdb', 'rnacmap_rnafold', 'rnacmap_dl'])
    parser.add_argument('--ncores', type=int, default=20, )
    parser.add_argument('--enable_mp', type=eval, default=True, choices=[True, False])
    parser.add_argument('--merge_pickles', type=eval, default=False, choices=[True, False])
    parser.add_argument('--include_grem_features', type=eval, default=True, choices=[True, False])
    args = parser.parse_args()

    if args.dataset == 'bprna':
        base_path = bprna_path
        scratch_path = '../../data/bprna_processed/'
    elif args.dataset == 'rnastralign':
        base_path = rnastralign_path
        scratch_path = '../data/rnastralign_processed/'
    else:
        base_path = rfam_dataset_path
        scratch_path = '../../data/rfam_ood_processed/'

    if args.dataset == 'rfam_external':
        all_split = ['rfam_external']
    else:
        if args.split == 'all':
            all_split = ['test', 'val', 'train']
        else:
            all_split = [args.split]

    if args.merge_pickles:
        # here we merge all pickle chunks to a hdf5 object
        for split in all_split:
            if args.dataset == 'rfam_external':
                full_data_path = base_path + 'all_data.pkl'
            else:
                full_data_path = base_path + '%s_split.pkl' % (split)

            all_rna = []
            rna_dataset = pickle.load(open(full_data_path, 'rb'))
            for rna_seq, pairings, pseudoknotted, filepath, _ in rna_dataset:
                rna_seq = rna_seq.upper()
                # for rfam external dataset, we only obtain evo features for a subset of the data
                if args.dataset == 'rfam_external' and len(rna_seq) <= 150:
                    continue
                all_rna.append([rna_seq, pairings, pseudoknotted, filepath])

            all_extracted_features = []
            print('merging these files for split %s:' % (split))
            all_to_merge_tmp = glob(base_path + '%s_split_evofeatures_%s_num_*.pkl' % (split, args.method))
            # first off remove all 'corrected' pickles, ref. rnacmap small case problem
            all_to_merge = []
            for fp in all_to_merge_tmp:
                if 'grem_updated' not in fp:
                    all_to_merge.append(fp)
            all_to_merge.sort(key=lambda fp: int(fp.split('_')[-1].split('.')[0]))
            for fp in all_to_merge:
                if os.path.exists('.'.join(fp.split('.')[:-1]) + '_grem_updated.pkl'):
                    fp = '.'.join(fp.split('.')[:-1]) + '_grem_updated.pkl'
                print(fp)
                all_extracted_features.extend(pickle.load(open(fp, 'rb')))

            assert len(all_rna) == len(all_extracted_features), \
                'size of original dataset %d not equal to number of extracted features %d' % (
                    len(all_rna), len(all_extracted_features))

            f = h5py.File(base_path + '%s_split_with_%s_evofeatures.hdf5' % (split, args.method), 'w')
            for i, data in tqdm(enumerate(all_extracted_features), total=len(all_extracted_features)):
                seq, pairings, pseudoknotted, filepath = all_rna[i]
                grp = f.create_group('%d_%s' % (i, filepath.split('/')[-1]))

                grp.create_dataset('seq', data=seq)
                grp.create_dataset('pairings', data=np.array(pairings, dtype=np.int32),
                                   compression="gzip", compression_opts=9)
                grp.create_dataset('contains_pseudoknot', data=pseudoknotted)
                grp.create_dataset('filepath', data=filepath)

                # store features
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

        exit(0)

    ncores = max(4, args.ncores)
    numba.set_num_threads(ncores)

    if args.enable_mp:
        pool = multiprocessing.get_context('spawn').Pool(4)

    for split in all_split:
        if args.dataset == 'rfam_external':
            full_data_path = base_path + 'all_data.pkl'
        else:
            full_data_path = base_path + '%s_split.pkl' % (split)
        print('processing:', full_data_path)
        data = pickle.load(open(full_data_path, 'rb'))
        if args.method == 'rnacmap_dl':
            if args.dataset == 'rfam_external':
                id_to_struct = parse_dl_struct_fasta(
                    os.path.join(base_path, 'all_struct.fa'), args.dataset)
            else:
                id_to_struct = parse_dl_struct_fasta(
                    os.path.join(base_path, '%s_split_dl_struct.fa' % (split)), args.dataset)

        all_rna = []
        for rna in data:
            rna_seq = rna[0].upper()
            # for rfam external dataset, we only obtain evo features for a subset of the data
            if args.dataset == 'rfam_external' and len(rna_seq) <= 150:
                continue
            rna_id = args.dataset + '-' + rna[-2].split('/')[-1]
            rna_id = rna_id.replace('(', '').replace(')', '')
            all_rna.append([rna_seq, rna_id, args.method, ncores])
        # cut into chunks
        all_rna_chunks = [all_rna[i:i + args.chunk_size] for i in range(0, len(all_rna), args.chunk_size)]

        if args.chunk == -1:
            for chunk_idx in range(len(all_rna_chunks)):
                print('processing chunk:', chunk_idx)
                start_idx = chunk_idx * args.chunk_size
                chunk = all_rna_chunks[chunk_idx]
                effective_chunk = []
                for i, rna_param in enumerate(chunk):
                    current_idx = start_idx + i
                    file_sig = '%s_split_evofeatures_%s_num_%d.pkl' % (split, args.method, current_idx)
                    scratch_save_path = scratch_path + file_sig
                    base_save_path = base_path + file_sig
                    if not os.path.exists(scratch_save_path):
                        if args.method == 'rnacmap_dl':
                            dl_struct = id_to_struct[rna_param[1]]
                        else:
                            dl_struct = None
                        effective_chunk.append(rna_param + [base_save_path, dl_struct, args.include_grem_features])

                if args.enable_mp:
                    all_ret = list(pool.imap(worker_func, effective_chunk))
                else:
                    all_ret = []
                    for rna_seq, rna_id, method, ncores, save_with_path, dl_struct, include_grem_features in effective_chunk:
                        all_ret.append(
                            compute_one_seq(rna_seq, rna_id, './' + rna_id, method, verbose=True, ncores=ncores,
                                            save_with_path=save_with_path, dl_struct=dl_struct,
                                            include_grem_features=include_grem_features))

        else:
            assert args.chunk >= 0 and args.chunk < len(all_rna_chunks)
            print('processing chunk:', args.chunk)
            start_idx = args.chunk * args.chunk_size
            chunk = all_rna_chunks[args.chunk]
            effective_chunk = []
            for i, rna_param in enumerate(chunk):
                current_idx = start_idx + i
                file_sig = '%s_split_evofeatures_%s_num_%d.pkl' % (split, args.method, current_idx)
                scratch_save_path = scratch_path + file_sig
                base_save_path = base_path + file_sig
                if not os.path.exists(scratch_save_path):
                    if args.method == 'rnacmap_dl':
                        dl_struct = id_to_struct[rna_param[1]]
                    else:
                        dl_struct = None
                    effective_chunk.append(rna_param + [base_save_path, dl_struct, args.include_grem_features])

            if args.enable_mp:
                all_ret = list(pool.imap(worker_func, effective_chunk))
            else:
                all_ret = []
                for rna_seq, rna_id, method, ncores, save_with_path, dl_struct, include_grem_features in effective_chunk:
                    all_ret.append(
                        compute_one_seq(rna_seq, rna_id, './' + rna_id, method, verbose=True, ncores=ncores,
                                        save_with_path=save_with_path, dl_struct=dl_struct,
                                        include_grem_features=include_grem_features))
