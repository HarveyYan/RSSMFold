import argparse
import torch
import os
import numpy as np
import subprocess
import pandas as pd
import pathlib
import yaml
import random
from tqdm import tqdm
import h5py
from itertools import product

torch.set_num_threads(1)

import RSSMFold
from RSSMFold.model.vision_transformer_unet_hier_evo import UNetVTModel, get_fc_block
from RSSMFold.lib.contact_map_utils import discretize_contact_map
from RSSMFold.utility_scripts.deployment_utils import download_weights, \
    cov_rssm_weights_link, cov_rssm_weights_path

NUC_VOCAB = ['A', 'C', 'G', 'U', 'N']
nb_type_node = 6
basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
linearpartition_executable = os.path.join(basedir, 'LinearPartition', 'linearpartition')
nb_ensemble = 5


def augment_linearpartition(seqs, cutoff, outdir):
    all_triu = []
    for seq in seqs:
        np.random.seed(random.seed())
        outfile = os.path.join(outdir, str(np.random.rand()) + '.lp_out')
        cmd = f'echo {seq} | {linearpartition_executable} -o {outfile} -c {cutoff} >/dev/null 2>&1'
        subprocess.call(cmd, shell=True)
        nb_nodes = len(seq)
        pred_mat = np.zeros((nb_nodes, nb_nodes, 1), dtype=np.float32)
        with open(outfile, 'r') as file:
            for line in file:
                ret = line.rstrip().split()
                if len(ret) == 0:
                    continue
                row = int(ret[0]) - 1
                col = int(ret[1]) - 1
                prob = float(ret[2])
                pred_mat[row, col] = prob
        all_triu.append(pred_mat[np.triu_indices(nb_nodes)])
        os.remove(outfile)
    return np.concatenate(all_triu, axis=0)


def cov_predictor_function(batch_seq_string, batch_seq, batch_cov_features, batch_len, model_ensemble, args):
    outdir = args.out_dir
    all_device = args.all_device
    map_threshold = args.map_threshold
    nonoverlapping = args.nonoverlapping
    constrained_pairing = args.constrained_pairing
    matrix_sampling_mode = args.matrix_sampling_mode

    enable_sliding_window = args.enable_sliding_window
    window_size = args.window_size
    window_move_increment = args.window_move_increment
    use_lp_pred = args.use_lp_pred
    save_contact_map_prob = args.save_contact_map_prob

    ret = {}
    list_model, list_fc_layer = model_ensemble
    with torch.no_grad():
        all_map_triu = []
        for model, all_fc_blocks in zip(list_model, list_fc_layer):

            max_len = batch_seq.shape[1]

            if not enable_sliding_window or max_len <= window_size:
                all_patch_map_triu = model(batch_seq, batch_len, batch_cov_features, enable_checkpoint=False,
                                           conv_backbone=False)
                contact_map_triu, batch_len_np = all_patch_map_triu[0]
                contact_map_triu = torch.sigmoid(all_fc_blocks[0](contact_map_triu))
            else:
                # safe to assume we would have batch size of 1

                full_cov_mat = torch.zeros(max_len, max_len, batch_cov_features.shape[-1]).to(batch_cov_features)
                full_cov_mat[torch.triu_indices(max_len, max_len)] = batch_cov_features

                cumsum_contact_map = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                count_contact_map = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                for start_idx in range(window_move_increment - window_size, max_len, window_move_increment):
                    end_idx = start_idx + window_size
                    if start_idx < 0:
                        start_idx = 0

                    windowed_x = batch_seq[:, start_idx: end_idx]
                    batch_len = windowed_x.shape[1]
                    windowed_batch_len = torch.as_tensor([batch_len]).to(all_device[0])
                    windowed_cov_mat = full_cov_mat[start_idx: end_idx, start_idx: end_idx][
                        torch.triu_indices(batch_len, batch_len)]

                    patch_map_triu = model(
                        windowed_x, windowed_batch_len, windowed_cov_mat, enable_checkpoint=False, conv_backbone=False)
                    contact_map_triu = torch.sigmoid(all_fc_blocks[0](patch_map_triu[0][0]))
                    x_idx, y_idx = np.triu_indices(batch_len)
                    cumsum_contact_map[x_idx + start_idx, y_idx + start_idx] += contact_map_triu
                    count_contact_map[x_idx + start_idx, y_idx + start_idx] += torch.ones_like(contact_map_triu)
                count_contact_map[count_contact_map == 0.] = 1.
                contact_map_triu = (cumsum_contact_map / count_contact_map)[np.triu_indices(max_len)]
                batch_len_np = np.array([max_len])

            all_map_triu.append(contact_map_triu)
            if save_contact_map_prob:
                ret['all_map_triu'] = all_map_triu

        if use_lp_pred:
            # practically adding LinearPartition into RSSM ensembles
            if not enable_sliding_window or max_len <= window_size:
                lp_triu = augment_linearpartition(batch_seq_string, 0., outdir)
                lp_triu = torch.tensor(lp_triu).to(all_device[-1])
            else:
                # safe to assume batch_size will be 1
                cumsum_lp_triu = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                count_lp_triu = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                for start_idx in range(window_move_increment - window_size, max_len, window_move_increment):
                    end_idx = start_idx + window_size
                    if start_idx < 0:
                        start_idx = 0
                    windowed_seqs = [seq[start_idx: end_idx] for seq in batch_seq_string]
                    batch_len = len(windowed_seqs[0])
                    x_idx, y_idx = np.triu_indices(batch_len)
                    lp_triu = augment_linearpartition(windowed_seqs, 0., outdir)
                    lp_triu = torch.as_tensor(lp_triu).to(all_device[-1])
                    cumsum_lp_triu[x_idx + start_idx, y_idx + start_idx] += lp_triu
                    count_lp_triu[x_idx + start_idx, y_idx + start_idx] += torch.ones_like(lp_triu)
                count_lp_triu[count_lp_triu == 0.] = 1.
                lp_triu = (cumsum_lp_triu / count_lp_triu)[np.triu_indices(max_len)]
            all_map_triu.append(lp_triu)
        contact_map_final = sum(all_map_triu) / len(all_map_triu)

        sampled_pairs = discretize_contact_map(
            contact_map_final, batch_seq, batch_len_np, map_threshold, nonoverlapping,
            constrained_pairing, matrix_sampling_mode, device=all_device[0], min_hairpin_span=1,
            ret_sampled_pairs=True)
        ret['sampled_pairs'] = sampled_pairs

    return ret


def cov_predictor(seqs, cov_features, ids, model_ensemble, args):
    out_dir = args.out_dir
    all_device = args.all_device
    save_contact_map_prob = args.save_contact_map_prob

    batch_seq = []
    all_len = [len(seq) for seq in seqs]
    max_len = max(all_len)

    for seq in seqs:
        nb_nodes = len(seq)
        node_features = list(map(lambda x: NUC_VOCAB.index(x), seq))
        batch_seq.append(node_features + [nb_type_node] * (max_len - nb_nodes))

    batch_seq = torch.tensor(np.array(batch_seq, dtype=np.long)).to(all_device[0])
    batch_len = torch.tensor(np.array(all_len, dtype=np.long)).to(all_device[0])
    batch_cov_features = torch.tensor(np.concatenate(cov_features, axis=0)).to(all_device[0])

    ret = cov_predictor_function(seqs, batch_seq, batch_cov_features, batch_len, model_ensemble, args)

    all_sampled_pairs = ret['sampled_pairs']
    for seq, seq_id, sampled_pairs in zip(seqs, ids, all_sampled_pairs):
        sampled_pairs_dict = {}
        for pair in sampled_pairs:
            sampled_pairs_dict[pair[0]] = pair[1] + 1
            sampled_pairs_dict[pair[1]] = pair[0] + 1

        with open(os.path.join(out_dir, f'{seq_id}.bpseq'), 'w') as file:
            for i, c in enumerate(seq):
                file.write(f'{i + 1} {c} {sampled_pairs_dict.get(i, 0)}\n')

    if save_contact_map_prob:
        all_map_triu = ret['all_map_triu']
        cumsum_triu_size = 0
        for seq, seq_id in zip(seqs, ids):
            seq_len = len(seq)
            triu_size = seq_len * (seq_len + 1) // 2
            seq_triu = []
            for i in range(nb_ensemble):
                seq_triu.append(all_map_triu[i][cumsum_triu_size: cumsum_triu_size + triu_size])
            np.save(os.path.join(out_dir, f'{seq_id}_prob.npy'), seq_triu)
            cumsum_triu_size += triu_size


def bulk_prediction_from_evo_lib(model_ensemble, args):
    evo_lib_path = args.input_evo_lib_path
    batch_size = args.batch_size

    all_ids, all_seqs, all_cov_features = [], [], []

    with h5py.File(evo_lib_path, 'r', libver='latest', swmr=True) as f:
        for rna_id in f.keys():
            seq = f[rna_id]['seq'][...]
            if type(seq) is np.ndarray:
                seq = seq.tolist()
            if type(seq) is bytes:
                seq = seq.decode('ascii')
            nb_nodes = len(seq)
            node_features = list(map(lambda x: NUC_VOCAB.index(x), seq))
            all_ids.append('_'.join(rna_id.split('_')[1:]))
            all_seqs.append(seq)

            # additional evolutionary features
            # note 1. may not have evolutionary features
            # note 2. target sequence in msa may not have the same length with the original sequence
            if 'full_msa' in f[rna_id]:
                target_seq_in_msa = f[rna_id]['full_msa'][...][0]
                length_msa_seq = len(target_seq_in_msa)
                # idx of 4 means gap in MSA
                idx_to_retain_in_msa_seq = \
                    [i for i, idx in enumerate(target_seq_in_msa) if idx != 4]  # 4 means gap here
                idx_corresponding_in_original_seq = \
                    [i for i, idx in enumerate(node_features) if idx != 4]  # 4 means N/unknown

                msa_seq_feature_mat = np.zeros((length_msa_seq, length_msa_seq, 16), dtype=np.float32)
                msa_seq_feature_mat[np.triu_indices(length_msa_seq, k=1)] = \
                    f[rna_id]['cov_triu_k1'][...][:, :4, :4].reshape(-1, 16)
                msa_seq_feature_mat[np.triu_indices(length_msa_seq, k=1)[::-1]] = \
                    f[rna_id]['cov_triu_k1'][...][:, :4, :4].reshape(-1, 16)

                all_indices = np.array(
                    list(product(idx_corresponding_in_original_seq, idx_corresponding_in_original_seq)))
                original_seq_feature_mat = np.zeros((nb_nodes, nb_nodes, 16), dtype=np.float32)

                original_seq_feature_mat[all_indices[:, 0], all_indices[:, 1]] = \
                    msa_seq_feature_mat[idx_to_retain_in_msa_seq, :][:, idx_to_retain_in_msa_seq].reshape(-1, 16)

                original_seq_feature_mat = original_seq_feature_mat[np.triu_indices(nb_nodes)]
                all_cov_features.append(original_seq_feature_mat)
            else:
                all_cov_features.append(np.zeros(((nb_nodes + 1) * nb_nodes // 2, 16), dtype=np.float32))

    if args.verbose:
        bar = tqdm(total=len(all_ids), position=0, leave=True)

    for i in range(0, len(all_ids), batch_size):
        cov_predictor(
            all_seqs[i: i + batch_size], all_cov_features[i: i + batch_size],
            all_ids[i: i + batch_size], model_ensemble, args)

        if args.verbose:
            bar.update(batch_size)


def bpseq_remove_pseudoknots(bpseq_path):
    subprocess.call('''export PERLLIB=./FreeKnot
        perl FreeKnot/remove_pseudoknot.pl -i bpseq -s bp {0} > {0}_freeknot'''.format(bpseq_path), shell=True)


def bpseq_to_dot_bracket(bpseq_path):
    bpseq_remove_pseudoknots(bpseq_path)
    file = pd.read_csv(bpseq_path + '_freeknot', delimiter=' ', header=None)
    seq = ''.join(list(file.iloc[:, 1]))
    struct = ['.'] * len(seq)
    for i, idx in enumerate(list(file.iloc[:, 2])):
        if idx != 0:
            if i < idx - 1:
                struct[i] = '('
            else:
                struct[i] = ')'
    return ''.join(struct)


def load_model_ensemble(config, all_device):
    list_model = []
    list_fc_layer = []
    loaded_weights = torch.load(os.path.join(basedir, 'RSSMFold', 'rssm_weights', config['weights']),
                                map_location='cpu')
    for model_idx in range(nb_ensemble):
        # models providing embeddings
        model = UNetVTModel(
            config['num_ds_steps'], config['t_emb_dim'], config['t_nhead'], all_device,
            map_concat_mode=config['map_concat_mode'], use_lw=config['enable_local_window'],
            patch_ds_stride=config['initial_ds_size'], use_conv_proj=config['use_conv_proj'],
            nb_pre_convs=config['nb_convs'], nb_post_convs=config['nb_convs'], include_coupling_features=False,
            enable_5x5_filter=config['enable_5x5_filter'], compute_norm=True)

        # fully connected models providing basepairing predictions
        num_fc_blocks = 2
        all_fc_blocks = []
        size_output = 1
        for i in range(config['num_ds_steps'] + 2):
            fc_blocks = None
            dim = config['t_emb_dim'] * 2 ** max(i - 1, 0)
            fc_dim = dim * 2
            for _ in range(num_fc_blocks):
                block = get_fc_block(dim, fc_dim, 0.5)
                if fc_blocks is None:
                    fc_blocks = block
                else:
                    fc_blocks = torch.nn.Sequential(*fc_blocks, *block)
                dim = fc_dim
            fc_blocks = torch.nn.Sequential(*fc_blocks, *get_fc_block(dim, size_output, 0.5))
            all_fc_blocks.append(fc_blocks)
        all_fc_blocks = torch.nn.ModuleList(all_fc_blocks).to(all_device[-1])

        # loading weights
        model_weights, fc_weights = loaded_weights[model_idx]
        model.load_state_dict(model_weights)
        all_fc_blocks.load_state_dict(fc_weights)
        print(f'{model_idx + 1}/{len(loaded_weights)} ensembles loaded')

        # setting to evaluation mode
        model.eval()
        all_fc_blocks.eval()
        list_model.append(model)
        list_fc_layer.append(all_fc_blocks)

    return list_model, list_fc_layer


def get_parser():
    parser = argparse.ArgumentParser(
        description='Covariance feature based RNA Secondary Structural Model predictor')

    basic_group = parser.add_argument_group('Basic input/output options')
    basic_group.add_argument('--input_evo_lib_path', type=str, metavar='example.hdf5', required=True,
                             help='Path to the input RNA evolutionary features library.')
    basic_group.add_argument('--out_dir', type=str, metavar='./output', required=True,
                             help='Output directory for prediction results. '
                                  'File names will be given by RNA ids in the feature library.')
    basic_group.add_argument('--use_gpu_device', nargs='*', type=int, default=[], metavar='0',
                             help='Specify a list of GPU devices to host RSSM. Default: only use CPU.')
    basic_group.add_argument('--verbose', type=eval, default=True, choices=[True, False],
                             help='Show a progress bar during RSSM inference. Default: True.')
    basic_group.add_argument('--batch_size', type=int, default=8, metavar='8',
                             help='Specify batch size for RSSM inference. '
                                  'Larger batch size may accelerate the computation, '
                                  'but runs the risk of exceeding GPU memory. Default: 8.')
    basic_group.add_argument('--generate_dot_bracket', type=eval, default=False, choices=[True, False],
                             help='Generate pseudoknot-free dot-bracket structural annotation in --out_dir. '
                                  'FreeKnot will be used for this purpose. Default: False')
    basic_group.add_argument('--save_contact_map_prob', type=eval, default=False, choices=[True, False],
                             help='Save the upper triangular portion of predicted RNA contact map'
                                  ' probabilities to --out_dir. Default: False.')

    sampling_group = parser.add_argument_group(
        "Sampling options for predicted RNA contact map basepairing probabilities")
    sampling_group.add_argument('--constrained_pairing', type=eval, default=False, choices=[True, False],
                                help='Option to only predict canonical RNA basepairs. Default: False.')
    sampling_group.add_argument('--nonoverlapping', type=eval, default=True, choices=[True, False],
                                help='Option to discard potential multiplets. Default: True.')
    sampling_group.add_argument('--map_threshold', type=float, default=0.2, metavar='0.2',
                                help='Discretization threshold for predicted RNA contact map basepairing probabilities.'
                                     'Default: 0.2.')
    sampling_group.add_argument('--matrix_sampling_mode', type=str, default='greedy',
                                choices=['greedy', 'greedy_bm', 'greedy_bm_shuffled',
                                         'conflict_sampling', 'diff_lp', 'no_sampling', 'probknot'],
                                help='Select a sampling method for discretizing '
                                     'the predicted RNA contact map probabilities. Default: greedy.')
    sampling_group.add_argument('--use_lp_pred', type=eval, default=False, choices=[True, False],
                                help='Add LinearPartition predictions into RSSM ensembles. Default: False.')

    window_group = parser.add_argument_group("Sliding window options for predicting long RNA sequences")
    window_group.add_argument('--enable_sliding_window', type=eval, default=False, choices=[True, False],
                              help='Only predicting local structures by enabling the sliding window.'
                                   'This option improves the accuracy of predicted local structures '
                                   'in long RNA sequences (>1000 nts), in addition to fitting long RNAs '
                                   'into GPU memory. Default: False.')
    window_group.add_argument('--window_size', type=int, default=1000, metavar='500',
                              help='Size of the sliding window. RSSM will not predict any basepairs '
                                   'spanning longer than this window size. Default: 1000.')
    window_group.add_argument('--window_move_increment', type=int, default=500, metavar='100',
                              help='Step size for moving the sliding window from 5\'UTR to 3\'UTR. Default: 500.')

    return parser


def run(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    # configurations
    config = yaml.safe_load(open(os.path.join(basedir, 'RSSMFold', 'rssm_configs', 'cov_rssm_config.yml')))
    if len(args.use_gpu_device) > 0 and torch.cuda.is_available():
        all_device = [torch.device('cuda:%d' % (i)) for i in args.use_gpu_device]
    else:
        all_device = [torch.device('cpu:0')]
    args.all_device = all_device

    if not os.path.exists(cov_rssm_weights_path):
        print(f'Downloading covariance feature based RSSM weights from {cov_rssm_weights_link}')
        download_weights(cov_rssm_weights_link, cov_rssm_weights_path)

    if args.use_lp_pred:
        if not os.path.exists(linearpartition_executable):
            raise ValueError(f'Need LinearPartition binary at {linearpartition_executable} when --use_lp_pred is True')

    # loading trained models
    list_model, list_fc_layer = load_model_ensemble(config, all_device)

    if args.enable_sliding_window:
        if args.batch_size != 1:
            print('setting batch size to 1, due to sliding window computation requirement')
            args.batch_size = 1

    # preparing output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print(f'results will be saved in {args.out_dir}')
    bulk_prediction_from_evo_lib([list_model, list_fc_layer], args)

    if args.generate_dot_bracket:
        print('Generating pseudoknot-free dot-bracket RNA secondary structures')
        all_ids = []
        with h5py.File(args.input_evo_lib_path, 'r', libver='latest', swmr=True) as f:
            for rna_id in f.keys():
                all_ids.append('_'.join(rna_id.split('_')[1:]))

        out_filename = args.input_evo_lib_path.split(os.path.sep)[-1].split('.')[0]
        with open(os.path.join(args.out_dir, f'{out_filename}.dot_bracket'), 'w') as file:
            for seq_id in all_ids:
                bpseq_path = os.path.join(args.out_dir, seq_id + '.bpseq')
                struct = bpseq_to_dot_bracket(bpseq_path)
                file.write('>%s\n%s\n' % (seq_id, struct))


if __name__ == "__main__":
    run()
