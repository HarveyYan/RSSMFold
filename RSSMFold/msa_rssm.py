import argparse
import torch
import os
import numpy as np
import pathlib
import yaml
from tqdm import tqdm
import h5py
import warnings

torch.set_num_threads(1)

import RSSMFold
from RSSMFold.model.joint_msa_transformer_rssm import JointEvoModel
from RSSMFold.lib.contact_map_utils import discretize_contact_map
from RSSMFold.utility_scripts.deployment_utils import download_weights, \
    msa_rssm_weights_link, msa_rssm_weights_path
from RSSMFold.utility_scripts.service_utils import load_fc_blocks, augment_linearpartition, bpseq_to_dot_bracket

NUC_VOCAB = ['A', 'C', 'G', 'U', 'N']
nb_type_node = 6  # the last dimension is for gaps
basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
linearpartition_executable = os.path.join(basedir, 'LinearPartition', 'linearpartition')


def get_alignment_idx(msa, target_seq_idx, original_node_features):
    target_seq_in_msa = msa[target_seq_idx]
    # idx of 4 means gap in MSA
    idx_to_retain_in_msa_seq = \
        [i for i, idx in enumerate(target_seq_in_msa) if idx != 4]  # 4 means gap here
    idx_corresponding_in_original_seq = \
        [i for i, idx in enumerate(original_node_features) if idx != 4]  # 4 means N/unknown

    return [idx_to_retain_in_msa_seq, idx_corresponding_in_original_seq]


def msa_predictor_function(seq_string, batch_seq, msa, batch_len, model, args):
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

    msa_depth_modulation_constant = args.msa_depth_modulation_constant
    max_nb_iters = args.max_nb_iters

    ret = {}
    model, fc_layer = model
    msa_depth, msa_length = msa.shape
    max_depth = 2 ** msa_depth_modulation_constant // msa_length
    max_len = batch_seq.shape[1]
    batch_len_np = np.array([max_len])

    with torch.no_grad():

        if msa_depth <= max_depth:
            nb_iters = 1
        else:
            nb_iters = max_nb_iters

        all_map_triu = []
        original_node_features = batch_seq.cpu().numpy().tolist()[0]
        for random_iter in range(nb_iters):

            # perform MSA subsampling
            if msa_depth <= max_depth:
                sampled_msa = msa
                target_seq_idx = [0]
            else:
                # random sampling
                sampled_idx = np.random.choice(msa_depth, max_depth, replace=False)
                # always include target sequence
                if 0 not in sampled_idx:
                    sampled_idx = np.array(sampled_idx.tolist()[:-1] + [0])
                sampled_msa = msa[sampled_idx]
                target_seq_idx = np.where(sampled_idx == 0)[0].tolist()

            alignment_idx = get_alignment_idx(sampled_msa, target_seq_idx[0], original_node_features)
            if not enable_sliding_window or max_len <= window_size:
                sampled_msa[sampled_msa == 4] = 5  # reassign the index of gap to 5
                all_patch_map_triu = model(
                    torch.as_tensor(sampled_msa, dtype=torch.long)[None, :, :].to(all_device[0]), [msa_length],
                    target_seq_idx, batch_seq, None, batch_len, [alignment_idx])
                contact_map_triu = torch.sigmoid(fc_layer[0](all_patch_map_triu[0][0]))
            else:
                cumsum_contact_map = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                count_contact_map = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                idx_to_retain_in_msa_seq, idx_corresponding_in_original_seq = alignment_idx

                for start_idx in range(window_move_increment - window_size, max_len, window_move_increment):
                    end_idx = start_idx + window_size
                    if start_idx < 0:
                        start_idx = 0

                    windowed_x = batch_seq[:, start_idx: end_idx]
                    batch_len = windowed_x.shape[1]
                    windowed_batch_len = torch.as_tensor([batch_len]).to(all_device[0])

                    idx_corresponding_in_original_seq = np.array(idx_corresponding_in_original_seq)
                    msa_start = np.sum(idx_corresponding_in_original_seq < start_idx)
                    msa_end = len(idx_to_retain_in_msa_seq) - np.sum(idx_corresponding_in_original_seq >= end_idx)
                    windowed_sampled_msa = sampled_msa[:, msa_start: msa_end]

                    windowed_alignment_idx = get_alignment_idx(
                        windowed_sampled_msa, target_seq_idx[0], original_node_features[start_idx: end_idx])
                    windowed_sampled_msa[windowed_sampled_msa == 4] = 5  # reassign the index of gap to 5
                    patch_map_triu = model(
                        torch.as_tensor(windowed_sampled_msa, dtype=torch.long)[None, :, :].to(all_device[0]),
                        [msa_end - msa_start], target_seq_idx, windowed_x, None, windowed_batch_len,
                        [windowed_alignment_idx])
                    contact_map_triu = torch.sigmoid(fc_layer[0](patch_map_triu[0][0]))
                    x_idx, y_idx = np.triu_indices(batch_len)
                    cumsum_contact_map[x_idx + start_idx, y_idx + start_idx] += contact_map_triu
                    count_contact_map[x_idx + start_idx, y_idx + start_idx] += torch.ones_like(contact_map_triu)

                count_contact_map[count_contact_map == 0.] = 1.
                contact_map_triu = (cumsum_contact_map / count_contact_map)[np.triu_indices(max_len)]

            all_map_triu.append(contact_map_triu)

        if save_contact_map_prob:
            ret['all_map_triu'] = all_map_triu

        if use_lp_pred:
            # practically adding LinearPartition into RSSM ensembles
            if not enable_sliding_window or max_len <= window_size:
                lp_triu = augment_linearpartition([seq_string], 0., outdir)
                lp_triu = torch.tensor(lp_triu).to(all_device[-1])
            else:
                cumsum_lp_triu = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                count_lp_triu = torch.zeros(max_len, max_len, 1).to(all_device[-1])
                for start_idx in range(window_move_increment - window_size, max_len, window_move_increment):
                    end_idx = start_idx + window_size
                    if start_idx < 0:
                        start_idx = 0
                    windowed_seqs = [seq_string[start_idx: end_idx]]
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


def msa_predictor(seq, msa, seq_id, model, args):
    out_dir = args.out_dir
    all_device = args.all_device
    save_contact_map_prob = args.save_contact_map_prob

    max_len = nb_nodes = len(seq)
    all_len = [nb_nodes]

    node_features = list(map(lambda x: NUC_VOCAB.index(x), seq))
    batch_seq = [node_features + [nb_type_node] * (max_len - nb_nodes)]
    batch_seq = torch.tensor(np.array(batch_seq, dtype=np.long)).to(all_device[0])
    batch_len = torch.tensor(np.array(all_len, dtype=np.long)).to(all_device[0])

    ret = msa_predictor_function(seq, batch_seq, msa, batch_len, model, args)

    sampled_pairs = ret['sampled_pairs'][0]
    sampled_pairs_dict = {}
    for pair in sampled_pairs:
        sampled_pairs_dict[pair[0]] = pair[1] + 1
        sampled_pairs_dict[pair[1]] = pair[0] + 1

    with open(os.path.join(out_dir, f'{seq_id}.bpseq'), 'w') as file:
        for i, c in enumerate(seq):
            file.write(f'{i + 1} {c} {sampled_pairs_dict.get(i, 0)}\n')

    if save_contact_map_prob:
        np.save(os.path.join(out_dir, f'{seq_id}_prob.npy'), ret['all_map_triu'])


def bulk_prediction_from_evo_lib(model, args):
    evo_lib_path = args.input_evo_lib_path

    all_ids, all_seqs, all_msa, all_alignment_idx = [], [], [], []

    with h5py.File(evo_lib_path, 'r', libver='latest', swmr=True) as f:
        for rna_id in f.keys():
            seq = f[rna_id]['seq'][...]
            if type(seq) is np.ndarray:
                seq = seq.tolist()
            if type(seq) is bytes:
                seq = seq.decode('ascii')
            seq = ''.join(list(map(lambda c: c if c in NUC_VOCAB else 'N', seq.upper().replace('T', 'U'))))
            node_features = list(map(lambda x: NUC_VOCAB.index(x), seq))
            all_ids.append(rna_id)
            # all_ids.append('_'.join(rna_id.split('_')[1:]))
            all_seqs.append(seq)

            if 'full_msa' in f[rna_id]:
                msa = f[rna_id]['full_msa'][...]
            elif 'msa' in f[rna_id]:
                msa = f[rna_id]['msa'][...]
            else:
                msa = torch.tensor(node_features)[None, :]
            all_msa.append(msa)

    if args.verbose:
        bar = tqdm(total=len(all_ids), position=0, leave=True)

    for i in range(len(all_ids)):
        msa_predictor(all_seqs[i], all_msa[i], all_ids[i], model, args)

        if args.verbose:
            bar.update(1)

    if args.verbose:
        bar.close()


def load_model(config, all_device):
    loaded_weights = torch.load(
        os.path.join(basedir, 'RSSMFold', 'rssm_weights', config['weights']), map_location='cpu')

    # models providing embeddings
    model = JointEvoModel(
        config['t_emb_dim'], config['t_nhead'], config['msa_transformer_nb_layers'],
        config['num_ds_steps'], all_device)

    # fully connected models providing basepairing predictions
    all_fc_blocks = load_fc_blocks(config)
    all_fc_blocks = torch.nn.ModuleList(all_fc_blocks).to(all_device[-1])

    # loading weights
    model_weights, fc_weights = loaded_weights
    model.load_state_dict(model_weights)
    all_fc_blocks.load_state_dict(fc_weights)
    print('MSA-RSSM loaded')

    # setting to evaluation mode
    model.eval()
    all_fc_blocks.eval()

    return model, all_fc_blocks


def get_parser():
    parser = argparse.ArgumentParser(
        description='MSA based RNA Secondary Structural Model predictor')

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
    basic_group.add_argument('--generate_dot_bracket', type=eval, default=False, choices=[True, False],
                             help='Generate pseudoknot-free dot-bracket structural annotation in --out_dir. '
                                  'FreeKnot will be used for this purpose. Default: False')
    basic_group.add_argument('--save_contact_map_prob', type=eval, default=False, choices=[True, False],
                             help='Save the upper triangular portion of predicted RNA contact map'
                                  ' probabilities to --out_dir. Default: False.')

    msa_group = parser.add_argument_group(
        "MSA subsampling options")
    msa_group.add_argument('--msa_depth_modulation_constant', type=int, default=13,
                           help='The size of subsampled MSA: 2**depth_constant/length.Default: 13.')
    msa_group.add_argument('--max_nb_iters', type=int, default=10,
                           help='Maximal iterations of MSA subsampling. Default: 10.')

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
    config = yaml.safe_load(open(os.path.join(basedir, 'RSSMFold', 'rssm_configs', 'msa_rssm_config.yml')))
    if len(args.use_gpu_device) > 0 and torch.cuda.is_available():
        all_device = [torch.device('cuda:%d' % (i)) for i in args.use_gpu_device]
    else:
        all_device = [torch.device('cpu:0')]
    args.all_device = all_device

    if not os.path.exists(msa_rssm_weights_path):
        print(f'Downloading MSA based RSSM weights from {msa_rssm_weights_link}')
        download_weights(msa_rssm_weights_link, msa_rssm_weights_path, args.verbose)

    if args.use_lp_pred:
        if not os.path.exists(linearpartition_executable):
            raise ValueError(f'Need LinearPartition binary at {linearpartition_executable} when --use_lp_pred is True')

    if args.enable_sliding_window and args.window_size < args.window_move_increment:
        warnings.warn('specified window_size smaller than window_move_increment')

    # loading trained models
    model, fc_layer = load_model(config, all_device)

    # preparing output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print(f'results will be saved in {args.out_dir}')
    bulk_prediction_from_evo_lib([model, fc_layer], args)

    if args.generate_dot_bracket:
        print('Generating pseudoknot-free dot-bracket RNA secondary structures')
        all_ids = []
        with h5py.File(args.input_evo_lib_path, 'r', libver='latest', swmr=True) as f:
            for rna_id in f.keys():
                all_ids.append(rna_id)
                # all_ids.append('_'.join(rna_id.split('_')[1:]))

        out_filename = args.input_evo_lib_path.split(os.path.sep)[-1].split('.')[0]
        with open(os.path.join(args.out_dir, f'{out_filename}.dot_bracket'), 'w') as file:
            for seq_id in all_ids:
                bpseq_path = os.path.join(args.out_dir, seq_id + '.bpseq')
                struct = bpseq_to_dot_bracket(bpseq_path)
                file.write('>%s\n%s\n' % (seq_id, struct))


if __name__ == "__main__":
    run()
