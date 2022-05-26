import os
import pathlib
import numpy as np
import subprocess
import random
import pandas as pd
import torch
import torch.nn as nn
import RSSMFold

NUC_VOCAB = ['A', 'C', 'G', 'U', 'N']
basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
linearpartition_executable = os.path.join(basedir, 'LinearPartition', 'linearpartition')


def read_fasta_file(fasta_path):
    all_ids, all_seqs = [], []
    with open(fasta_path, 'r') as file:
        read_seq = ''
        for line in file:
            line = line.rstrip()
            if line[0] == '>':
                all_ids.append(line[1:].rstrip())
                if len(read_seq) > 0:
                    all_seqs.append(read_seq)
                    read_seq = ''
            else:
                seq_one_line = line.upper().replace('T', 'U')
                seq_one_line = ''.join(list(map(lambda c: c if c in NUC_VOCAB else 'N', seq_one_line)))
                read_seq += seq_one_line
        if len(read_seq) > 0:
            all_seqs.append(read_seq)
    return all_ids, all_seqs


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


def define_fc_block(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.CELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim)
    )


def load_fc_blocks(config):
    num_fc_blocks = 2
    all_fc_blocks = []
    size_output = 1

    for i in range(config['num_ds_steps'] + 2):
        fc_blocks = None
        dim = config['t_emb_dim'] * 2 ** max(i - 1, 0)
        fc_dim = dim * 2
        for _ in range(num_fc_blocks):
            block = define_fc_block(dim, fc_dim, 0.5)
            if fc_blocks is None:
                fc_blocks = block
            else:
                fc_blocks = torch.nn.Sequential(*fc_blocks, *block)
            dim = fc_dim
        fc_blocks = torch.nn.Sequential(*fc_blocks, *define_fc_block(dim, size_output, 0.5))
        all_fc_blocks.append(fc_blocks)
    return all_fc_blocks
