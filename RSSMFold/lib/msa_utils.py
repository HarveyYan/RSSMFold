# This assumes that you have installed RNAfold, blastn and Infernal in your system (callable from shell)
import os
import pathlib
import subprocess
import re
import gzip
import numpy as np
import logging

import RSSMFold
from RSSMFold.lib.dca_support import get_msa_eff, get_single_site_stats, get_pair_site_stats

basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()

rfam_v14_path = os.path.join(os.path.dirname(basedir), 'RFAMv14.5')
rnacmap_base = os.path.join(os.path.dirname(basedir), 'RNAcmap')
blastn_database_path = os.path.join(rnacmap_base, 'nt_database', 'nt')
# compute canada's ncbi nucleotide library is slow and outofdated as hell
# '/cvmfs/ref.mugqic/genomes/blast_db/LATEST/nt'
# modified version which retains MSA columns that contain target sequence residuals

VOCAB = ['A', 'C', 'G', 'U', '-']
N_VOCAB = len(VOCAB)

if not os.path.exists(rfam_v14_path):
    logging.warning(f'Missing RFAMv14.5 library in path {rfam_v14_path}. '
                    f'Alignment based RSSM may not function properly')

if not os.path.exists(rnacmap_base):
    logging.warning(f'Missing RNAcmap library in path {rfam_v14_path}. '
                    f'Alignment based RSSM may not function properly')

if not os.path.exists(rnacmap_base):
    logging.warning(f'Missing NCBI nucleotide database in path {rfam_v14_path}. '
                    f'Alignment based RSSM may not function properly')


def match_to_rfam_id(seq, seq_id='tmp', out_dir='./', ncores=20):
    '''
    use cmscan to see if the target sequence can match an existing RNA family in RFAM
    '''

    with open(os.path.join(out_dir, '%s.seq' % (seq_id)), 'w') as file:
        file.write('>%s\n%s\n' % (seq_id, seq))

    subprocess.call(
        'cmscan --nohmmonly --rfam --cut_ga --fmt 2 --oclan --oskip --cpu={2} --clanin {0}/Rfam.clanin '
        '-o {1}.cmscan_out --tblout {1}.tblout {0}/Rfam.cm {1}.seq'
            .format(rfam_v14_path, os.path.join(out_dir, seq_id), ncores), shell=True)

    with open('%s.tblout' % (os.path.join(out_dir, seq_id)), 'r') as file:
        rfam_id = None
        for line in file:
            if not line.startswith('#'):
                res = re.sub('\s+', ' ', line).split(' ')
                rfam_id = res[2]
                break  # return the highest match

    return rfam_id


def stk_to_msa_with_target(seq_id, out_dir='./'):
    '''
    stockholm format to a simpler MSA format
    note that we shall keep inserted columns (w.r.t. the covariance model)
    as long as those columns contain target sequence residuals, and the target
    sequence shall be the first sequence in the alignment
    '''
    with open(os.path.join(out_dir, '%s.stk' % (seq_id)), 'r') as file:
        ids, desc, id2seq = [], [], {}
        for line in file:
            line = line.rstrip()
            if line.startswith('#=GS'):
                res = re.sub('\s+', ' ', line).split(' ')
                ids.append(res[1])
                desc.append(' '.join(res[3:]))

            if len(line) > 0 and not line.startswith('#') and not line.startswith('//'):
                cur_id, seq = re.sub('\s+', ' ', line).split(' ')
                if cur_id not in id2seq:
                    id2seq[cur_id] = seq
                else:
                    id2seq[cur_id] += seq

        seqs = [id2seq[cur_id] for cur_id in ids]
        # remove redundant sequences
        _, idx = np.unique(seqs, return_index=True)
        seqs = np.array(seqs)[np.sort(idx)]
    # '.' indicates insertion with regard to the covariance model
    # delete inserted columns unless they contain target nucleotides
    idx = np.where(np.array(list(seqs[0])) != '.')

    with open(os.path.join(out_dir, '%s.a2m' % (seq_id)), 'w') as file:
        for id, desc, seq in zip(ids, desc, seqs):
            file.write('>%s %s\n%s\n' % (
                id, desc, ''.join(np.array(list(seq))[idx]).replace('.', '-').upper()))

    return os.path.join(out_dir, '%s.a2m' % (seq_id))


def save_rfam_msa(seq, seq_id, rfam_id, out_dir='./', cap_cmalign_msa_depth=np.inf, ncores=20):
    '''
    obtain MSA of *all* RNA sequences in the same RFAM family as the target sequence
    '''
    with gzip.open(os.path.join(rfam_v14_path, 'fasta_files', rfam_id + '.fa.gz'), 'rb') as in_file:
        all_id, all_seq = [], []
        for line in in_file:
            line = line.decode('ascii').rstrip()
            if line.startswith('>'):
                all_id.append(line)
            elif len(line) > 0:
                all_seq.append(line.replace('T', 'U'))

        if len(all_seq) > cap_cmalign_msa_depth:
            selected_idx = np.sort(np.random.choice(np.arange(len(all_seq)), cap_cmalign_msa_depth, False))
            all_id = np.array(all_id)[selected_idx]
            all_seq = np.array(all_seq)[selected_idx]

        with open(os.path.join(out_dir, '%s.fa' % (seq_id)), 'w') as out_file:
            out_file.write('>target_seq %s\n%s\n' % (seq_id, seq.upper()))
            for id_, seq_ in zip(all_id, all_seq):
                if len(seq_) <= 2000:
                    # need to restrict sequence length, vis-a-vis RFAM 02541
                    out_file.write('%s\n%s\n' % (id_, seq_))

    # align these sequences using the family's covariance model
    subprocess.call('cmalign --cpu={3} {0}/all_rfam_cm/{1}.cm {2}.fa > {2}.stk'.format(
        rfam_v14_path, rfam_id, os.path.join(out_dir, seq_id), ncores), shell=True)

    # converting the format
    msa_filepath = stk_to_msa_with_target(seq_id, out_dir)

    return msa_filepath


def save_rnacmap_msa(seq, seq_id, out_dir='./', cap_rnacmap_msa_depth=50000, ncores=20, dl_struct=None, ret_size=False):
    # RNAcmap but executed in python (originally with shell)

    with open(os.path.join(out_dir, '%s.seq' % (seq_id)), 'w') as file:
        file.write('>%s\n%s\n' % (seq_id, seq))

    # first round blastn search and reformat its output
    # GC RF determines which are insertions relative to the consensus
    cmd = '''
        blastn -db {0} -query {2}.seq -out {2}.bla -evalue 0.001 -num_descriptions 1 -num_threads {3} -line_length 1000 -num_alignments {4}
        {1}/parse_blastn_local.pl {2}.bla {2}.seq {2}.aln
        {1}/reformat.pl fas sto {2}.aln {2}.sto
        '''.format(blastn_database_path, rnacmap_base, os.path.join(out_dir, seq_id), ncores, cap_rnacmap_msa_depth)
    subprocess.call(cmd, shell=True)

    if dl_struct is None:
        # RNAfold for only the target sequence
        cmd = '''
        RNAfold {0}.seq | awk '{{print $1}}' | tail -n +3 > {0}.dbn
        for i in `awk '{{print $2}}' {0}.sto | head -n5 | tail -n1 | grep -b -o - | sed 's/..$//'`; do sed -i "s/./&-/$i" {0}.dbn; done
        head -n -1 {0}.sto > {1}.sto
        echo "#=GC SS_cons                     "`cat {0}.dbn` > {1}.txt
        cat {1}.sto {1}.txt > {0}.sto
        echo "//" >> {0}.sto
        '''.format(os.path.join(out_dir, seq_id), os.path.join(out_dir, 'temp'))
        subprocess.call(cmd, shell=True)
    else:
        # deep learning predicted structure
        cmd = '''
        echo "{2}" > {0}.dbn
        for i in `awk '{{print $2}}' {0}.sto | head -n5 | tail -n1 | grep -b -o - | sed 's/..$//'`; do sed -i "s/./&-/$i" {0}.dbn; done
        head -n -1 {0}.sto > {1}.sto
        echo "#=GC SS_cons                     "`cat {0}.dbn` > {1}.txt
        cat {1}.sto {1}.txt > {0}.sto
        echo "//" >> {0}.sto
        '''.format(os.path.join(out_dir, seq_id), os.path.join(out_dir, 'temp'), dl_struct)
        subprocess.call(cmd, shell=True)


    # option 2 todo, use RNAalifold to obtain css

    # second round covariance model search
    cmd = '''
    cmbuild --hand -F {0}.cm {0}.sto
	cmcalibrate {0}.cm
	cmsearch -o {0}.out -A {0}.msa --cpu {1} --incE 10.0 {0}.cm {2}
	{3} --replace acgturyswkmbdhvn:................ a2m {0}.msa > {4}.a2m
    '''.format(os.path.join(out_dir, seq_id), ncores, blastn_database_path, os.path.join(rfam_v14_path, 'esl-reformat'),
               os.path.join(out_dir, 'temp'))
    ret_code = subprocess.call(cmd, shell=True)

    if ret_code == 0:
        # constrain maximal depth of the MSA
        with open(os.path.join(out_dir, 'temp.a2m'), 'r') as in_file:
            all_id, all_seq = [], []
            seq_ = ''
            for line in in_file:
                if line.startswith('>'):
                    all_id.append(line.rstrip())
                    if len(seq_) > 0:
                        all_seq.append(seq_)
                        seq_ = ''
                elif len(line) > 0:
                    seq_ += line.rstrip().replace('T', 'U').upper()
            if len(seq_) > 0:
                all_seq.append(seq_)

            if len(all_seq) > cap_rnacmap_msa_depth:
                selected_idx = np.sort(np.random.choice(np.arange(len(all_seq)), cap_rnacmap_msa_depth, False))
                all_id = np.array(all_id)[selected_idx]
                all_seq = np.array(all_seq)[selected_idx]
    else:
        # when <seq_id>.msa is empty
        all_id, all_seq = [], []

    with open(os.path.join(out_dir, '%s.a2m' % (seq_id)), 'w') as out_file:
        out_file.write('>target_seq %s\n%s\n' % (seq_id, seq.upper()))  # imperative to call an `upper' here
        for id_, seq_ in zip(all_id, all_seq):
            if len(seq_) <= 2000:
                # need to restrict sequence length, vis-a-vis RFAM 02541
                out_file.write('%s\n%s\n' % (id_, seq_))

    if ret_size:
        return os.path.join(out_dir, '%s.a2m' % (seq_id)), len(all_seq)
    else:
        return os.path.join(out_dir, '%s.a2m' % (seq_id))


def filter_gaps(msa, gap_cutoff=0.5):
    '''
    filter alignment to remove gappy positions
    '''
    frac_gaps = np.mean((msa == N_VOCAB - 1).astype(np.float), 0)
    # mostly non-gaps, or containing target residuals
    col_idx = np.where((frac_gaps < gap_cutoff) | (msa[0] != (N_VOCAB - 1)))[0]
    return msa[:, col_idx], col_idx


def read_msa(msa_filepath, cap_msa_depth=np.inf, eff_cutoff=0.8, gap_cutoff=0.5):
    '''
    read and preprocess MSA
    - set non-vocab characters to gaps
    - char to int
    - cap MSA depth by sampling
    - filter gaps
    - get sequence weights
    '''
    msa = []
    with open(msa_filepath, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                # non-vocab characters to gaps
                msa.append([char if char in VOCAB else VOCAB[-1] for char in line.rstrip()])
    msa = np.array(msa)

    # char to int
    for n in range(len(VOCAB)):
        msa[msa == VOCAB[n]] = n
    msa = msa.astype(np.int)

    # 1. cap MSA depth if necessary
    # 2. remove gaps if necessary
    if msa.shape[0] > cap_msa_depth:
        # for weighted sampling — more unique sequences
        msa_w = get_msa_eff(msa[1:], eff_cutoff)
        sampled_idx = np.random.choice(np.arange(1, msa.shape[0]), cap_msa_depth - 1, False, msa_w / np.sum(msa_w))
        sampled_idx = np.sort(sampled_idx)
        msa = np.concatenate([msa[:1, :], msa[sampled_idx]], axis=0)

    msa, v_idx = filter_gaps(msa, gap_cutoff)
    msa_w = get_msa_eff(msa, eff_cutoff)

    return msa, msa_w, v_idx


def get_msa_covariance(msa, msa_w, pse_count=None):
    msa_length = msa.shape[1]
    # effective MSA depth
    m_eff = msa_w.sum()

    if pse_count is None:
        pse_count = m_eff

    # local statistics
    local_stats = get_single_site_stats(msa, msa_w, pse_count)

    # pairwise statistics, upper triangular offset 1
    pairwise_stats_triu_k1 = get_pair_site_stats(msa, msa_w, pse_count)
    # to full covariance matrix
    pairwise_stats = np.zeros((msa_length, msa_length, N_VOCAB, N_VOCAB))
    pairwise_stats[np.triu_indices(msa_length, k=1)] = pairwise_stats_triu_k1
    # care: joint-single index must transpose at the same time
    pairwise_stats = pairwise_stats + pairwise_stats.transpose((1, 0, 3, 2))
    # adding diagonal content to the covariance matrix
    diagonal = np.zeros((msa_length, N_VOCAB, N_VOCAB))
    # diagonal.fill(pse_count / (pse_count + m_eff) / N_VOCAB ** 2)
    diagonal[:, np.arange(N_VOCAB), np.arange(N_VOCAB)] = \
        local_stats  # + pse_count * (1 - N_VOCAB) / N_VOCAB ** 2 / (m_eff + pse_count)
    pairwise_stats[np.arange(msa_length), np.arange(msa_length)] = diagonal

    # covariance
    cov = pairwise_stats.reshape((msa_length, msa_length, N_VOCAB, N_VOCAB)) - \
          np.matmul(local_stats[:, None, :, None], local_stats[None, :, None, :])
    cov = cov.transpose(0, 2, 1, 3)  # l x N_VOCAB x l x N_VOCAB

    return {
        'single_site_stats': local_stats,
        'pairwise_stats_triu_k1': pairwise_stats_triu_k1,
        'covariance_mat': cov
    }
