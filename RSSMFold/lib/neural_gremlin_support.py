# ------------------------------------------------------------------
# "THE BEERWARE LICENSE" (Revision 42)
# ------------------------------------------------------------------
# <so@g.harvard.edu> and <pkk382@g.harvard.edu> wrote this code.
# As long as you retain this notice, you can do whatever you want
# with this stuff. If we meet someday, and you think this stuff
# is worth it, you can buy us a beer in return.
# --Sergey Ovchinnikov and Peter Koo
# ------------------------------------------------------------------
# The original MATLAB code for GREMLIN was written by Hetu Kamisetty
# ------------------------------------------------------------------
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import subprocess
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Zeros, Constant

################
# note: if you are modifying the alphabet
# make sure last character is "-" (gap)
################
alphabet = "ACGU-"
states = len(alphabet)

# map amino acids to integers (A->0, R->1, etc)
a2n = dict((a, n) for n, a in enumerate(alphabet))
aa2int = lambda x: a2n.get(x, a2n['-'])

gremlin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../RNAcmap/GREMLIN_CPP_MOD')


def get_gremlin_mrf(seq_id, out_dir='./'):
    subprocess.call('OMP_NUM_THREADS=10 {0}/gremlin_cpp -alphabet rna -i {1}.a2m -o {1}.dca -mrf_o {1}.mrf'.format(
        gremlin_path, os.path.join(out_dir, seq_id)), shell=True)

    with open('%s.a2m' % (os.path.join(out_dir, seq_id)), 'r') as file:
        file.readline()
        msa_length = len(file.readline().rstrip())

    dca_mat = np.zeros((msa_length, msa_length, 25))
    conserv_vector = np.zeros((msa_length, 5))

    with open('%s.mrf' % (os.path.join(out_dir, seq_id)), 'r') as file:
        for line in file:
            ret = line.rstrip().split(' ')
            if ret[0].startswith('V'):
                # local fields — nucleic conservation
                site = int(ret[0][2:-1])
                conserv_vector[site] = [float(num) for num in ret[1:]]
            elif ret[0].startswith('W'):
                # two-sites interaction — coupling scores
                site_a, site_b = [int(site[:-1]) for site in ret[0].split('[')[1:]]
                dca_mat[site_a, site_b] = [float(num) for num in ret[1:]]

    return {
        'msa_length': msa_length,
        'pairwise_coupling_triu_k1': dca_mat[np.triu_indices(msa_length, k=1)],
        'local_conservation': conserv_vector,
    }


def get_neural_gremlin_mrf(msa, msa_w):
    '''
    all preprocessing:
    - sample MSA (when it is too deep)
    - remove gaps
    '''
    mrf = GREMLIN({
        'nrow': msa.shape[0],
        'ncol': msa.shape[1],
        'weights': msa_w,
        'neff': np.sum(msa_w),
        'msa': msa,
    })

    return mrf['w'], mrf['v']


# optimizer
def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
    # adam optimizer
    # Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
    # with sum(g*g) instead of (g*g). Furthermore, we find that disabling the bias correction
    # (b_fix=False) speeds up convergence for our case.

    if var_list is None: var_list = tf.trainable_variables()
    gradients = tf.gradients(loss, var_list)
    if b_fix: t = tf.Variable(0.0, "t")
    opt = []
    for n, (x, g) in enumerate(zip(var_list, gradients)):
        if g is not None:
            ini = dict(initializer=tf.zeros_initializer, trainable=False)
            mt = tf.get_variable(name + "_mt_" + str(n), shape=list(x.shape), **ini)
            vt = tf.get_variable(name + "_vt_" + str(n), shape=[], **ini)

            mt_tmp = b1 * mt + (1 - b1) * g
            vt_tmp = b2 * vt + (1 - b2) * tf.reduce_sum(tf.square(g))
            lr_tmp = lr / (tf.sqrt(vt_tmp) + 1e-8)

            if b_fix: lr_tmp = lr_tmp * tf.sqrt(1 - tf.pow(b2, t)) / (1 - tf.pow(b1, t))

            opt.append(x.assign_add(-lr_tmp * mt_tmp))
            opt.append(vt.assign(vt_tmp))
            opt.append(mt.assign(mt_tmp))

    if b_fix: opt.append(t.assign_add(1.0))
    return (tf.group(opt))


def GREMLIN(msa,
            opt_iter=100,
            opt_rate=1.0,
            batch_size=None,
            lam_v=0.01,
            lam_w=0.01,
            scale_lam_w=True,
            v=None,
            w=None,
            ignore_gap=True):
    '''fit params of MRF (Markov Random Field) given MSA (multiple sequence alignment)'''

    ########################################
    # SETUP COMPUTE GRAPH
    ########################################
    # reset tensorflow graph
    tf.reset_default_graph()

    # length of sequence
    ncol = msa["ncol"]

    # input msa (multiple sequence alignment)
    MSA = tf.placeholder(tf.int32, shape=(None, ncol), name="msa")

    # input msa weights
    MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")

    # one-hot encode msa
    OH_MSA = tf.one_hot(MSA, states)

    if ignore_gap:
        ncat = states - 1
        NO_GAP = 1.0 - OH_MSA[..., -1]
        OH_MSA = OH_MSA[..., :ncat]

    else:
        ncat = states

    ########################################
    # V: 1-body-term of the MRF
    ########################################
    V = tf.get_variable(name="V",
                        shape=[ncol, ncat],
                        initializer=tf.zeros_initializer)

    ########################################
    # W: 2-body-term of the MRF
    ########################################
    W_tmp = tf.get_variable(name="W",
                            shape=[ncol, ncat, ncol, ncat],
                            initializer=tf.zeros_initializer)

    # symmetrize W
    W = W_tmp + tf.transpose(W_tmp, [2, 3, 0, 1])

    # set diagonal to zero
    W = W * (1 - np.eye(ncol))[:, None, :, None]

    ########################################
    # Pseudo-Log-Likelihood
    ########################################
    # V + W
    VW = V + tf.tensordot(OH_MSA, W, 2)

    # hamiltonian
    H = tf.reduce_sum(OH_MSA * VW, -1)

    # local Z (parition function)
    Z = tf.reduce_logsumexp(VW, -1)

    PLL = H - Z
    if ignore_gap:
        PLL = PLL * NO_GAP

    PLL = tf.reduce_sum(PLL, -1)
    PLL = tf.reduce_sum(MSA_weights * PLL) / tf.reduce_sum(MSA_weights)

    ########################################
    # Regularization
    ########################################
    L2 = lambda x: tf.reduce_sum(tf.square(x))
    L2_V = lam_v * L2(V)
    L2_W = lam_w * L2(W) * 0.5

    if scale_lam_w:
        L2_W = L2_W * (ncol - 1) * (states - 1)

    ########################################
    # Loss Function
    ########################################
    # loss function to minimize
    loss = -PLL + (L2_V + L2_W) / msa["neff"]

    # optimizer
    opt = opt_adam(loss, "adam", lr=opt_rate)

    ########################################
    # Input Generator
    ########################################
    all_idx = np.arange(msa["nrow"])

    def feed(feed_all=False):
        if batch_size is None or feed_all:
            return {MSA: msa["msa"], MSA_weights: msa["weights"]}
        else:
            batch_idx = np.random.choice(all_idx, size=batch_size)
            return {MSA: msa["msa"][batch_idx], MSA_weights: msa["weights"][batch_idx]}

    ########################################
    # OPTIMIZE
    ########################################
    gpu_options = tf.GPUOptions()
    gpu_options.visible_device_list = '0'
    gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # initialize variables V and W
        sess.run(tf.global_variables_initializer())

        # initialize V
        if v is None:
            oh_msa = np.eye(states)[msa["msa"]]
            if ignore_gap: oh_msa = oh_msa[..., :-1]

            pseudo_count = 0.01 * np.log(msa["neff"])
            f_v = np.einsum("nla,n->la", oh_msa, msa["weights"])
            V_ini = np.log(f_v + pseudo_count)
            if lam_v > 0:
                V_ini = V_ini - np.mean(V_ini, axis=-1, keepdims=True)
            sess.run(V.assign(V_ini))

        else:
            sess.run(V.assign(v))

        # initialize W
        if w is not None:
            sess.run(W_tmp.assign(w * 0.5))

        # compute loss across all data
        # get_loss = lambda: np.round(sess.run(loss, feed(True)) * msa["neff"], 2)

        for i in range(opt_iter):
            sess.run(opt, feed())
        #     if (i + 1) % int(opt_iter / 10) == 0:
        #         print("iter", (i + 1), get_loss())

        # save the V and W parameters of the MRF
        V_ = sess.run(V)
        W_ = sess.run(W)

    ########################################
    # return MRF
    ########################################
    no_gap_states = states - 1
    mrf = {
        "v": V_[:, :no_gap_states],
        "w": W_[:, :no_gap_states, :, :no_gap_states]}
    return mrf


def GREMLIN_simple(msa, msa_weights=None, lam=0.01,
                   opt=None, opt_rate=None,
                   opt_batch=None, opt_epochs=100):
    msa = np.eye(states)[msa]
    N, L, A = msa.shape

    # reset any open sessions/graphs
    K.clear_session()

    gpu_options = tf.GPUOptions()
    gpu_options.visible_device_list = '0'
    gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        K.set_session(sess)

        # constraints
        def cst_w(x):
            '''symmetrize, set diagonal to zero'''
            x = (x + K.transpose(x)) / 2
            zero_mask = K.constant((1 - np.eye(L))[:, None, :, None], dtype=tf.float32)
            x = K.reshape(x, (L, A, L, A)) * zero_mask
            return K.reshape(x, (L * A, L * A))

        # initialiation
        if msa_weights is None:
            Neff = N
            pssm = msa.sum(0)
        else:
            Neff = msa_weights.sum()
            pssm = (msa.T * msa_weights).sum(-1).T

        ini_v = np.log(pssm + lam * np.log(Neff))
        ini_v = Constant(ini_v - ini_v.mean(-1, keepdims=True))
        ini_w = Zeros

        # regularization
        lam_v = l2(lam / N)
        lam_w = l2(lam / N * (L - 1) * (A - 1) / 2)

        # model
        model = Sequential()
        model.add(Flatten(input_shape=(L, A)))
        model.add(Dense(units=L * A,
                        kernel_initializer=ini_w,
                        kernel_regularizer=lam_w,
                        kernel_constraint=cst_w,
                        bias_initializer=ini_v,
                        bias_regularizer=lam_v))
        model.add(Reshape((L, A)))
        model.add(Activation("softmax"))

        @tf.function
        def CCE(true, pred):
            return K.sum(-true * K.log(pred + 1e-8), axis=(1, 2))

        # optimizer settings
        if opt is None: opt = Adam
        if opt_rate is None: opt_rate = 0.1 * np.log(Neff) / L
        if opt_batch is None: opt_batch = N

        model.compile(opt(opt_rate), CCE)

        model.fit(msa, msa, sample_weight=msa_weights,
                  batch_size=opt_batch, epochs=opt_epochs,
                  verbose=True)

        # report loss
        loss = model.evaluate(msa, msa, sample_weight=msa_weights, verbose=False) * N
        print(f"loss: {loss}")

        w, v = model.get_weights()

    return v.reshape((L, A)), w.reshape((L, A, L, A))
