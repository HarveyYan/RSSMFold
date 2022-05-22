import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy.sparse import diags


def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


def soft_sign(x):
    k = 1
    return 1.0 / (1.0 + torch.exp(-2 * k * x))


def postprocess(u, batch_len, threshold, lr_min, lr_max, num_itr=np.inf, rho=0.0, with_l1=False):
    """
    :param u: logits
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    # regularity constraint
    m = torch.zeros_like(u)
    for i, seq_len in enumerate(batch_len):
        m[i, :seq_len, :seq_len] = 1.
    mask = diags([1] * 3, [-1, 0, 1], shape=(m.shape[-2], m.shape[-1])).toarray()
    m = m.masked_fill(torch.Tensor(mask).bool(), 0)

    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - math.log(10 * threshold)) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - math.log(10 * threshold)).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    all_ret = [None] * len(batch_len)
    viable_batch_idx = list(range(len(batch_len)))

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

        # intermediate results
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m

        max_idx = a.max(dim=-1)[1].cpu().numpy()  # batch_size, max_len
        batch_viable_idx = list(range(max_idx.shape[0]))
        batch_idx_to_keep = []
        actual_batch_idx_to_delete = []
        for batch_idx in batch_viable_idx:
            actual_batch_idx = viable_batch_idx[batch_idx]
            seq_len = batch_len[batch_idx]
            ret = torch.zeros(seq_len, seq_len)
            viable_row_idx, viable_col_idx = [], []
            for seq_idx in range(seq_len):
                if a[batch_idx, seq_idx, max_idx[batch_idx, seq_idx]] > 0:
                    viable_row_idx.append(seq_idx)
                    viable_col_idx.append(max_idx[batch_idx, seq_idx])
            if len(viable_row_idx) > 0:
                ret[torch.as_tensor(viable_row_idx), torch.as_tensor(viable_col_idx)] = 1.

            if (ret.transpose(0, 1) == ret).all():
                all_ret[actual_batch_idx] = ret
                actual_batch_idx_to_delete.append(actual_batch_idx)
                print(actual_batch_idx)
            else:
                batch_idx_to_keep.append(batch_idx)

        for actual_batch_idx in actual_batch_idx_to_delete:
            viable_batch_idx.remove(actual_batch_idx)

        if len(batch_idx_to_keep) == 0:
            return all_ret
        else:
            a_hat = torch.index_select(a_hat, 0, torch.as_tensor(batch_idx_to_keep))
            lmbd = torch.index_select(lmbd, 0, torch.as_tensor(batch_idx_to_keep))
            m = torch.index_select(m, 0, torch.as_tensor(batch_idx_to_keep))
            u = torch.index_select(u, 0, torch.as_tensor(batch_idx_to_keep))

    return all_ret


def postprocess_vanilla(logits_map, seq_len, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, threshold=0.9):
    m = torch.zeros_like(logits_map)
    m[:seq_len, :seq_len] = 1.
    mask = diags([1] * 3, [-1, 0, 1], shape=(m.shape[-2], m.shape[-1])).toarray()
    m = m.masked_fill(torch.Tensor(mask).bool().to(m.device), 0)

    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(logits_map - math.log(threshold / (1 - threshold))) * logits_map

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - math.log(threshold / (1 - threshold))).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


