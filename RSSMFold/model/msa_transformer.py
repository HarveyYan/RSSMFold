# Reference to MSA-transformer
# Adaptation made from RM Rao's github repository: github.com/rmrao/msa-transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

from RSSMFold.lib.utils import get_original_pe

nb_type_node = 6  # A, C, G, U, N/mask and gaps


class RowSelfAttention(nn.Module):

    def __init__(self, emb_dim, nhead, activation="relu"):
        super(RowSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.scaling = self.head_dim ** -0.5
        self.activation = activation

        self.q_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(self.emb_dim, self.emb_dim, kernel_size=1)

    def compute_self_attention_weights(self, x, scaling, key_padding_mask, pos_emb=None):
        # x is a MSA with shape: batch_size, emb_dim, nrows, ncols
        # key_padding_mask shape: batch_size, nrows, ncols
        batch_size, emb_dim, nrows, ncols = x.shape

        if pos_emb is not None:
            x = x + pos_emb.permute(0, 3, 1, 2)

        q = self.q_conv(x)
        k = self.k_conv(x)

        q = q * scaling
        q = q.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)
        k = k.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)

        # important note, zero out padded rows in advance,
        # since we are summing over rows in row-wise self-attention
        q = q * (1. - key_padding_mask.unsqueeze(1).unsqueeze(2))

        attn_output_weights = torch.matmul(q.permute(0, 1, 3, 4, 2), k.permute(0, 1, 3, 2, 4)).sum(dim=2)
        # bs, nhead, ncols, ncols

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask[:, 0, :].type(torch.bool).unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        return attn_output_weights

    def compute_self_attention_update(self, x, attn_weights, pos_emb=None):
        batch_size, emb_dim, nrows, ncols = x.shape
        if pos_emb is not None:
            x = x + pos_emb.permute(0, 3, 1, 2)
        attn_output_prob = torch.softmax(attn_weights, dim=-1)
        v = self.v_conv(x)
        v = v.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)
        attn_output = torch.matmul(attn_output_prob.unsqueeze(2), v.permute(0, 1, 3, 4, 2))
        # bs, nhead, 1, ncols, ncols <dot> bs, nheads, nrows, ncols, dim <ret> bs, nhead, nrows, ncols, dim
        attn_output = attn_output.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, emb_dim, nrows, ncols)
        x = self.output_conv(attn_output)
        return x

    def forward(self, src, key_padding_mask, pos_emb=None, ret_attn_weights=False):
        _, emb_dim, nrows, _ = src.shape
        batch_eff_nrows = ((1 - key_padding_mask).sum(-1) > 0).sum(-1)[:, None, None, None]
        scaling = self.scaling * batch_eff_nrows ** -0.5
        attn_weights = self.compute_self_attention_weights(src, scaling, key_padding_mask, pos_emb)
        src = self.compute_self_attention_update(src, attn_weights, pos_emb)
        if ret_attn_weights:
            return src, attn_weights
        else:
            return src


class ColSelfAttention(nn.Module):
    def __init__(self, emb_dim, nhead, activation="relu"):
        super(ColSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.scaling = self.head_dim ** -0.5
        self.activation = activation

        self.q_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(self.emb_dim, self.emb_dim, kernel_size=1)

    def compute_self_attention_weights(self, x, scaling, key_padding_mask, pos_emb=None):
        # x is a MSA with shape: batch_size, emb_dim, nrows, ncols
        # key_padding_mask shape: batch_size, nrows, ncols
        batch_size, emb_dim, nrows, ncols = x.shape

        if pos_emb is not None:
            x = x + pos_emb.permute(0, 3, 1, 2)

        q = self.q_conv(x)
        k = self.k_conv(x)

        q = q * scaling
        q = q.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)
        k = k.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)

        attn_output_weights = torch.matmul(q.permute(0, 1, 4, 3, 2), k.permute(0, 1, 4, 2, 3))
        # bs, nhead, ncols, nrows, nrows

        selected_index = torch.where(key_padding_mask.sum(dim=1) == nrows)
        key_padding_mask[selected_index[0], :, selected_index[1]] = 0.

        if key_padding_mask is not None:  # we might have empty columns here
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.type(torch.bool).transpose(1, 2).unsqueeze(1).unsqueeze(3),
                float('-inf'),
            )

        return attn_output_weights

    def compute_self_attention_update(self, x, attn_weights, pos_emb=None):
        batch_size, emb_dim, nrows, ncols = x.shape
        if pos_emb is not None:
            x = x + pos_emb.permute(0, 3, 1, 2)
        attn_output_prob = torch.softmax(attn_weights, dim=-1)
        v = self.v_conv(x)
        v = v.contiguous().view(batch_size, self.nhead, self.head_dim, nrows, ncols)
        attn_output = torch.matmul(attn_output_prob, v.permute(0, 1, 4, 3, 2))
        # bs, nhead, ncols, nrows, nrows <dot> bs, nheads, ncols, nrows, dim <ret> bs, nhead, ncols, nrows, dim
        attn_output = attn_output.permute(0, 1, 4, 3, 2).contiguous().view(batch_size, emb_dim, nrows, ncols)
        x = self.output_conv(attn_output)
        return x

    def forward(self, src, key_padding_mask, pos_emb=None):
        _, emb_dim, nrows, _ = src.shape
        attn_weights = self.compute_self_attention_weights(src, self.scaling, key_padding_mask, pos_emb)
        src = self.compute_self_attention_update(src, attn_weights, pos_emb)
        return src


class AxialTransformerLayer(nn.Module):

    def __init__(self, emb_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 device=torch.device('cpu')):
        super(AxialTransformerLayer, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation = activation
        self.device = device

        # Row-wise and col-wise self-attention
        self.row_sa = RowSelfAttention(emb_dim, nhead, activation)
        self.row_norm = nn.LayerNorm(emb_dim)
        self.row_dropout = nn.Dropout(dropout)

        self.col_sa = ColSelfAttention(emb_dim, nhead, activation)
        self.col_norm = nn.LayerNorm(emb_dim)
        self.col_dropout = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(emb_dim, dim_feedforward, kernel_size=1)
        self.linear2 = nn.Conv2d(dim_feedforward, emb_dim, kernel_size=1)
        self.mlp_norm = nn.LayerNorm(emb_dim)
        self.mlp_dropout_1 = nn.Dropout2d(dropout)
        self.mlp_dropout_2 = nn.Dropout2d(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def forward(self, src, key_padding_mask, row_pos_emb=None, col_pos_emb=None, ret_attn_weights=False):

        # channel first to channel last in order to use layernorm
        src_row_sa = self.row_norm(src.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        src_row_sa, attn_weights = self.row_sa(src_row_sa, key_padding_mask, row_pos_emb, ret_attn_weights=True)
        src = src + self.row_dropout(src_row_sa)

        src_col_sa = self.col_norm(src.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        src_col_sa = self.col_sa(src_col_sa, key_padding_mask, col_pos_emb)
        src = src + self.col_dropout(src_col_sa)

        # begin feedforward model
        src_mlp = self.mlp_norm(src.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        src_mlp = self.linear2(self.mlp_dropout_1(self.activation(self.linear1(src_mlp))))
        src = src + self.mlp_dropout_2(src_mlp)

        if ret_attn_weights:
            return src, attn_weights
        else:
            return src


class PositionalEmbedding(nn.Module):

    def __init__(self, emb_dim, padding_idx, dropout):
        super(PositionalEmbedding, self).__init__()

        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.dropout = dropout

        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, x):
        # padding should only appear at the end of the sequence
        batch_size, nrows, ncols = x.shape
        mask = x.reshape(-1, ncols).ne(self.padding_idx)  # bsz x nrows, ncols
        seq_lens = mask.sum(dim=-1)

        pe = get_original_pe(seq_lens, ncols, self.emb_dim, return_flattened=False).to(x.device)
        # ncols, dim

        pe = mask.unsqueeze(2) * pe
        learned_pe = self.emb_mlp(pe)
        learned_pe = learned_pe.reshape(batch_size, nrows, ncols, self.emb_dim)

        return learned_pe


class MSATransformer(nn.Module):

    def __init__(self, emb_dim, nhead, nb_layers, all_device, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super(MSATransformer, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.nb_layers = nb_layers
        self.feedforward_dim = feedforward_dim
        self.dropot = dropout
        self.activation = activation

        if type(all_device) is torch.device:
            self.all_device = [all_device]
        else:
            assert len(all_device) > 0
            self.all_device = all_device

        nb_device = len(self.all_device)
        nb_layer_per_device = nb_layers // nb_device
        self.layer_split_idx = list(range(nb_layer_per_device - 1, nb_layers - 1, nb_layer_per_device))
        self.x_embedding = nn.Embedding(nb_type_node + 1, emb_dim, padding_idx=nb_type_node).to(self.all_device[0])
        self.row_wise_pos_emb = PositionalEmbedding(emb_dim, nb_type_node, dropout).to(self.all_device[0])
        self.col_wise_pos_emb = PositionalEmbedding(emb_dim, nb_type_node, dropout).to(self.all_device[0])

        layers = []
        for i in range(nb_layers):
            device = None
            for j, layer_split_idx in enumerate(self.layer_split_idx):
                if i <= layer_split_idx:
                    device = self.all_device[j]
                    break
            if device is None:
                device = self.all_device[-1]
            layer = AxialTransformerLayer(emb_dim, nhead, feedforward_dim, dropout, activation, device=device).to(
                device)
            print(i, device)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):

        all_attn_weights = []
        key_padding_mask = x.eq(nb_type_node).int()

        row_pos_emb = self.row_wise_pos_emb(x)
        col_pos_emb = self.col_wise_pos_emb(x.transpose(1, 2)).transpose(1, 2)

        x = self.x_embedding(x)
        x = x + row_pos_emb + col_pos_emb
        # bsz, nrows, ncols, emb

        x = x.permute(0, 3, 1, 2)
        for layer in self.layers:
            if x.device != layer.device:
                x = x.to(layer.device)
                col_pos_emb = col_pos_emb.to(layer.device)
                row_pos_emb = row_pos_emb.to(layer.device)
                key_padding_mask = key_padding_mask.to(layer.device)
            x, attn_weights = layer(x, key_padding_mask, col_pos_emb, row_pos_emb, ret_attn_weights=True)
            all_attn_weights.append(attn_weights.to(self.all_device[-1]))

        all_attn_weights = torch.cat(all_attn_weights, dim=1)
        x = x.permute(0, 2, 3, 1)
        return x, all_attn_weights
