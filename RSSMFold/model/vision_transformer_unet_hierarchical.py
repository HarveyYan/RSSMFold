import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.utils import checkpoint
from RSSMFold.lib.utils import get_original_pe

# A, C, G, U, N/mask and gaps
nb_type_node = 6


def get_conv_block(in_dim, out_dim, kernel_size, dropout, stride, padding):
    return nn.Sequential(
        MaskedLayerNorm(in_dim),
        nn.CELU(),
        nn.Dropout2d(dropout),
        nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
    )


def get_transposed_conv_block(in_dim, out_dim, kernel_size, dropout, stride, padding):
    return nn.Sequential(
        MaskedLayerNorm(in_dim),
        nn.CELU(),
        nn.Dropout2d(dropout),
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
    )


def get_fc_block(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.CELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim)
    )


class MaskedLayerNorm(nn.LayerNorm):

    def __init__(self, num_features):
        self.num_features = num_features
        super(MaskedLayerNorm, self).__init__(num_features, elementwise_affine=True)

    def forward(self, pair_map_mask):
        map, mask = pair_map_mask
        # map: NCHW, mask: N1HW (True indicates padding)

        mask = 1. - mask.float()
        n = torch.sum(mask, dim=[1, 2, 3], keepdim=True) * self.num_features
        mean = torch.sum(map * mask, dim=[1, 2, 3], keepdim=True) / n
        var = torch.sum(((map - mean) * mask) ** 2, dim=[1, 2, 3], keepdim=True) / n

        normed_map = (map - mean) / torch.sqrt(var + self.eps) * self.weight[None, :, None, None] + \
                     self.bias[None, :, None, None]
        normed_map = normed_map.masked_fill(pair_map_mask[1], 0.)
        return normed_map


class UNetVTEncoder(nn.Module):

    def __init__(self, all_device, layer_split_idx, nb_stride2_downsampling, emb_dim, dropout,
                 map_concat_mode='sum', patch_ds_stride=20, **kwargs):
        super(UNetVTEncoder, self).__init__()

        self.all_device = all_device
        self.layer_split_idx = layer_split_idx
        self.nb_stride2_downsampling = nb_stride2_downsampling
        self.emb_dim = emb_dim
        self.patch_ds_stride = patch_ds_stride  # for down-sampling original contact maps to patches

        self.nhead = kwargs.get('nhead', 4)
        # self.dim_feedforward = kwargs.get('dim_feedforward', 2048)
        self.use_conv_proj = kwargs.get('use_conv_proj', False)
        self.use_lw = kwargs.get('use_lw', False)
        self.nb_pre_convs = kwargs.get('nb_pre_convs', 10)
        self.nb_post_convs = kwargs.get('nb_post_convs', 10)
        self.enable_5x5_filter = kwargs.get('enable_5x5_filter', False)

        pre_convs = []
        for i in range(self.nb_pre_convs):  # padding set to maintain spatial dimensionality
            if self.enable_5x5_filter:
                if i % 2 == 0:
                    filter_size = 3
                else:
                    filter_size = 5
            else:
                filter_size = 3
            pre_convs.append(get_conv_block(
                emb_dim, emb_dim, filter_size, dropout, 1, (filter_size - 1) // 2).to(all_device[0]))
        self.pre_convs = nn.ModuleList(pre_convs)

        post_convs = []
        for i in range(self.nb_post_convs):  # padding set to maintain spatial dimensionality
            if self.enable_5x5_filter:
                if i % 2 == 0:
                    filter_size = 3
                else:
                    filter_size = 5
            else:
                filter_size = 3
            post_convs.append(get_conv_block(
                emb_dim, emb_dim, filter_size, dropout, 1, (filter_size - 1) // 2).to(all_device[-1]))
        self.post_convs = nn.ModuleList(post_convs)

        self.patch_ds_conv = get_conv_block(  # to manually add paddings depending on patch_ds_stride
            emb_dim, emb_dim, patch_ds_stride, dropout, stride=patch_ds_stride, padding=0).to(all_device[0])

        t_layers = []
        stride2_ds_convs = []
        # stride2_us_convs = []
        stride3_merge_convs = []

        dim = self.emb_dim
        for i in range(nb_stride2_downsampling * 2 + 1):
            device = None
            for j, split_idx in enumerate(layer_split_idx):
                if i <= split_idx:
                    device = self.all_device[j]
                    break
            if device is None:
                device = self.all_device[-1]

            t_layers.append(VTBlock(dim, self.nhead, dim * 2, dropout, use_conv_proj=self.use_conv_proj,
                                    use_lw=self.use_lw, use_cyclic_shift=False).to(device))
            t_layers.append(VTBlock(dim, self.nhead, dim * 2, dropout, use_conv_proj=self.use_conv_proj,
                                    use_lw=self.use_lw, use_cyclic_shift=True).to(device))
            if i < nb_stride2_downsampling:
                stride2_ds_convs.append(get_conv_block(dim, dim * 2, 2, dropout, stride=2, padding=0).to(device))
                dim *= 2
            elif i < nb_stride2_downsampling * 2:
                # to maintain spatial dimensionality
                # stride2_us_convs.append(
                #     get_transposed_conv_block(dim, dim // 2, 2, dropout, stride=2, padding=0).to(device))
                # stride3_merge_convs.append(get_conv_block(dim, dim // 2, 3, dropout, stride=1, padding=1).to(device))
                stride3_merge_convs.append(
                    get_conv_block(3 * dim // 2, dim // 2, 3, dropout, stride=1, padding=1).to(device))
                dim //= 2
            else:
                # patch us conv
                # self.patch_us_conv = get_transposed_conv_block(
                #     emb_dim, emb_dim, patch_ds_stride, dropout, stride=patch_ds_stride, padding=0).to(all_device[-1])
                self.patch_merge_conv = get_conv_block(
                    emb_dim * 2, emb_dim, 3, dropout, stride=1, padding=1).to(all_device[-1])

        self.t_layers = nn.ModuleList(t_layers)
        self.stride2_ds_convs = nn.ModuleList(stride2_ds_convs)
        # self.stride2_us_convs = nn.ModuleList(stride2_us_convs)
        self.stride3_merge_convs = nn.ModuleList(stride3_merge_convs)
        self.map_concat_mode = map_concat_mode

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module((inputs[0], inputs[1]))
            return inputs

        return custom_forward

    def forward(self, src, padding_mask, enable_checkpoint, conv_backbone=True):
        contact_map = obtain_contact_map(self.map_concat_mode, src).permute(0, -1, 1, 2)
        batch_size, dim, length, _ = contact_map.shape
        all_padding_mask = [padding_mask]

        for conv_block in self.pre_convs:
            if enable_checkpoint and length > 1000:
                contact_map = contact_map + checkpoint.checkpoint(self.custom(conv_block), contact_map, padding_mask)
            else:
                contact_map = contact_map + conv_block((contact_map, padding_mask))

        if conv_backbone:
            # bypass vision transformers
            device = self.all_device[-1]
            contact_map = contact_map.to(device)
            padding_mask = padding_mask.to(device)

            for conv_block in self.post_convs:
                if enable_checkpoint and length > 1000:
                    contact_map = contact_map + checkpoint.checkpoint(
                        self.custom(conv_block), contact_map, padding_mask)
                else:
                    contact_map = contact_map + conv_block((contact_map, padding_mask))

            return [((contact_map + contact_map.transpose(-1, -2)) / 2, (~padding_mask).sum(dim=[1, -1])[:, 0])]

        if length % self.patch_ds_stride == 0:
            contact_map_paddings = 0
            nb_patches = length // self.patch_ds_stride
        else:
            contact_map_paddings = self.patch_ds_stride - length % self.patch_ds_stride
            nb_patches = length // self.patch_ds_stride + 1

        padding_mask = F.pad(padding_mask, (0, contact_map_paddings, 0, contact_map_paddings), value=True)
        patch_map = self.patch_ds_conv(
            (F.pad(contact_map, (0, contact_map_paddings, 0, contact_map_paddings)), padding_mask))  # to patches
        padding_mask = rearrange(
            padding_mask, 'b 1 (nw_h w_h) (nw_w w_w) -> b 1 nw_h nw_w (w_h w_w)',
            w_h=self.patch_ds_stride, w_w=self.patch_ds_stride).prod(dim=-1).bool()
        patch_map = patch_map.masked_fill(padding_mask, 0.)

        layer_split_idx = 0
        ds_path_cached_maps = []
        ds_path_paddings = []
        all_patch_maps = []

        for i in range(self.nb_stride2_downsampling * 2 + 1):

            t_layer_1, t_layer_2 = self.t_layers[i * 2], self.t_layers[i * 2 + 1]

            # goes in and out with shape: b, d, l, l
            patch_map = t_layer_2(t_layer_1(patch_map, padding_mask), padding_mask)

            if i < self.nb_stride2_downsampling:  # down-sampling path

                all_padding_mask.append(padding_mask)
                ds_path_cached_maps.append(patch_map)

                if nb_patches % 2 != 0:
                    padding = 1
                else:
                    padding = 0

                patch_map = F.pad(patch_map, (0, padding, 0, padding))
                nb_patches += padding
                ds_path_paddings.append(padding)

                padding_mask = F.pad(padding_mask, (0, padding, 0, padding), value=True)
                patch_map = self.stride2_ds_convs[i]((patch_map, padding_mask))
                padding_mask = rearrange(
                    padding_mask, 'b 1 (nw_h w_h) (nw_w w_w) -> b 1 nw_h nw_w (w_h w_w)',
                    w_h=2, w_w=2).prod(dim=-1).bool()
                patch_map = patch_map.masked_fill(padding_mask, 0.)
                nb_patches //= 2
                dim *= 2

            elif i < self.nb_stride2_downsampling * 2:  # up-sampling path

                all_patch_maps.append((
                    (patch_map + patch_map.transpose(-1, -2)) / 2, (~padding_mask).sum(dim=[1, -1])[:, 0]))
                # patch_map = self.stride2_us_convs[i - self.nb_stride2_downsampling]((patch_map, padding_mask))
                patch_map = F.interpolate(patch_map, scale_factor=2, mode='nearest')
                nb_patches *= 2
                dim //= 2

                padding_mask = all_padding_mask.pop().to(patch_map.device)
                padding = ds_path_paddings.pop()
                nb_patches -= padding
                if padding > 0:
                    patch_map = patch_map[:, :, :-padding, :-padding]

                # concatenate and convolve
                # patch_map = patch_map + ds_path_cached_maps.pop().to(patch_map.device)
                patch_map = self.stride3_merge_convs[i - self.nb_stride2_downsampling](
                    (torch.cat([patch_map, ds_path_cached_maps.pop().to(patch_map.device)], dim=1), padding_mask))
                patch_map = patch_map.masked_fill(padding_mask, 0.)

            else:

                all_patch_maps.append((
                    (patch_map + patch_map.transpose(-1, -2)) / 2, (~padding_mask).sum(dim=[1, -1])[:, 0]))
                # upsampled_contact_map = self.patch_us_conv((patch_map, padding_mask))
                upsampled_contact_map = F.interpolate(patch_map, scale_factor=self.patch_ds_stride, mode='nearest')

                padding_mask = all_padding_mask.pop().to(patch_map.device)
                if contact_map_paddings > 0:
                    upsampled_contact_map = upsampled_contact_map[:, :, :-contact_map_paddings, :-contact_map_paddings]

                # contact_map = upsampled_contact_map + contact_map.to(upsampled_contact_map.device)
                contact_map = self.patch_merge_conv(
                    (torch.cat([upsampled_contact_map, contact_map.to(upsampled_contact_map.device)], dim=1),
                     padding_mask))
                contact_map = contact_map.masked_fill(padding_mask, 0.)

            if len(self.layer_split_idx) > layer_split_idx and i == self.layer_split_idx[layer_split_idx]:
                layer_split_idx += 1
                if len(self.layer_split_idx) > layer_split_idx:
                    device = self.all_device[layer_split_idx]
                else:
                    device = self.all_device[-1]

                patch_map = patch_map.to(device)
                padding_mask = padding_mask.to(device)

        for conv_block in self.post_convs:
            if enable_checkpoint and length > 1000:
                contact_map = contact_map + checkpoint.checkpoint(self.custom(conv_block), contact_map, padding_mask)
            else:
                contact_map = contact_map + conv_block((contact_map, padding_mask))

        all_patch_maps.append((
            (contact_map + contact_map.transpose(-1, -2)) / 2, (~padding_mask).sum(dim=[1, -1])[:, 0]))

        return all_patch_maps


class VTBlock(nn.Module):

    def __init__(self, emb_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_conv_proj=False, use_lw=False, **kwargs):
        super(VTBlock, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation = activation
        # choose between linear projection (ordinary transformer) or convolution projection
        self.use_conv_proj = use_conv_proj

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(emb_dim, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, emb_dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

        if self.use_conv_proj:
            self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
        else:
            self.conv_kernel_size = 1
        self.qkv_conv = nn.Conv2d(emb_dim, 3 * emb_dim, self.conv_kernel_size, padding=(self.conv_kernel_size - 1) // 2)
        self.output_conv = nn.Conv2d(self.emb_dim, self.emb_dim, kernel_size=1)

        self.use_lw = use_lw
        if use_lw:
            self.lw_size = kwargs.get('lw_size', 20)
            self.use_cyclic_shift = kwargs.get('use_cyclic_shift', False)
            if self.use_cyclic_shift:
                self.shift_offset = kwargs.get('shift_offset', self.lw_size // 2) % self.lw_size
            else:
                self.shift_offset = 0

            self.rel_pos_bias = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(2 * self.lw_size - 1, 2 * self.lw_size - 1, self.nhead)))

            indices = np.array([(x, y) for x in range(self.lw_size) for y in range(self.lw_size)]).astype(np.long)
            self.all_rel_distance = indices[None, :, :] - indices[:, None, :]

    def forward(self, src, padding_mask=None):
        # padding_mask should indicate which patches in src are paddings
        batch_size, dim, nb_patches, _ = src.shape
        # channel first to channel last in order to use layernorm
        src2 = self.norm1(src.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_lw:

            if self.shift_offset == 0:
                patches_l_pad = 0
            else:
                patches_l_pad = self.lw_size - self.shift_offset
            r_left_over = (nb_patches - self.shift_offset) % self.lw_size
            if r_left_over == 0:
                patches_r_pad = 0
            else:
                patches_r_pad = self.lw_size - r_left_over

            nb_windows = (nb_patches + patches_l_pad + patches_r_pad) // self.lw_size
            patch_map = rearrange(
                F.pad(src2, (patches_l_pad, patches_r_pad, patches_l_pad, patches_r_pad)),
                'b d (nw_h w_h) (nw_w w_w) -> (b nw_h nw_w) d w_h w_w',
                w_h=self.lw_size, w_w=self.lw_size)

            # prepare masks
            if padding_mask is None:
                key_padding_mask = rearrange(
                    F.pad(torch.zeros(batch_size, nb_patches, nb_patches).to(patch_map.device),
                          (patches_l_pad, patches_r_pad, patches_l_pad, patches_r_pad), value=1.),
                    'b (nw_h w_h) (nw_w w_w) -> (b nw_h nw_w) (w_h w_w)',
                    w_h=self.lw_size, w_w=self.lw_size).bool()
            else:
                key_padding_mask = rearrange(
                    F.pad(padding_mask.squeeze(1), (patches_l_pad, patches_r_pad, patches_l_pad, patches_r_pad),
                          value=True),
                    'b (nw_h w_h) (nw_w w_w) -> (b nw_h nw_w) (w_h w_w)',
                    w_h=self.lw_size, w_w=self.lw_size)

            ''' [src_mask] If a FloatTensor is provided, it will be added to the attention weight '''
            rel_pos_emb = self.rel_pos_bias[
                self.all_rel_distance[:, :, 0], self.all_rel_distance[:, :, 1]].to(patch_map.device)
        else:
            patch_map = src2
            # batch_size, effective_attn_field
            key_padding_mask = padding_mask.view(batch_size, -1) if padding_mask is not None else None
            rel_pos_emb = None

        eff_bsz, _, eff_l, _ = patch_map.shape
        eff_attn_field = eff_l ** 2
        head_dim = self.emb_dim // self.nhead
        scaling = float(head_dim) ** -0.5

        q, k, v = self.qkv_conv(patch_map).chunk(3, dim=1)  # b, d, l, l

        q = q * scaling
        q = q.contiguous().view(eff_bsz, self.nhead, head_dim, eff_attn_field)
        k = k.contiguous().view(eff_bsz, self.nhead, head_dim, eff_attn_field)
        v = v.contiguous().view(eff_bsz, self.nhead, head_dim, eff_attn_field)

        attn_output_weights = torch.matmul(q.transpose(2, 3), k)  # b, nhead, l, l

        # # when a local window is entirely filled with paddings — possible in a batch setting,
        # # some nans would reside in the attn_weights
        # selected_index = ~(key_padding_mask.sum(-1) == eff_attn_field)
        # attn_output_weights = attn_output_weights[selected_index]
        selected_index = torch.where(key_padding_mask.sum(-1) == eff_attn_field)[0]
        key_padding_mask.index_fill_(0, selected_index, False)

        if key_padding_mask is not None:
            # key_padding_mask = key_padding_mask[selected_index]
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        if rel_pos_emb is not None:
            attn_output_weights = attn_output_weights + rel_pos_emb.permute(2, 0, 1).unsqueeze(0)

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        ########################################################################################
        ##
        ##      these do not work:
        ##      attn_output_weights = attn_output_weights.masked_fill(torch.isnan(attn_output_weights), 0.)
        ##      attn_output_weights[torch.isnan(attn_output_weights)] = 0.
        ##
        ########################################################################################

        # attn_output_weights_new = torch.empty(eff_bsz, self.nhead, eff_attn_field, eff_attn_field).to(
        #     attn_output_weights.device)
        # attn_output_weights_new[selected_index] = attn_output_weights
        # attn_output_weights = attn_output_weights_new

        attn_output = torch.matmul(attn_output_weights, v.transpose(2, 3))
        # eff_bsz, nhead, size_attn_field, head_dim

        attn_output = attn_output.transpose(2, 3).contiguous().view(eff_bsz, self.emb_dim, eff_l, eff_l)
        src2 = self.output_conv(attn_output)  # eff_bsz, dim, eff_l, eff_l

        if self.use_lw:
            src2 = rearrange(src2, '(b nw_h nw_w) d w_h w_w -> b d (nw_h w_h) (nw_w w_w)',
                             nw_h=nb_windows, nw_w=nb_windows)
            src2 = src2[:, :, patches_l_pad:nb_patches + patches_l_pad, patches_l_pad:nb_patches + patches_l_pad]

        src = src + self.dropout1(src2)

        # begin feedforward model
        src2 = self.norm2(src.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if padding_mask is not None:
            src = src.masked_fill(padding_mask, 0.)

        return src


def obtain_contact_map(contact_map_mode, node_emb):
    batch_size, max_len, dim = node_emb.shape

    if contact_map_mode == 'concat':
        mat = torch.cat([
            node_emb[:, :, None, :].repeat(1, 1, max_len, 1),
            node_emb[:, None, :, :].repeat(1, max_len, 1, 1)], dim=-1)
    elif contact_map_mode == 'concat_half':
        mat = torch.cat([
            node_emb[:, :, None, :dim // 2].repeat(1, 1, max_len, 1),
            node_emb[:, None, :, dim // 2:].repeat(1, max_len, 1, 1)], dim=-1)
    elif contact_map_mode == 'sum':
        mat = node_emb[:, :, None, :] + node_emb[:, None, :, :]
    elif contact_map_mode == 'dotprod':
        mat = torch.bmm(node_emb, node_emb.transpose(-1, -2))[:, :, :, None]
    else:
        raise ValueError('Unknown mode:', contact_map_mode)

    return mat


class UNetVTModel(nn.Module):

    def __init__(self, nb_stride2_downsampling, emb_dim, nhead, device, **kwargs):
        super(UNetVTModel, self).__init__()
        self.nb_stride2_downsampling = nb_stride2_downsampling
        self.emb_dim = emb_dim
        self.nhead = nhead

        if type(device) is torch.device:
            # preferably a large GPU device — 32 GB memory
            self.all_device = [device]
        else:
            # two small GPUs — 2 x 16 GB
            assert len(device) > 0
            self.all_device = device

        if nb_stride2_downsampling == 4 and len(self.all_device) == 4:
            layer_split_idx = [0, 4, 7]
        elif nb_stride2_downsampling == 4 and len(self.all_device) == 3:
            layer_split_idx = [0, 7]
        else:
            layer_split_idx = list(
                np.linspace(-1, nb_stride2_downsampling * 2 + 1, len(self.all_device) + 1)[1: -1].astype(np.int32))
        self.layer_split_idx = layer_split_idx

        # transformer encoding layers
        self.t_dropout = kwargs.get('t_dropout', 0.1)
        self.map_concat_mode = 'sum'
        self.patch_ds_stride = kwargs.get('patch_ds_stride', 20)
        self.use_conv_proj = kwargs.get('use_conv_proj', True)
        self.use_lw = kwargs.get('use_lw', False)
        print('local window in vision transformers:', self.use_lw)

        self.nb_pre_convs = kwargs.get('nb_pre_convs', 10)
        self.nb_post_convs = kwargs.get('nb_post_convs', 10)
        self.enable_5x5_filter = kwargs.get('enable_5x5_filter', False)
        self.transformer_encoder = UNetVTEncoder(
            self.all_device, self.layer_split_idx, self.nb_stride2_downsampling, emb_dim,
            self.t_dropout, self.map_concat_mode, patch_ds_stride=self.patch_ds_stride, nhead=4,
            use_conv_proj=self.use_conv_proj, use_lw=self.use_lw, nb_pre_convs=self.nb_pre_convs,
            nb_post_convs=self.nb_post_convs, enable_5x5_filter=self.enable_5x5_filter)

        # embedding layer
        self.x_embedding = nn.Embedding(nb_type_node + 1, emb_dim, padding_idx=nb_type_node).to(self.all_device[0])
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
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

    def forward(self, x, batch_len, enable_checkpoint=True, conv_backbone=False):
        batch_size, max_len = x.shape
        # use fixed pe from the original transformer paper, absolute positional encodings
        pe = get_original_pe(
            batch_len.cpu().numpy(), max_len, self.emb_dim, return_flattened=False).to(self.all_device[0])
        x = self.x_embedding(x) + pe  # batch_size, length, dimension

        batch_padding_mask = torch.ones((batch_size, max_len)).cumsum(dim=1).to(self.all_device[0]) > batch_len[:, None]
        batch_padding_mask = batch_padding_mask[:, :, None] + batch_padding_mask[:, None, :]

        all_patch_maps = self.transformer_encoder(x, batch_padding_mask.unsqueeze_(1), enable_checkpoint, conv_backbone)
        # contact_map should be on the last device

        all_patch_map_triu = []

        for patch_map, batch_len in all_patch_maps:

            patch_map = patch_map.permute(0, 2, 3, 1).to(self.all_device[-1])
            batch_len_np = batch_len.detach().cpu().numpy()

            contact_map_triu = []
            cumsum_nodes = 0
            for i in range(batch_size):
                indices = torch.triu_indices(batch_len_np[i], batch_len_np[i]).to(self.all_device[-1])
                contact_map_triu.append(patch_map[i, indices[0], indices[1], :])
                cumsum_nodes += batch_len_np[i]
            contact_map_triu = torch.cat(contact_map_triu, dim=0)

            all_patch_map_triu.append((contact_map_triu, batch_len_np))

        return all_patch_map_triu[::-1]
