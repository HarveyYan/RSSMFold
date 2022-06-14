import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product

from RSSMFold.model.msa_transformer import MSATransformer
from RSSMFold.model.vision_transformer_unet_hier_evo import UNetVTModel, obtain_contact_map

nb_type_node = 6  # A, C, G, U, N/mask and gaps


class JointEvoModel(nn.Module):

    def __init__(self, emb_dim, nhead, msa_transformer_nb_layers, num_ds_steps, all_device, dropout=0.1,
                 activation="relu", **kwargs):
        super(JointEvoModel, self).__init__()
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.msa_transformer_nb_layers = msa_transformer_nb_layers
        self.all_device = all_device
        self.dropout = dropout

        # feedforward dim defaults to 2048
        self.msa_transformer = MSATransformer(
            emb_dim, nhead, msa_transformer_nb_layers, dropout=dropout, all_device=all_device,
            activation=activation)

        # feedforward dim defaults to emb_dim * 2
        self.contact_model = UNetVTModel(
            num_ds_steps, emb_dim, nhead, all_device[-1:], t_dropout=dropout, map_concat_mode='concat', use_lw=True,
            patch_ds_stride=20, use_conv_proj=False, nb_pre_convs=10, nb_post_convs=10, enable_5x5_filter=True)



        # 8 for the outer concatenated sequence one-hot encodings, 16 for the covariance features
        # the other is for msa transformer extracted features
        self.msa_feature_dim = emb_dim * 2 + msa_transformer_nb_layers * nhead
        self.joint_initial_feature_dim = 8 + self.msa_feature_dim
        # joint_feature_merge_conv should replace the initial_feature_merge_conv in the contact_map model
        self.joint_feature_merge_conv = nn.Conv2d(
            self.joint_initial_feature_dim, emb_dim, 3, stride=1, padding=1).to(self.all_device[-1])

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

    def forward(self, msa, batch_msa_length, target_seq_indices, target_seqs, cov_features, batch_len,
                all_alignment_idx, return_learned_msa=False):
        batch_size, max_seq_len = target_seqs.shape
        batch_len_np = batch_len.cpu().numpy()

        # MSA transformer acquire features
        learned_msa, all_attn_weights = self.msa_transformer(msa)
        all_msa_feature_mat = []
        for i, (msa_length, seq_len, target_seq_idx) in enumerate(
                zip(batch_msa_length, batch_len_np, target_seq_indices)):
            rna_seq = learned_msa[i, target_seq_idx, :msa_length]
            attn_weight = all_attn_weights[i, :, :msa_length, :msa_length].permute(1, 2, 0)  # length, length, L*nhead
            sym_attn_weight = (attn_weight.transpose(0, 1) + attn_weight) / 2
            msa_feature_mat = torch.cat([
                rna_seq[:, None, :].repeat(1, msa_length, 1),
                rna_seq[None, :, :].repeat(msa_length, 1, 1),
                sym_attn_weight], dim=-1)
            if msa_length != seq_len:
                idx_to_retain_in_msa_seq, idx_corresponding_in_original_seq = all_alignment_idx[i]
                all_indices = np.array(
                    list(product(idx_corresponding_in_original_seq, idx_corresponding_in_original_seq)))
                original_seq_feature_mat = torch.zeros(
                    (seq_len, seq_len, self.msa_feature_dim), dtype=torch.float32).to(msa_feature_mat)
                original_seq_feature_mat[all_indices[:, 0], all_indices[:, 1]] = \
                    msa_feature_mat[idx_to_retain_in_msa_seq, :][:, idx_to_retain_in_msa_seq]. \
                        reshape(-1, self.msa_feature_dim)
                msa_feature_mat = original_seq_feature_mat
            all_msa_feature_mat.append(
                F.pad(msa_feature_mat.permute(2, 0, 1), (0, max_seq_len - seq_len, 0, max_seq_len - seq_len)))
        all_msa_feature_mat = torch.stack(all_msa_feature_mat, dim=0)  # batch_size, dim, max_len, max_len

        # Contact map model
        x = torch.eye(nb_type_node + 1).to(self.all_device[-1])[target_seqs][:, :, :4]
        contact_map = obtain_contact_map('concat', x).permute(0, -1, 1, 2)
        joint_contact_map = torch.cat([contact_map, all_msa_feature_mat.to(self.all_device[-1])], dim=1)
        contact_map = self.joint_feature_merge_conv(joint_contact_map)

        batch_padding_mask = torch.ones((batch_size, max_seq_len)). \
                                 cumsum(dim=1).to(self.all_device[-1]) > batch_len[:, None]
        batch_padding_mask = batch_padding_mask[:, :, None] + batch_padding_mask[:, None, :]

        all_evo_features = []
        all_patch_maps = self.contact_model.transformer_encoder(
            contact_map, all_evo_features, batch_padding_mask.unsqueeze_(1))
        # contact_map should be on the last device

        all_patch_map_triu = []
        for patch_map, batch_len in all_patch_maps:
            patch_map = patch_map.permute(0, 2, 3, 1).to(self.all_device[-1])
            batch_len_np = batch_len.detach().cpu().numpy()
            contact_map_triu = []
            cumsum_nodes = 0
            for i in range(batch_size):
                indices = torch.triu_indices(int(batch_len_np[i]), int(batch_len_np[i])).to(self.all_device[-1])
                contact_map_triu.append(patch_map[i, indices[0], indices[1], :])
                cumsum_nodes += batch_len_np[i]
            contact_map_triu = torch.cat(contact_map_triu, dim=0)
            all_patch_map_triu.append((contact_map_triu, batch_len_np))

        if return_learned_msa:
            return learned_msa, all_patch_map_triu[::-1]
        else:
            return all_patch_map_triu[::-1]
