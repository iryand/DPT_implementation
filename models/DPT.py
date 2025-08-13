from xml.dom import WrongDocumentErr

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import *

from utils.metrics import cal_norm_mask
from utils.misc import init_weights



class DPT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bs = args.bs
        self.len_trim = args.len_trim
        self.n_item = args.n_item
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b
        self.n_neg = args.n_neg
        self.temp = args.temp
        self.args = args
        self.d_embed = args.d_embed
        self.dropout = nn.Dropout(args.dropout)   

        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # self.alpha1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.alpha3 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # feature_extration
        self.target_feature_attn = MultiHeadAttention(args)
        self.source_feature_attn = MultiHeadAttention(args)

        # preference_extration
        self.cross = Cross_Self_Attention(args)

        #FFN and norm
        self.preference_ffn = FeedForward(self.d_embed)
        self.preference_norm = nn.LayerNorm(self.d_embed)

        #projector
        self.proj_i = FeedForward(self.d_embed)
        self.proj_a = FeedForward(self.d_embed)
        self.proj_b = FeedForward(self.d_embed)

        self.norm_i2a = nn.LayerNorm(self.d_embed)
        self.norm_i2b = nn.LayerNorm(self.d_embed)
        self.norm_a2a = nn.LayerNorm(self.d_embed)
        self.norm_b2b = nn.LayerNorm(self.d_embed)

        self.apply(init_weights)
        

    def forward(self, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b):
        # embedding
        e_x = self.dropout((self.ei(seq_x) + self.ep(pos_x)) * mask_x)
        e_a = self.dropout((self.ei(seq_a) + self.ep(pos_a)) * mask_a)
        e_b = self.dropout((self.ei(seq_b) + self.ep(pos_b)) * mask_b)

        # feature extraction
        source_feature_a = self.source_feature_attn(e_a, mask_a)
        source_feature_b = self.source_feature_attn(e_b, mask_b)
        source_feature_x = self.source_feature_attn(e_x, mask_x) 

        target_feature_x = self.target_feature_attn(e_x, mask_x)
        target_feature_a = self.target_feature_attn(e_a, mask_a)
        target_feature_b = self.target_feature_attn(e_b, mask_b)

        # preference extraction
        preference_x = self.cross(target_feature_x, source_feature_x, source_feature_x, mask_x)
        preference_a = self.cross(target_feature_a, source_feature_b, source_feature_b, mask_a)
        preference_b = self.cross(target_feature_b, source_feature_a, source_feature_a, mask_b)

        preference_x = self.preference_norm(preference_x + self.dropout(self.preference_ffn(preference_x)))
        preference_a = self.preference_norm(preference_a + self.dropout(self.preference_ffn(preference_a)))
        preference_b = self.preference_norm(preference_b + self.dropout(self.preference_ffn(preference_b)))

        # switch training / evaluating
        if self.training:
            mask_gt_a = mask_gt_a.unsqueeze(-1)
            mask_gt_b = mask_gt_b.unsqueeze(-1)
        else:
            mask_x = mask_a = mask_b = 1
            preference_x = preference_x[:, -1]
            preference_a = preference_a[:, -1]
            preference_b = preference_b[:, -1]

        # projector
        p_i = self.proj_i(preference_x)

        h_a2a = self.norm_a2a((preference_a +
                                self.dropout(self.proj_a(preference_a))) * mask_gt_a)

        h_b2b = self.norm_b2b((preference_b +
                                self.dropout(self.proj_b(preference_b))) * mask_gt_b)

        h_i2b = self.norm_i2b((preference_x +
                        self.dropout(p_i)) * mask_gt_b)
        
        h_i2a = self.norm_i2a((preference_x +
                                self.dropout(p_i)) * mask_gt_a) 
               
        h_a = h_i2a + h_a2a
        h_b = h_i2b + h_b2b

        h = h_a * mask_gt_a + h_b * mask_gt_b
        return h, preference_x, preference_a, preference_b

    def cal_rec_loss(self, h, gt, gt_neg, mask_gt_a, mask_gt_b, emb_grad=True):
        """ InfoNCE """
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg)
        if not emb_grad:
            e_gt = e_gt.detach()
            e_neg = e_neg.detach()

        logits = torch.cat(((h * e_gt).unsqueeze(-2).sum(-1),
                            (h.unsqueeze(-2) * e_neg).sum(-1)), dim=-1).div(self.temp)

        loss = -F.log_softmax(logits, dim=2)[:, :, 0]
        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()
        return loss_a, loss_b

    @staticmethod
    def cal_domain_rank(h, e_gt, e_mtc, mask_gt_a, mask_gt_b):
        """ calculate domain rank via inner-product similarity """
        logit_gt = (h * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logit_mtc = (h.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logit_mtc - logit_gt).gt(0).sum(-1).add(1)
        ranks_a = ranks[mask_gt_a == 1].tolist()
        ranks_b = ranks[mask_gt_b == 1].tolist()

        return ranks_a, ranks_b

    def cal_rank(self, h_f, h_c, h_a, h_b, gt, gt_mtc, mask_gt_a, mask_gt_b):
        """ rank via inner-product similarity """
        mask_gt_a = mask_gt_a.squeeze(-1)
        mask_gt_b = mask_gt_b.squeeze(-1)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        ranks_f2a, ranks_f2b = self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_c2a, ranks_c2b = self.cal_domain_rank(h_c, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_a2a, ranks_a2b = self.cal_domain_rank(h_a, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_b2a, ranks_b2b = self.cal_domain_rank(h_b, e_gt, e_mtc, mask_gt_a, mask_gt_b)

        return ranks_f2a, ranks_f2b, ranks_c2a, ranks_c2b, ranks_a2a, ranks_a2b, ranks_b2a, ranks_b2b
