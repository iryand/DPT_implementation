import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """ SwiGLU-FFN """
    def __init__(self, d_embed, d_ffn=680):
        super().__init__()
        self.d_embed = d_embed
        self.d_ffn = d_ffn

        self.fc_1 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_embed, bias=False)

    def forward(self, h):
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h



class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = nn.MultiheadAttention(args.d_embed, args.n_head, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_mha = nn.LayerNorm(args.d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((args.len_trim,
                                                    args.len_trim), True), diagonal=1),
                             persistent=False)

    def forward(self, h, mask):
        h_mha = self.norm_mha(h + self.dropout(self.mha(h, h, h,
                                                    attn_mask=self.mask_causal,
                                                    #is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_mha
    
    

class Cross_Self_Attention(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(args.d_embed, args.n_head, batch_first=True)
        self.mha2 = nn.MultiheadAttention(args.d_embed, args.n_head, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_mha = nn.LayerNorm(args.d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((args.len_trim,
                                                    args.len_trim), True), diagonal=1),
                             persistent=False)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.T = args.T
        if self.T == 0:
            self.cur_alpha = 1
        elif self.T == 1e-9:
            self.cur_alpha = 0
        else:
            self.cur_alpha = torch.sigmoid(self.alpha / self.T)


    def forward(self, q, k, v, mask):

        if self.T != 0 and self.T != 1e-9:
            self.cur_alpha = torch.sigmoid(self.alpha / self.T)

        # output = q + α·CrossAttn(q, k_enc, v_enc) + (1-α)·SelfAttn(q) 
        attn = self.cur_alpha * self.mha1(q, k, v, attn_mask=self.mask_causal,
                                                    #is_causal=True,
                                                    need_weights=False)[0] \
            + (1 - self.cur_alpha) * self.mha2(q, q, q, attn_mask=self.mask_causal,
                                                    #is_causal=True,
                                                    need_weights=False)[0]
        h_mha = self.norm_mha(q + self.dropout(attn)) * mask
        return h_mha

class Cross_Self_Attention2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(args.d_embed, args.n_head, batch_first=True)
        self.mha2 = nn.MultiheadAttention(args.d_embed, args.n_head, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_mha = nn.LayerNorm(args.d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((args.len_trim,
                                                    args.len_trim), True), diagonal=1),
                             persistent=False)
        self.T = args.T
        if self.T == 0:
            self.cur_alpha = 1
        elif self.T == 1e-9:
            self.cur_alpha = 0
        else:
            self.cur_alpha = 0.5


    def forward(self, q, k, v, mask, alpha):

        if self.T != 0 and self.T != 1e-9:
            self.cur_alpha = torch.sigmoid(alpha / self.T)

        # output = q + α·CrossAttn(q, k_enc, v_enc) + (1-α)·SelfAttn(q) 
        attn = self.cur_alpha * self.mha1(q, k, v, attn_mask=self.mask_causal,
                                                    #is_causal=True,
                                                    need_weights=False)[0] \
            + (1 - self.cur_alpha) * self.mha2(q, q, q, attn_mask=self.mask_causal,
                                                    #is_causal=True,
                                                    need_weights=False)[0]
        h_mha = self.norm_mha(q + self.dropout(attn)) * mask
        return h_mha

    