import torch
import torch.nn as nn
import math

class Cross_Attention_Sparse(nn.Module):
    def __init__(self, dim, num_heads=8, ratio=0.5, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Cross_Attention_Sparse, self).__init__()
        self.num_heads = num_heads
        self.ratio = ratio
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k_v):
        B, N, C = q.shape

        # q, k, v 投影 [B, num_heads, N, head_dim]
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(k_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力分数 [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        max_neg_value = -torch.finfo(attn.dtype).max

        K = max(1, math.ceil(N * (1 - self.ratio)))

        # 1. 行稀疏：每行保留K个最大的（dim=-1）
        attn_row_sparse = torch.full_like(attn, max_neg_value)
        row_topk_values, row_topk_indices = torch.topk(attn, K, dim=-1)
        attn_row_sparse.scatter_(-1, row_topk_indices, row_topk_values)

        # 2. 列稀疏：每列保留K个最大的（dim=-2）
        attn_col_sparse = torch.full_like(attn, max_neg_value)
        col_topk_values, col_topk_indices = torch.topk(attn, K, dim=-2)
        attn_col_sparse.scatter_(-2, col_topk_indices, col_topk_values)

        # 3. 合并行和列稀疏（逻辑或）
        combined_mask = (attn_row_sparse > max_neg_value) | (attn_col_sparse > max_neg_value)
        attn_sparse = torch.where(combined_mask, attn, torch.full_like(attn, max_neg_value))

        # softmax和输出计算
        attn_sparse = attn_sparse.softmax(dim=-1)
        # attn_sparse = self.attn_drop(attn_sparse)

        out = (attn_sparse @ v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

if __name__ == '__main__':
    input_q  = torch.rand(2, 4, 4).cuda()  # B,H*W,C
    input_vk = torch.rand(2, 4, 4).cuda()
    #mymodel = Cross_Attention_Sparse(dim=64, out_dim=64, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.).cuda()
    mymodel = Cross_Attention_Sparse(dim=4, num_heads=1, ratio=0.3).cuda()
    output = mymodel(input_q,input_vk)
    print(output)
