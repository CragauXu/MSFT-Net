import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from .vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,DropPath, to_2tuple, trunc_normal_

class Mixed_scale_convolutional_module(nn.Module):
    def __init__(self,img_size, in_dim,out_dim,patch_size):
        super(Mixed_scale_convolutional_module, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_dim = 36
        # 定义输入投影层1x1卷积升维
        self.project_in = nn.Conv2d(in_dim, self.hidden_dim, kernel_size=1)

        # 定义3x3和5x5的深度可分离卷积
        self.dwconv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, groups=self.hidden_dim)
        self.dwconv5x5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1, padding=2, groups=self.hidden_dim)
        self.dwconv7x7 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=7, stride=1, padding=3, groups=self.hidden_dim)
        self.relu3 = nn.ReLU()  # 3x3卷积后的ReLU激活
        self.relu5 = nn.ReLU()  # 5x5卷积后的ReLU激活
        self.relu7 = nn.ReLU()

        # 定义第二层3x3和5x5的深度可分离卷积
        self.dwconv3x3_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim // 3, kernel_size=3, stride=1, padding=1, groups=self.hidden_dim // 3)
        self.dwconv5x5_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim // 3, kernel_size=5, stride=1, padding=2, groups=self.hidden_dim // 3)
        self.dwconv7x7_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim // 3, kernel_size=7, stride=1, padding=3, groups=self.hidden_dim // 3)

        self.relu3_1 = nn.ReLU()  # 第二层3x3卷积后的ReLU激活
        self.relu5_1 = nn.ReLU()  # 第二层5x5卷积后的ReLU激活
        self.relu7_1 = nn.ReLU()  # 第二层5x5卷积后的ReLU激活

        # 定义输出投影层
        self.project_out = nn.Conv2d(self.hidden_dim, out_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.project_in(x)  # 输入投影

        # 通过3x3卷积和ReLU激活，然后分块
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        # 通过5x5卷积和ReLU激活，然后分块
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)
        # 通过7x7卷积和ReLU激活，然后分块
        x1_7, x2_7 = self.relu7(self.dwconv7x7(x)).chunk(2, dim=1)

        # 将3x3和5x5的输出拼接
        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_5, x1_7], dim=1)
        x3 = torch.cat([x2_3, x2_7], dim=1)

        # 通过第二层3x3卷积和ReLU激活
        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        # 通过第二层5x5卷积和ReLU激活
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        # 通过第三层7x7卷积和ReLU激活
        x3 = self.relu7_1(self.dwconv7x7_1(x3))

        # 将三个输出拼接
        x = torch.cat([x1, x2, x3], dim=1)

        # 输出投影
        x = self.project_out(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class FourierPositionalEncoding(torch.nn.Module):
    def __init__(self, num_frames, embed_dim):
        super(FourierPositionalEncoding, self).__init__()
        # 初始化傅里叶特征位置编码
        self.time_embed = self.generate_fourier_features(num_frames, embed_dim)

    def generate_fourier_features(self, num_frames, embed_dim):
        # 创建位置索引
        pos = torch.arange(num_frames).unsqueeze(1)  # 形状: (num_frames, 1)

        # 创建频率索引
        i = torch.arange(0, embed_dim, 2)  # 形状: (embed_dim // 2,)
        freq = 10000 ** (i / embed_dim)  # 计算频率

        # 计算正弦和余弦值
        pe = torch.zeros(num_frames, embed_dim)
        pe[:, 0::2] = torch.sin(pos / freq)  # 偶数维度 (2i)
        pe[:, 1::2] = torch.cos(pos / freq)  # 奇数维度 (2i+1)

        # 返回位置编码
        return pe.unsqueeze(0)  # 形状: (1, num_frames, embed_dim)

    def forward(self):
        return self.time_embed

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block_Split(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='Spatial'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['Spatial', 'Temporal'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
          dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        # x->B((HW)T+1)C
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W
        init_cls_token = x[:, 0, :].unsqueeze(1)
        if self.attention_type in ['joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'Spatial':
            ## Spatial
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            # B(HWT)C->(BT)(HW)C
            xs = x[:, 1:, :]
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            # (BT)(HW)C->(BT)(HW+1)C
            xs = torch.cat((cls_token, xs), 1)
            # attention不变
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            # 将cls_token = init_cls_token.repeat(1, T, 1)恢复
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            # (BT)(HW+1)C->(BT)(HW)C
            res_spatial = res_spatial[:,1:,:]
            # (BT)(HW)C->B(HWT)C
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            # 残差连接(可选)
            res_spatial = x[:,1:,:] + res_spatial
            # 恢复cls_tokenB(HWT)C->B(HWT+1)C
            x = torch.cat((cls_token, res_spatial), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'Temporal':
            ## Temporal
            # B((HW)T+1)C->B((HW)T)C
            xt = x[:,1:,:]
            # B((HW)T)C->(BHW)TC
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            # attention不变
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            # (BHW)TC->B(HWT)C
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            # feedforward不变
            res_temporal = self.temporal_fc(res_temporal)
            # 残差连接不变
            xt = x[:,1:,:] + res_temporal
            # 恢复cls_tokenB(HWT)C->B(HWT+1)C
            x = torch.cat((init_cls_token, xt), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        # x->B((HW)T+1)C
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            # B((HW)T+1)C->B((HW)T)C
            xt = x[:,1:,:]
            # B((HW)T)C->(BHW)TC
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            # attention不变
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            # (BHW)TC->B(HWT)C
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            # feedforward不变
            res_temporal = self.temporal_fc(res_temporal)
            # 残差连接不变
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            # B(HWT)C->(BT)(HW)C
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            # (BT)(HW)C->(BT)(HW+1)C
            xs = torch.cat((cls_token, xs), 1)
            # attention不变
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            # 将cls_token = init_cls_token.repeat(1, T, 1)恢复
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            # (BT)(HW+1)C->(BT)(HW)C
            res_spatial = res_spatial[:,1:,:]
            # (BT)(HW)C->B(HWT)C
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            # 时间注意力和空间注意力相加,并恢复cls_tokenB(HWT)C->B(HWT+1)C
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class Attention_Split(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='joint_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = Mixed_scale_convolutional_module(img_size=img_size,in_dim=in_chans ,out_dim=embed_dim, patch_size=patch_size)
        num_patches = (img_size//patch_size) * (img_size//patch_size)

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        ## FourierPositionalEncoding
        self.fourier_pe = FourierPositionalEncoding(num_frames, embed_dim)
        self.fourier_embed = self.fourier_pe().cuda()

        if self.attention_type != 'space_only':
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_Split(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'joint_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # BCTHW->(BT)(HW)C
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # (BT)(HW)C->(BT)(HW+1)C
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            # 不变
            x = x + self.pos_embed
        # 不变
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'joint_space_time':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            # (BT)(HW+1)C->(BT)(HW)C
            x = x[:,1:]
            # (BT)(HW)C->(B(HW))TC
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.fourier_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.fourier_embed
            x = self.time_drop(x)
            # (B(HW))TC->B((HW)T)C
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            # B((HW)T)C->B((HW)T+1)C
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            # 不变
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'joint_space_time':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        # 取出cls_token用作分类
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]
        x = self.head(x)
        return x

class Attention_joint(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=16, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = Mixed_scale_convolutional_module(img_size=img_size,in_dim=in_chans ,out_dim=embed_dim, patch_size=patch_size)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        # BCTHW->(BT)(HW)C
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # (BT)(HW)C->(BT)(HW+1)C
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            # 不变
            x = x + self.pos_embed
        # 不变
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            # (BT)(HW+1)C->(BT)(HW)C
            x = x[:,1:]
            # (BT)(HW)C->(B(HW))TC
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            # (B(HW))TC->B((HW)T)C
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            # B((HW)T)C->B((HW)T+1)C
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            # 不变
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        # 取出cls_token用作分类
        return x

if __name__ == '__main__':
    input = torch.rand(8,3,16,224,224).cuda()
    mymodel = Attention_Split(img_size=224, num_classes=2, patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=16, attention_type='Spatial').cuda()
    output = mymodel(input)
    print(output.shape)
