from timesformer.Attention import Attention_joint,Attention_Split,Mlp
from timesformer.Cross_Attention import Cross_Attention_Sparse
import torch.nn as nn
from functools import partial
import torch

class MSFT_Net(nn.Module):
    def __init__(self,img_size,num_classes,patch_size,embed_dim,num_heads,sparse_ratio1,sparse_ratio2):
        super().__init__()
        self.head = Mlp(in_features=embed_dim,hidden_features=int(embed_dim/2),out_features=num_classes)
        self.mlp = Mlp(in_features=embed_dim,hidden_features=embed_dim,out_features=embed_dim)
        self.b_Attention1 = Attention_joint(depth=6)
        self.b_Attention2 = Attention_joint(depth=6)
        self.c_Attention = Attention_Split(img_size=img_size, num_classes=2, patch_size=patch_size, embed_dim=embed_dim,
                                              depth=2, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0.,
                                              drop_path_rate=0.1, num_frames=16, attention_type='Temporal')
        self.e_Attention = Attention_Split(img_size=img_size, num_classes=2, patch_size=16, embed_dim=embed_dim,
                                              depth=2, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0.,
                                              drop_path_rate=0.1, num_frames=16, attention_type='Spatial')
        self.Cross_Attention1 = Cross_Attention_Sparse(dim=embed_dim,num_heads=num_heads,ratio=sparse_ratio1)
        self.Cross_Attention2 = Cross_Attention_Sparse(dim=embed_dim,num_heads=num_heads,ratio=sparse_ratio1)

    def forward(self, b, c, e):
        b_Temporal = self.b_Attention1(b)
        b_Spatial = self.b_Attention2(b)
        c_Temporal = self.c_Attention.forward_features(c)
        e_Spatial = self.e_Attention.forward_features(e)
        x_Temporal = self.Cross_Attention1(c_Temporal,b_Temporal)
        x_Spatial  = self.Cross_Attention2(e_Spatial,b_Spatial)
        x = torch.cat((x_Temporal,x_Spatial),1)
        print(x.shape)
        x = self.mlp(x)
        x = x[:, 0,:]
        x = self.head(x)
        return x


if __name__ == '__main__':
    input_b  = torch.rand(2, 3, 16, 224, 224).cuda()
    input_c  = torch.rand(2, 3, 16, 224, 224).cuda()
    input_e  = torch.rand(2, 3, 16, 224, 224).cuda()
    mymodel = MSFT_Net(img_size=224,num_classes=2,patch_size=16,embed_dim=384,num_heads=4,sparse_ratio1=0.3,sparse_ratio2=0.3).cuda()
    output = mymodel(input_b,input_c,input_e)
    print(output.shape)

    # for i in mymodel.children():
    #     print(i)


