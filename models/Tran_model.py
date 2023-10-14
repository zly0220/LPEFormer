import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, drop_path, to_2tuple
from timm.models.registry import register_model

import itertools
import math
from functools import partial


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class Conv1d(nn.Module):
    default_act = nn.ReLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,norm=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = nn.LayerNorm(c2) if norm is True else nn.Identity()      
        # self.norm = nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x).permute(0,2,1)
        return self.act(self.norm(x))
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def set_rel_pos_index(self,seq_len):
        if self.window_size:
            window_size = (1,seq_len)
        # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1], ) * 2, dtype=relative_coords.dtype)
            relative_position_index[:, :] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index = torch.clamp(relative_position_index, 0, self.num_relative_distance-1)
            # relative_position_index[0, 0:] = self.num_relative_distance - 3
            # relative_position_index[0:, 0] = self.num_relative_distance - 2
            # relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        self.set_rel_pos_index(N)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerLayer_rel_pos(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.norm1(x)
        x = self.fc2(self.fc1(x)) + x
        x = self.norm2(x)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers, relative_pos=True):
        super().__init__()
        self.conv = None
        self.relative_pos = relative_pos
        if c1 != c2:
            self.conv = Conv1d(c1,c2,7,act=nn.Identity())
        if relative_pos:
            self.tr = nn.Sequential(*(TransformerLayer_rel_pos(dim=c2,num_heads=num_heads,qkv_bias=True,drop_path=0.,window_size=(1,200)) for _ in range(num_layers)))
        else:
            self.tr = nn.Sequential(*(TransformerLayer(c=c2,num_heads=num_heads) for _ in range(num_layers)))
            self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.c2 = c2
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        if self.relative_pos:
            p = x
        else:
            p = x + self.linear(x)
        p = self.norm(self.tr(p))
        return p
    
class TextCNN(nn.Module):
    def __init__(self, c1, c2, kernal_size=(3,5,7)):
        super().__init__()
        num_kernal = len(kernal_size)
        self.conv_list = nn.ModuleList([Conv1d(c1, c2, k) for k in kernal_size])
        self.max_pl = nn.MaxPool1d(num_kernal, num_kernal)
        
    def forward(self, x):
        fea_list = []
        for conv in self.conv_list:
            fea_list.append(F.relu(conv(x)))
        x = torch.cat(fea_list, dim=1)
        return self.max_pl(x.permute(0,2,1)).permute(0,2,1)
    
class RCNN(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.conv = Conv1d(c1,c2*3,3)
        self.max_pl = nn.MaxPool1d(3,3)
        self.gru = nn.GRU(c2,int(c2//2),1,bidirectional=True,batch_first=True)
    
    def forward(self, x):
        x = self.max_pl(self.conv(x))
        output, _ = self.gru(x)
        return output
    
class Tran_block(nn.Module):
    def __init__(self, in_channels, width, relative_pos=True, key_analysis=False, method="self_attention"):
        super().__init__()
        self.key_analysis = key_analysis
        if method == "self_attention":
            self.tr_net = TransformerBlock(c1=in_channels, c2=width, num_heads=8, num_layers=1,relative_pos=relative_pos) #em1b 1280 prot5 1024 word2vec 512 bepler 121
        elif method == "lstm":
            self.tr_net = nn.LSTM(input_size=in_channels, hidden_size=width, num_layers=1, batch_first=True)
        elif method == "textcnn":
            self.tr_net = TextCNN(c1=in_channels,c2=width)
        elif method =="rcnn":
            self.tr_net = RCNN(c1=in_channels, c2=width)
        else:
            raise ValueError('no such method')
        self.method = method
        # self.tr_net_2 = TransformerBlock(1024, 320, 4, 1)
        # self.tr_net = Meta3D(128,mlp_ratio=1)
        self.conv_feat_1 = nn.Sequential(*(
                                            Conv1d(width, int(width/2), 7, act=nn.ReLU()),
                                            Conv1d(int(width/2), int(width/4), 9, act=nn.ReLU()),
                                            Conv1d(int(width/4), 1, 11,norm=False, act=nn.ReLU())))
        
        self.conv_feat_2 = nn.Sequential(*(
                                            Conv1d(width, int(width/2), 7, act=nn.ReLU()),
                                            Conv1d(int(width/2), int(width/4), 9, act=nn.ReLU()),
                                            Conv1d(int(width/4), 1, 11,norm=False, act=nn.ReLU())))
        if self.key_analysis:
            self.add_weight = nn.Parameter(torch.ones(1,width))
        
    def forward(self, seq_a, seq_b, add_a=None, add_b=None):
        if self.method == "self_attention":
            a_tr = self.tr_net(seq_a) #[1,N,emb]
            b_tr = self.tr_net(seq_b)
        elif self.method == "lstm":
            a_tr, _ = self.tr_net(seq_a) #[1,N,emb]
            b_tr, _ = self.tr_net(seq_b)
        elif self.method =="textcnn" :
            a_tr = self.tr_net(seq_a) #[1,N,emb]
            b_tr = self.tr_net(seq_b)
        elif self.method =="rcnn":
            a_tr = self.tr_net(seq_a) #[1,N,emb]
            b_tr = self.tr_net(seq_b)
        a_feat = torch.mul(F.softmax(self.conv_feat_1(a_tr),dim=1), a_tr)
        a_feat = torch.mean(a_feat, dim=1) #[1,emb]
        if add_a != None:
            if self.key_analysis:
                a_feat += add_a * self.add_weight
            else:
                a_feat += add_a
        b_feat = torch.mul(F.softmax(self.conv_feat_2(b_tr),dim=1), b_tr)
        b_feat = torch.mean(b_feat, dim=1)
        if add_b != None:
            if self.key_analysis:
                b_feat += add_b * self.add_weight
            else:
                b_feat += add_b
        out = torch.cat([a_feat, b_feat], dim=1)
        return out
    
class mlp_block(nn.Module):
    def __init__(self,c1,c2,p,act='ReLU',norm=True):
        super().__init__()
        self.linear = nn.Linear(c1,c2)
        if act == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif act == "GELU":
            self.act = nn.GELU()
        elif act == "none":
            self.act = nn.Identity()
        else:
            print("error act")
            exit(0)
        self.dropout = nn.Dropout(p,inplace=True)
        self.norm = norm
        
    def forward(self, x):
        if self.norm:
            return self.act(self.dropout(F.normalize(self.linear(x),dim=1)))
        else:
            return self.act(self.dropout(self.linear(x)))

class CLS_Head(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net = nn.Sequential(*(
                                    mlp_block(int(width), int(width/2), 0.3, act="ReLU"),
                                    mlp_block(int(width/2), int(width/4), 0.3, act="ReLU"),
                                    mlp_block(int(width/4), 1, 0.3, act="none",norm=False),
                                    # nn.Sigmoid()
                                   ))    
        
    def forward(self,x):
        return self.net(x)
    
class Tran_PPI(nn.Module):
    def __init__(self, in_channels, width, relative_pos=True, key_analysis=False, method="self_attention"):
        super().__init__()
        self.tran_block = Tran_block(in_channels, width, relative_pos, key_analysis, method)
        self.cls_head = CLS_Head(width*2)
    
    def forward(self, seq_a, seq_b, add_a=None,add_b=None):
        outs = []
        batch_size = len(seq_a)
        for idx in range(batch_size):
            a = torch.from_numpy(seq_a[idx]).unsqueeze(0).cuda()
            b = torch.from_numpy(seq_b[idx]).unsqueeze(0).cuda()
            if add_a != None:
                outs.append(self.tran_block(a, b, add_a[idx].cuda(), add_b[idx].cuda()))
            else:
                outs.append(self.tran_block(a, b))
        outs = torch.cat(outs)
        outs = self.cls_head(outs)
        return outs

    
if __name__ == "__main__":
    # model = TransformerBlock(1024,320,8,1)
    # a = torch.randn((1,50,1024))
    model = Tran_PPI(1024, 320)
    # a = torch.randn((50,1280))
    # a = model(a)
    # print(a.shape)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:{:.3f}M'.format(n_parameters/1e6))
    n_parameters = sum(p.numel()
                       for p in model.cls_head.parameters() if p.requires_grad)
    print('number of params:{:.3f}M'.format(n_parameters/1e6))


# x = torch.randn((1,3,128))
# #print(x.numpy().shape)
# model = Tran_PPI()
# seq_dataset = Seq_Pair_dataset("w2v.model", "dataset/1/cv_train_1.csv")
# loader = DataLoader(seq_dataset,100,collate_fn=collate,shuffle=True)
# for A,B,C,D,E in loader:
#     preds = model(A, B)
#     loss = sigmoid_focal_loss_jit(preds, E, 0.25, 2, reduction="mean")
#     print(loss)
#     break



