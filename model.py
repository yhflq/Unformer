from ptflops import get_model_complexity_info
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from functools import partial
import numpy as np
from torch import einsum, nn


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):

    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),)

    def forward(self, x):
        return downshuffle(self.body(x),2)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = self.patch_conv(x)

        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvRelPosEnc(nn.Module):
    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1 
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        q_img = q
        v_img = v

        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        crpe = self.crpe(q, v, size=size)

        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x



class MHCAEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=None,
        crpe_window={
            3: 2,
            5: 3,
            7: 3
        },
    ):
        super().__init__()

        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)

        self.MHCA_layer = MHCABlock(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=0.,
            qk_scale=qk_scale,
            shared_cpe=self.cpe,
            shared_crpe=self.crpe,
        )

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        
        x = self.MHCA_layer(x, (H, W))
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x


def dpr_generator(drop_path_rate, num_layers, num_stages):
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class Patch_Embed_stage(nn.Module):
    def __init__(self, embed_dim, num_path=4, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs

class MHCA_stage(nn.Module):

    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        drop_path_list=0.,
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_heads,
                mlp_ratio,
            ) for _ in range(num_path)
        ])

        self.eca_layer = eca_layer(channel=embed_dim)
        self.aggregate = Conv2d_BN(embed_dim * (num_path + 1),
                                   out_embed_dim,
                                   act_layer=nn.Hardswish)

    def forward(self, inputs):

        att_outputs = [self.eca_layer(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)
        return out

class Trans_DWcov(nn.Module):
    def __init__(
        self,
        num_stages=3,
        num_path=[4],
        num_layers=[1],
        embed_dims=[64],
        mlp_ratios=[8],
        num_heads=[8],
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
        
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=False if idx == 0 else True,
            ) for idx in range(self.num_stages)
        ])
        
        self.mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])
    def forward(self, x):
        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)
        
        return x


class DWConv2D(nn.Conv2d):
        def __init__(self, in_channels, kernel_size, bias=True):
            super().__init__(in_channels, in_channels, kernel_size, stride=1,
                             padding= kernel_size//2,  groups=in_channels, bias=bias)

class IFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.IFFN = nn.Sequential(
            DWConv2D(dim, kernel_size=3),
            nn.Conv2d(dim, dim//8, 1), 
            nn.GELU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.IFFN(x)


class GlobalContext(nn.Module):

    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim//8),
            act_layer(),
            nn.Linear(dim//8, dim)
        )
        self.head = 8
        self.scale = (dim//self.head) ** -0.5
        self.rescale_weight = nn.Parameter(torch.ones(self.head))
        self.rescale_bias = nn.Parameter(torch.zeros(self.head))
        self.epsilon = 1e-5

    def _get_gc(self, gap):
        return self.fc(gap)

    def forward(self,x): 
        b,c,w,h = x.size()
        x = rearrange(x,"b c x y -> b c (x y)")  
        gap = x.mean(dim=-1, keepdim=True) 
        q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap]) 
        sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale
        std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True) 
        sim = (sim-mean)/(std+self.epsilon) 
        sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
        sim = sim.reshape(b,self.head,1, w, h)
        gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b,self.head,-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gc = rearrange(sim*gc, "b h d x y -> b (h d) x y")
        return gc



class HAM(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__() 
        self.act = act_layer()

        self.gc1 = GlobalContext(dim, act_layer=act_layer) 
        self.dw1 = DWConv2D(dim, 11) 
        self.eca_layer1 = eca_layer(dim)

        self.fc1 = nn.Sequential( 
            nn.Conv2d(dim, max(dim//8, 16), 1),
            self.act,
            nn.Conv2d(max(dim//8, 16), dim, 1)
        )

        self.gc2 = GlobalContext(dim, act_layer=act_layer)
        self.dw2 = DWConv2D(dim, 11) 

        self.fc2 = nn.Sequential( 
            nn.Conv2d(dim, max(dim//8, 16), 1),
            self.act,
            nn.Conv2d(max(dim//8, 16), dim, 1)
        )

        self.IFFN = IFFN(dim=dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02) 
            if isinstance(m, nn.Linear) and m.bias is not None: 
                nn.init.constant_(m.bias, 0) 
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)): 
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)  
        elif isinstance(m, nn.Conv2d):  
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
            fan_out //= m.groups 
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) 
            if m.bias is not None: 
                m.bias.data.zero_()  

    def forward(self, x): 
        gc1 = self.gc1(x)  
        eca1 = self.eca_layer1(x)
        x = eca1 +  gc1 
        
        x = self.act(self.fc1(self.dw1(x))) 

        gc2 = self.gc2(x) 
        eca2 = self.eca_layer1(x)
        x = eca2 +gc2
        x = self.act(self.fc2(self.dw2(x)))

        x = self.IFFN(x)

        return x 



class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class ChannelAtt(nn.Module):
    def __init__(self, act_layer=nn.GELU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        size_1 =3
        size_2 =5
        self.channelConv1 = nn.Conv1d(1, 1, size_1, padding=size_1//2) 
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=size_2, padding=size_2//2) 
        self.act = act_layer()

    def forward(self, x):
        res = x.clone()
        x = self.avg_pool(self.act(x)) 
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2)) 
        x = self.act(x)  
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)  

        return res + x 

class Ch_att(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.act = act_layer()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)  
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.fc2 =nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)
        self.channel_att = ChannelAtt(act_layer=act_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.channel_att(x)
        return x

class encoder_module(nn.Module):

    def __init__(self, dim,  mlp_ratio=4.,
                 act_layer=nn.GELU, num_groups=64,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=1e-4):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, dim) 
        self.HAM = HAM(dim=dim, act_layer=act_layer) 
        self.norm2 = nn.GroupNorm(num_groups, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = Ch_att(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) 
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) 

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.HAM(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.channel_mixer(self.norm2(x))) 
        return x


class Channelatt(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()  
        hidden_dim = hidden_dim or dim 
        self.act = act_layer() 
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1) 
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim) 
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1) 
        self.drop = nn.Dropout(drop) 

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channelConv1 = nn.Conv1d(1, 1, 3, padding=1) 
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2) 

        self.apply(self._init_weights) 

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) 
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)): 
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): 
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            fan_out //= m.groups 
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) 
            if m.bias is not None:
                m.bias.data.zero_() 

    def forward(self, x): 
        x = self.fc1(x)  
        x = self.dwconv(x)  
        x = self.act(x)  
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x)

        res = x.clone()
        x = self.avg_pool(self.act(x)) 
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2)) 
        x = self.act(x)  
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)  

        return res + x 



class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = torch.nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class IMSA(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(IMSA, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FFN(dim, dim*ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Trans_chatt(nn.Module):

    def __init__(self, in_channel,num_heads=8,ffn_expansion_factor=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.IMSA = IMSA(dim=in_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_channel*2,out_channels=in_channel,kernel_size=1,stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.Channelatt = Channelatt(dim=in_channel, hidden_dim=in_channel, act_layer=nn.GELU, drop=0.)
    def forward(self, x):
        chatt = self.Channelatt(x)
        IMSA = self.IMSA(x)
        x = torch.cat([chatt, IMSA], 1)
        x = self.conv2(x)
        x = self.lrelu(self.Conv_out(x))
        return x


class Featurefusion1(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse):
        super(Featurefusion1, self).__init__()
        self.channels_fuse = channels_fuse
        self.conv_fuse = nn.Conv2d(2 * channels_fuse, 3* channels_fuse, kernel_size=1)  
        self.FFN = FFN(channels_fuse, channels_fuse, channels_fuse)
        self.softmax = nn.Softmax(dim=0)
        self.conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, kernel_size=1)

    def forward(self, x, y):
        residual = self.conv(torch.cat((x, y), dim=1))

        x_ori, y_ori = x, y
        xy_fuse = self.conv_fuse(torch.cat((x, y), 1))
        xy_split = torch.split(xy_fuse, self.channels_fuse, dim=1) 
        x1 = torch.sigmoid(self.FFN(xy_split[0]))
        x2 = torch.sigmoid(self.FFN(xy_split[1]))
        x3 = torch.sigmoid(self.FFN(xy_split[2])) 
        weights = self.softmax(torch.stack((x1, x2, x3), 0)) 
        out = weights[0] * x_ori + weights[1] * y_ori + weights[2] * residual
        return out





class FeatureFusion2(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeatureFusion2, self).__init__()
        self.fuse_conv1 = nn.Conv2d(4 * in_channels_list, out_channels, kernel_size=3, stride=1, padding=1)
        self.fuse_conv2 = nn.Conv2d(2 * in_channels_list, out_channels, kernel_size=3, stride=1, padding=1)
        self.fuse_conv3 = nn.Conv2d( in_channels_list, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):
        x1 = self.fuse_conv1(F.interpolate(x1, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False))
        x2 = self.fuse_conv2(F.interpolate(x2, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False))
        x3 = self.fuse_conv3(F.interpolate(x3, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False))
        x4 = self.fuse_conv3(F.interpolate(x4, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False))
        x = torch.cat([x1,x2,x3, x4], dim=1)
        return  x



class underFormer(nn.Module):
    def __init__(self,inp_channels=3,out_channels=3,dim=48,num_heads=[8,8,8,8],ffn_expansion_factor=2):
        super(underFormer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.embedding = nn.Conv2d(inp_channels*4,dim,kernel_size=3, stride=1, padding=1)
        self.encoder1 = encoder_module(dim=dim, mlp_ratio=8,num_groups=dim)
        self.down1 = Downsample(dim)
        self.encoder2 = encoder_module(dim=dim*2, mlp_ratio=8,num_groups=dim*2)
        self.down2 = Downsample(dim*2)
        self.encoder3 = encoder_module(dim=dim*4, num_groups=dim*4)
        self.down3 = Downsample(dim*4)
        self.enc_dec = Trans_chatt(dim*8,num_heads[3],ffn_expansion_factor)


        self.up1 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.decoder3 = Trans_DWcov(num_stages=1,num_path=[3],num_layers=[1],embed_dims=[dim*4],
                        mlp_ratios=[4],num_heads=[8],drop_path_rate=0.0,)
        
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.decoder2 = Trans_DWcov(num_stages=1,num_path=[3],num_layers=[1],embed_dims=[dim*2],
                        mlp_ratios=[8],num_heads=[8],drop_path_rate=0.0,)
        
        self.up3 = nn.ConvTranspose2d(dim*2, dim*1, 2, stride=2)
        self.Conv_c = nn.Conv2d(dim * 2, dim * 1, 1, 1)

        self.decoder1 = Trans_DWcov(num_stages=1,num_path=[3],num_layers=[1],
                        embed_dims=[dim],mlp_ratios=[8],num_heads=[8],drop_path_rate=0.0,)
        
        self.conv_out = nn.Conv2d(dim, out_channels*4, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        
        self.feature_fusion1 = Featurefusion1(dim*4, dim*4, dim*4)
        self.feature_fusion2 = Featurefusion1(dim*2, dim*2, dim*2)
        self.feature_fusion3 = Featurefusion1(dim, dim, dim)
        self.feature_fusion_final = FeatureFusion2(dim, out_channels*4)

    def forward(self, x):

        x = downshuffle(x, 2)
        x = self.embedding(x)

        encoder1 = self.encoder1(x)
        pool1 = self.down1(encoder1)

        encoder2 = self.encoder2(pool1)
        pool2 = self.down2(encoder2)

        encoder3 = self.encoder3(pool2)
        pool3 = self.down3(encoder3)

        enc_dec = self.enc_dec(pool3)

        up1 = self.up1(enc_dec)
        concat1 = self.feature_fusion1(up1, encoder3)
        decoder3 = self.decoder3(concat1)

        up2 = self.up2(decoder3)
        concat2 = self.feature_fusion2(up2, encoder2)
        decoder2 = self.decoder2(concat2)

        up3 = self.up3(decoder2)
        concat3 = torch.cat([up3, encoder1], 1)
        concat3 = self.Conv_c(concat3)
        decoder1 = self.decoder1(concat3)

        decoder1=self.feature_fusion_final(concat1,concat2,concat3,decoder1)

        conv_out = self.lrelu(self.conv_out(decoder1))
        
        out = self.pixelshuffle(conv_out)

        return out

#-------------------------------------

if __name__ == "__main__":
    model = underFormer(dim=48)
    ops, params = get_model_complexity_info(model, (3,128,128), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(ops, params)
    #print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))